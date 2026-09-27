import { createHash } from "node:crypto";
import { Router, type Request, type Response } from "express";
import { DeepSeekClient, AuthExpiredError } from "../deepseek/client.js";
import type {
  AuthData,
  CompletionUsage,
  OpenAIChatRequest,
  OpenAIMessage,
  OpenAIToolCall,
} from "../shared/types.js";
import { SessionManager } from "./sessions.js";
import { KeyedSerialQueue } from "./queue.js";
import {
  buildPrompt,
  checkToolChoice,
  contentToText,
  createStreamState,
  applyStreamEvent,
  extractToolCalls,
  fingerprintMessages,
  makeChunk,
  makeToolCallChunk,
  makeUsageChunk,
  newCompletionMeta,
  mapModel,
} from "../shared/convert.js";

/**
 * Coerce the client-supplied `thinking` field to a strict boolean.
 * OpenAI-compatible clients may send objects/maps, strings, or numbers,
 * which DeepSeek's API rejects when passed through as-is.
 */
function resolveThinking(body: OpenAIChatRequest, modelId: string): boolean {
  const v = (body as unknown as { thinking?: unknown }).thinking;
  if (typeof v === "boolean") return v;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (s === "true" || s === "1") return true;
    if (s === "false" || s === "0") return false;
  }
  if (typeof v === "number") return v !== 0;
  return modelId === "deepseek-reasoner";
}

/** OpenAI-shaped error payload used for both HTTP and SSE failures. */
function errorBody(message: string, type: string, code?: string) {
  return { error: { message, type, ...(code ? { code } : {}) } };
}

/**
 * Per-request usage for a turn. DeepSeek reports tokens accumulated over the
 * whole chat session, so the total is diffed against the previous turn;
 * otherwise a long agent run re-reports every earlier turn.
 */
function requestUsage(
  sessions: SessionManager,
  convKey: string,
  usage: CompletionUsage | null
): CompletionUsage | null {
  if (!usage) return null;
  const total = sessions.usageDelta(convKey, usage.total_tokens);
  return total == null ? null : { ...usage, total_tokens: total };
}

// Tracks DeepSeek sessions across turns, keyed by conversation.
const sessions = new SessionManager();

// One turn at a time per conversation, so concurrent requests cannot race on
// the session's parent pointer and forwarding state.
const turns = new KeyedSerialQueue();

function extractExplicitSessionId(body: OpenAIChatRequest): string | undefined {
  const req = body as unknown as {
    chat_session_id?: unknown;
    chatSessionId?: unknown;
    session_id?: unknown;
  };
  for (const field of [req.chat_session_id, req.chatSessionId, req.session_id]) {
    if (typeof field === "string" && field.trim().length > 0) {
      return field.trim();
    }
  }
  return undefined;
}

/**
 * Derive a stable conversation key. Prefer explicit chat session IDs or
 * the optional `user` field (the OpenAI convention for a stable identity);
 * otherwise fall back to a hash of the first user message.
 */
function conversationKey(body: OpenAIChatRequest): string {
  const explicit = extractExplicitSessionId(body);
  if (explicit) return explicit;
  if (typeof body.user === "string" && body.user.length > 0) return body.user;
  const first = body.messages.find((m) => m.role === "user");
  // contentToText keeps the key stable whether the client sends a plain string
  // or multimodal content parts, so a turn cannot fork into a new session.
  const seed = contentToText(first?.content);
  return createHash("sha256").update(seed).digest("hex").slice(0, 32);
}

/**
 * Select only the messages the DeepSeek session has not forwarded yet,
 * so history is not duplicated on every turn. Falls back to the last
 * message when nothing new is pending (e.g. the client re-sent history).
 *
 * When continuing an existing session (`forwarded > 0`), leading assistant
 * messages in `delta` are skipped because the DeepSeek session already holds
 * the assistant's previous response as the parent message node.
 */
export interface ForwardPlan {
  /** The messages to send to DeepSeek this turn. */
  delta: OpenAIMessage[];
  /**
   * True when the client rewrote history it had already sent, so the DeepSeek
   * session no longer describes it and must be rebuilt from scratch.
   */
  replay: boolean;
}

/**
 * Decide which messages to forward, given what the DeepSeek session has
 * already seen.
 *
 * Slicing by message count alone silently corrupts a conversation whose client
 * compacts its history: the slice lands in the wrong place, and the fallback
 * drops the user's actual new turn. So the already-forwarded prefix is
 * fingerprinted and compared — an exact match means the client only appended,
 * and anything else means history was rewritten.
 *
 * When continuing a session, leading assistant messages in `delta` are skipped
 * because the DeepSeek session already holds the assistant's previous response
 * as the parent message node. A replay has no such parent, so it forwards
 * everything.
 */
export function planForward(
  messages: OpenAIMessage[],
  entry: { lastMessageCount: number; historyDigest: string | null }
): ForwardPlan {
  const forwarded = entry.lastMessageCount;
  if (forwarded === 0) return { delta: messages, replay: false };

  const prefix = messages.slice(0, forwarded);
  const intact =
    prefix.length === forwarded &&
    entry.historyDigest !== null &&
    fingerprintMessages(prefix) === entry.historyDigest;

  if (!intact) return { delta: messages, replay: true };

  let delta = messages.slice(forwarded);
  while (delta.length > 0 && delta[0].role === "assistant") {
    delta = delta.slice(1);
  }
  // Nothing new was appended (e.g. the client retried): re-ask the last turn.
  return { delta: delta.length > 0 ? delta : messages.slice(-1), replay: false };
}

export function chatRouter(getClient: () => DeepSeekClient): Router {
  const router = Router();

  router.post("/v1/chat/completions", async (req: Request, res: Response) => {
    const body = req.body as OpenAIChatRequest;

    if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
      res.status(400).json({ error: { message: "messages[] is required" } });
      return;
    }

    const client = getClient();
    const modelId = body.model ?? "deepseek-chat";
    const dsModelType = mapModel(modelId);
    const thinkingEnabled = resolveThinking(body, modelId);
    // Per the OpenAI spec the default is a single JSON response; clients that
    // omit `stream` cannot parse an SSE body.
    const stream = body.stream ?? false;

    // A tool_choice that names a function is meaningless without tools.
    const needsTools =
      body.tool_choice === "required" ||
      (typeof body.tool_choice === "object" && !!body.tool_choice?.function?.name);
    if (needsTools && !body.tools?.length) {
      res.status(400).json(
        errorBody(
          `tool_choice is ${JSON.stringify(body.tool_choice)} but no tools were provided`,
          "invalid_request_error",
          "tool_choice_without_tools"
        )
      );
      return;
    }

    const convKey = conversationKey(body);
    const abort = new AbortController();
    let clientGone = false;
    // Fires on a normal res.end() too, so the finished check is what
    // distinguishes a disconnect from a completed response.
    const onClose = () => {
      if (!res.writableFinished) {
        clientGone = true;
        abort.abort();
      }
    };
    res.on("close", onClose);

    try {
      await turns.run(convKey, async () => {
        const explicitSessionId = extractExplicitSessionId(body);
        let entry = await sessions.getOrCreate(client, convKey, explicitSessionId);
        let plan = planForward(body.messages, entry);

        if (plan.replay) {
          // The client rewrote history it had already sent, so the DeepSeek
          // session no longer matches it. Start a fresh session and replay
          // everything, which keeps the context instead of silently sending a
          // misaligned slice.
          entry = await sessions.restart(client, convKey, explicitSessionId);
          plan = { delta: body.messages, replay: false };
        }

        const prompt = buildPrompt(plan.delta, body.tools, body.tool_choice);
        const hasTools = !!body.tools?.length;
        let parentId: number | null = entry.parentMessageId;
        // One id/created pair for the whole completion, as OpenAI does.
        const completion = newCompletionMeta();
        const chainState = {
          messageCount: body.messages.length,
          historyDigest: fingerprintMessages(body.messages),
        };
        let chained = false;

        /** Persist the chaining state once DeepSeek reveals the response id. */
        const chainIfKnown = (responseMessageId: number | null): void => {
          if (chained || responseMessageId == null) return;
          chained = true;
          parentId = responseMessageId;
          // Persisted immediately rather than after the stream finishes: a turn
          // cut short by a client disconnect has still created a message in the
          // DeepSeek session, and the next turn must chain from it instead of
          // orphaning it.
          sessions.update(convKey, { ...chainState, parentMessageId: responseMessageId });
        };

        if (stream) {
          // ── Streaming response ──────────────────────────────
          res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
          res.setHeader("Cache-Control", "no-cache");
          res.setHeader("Connection", "keep-alive");
          res.setHeader("X-Accel-Buffering", "no");

          // Send initial role chunk
          const initChunk = makeChunk(completion, modelId, { role: "assistant" });
          res.write(`data: ${JSON.stringify(initChunk)}\n\n`);

          const state = createStreamState(hasTools, body.tools);

          for await (const event of client.chatCompletion({
            chatSessionId: entry.sessionId,
            parentMessageId: parentId,
            prompt,
            thinkingEnabled,
            modelType: dsModelType,
            signal: abort.signal,
          })) {
            const delta = applyStreamEvent(state, event);

            // DeepSeek reports the response id on its opening frame, long before
            // the answer finishes.
            chainIfKnown(state.responseMessageId);

            if (delta.content || delta.reasoning) {
              const chunk = makeChunk(completion, modelId, {
                ...(delta.content ? { content: delta.content } : {}),
                ...(delta.reasoning ? { reasoning_content: delta.reasoning } : {}),
              });
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            }
          }

          sessions.update(convKey, {
            parentMessageId:
              state.responseMessageId != null
                ? state.responseMessageId
                : parentId,
            ...chainState,
          });

          // In tool mode, the buffered response is only interpretable once the
          // model has finished emitting it.
          const parsed = extractToolCalls(state);
          const usage = requestUsage(sessions, convKey, state.usage);
          const choiceError = checkToolChoice(body.tool_choice, body.tools ?? [], parsed);

          // A malformed tool call must not look like a finished turn: the SDK
          // raises APIError on an `error` frame, so the loop stops loudly.
          if (parsed.error) {
            res.write(
              `data: ${JSON.stringify(
                errorBody(parsed.error, "upstream_error", "tool_call_parse_failed")
              )}\n\n`
            );
            res.end();
            return;
          }
          if (choiceError) {
            res.write(
              `data: ${JSON.stringify(
                errorBody(choiceError, "upstream_error", "tool_choice_violation")
              )}\n\n`
            );
            res.end();
            return;
          }

          if (parsed.calls.length > 0) {
            if (parsed.leadingText) {
              const leadingChunk = makeChunk(completion, modelId, {
                content: parsed.leadingText,
              });
              res.write(`data: ${JSON.stringify(leadingChunk)}\n\n`);
            }
            const chunk = makeToolCallChunk(completion, modelId, parsed.calls);
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          } else if (state.hasTools && state.content) {
            const chunk = makeChunk(completion, modelId, { content: state.content });
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          }

          if (body.stream_options?.include_usage && usage) {
            const usageChunk = makeUsageChunk(completion, modelId, usage);
            res.write(`data: ${JSON.stringify(usageChunk)}\n\n`);
          }

          // Send final chunk. finish_reason is set exactly once per stream.
          const finishReason = parsed.calls.length > 0 ? "tool_calls" : "stop";
          const finalChunk = makeChunk(completion, modelId, {}, finishReason);
          res.write(`data: ${JSON.stringify(finalChunk)}\n\n`);
          res.write("data: [DONE]\n\n");
          res.end();
        } else {
          // ── Non-streaming response ──────────────────────────
          const state = createStreamState(hasTools, body.tools);

          for await (const event of client.chatCompletion({
            chatSessionId: entry.sessionId,
            parentMessageId: parentId,
            prompt,
            thinkingEnabled,
            modelType: dsModelType,
            signal: abort.signal,
          })) {
            applyStreamEvent(state, event);
            chainIfKnown(state.responseMessageId);
          }

          sessions.update(convKey, {
            parentMessageId:
              state.responseMessageId != null
                ? state.responseMessageId
                : parentId,
            ...chainState,
          });

          const parsed = extractToolCalls(state);
          const usage = requestUsage(sessions, convKey, state.usage);

          // Fail loudly rather than returning a clean stop that an agent loop
          // would read as "the model finished without calling a tool".
          if (parsed.error) {
            res
              .status(502)
              .json(
                errorBody(parsed.error, "upstream_error", "tool_call_parse_failed")
              );
            return;
          }

          const choiceError = checkToolChoice(body.tool_choice, body.tools ?? [], parsed);
          if (choiceError) {
            res
              .status(502)
              .json(errorBody(choiceError, "upstream_error", "tool_choice_violation"));
            return;
          }

          const reasoning = state.thinking || undefined;
          const message: OpenAIMessage =
            parsed.calls.length > 0
              ? {
                  role: "assistant",
                  content: parsed.leadingText || null,
                  ...(reasoning ? { reasoning_content: reasoning } : {}),
                  tool_calls: parsed.calls.map(
                    (tool) =>
                      ({
                        id: tool.id,
                        type: "function",
                        function: {
                          name: tool.name,
                          arguments: tool.arguments,
                        },
                      }) satisfies OpenAIToolCall
                  ),
                }
              : {
                  role: "assistant",
                  content: state.content,
                  ...(reasoning ? { reasoning_content: reasoning } : {}),
                };
          const finishReason = parsed.calls.length > 0 ? "tool_calls" : "stop";

          res.json({
            id: completion.id,
            object: "chat.completion",
            created: completion.created,
            model: modelId,
            choices: [
              {
                index: 0,
                message,
                finish_reason: finishReason,
              },
            ],
            usage: usage ?? {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0,
            },
          });
        }
      });
    } catch (err) {
      // A client that hung up is not a broken conversation: the response id was
      // already chained when it arrived, so the next turn continues correctly.
      // Resetting here would throw that context away.
      if (!clientGone) sessions.reset(convKey);

      const authExpired = err instanceof AuthExpiredError;
      const message = authExpired
        ? "DeepSeek auth expired. Run `ds auth` to re-authenticate."
        : (err as Error).message ?? "Internal server error";
      const type = authExpired ? "auth_error" : "server_error";

      if (!clientGone) console.error("Chat completion error:", err);

      if (clientGone) {
        res.off("close", onClose);
        return;
      }

      if (res.headersSent) {
        // Already streaming. An `error` frame makes the SDK raise APIError;
        // deliberately no [DONE], so a client cannot mistake a failure for a
        // cleanly finished turn.
        res.write(`data: ${JSON.stringify(errorBody(message, type))}\n\n`);
        res.end();
        return;
      }

      res
        .status(authExpired ? 401 : 500)
        .json(errorBody(message, type));
    } finally {
      res.off("close", onClose);
    }
  });

  return router;
}
