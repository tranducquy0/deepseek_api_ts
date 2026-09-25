import { createHash } from "node:crypto";
import { Router, type Request, type Response } from "express";
import { DeepSeekClient, AuthExpiredError } from "../deepseek/client.js";
import type {
  AuthData,
  OpenAIChatRequest,
  OpenAIMessage,
  OpenAIToolCall,
} from "../shared/types.js";
import { SessionManager } from "./sessions.js";
import {
  buildPrompt,
  createStreamState,
  applyStreamEvent,
  makeChunk,
  makeToolCallChunk,
  extractToolCall,
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

// Tracks DeepSeek sessions across turns, keyed by conversation.
const sessions = new SessionManager();

/**
 * Derive a stable conversation key. Prefer the optional `user` field
 * (the OpenAI convention for a stable identity); otherwise fall back to
 * a hash of the first user message, which never changes across turns.
 */
function conversationKey(body: OpenAIChatRequest): string {
  if (typeof body.user === "string" && body.user.length > 0) return body.user;
  const first = body.messages.find((m) => m.role === "user");
  const raw = first?.content ?? "";
  const seed = Array.isArray(raw)
    ? raw.map((p) => (typeof p === "string" ? p : p.text ?? "")).join("")
    : raw;
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
export function forwardMessages(
  messages: OpenAIMessage[],
  forwarded: number
): OpenAIMessage[] {
  let delta = messages.slice(forwarded);
  if (forwarded > 0) {
    while (delta.length > 0 && delta[0].role === "assistant") {
      delta = delta.slice(1);
    }
  }
  return delta.length > 0 ? delta : messages.slice(-1);
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
    const stream = body.stream ?? true;

    try {
      const convKey = conversationKey(body);
      const entry = await sessions.getOrCreate(client, convKey);
      const prompt = buildPrompt(
        forwardMessages(body.messages, entry.lastMessageCount),
        body.tools,
        body.tool_choice
      );
      const hasTools = !!body.tools?.length;
      let parentId: string | null = entry.parentMessageId;

      if (stream) {
        // ── Streaming response ──────────────────────────────
        res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");

        // Send initial role chunk
        const initChunk = makeChunk(modelId, { role: "assistant" }, null);
        res.write(`data: ${JSON.stringify(initChunk)}\n\n`);

        const state = createStreamState(hasTools);

        for await (const event of client.chatCompletion({
          chatSessionId: entry.sessionId,
          parentMessageId: parentId,
          prompt,
          thinkingEnabled,
          modelType: dsModelType,
        })) {
          const delta = applyStreamEvent(state, event);

          // Capture parent_message_id for subsequent turns
          if (event.p === "response/message_id" && event.v) {
            parentId = String(event.v);
          }

          if (delta) {
            const chunk = makeChunk(modelId, { content: delta });
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          }

          if (state.finished) break;
        }

        sessions.update(convKey, {
          parentMessageId:
            state.responseMessageId != null
              ? String(state.responseMessageId)
              : parentId,
          messageCount: body.messages.length,
        });

        // In tool mode, inspect the full buffered state
        const toolCall = extractToolCall(state);
        if (toolCall) {
          if (toolCall.leadingText) {
            const leadingChunk = makeChunk(modelId, { content: toolCall.leadingText });
            res.write(`data: ${JSON.stringify(leadingChunk)}\n\n`);
          }
          const chunk = makeToolCallChunk(modelId, toolCall);
          res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        } else if (state.hasTools && state.content) {
          const chunk = makeChunk(modelId, { content: state.content });
          res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        }

        // Send final chunk
        const finishReason = toolCall ? "tool_calls" : "stop";
        const finalChunk = makeChunk(modelId, {}, finishReason);
        res.write(`data: ${JSON.stringify(finalChunk)}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
      } else {
        // ── Non-streaming response ──────────────────────────
        const state = createStreamState(hasTools);

        for await (const event of client.chatCompletion({
          chatSessionId: entry.sessionId,
          parentMessageId: parentId,
          prompt,
          thinkingEnabled,
          modelType: dsModelType,
        })) {
          applyStreamEvent(state, event);
          if (event.p === "response/message_id" && event.v) {
            parentId = String(event.v);
          }
          if (state.finished) break;
        }

        sessions.update(convKey, {
          parentMessageId:
            state.responseMessageId != null
              ? String(state.responseMessageId)
              : parentId,
          messageCount: body.messages.length,
        });

        const toolCall = extractToolCall(state);
        const message: OpenAIMessage =
          toolCall
            ? {
                role: "assistant",
                content: toolCall.leadingText || null,
                tool_calls: [
                  {
                    id: toolCall.id,
                    type: "function",
                    function: {
                      name: toolCall.name,
                      arguments: toolCall.arguments,
                    },
                  } satisfies OpenAIToolCall,
                ],
              }
            : { role: "assistant", content: state.content };
        const finishReason = toolCall ? "tool_calls" : "stop";

        res.json({
          id: `chatcmpl-${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: modelId,
          choices: [
            {
              index: 0,
              message,
              finish_reason: finishReason,
            },
          ],
          usage: {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
          },
        });
      }
    } catch (err) {
      const authExpired = err instanceof AuthExpiredError;
      const message = authExpired
        ? "DeepSeek auth expired. Run `ds auth` to re-authenticate."
        : (err as Error).message ?? "Internal server error";
      const type = authExpired ? "auth_error" : "server_error";

      console.error("Chat completion error:", err);

      if (res.headersSent) {
        // Already streaming — emit an SSE error event and close
        res.write(`data: ${JSON.stringify({ error: { message, type } })}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
        return;
      }

      res.status(authExpired ? 401 : 500).json({ error: { message, type } });
    }
  });

  return router;
}
