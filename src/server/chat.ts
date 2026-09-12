import { Router, type Request, type Response } from "express";
import { DeepSeekClient, AuthExpiredError } from "../deepseek/client.js";
import type {
  AuthData,
  OpenAIChatRequest,
  OpenAIMessage,
  OpenAIToolCall,
} from "../shared/types.js";
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

// Session map: tracks DeepSeek session IDs for multi-turn conversations
// Keyed by a conversation hash derived from the first message
const sessionMap = new Map<string, string>();
const SESSION_TTL = 60 * 60 * 1000; // 1 hour
const sessionTimestamps = new Map<string, number>();

function getOrCreateSession(
  client: DeepSeekClient,
  conversationKey: string
): Promise<string> {
  const existing = sessionMap.get(conversationKey);
  if (existing) {
    const ts = sessionTimestamps.get(conversationKey) ?? 0;
    if (Date.now() - ts < SESSION_TTL) return Promise.resolve(existing);
  }
  return client.createSession().then((id) => {
    sessionMap.set(conversationKey, id);
    sessionTimestamps.set(conversationKey, Date.now());
    return id;
  });
}

function cleanupSessions(): void {
  const now = Date.now();
  for (const [key, ts] of sessionTimestamps) {
    if (now - ts > SESSION_TTL) {
      sessionMap.delete(key);
      sessionTimestamps.delete(key);
    }
  }
}

// Periodic cleanup
setInterval(cleanupSessions, 5 * 60 * 1000);

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
      // Build a conversation key from the first user message
      const convKey = body.messages
        .filter((m: OpenAIMessage) => m.role === "user")
        .map((m: OpenAIMessage) => m.content ?? "")
        .join("|")
        .slice(0, 100);

      const sessionId = await getOrCreateSession(client, convKey);
      const prompt = buildPrompt(body.messages, body.tools, body.tool_choice);
      const hasTools = !!body.tools?.length;

      if (stream) {
        // ── Streaming response ──────────────────────────────
        res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");

        // Send initial role chunk
        const initChunk = makeChunk(modelId, { role: "assistant" }, null);
        res.write(`data: ${JSON.stringify(initChunk)}\n\n`);

        let parentId: string | null = null;
        const state = createStreamState(hasTools);

        for await (const event of client.chatCompletion({
          chatSessionId: sessionId,
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

        // In tool mode the raw JSON was buffered — emit it as a tool call
        const toolCall = extractToolCall(state);
        if (toolCall) {
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
        let parentId: string | null = null;
        const state = createStreamState(hasTools);

        for await (const event of client.chatCompletion({
          chatSessionId: sessionId,
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

        const toolCall = extractToolCall(state);
        const message: OpenAIMessage =
          toolCall
            ? {
                role: "assistant",
                content: null,
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
