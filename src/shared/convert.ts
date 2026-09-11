import { randomUUID } from "node:crypto";
import type {
  DSStreamEvent,
  OpenAIChunk,
  OpenAIChatRequest,
  OpenAIModelList,
  OpenAIModel,
  OpenAIMessage,
} from "./types.js";
import { MODELS, MODEL_MAP, type ModelId } from "./config.js";

// ── Prompt builder ──────────────────────────────────────────────────

/**
 * Convert OpenAI messages[] into a single DeepSeek prompt string.
 * System messages are prepended, conversation history is flattened.
 */
export function buildPrompt(messages: OpenAIMessage[]): string {
  const parts: string[] = [];

  for (const msg of messages) {
    switch (msg.role) {
      case "system":
        parts.push(`[System]: ${msg.content}`);
        break;
      case "user":
        parts.push(msg.content);
        break;
      case "assistant":
        parts.push(`[Assistant]: ${msg.content}`);
        break;
    }
  }

  return parts.join("\n\n");
}

// ── SSE → OpenAI chunk converter ────────────────────────────────────

export interface DSStreamState {
  content: string;
  thinking: string;
  /** The DeepSeek response message ID (for parent_message_id chaining) */
  responseMessageId: number | null;
  finished: boolean;
}

/**
 * Initialize a new stream state.
 */
export function createStreamState(): DSStreamState {
  return {
    content: "",
    thinking: "",
    responseMessageId: null,
    finished: false,
  };
}

/**
 * Apply a DeepSeek stream event to the state.
 * Returns the delta content (if any) for this event.
 */
export function applyStreamEvent(
  state: DSStreamState,
  event: DSStreamEvent
): string {
  // Capture response message ID for multi-turn chaining
  if (event.response_message_id != null) {
    state.responseMessageId = event.response_message_id;
  }

  if (!event.p || !event.o) return "";

  // Check for finish
  if (event.p === "response/status" && event.v === "FINISHED") {
    state.finished = true;
    return "";
  }

  // Handle content append
  if (event.p.includes("/content") && !event.p.includes("thinking")) {
    if (event.o === "APPEND" && typeof event.v === "string") {
      state.content += event.v;
      return event.v;
    }
  }

  // Handle thinking content (we collect it but don't emit as content)
  if (event.p.includes("thinking_content")) {
    if (event.o === "APPEND" && typeof event.v === "string") {
      state.thinking += event.v;
    }
  }

  return "";
}

/**
 * Create an OpenAI-compatible chunk from a delta.
 */
export function makeChunk(
  model: string,
  delta: Partial<OpenAIMessage>,
  finishReason: "stop" | "length" | null = null,
  index = 0
): OpenAIChunk {
  return {
    id: `chatcmpl-${randomUUID().replace(/-/g, "").slice(0, 24)}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index,
        delta,
        finish_reason: finishReason,
      },
    ],
  };
}

// ── Model list ──────────────────────────────────────────────────────

export function buildModelList(): OpenAIModelList {
  const data: OpenAIModel[] = MODELS.map((id): OpenAIModel => ({
    id,
    object: "model",
    created: 1700000000,
    owned_by: "deepseek",
  }));

  return { object: "list", data };
}

/**
 * Map an OpenAI model ID to DeepSeek's internal model_type.
 */
export function mapModel(model: string): string {
  const key = model as ModelId;
  return MODEL_MAP[key]?.ds ?? "default";
}
