import { randomUUID } from "node:crypto";
import type {
  DSStreamEvent,
  OpenAIChunk,
  OpenAIChatRequest,
  OpenAITool,
  OpenAIToolCall,
  OpenAIModelList,
  OpenAIModel,
  OpenAIMessage,
} from "./types.js";
import { MODELS, MODEL_MAP, type ModelId } from "./config.js";

// ── Prompt builder ──────────────────────────────────────────────────

/**
 * Convert OpenAI messages[] (and optional tool definitions) into a
 * single DeepSeek prompt string. System messages are prepended,
 * conversation history is flattened, and tool definitions are injected
 * as a structured instruction block.
 */
export function buildPrompt(
  messages: OpenAIMessage[],
  tools?: OpenAITool[],
  toolChoice?: OpenAIChatRequest["tool_choice"]
): string {
  const parts: string[] = [];

  const toolBlock = buildToolBlock(tools, toolChoice);
  if (toolBlock) parts.push(toolBlock);

  for (const msg of messages) {
    switch (msg.role) {
      case "system":
        if (msg.content) parts.push(`[System]: ${msg.content}`);
        break;
      case "user":
        parts.push(msg.content ?? "");
        break;
      case "assistant":
        if (msg.tool_calls?.length) {
          for (const tc of msg.tool_calls) {
            parts.push(
              `[Tool call]: ${tc.function.name}(${tc.function.arguments})`
            );
          }
        }
        if (msg.content) parts.push(`[Assistant]: ${msg.content}`);
        break;
      case "tool":
        parts.push(`[Tool result for ${msg.tool_call_id ?? "tool"}]: ${msg.content ?? ""}`);
        break;
    }
  }

  return parts.join("\n\n");
}

/**
 * Build the tool-calling instruction block injected into the prompt.
 * DeepSeek's web API has no native tool support, so functions are
 * described to the model and calls are made via structured JSON output.
 */
function buildToolBlock(
  tools: OpenAITool[] | undefined,
  toolChoice: OpenAIChatRequest["tool_choice"]
): string {
  if (!tools || tools.length === 0) return "";

  const lines = [
    "# Available Tools",
    "You can call a function by responding with EXACTLY one JSON object on its own line:",
    '{"name": "<function name>", "arguments": { ... }}',
    "",
    "Rules:",
    "- The arguments object must match the function's JSON schema.",
    "- Do not wrap it in markdown, add extra text, or explain.",
    "- Call a tool only when the user's request requires it.",
  ];

  switch (toolChoice) {
    case "none":
      lines.push("- Do NOT call any tool in this turn; answer directly.");
      break;
    case "required":
    case "auto":
      break;
    default:
      if (
        typeof toolChoice === "object" &&
        toolChoice?.function?.name
      ) {
        lines.push(
          `- You MUST call the function "${toolChoice.function.name}" in this turn.`
        );
      }
  }

  lines.push(
    "",
    "Functions:",
    JSON.stringify(tools.map((t) => t.function), null, 2)
  );

  return lines.join("\n");
}

// ── SSE → OpenAI chunk converter ────────────────────────────────────

export interface DSStreamState {
  content: string;
  thinking: string;
  /** True when the request included tool definitions — content is buffered instead of streamed */
  hasTools: boolean;
  /** The DeepSeek response message ID (for parent_message_id chaining) */
  responseMessageId: number | null;
  finished: boolean;
}

/**
 * Initialize a new stream state.
 */
export function createStreamState(hasTools = false): DSStreamState {
  return {
    content: "",
    thinking: "",
    hasTools,
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
      // In tool mode, buffer content and emit it (raw JSON) only at the end.
      return state.hasTools ? "" : event.v;
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

// ── Tool call parsing ───────────────────────────────────────────────

export interface ParsedToolCall {
  id: string;
  name: string;
  arguments: string;
}

/**
 * Parse a DeepSeek completion as a structured tool call.
 * The model is instructed to emit a single JSON object; tolerate
 * surrounding whitespace and markdown fences.
 */
export function parseToolCall(content: string): ParsedToolCall | null {
  let text = content.trim();
  text = text.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");

  const start = text.indexOf("{");
  if (start === -1) return null;

  const end = text.lastIndexOf("}");
  if (end === -1 || end < start) return null;

  try {
    const obj = JSON.parse(text.slice(start, end + 1)) as {
      name?: unknown;
      arguments?: unknown;
      params?: unknown;
      parameters?: unknown;
    };
    if (typeof obj.name !== "string" || obj.name.length === 0) return null;

    const args = obj.arguments ?? obj.params ?? obj.parameters ?? {};
    const argsStr =
      typeof args === "string" ? args : JSON.stringify(args ?? {});

    return {
      id: `call_${randomUUID().replace(/-/g, "").slice(0, 24)}`,
      name: obj.name,
      arguments: argsStr,
    };
  } catch {
    return null;
  }
}

/**
 * Create an OpenAI-compatible streaming chunk carrying a tool call.
 */
export function makeToolCallChunk(
  model: string,
  tool: ParsedToolCall
): OpenAIChunk {
  return makeChunk(
    model,
    {
      tool_calls: [
        {
          index: 0,
          id: tool.id,
          type: "function",
          function: { name: tool.name, arguments: tool.arguments },
        } satisfies OpenAIToolCall,
      ],
    },
    "tool_calls"
  );
}

/**
 * Extract a tool call (if any) from buffered state content.
 */
export function extractToolCall(state: DSStreamState): ParsedToolCall | null {
  return state.hasTools ? parseToolCall(state.content) : null;
}

/**
 * Create an OpenAI-compatible chunk from a delta.
 */
export function makeChunk(
  model: string,
  delta: Partial<OpenAIMessage>,
  finishReason: "stop" | "length" | "tool_calls" | null = null,
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
