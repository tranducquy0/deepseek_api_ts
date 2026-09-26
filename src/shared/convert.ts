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
        if (msg.content) parts.push(`[Assistant]: ${msg.content}`);
        if (msg.tool_calls?.length) {
          for (const tc of msg.tool_calls) {
            let argsObj: unknown = tc.function.arguments;
            if (typeof tc.function.arguments === "string") {
              try {
                argsObj = JSON.parse(tc.function.arguments);
              } catch {
                argsObj = tc.function.arguments;
              }
            }
            parts.push(
              JSON.stringify({
                name: tc.function.name,
                arguments: argsObj,
              })
            );
          }
        }
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

/** A DeepSeek response fragment (one THINK / RESPONSE / SEARCH segment). */
export interface DSFragment {
  id?: number;
  type?: string;
  content?: string;
}

export interface DSStreamState {
  content: string;
  thinking: string;
  /** True when the request included tool definitions */
  hasTools: boolean;
  /** The DeepSeek response message ID (for parent_message_id chaining) */
  responseMessageId: number | null;
  finished: boolean;
  /**
   * Last explicitly declared patch path. DeepSeek's stream is a JSON patch:
   * `p`/`o` are declared once and every following bare `{"v":…}` frame reuses
   * them, so the path has to be remembered across events.
   */
  path: string | null;
  /** Last explicitly declared operation ("APPEND" | "SET" | "REPLACE" | …). */
  op: string | null;
  /** Fragments announced so far; the `fragments/-1` path resolves to the last. */
  fragments: DSFragment[];
  /** True when the active fragment is a reasoning (THINK) fragment. */
  reasoning: boolean;
  /** The opening snapshot has already been consumed. */
  seeded: boolean;
}

/** Text produced by a single stream event. */
export interface StreamDelta {
  content: string;
  reasoning: string;
}

const NO_DELTA: StreamDelta = { content: "", reasoning: "" };

/** `response/fragments/<index>/content` — where token deltas arrive. */
const FRAGMENT_CONTENT_PATH = /^response\/fragments\/(-?\d+)\/content$/;

/** Path used for a fragment's text once the fragment has been announced. */
const CURRENT_FRAGMENT_CONTENT = "response/fragments/-1/content";

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
    path: null,
    op: null,
    fragments: [],
    reasoning: false,
    seeded: false,
  };
}

/**
 * Apply a DeepSeek stream event to the state.
 * Returns the text this event added (empty strings when it carried none).
 */
export function applyStreamEvent(
  state: DSStreamState,
  event: DSStreamEvent
): StreamDelta {
  // Capture response message ID for multi-turn chaining
  if (event.response_message_id != null) {
    state.responseMessageId = event.response_message_id;
  }

  // Opening snapshot: {"v":{"response":{…,"fragments":[{type,content}]}}}
  if (
    !state.seeded &&
    !event.p &&
    state.content === "" &&
    state.thinking === "" &&
    isSnapshot(event.v)
  ) {
    state.seeded = true;
    return seedFromSnapshot(state, event.v as { response?: { fragments?: unknown } });
  }

  // An explicit `p` re-anchors the patch; bare frames inherit the last one.
  if (typeof event.p === "string" && event.p.length > 0) {
    state.path = event.p;
    state.op = typeof event.o === "string" ? event.o : null;
  }
  const path = state.path;
  if (!path) return NO_DELTA;
  const op = state.op ?? undefined;

  // Batch: {"p":"response","o":"BATCH","v":[{"p":"…/…","v":…}, …]}
  if (event.o === "BATCH" && Array.isArray(event.v)) {
    for (const sub of event.v as DSStreamEvent[]) {
      if (sub && typeof sub === "object" && typeof sub.p === "string") {
        applyStreamEvent(state, {
          p: `${path}/${sub.p}`,
          o: sub.o,
          v: sub.v,
        });
      }
    }
    return NO_DELTA;
  }

  if (path === "response/status") {
    if (event.v === "FINISHED") state.finished = true;
    return NO_DELTA;
  }

  if (path === "response/fragments") {
    return applyFragmentList(state, event);
  }

  const fragmentMatch = FRAGMENT_CONTENT_PATH.exec(path);
  if (fragmentMatch) {
    state.reasoning = fragmentAt(state, Number(fragmentMatch[1]))?.type === "THINK";
    return applyTextDelta(state, event.v, op, state.reasoning);
  }

  // Legacy/alternate path for reasoning text.
  if (path.endsWith("/thinking_content")) {
    state.reasoning = true;
    return applyTextDelta(state, event.v, op, true);
  }

  // Whole-message content path.
  if (path === "response/content") {
    state.reasoning = false;
    return applyTextDelta(state, event.v, op, false);
  }

  // Anything else (elapsed_secs, accumulated_token_usage, title, …).
  return NO_DELTA;
}

/** True for the full response snapshot frame that opens the stream. */
function isSnapshot(v: unknown): v is { response?: { fragments?: unknown } } {
  return (
    typeof v === "object" &&
    v !== null &&
    !Array.isArray(v) &&
    typeof (v as { response?: unknown }).response === "object"
  );
}

/**
 * Seed buffers from the opening snapshot. Its first fragment already holds the
 * opening token(s), which DeepSeek never re-sends as deltas.
 */
function seedFromSnapshot(
  state: DSStreamState,
  snapshot: { response?: { fragments?: unknown } }
): StreamDelta {
  const raw = snapshot.response?.fragments;
  if (!Array.isArray(raw) || raw.length === 0) return NO_DELTA;

  state.fragments = (raw as DSFragment[]).slice();
  const last = state.fragments[state.fragments.length - 1];
  state.reasoning = last?.type === "THINK";
  return applyTextDelta(state, last?.content, state.op ?? undefined, state.reasoning);
}

/**
 * Handle `response/fragments` announcements. Each new fragment arrives with the
 * text produced so far — those tokens are never streamed separately, so they
 * are emitted here instead of being lost.
 */
function applyFragmentList(state: DSStreamState, event: DSStreamEvent): StreamDelta {
  const incoming = Array.isArray(event.v) ? (event.v as DSFragment[]) : [];

  // A SET carries the full list, i.e. text that was already streamed.
  if (event.o === "SET") {
    state.fragments = incoming.slice();
    return NO_DELTA;
  }

  let content = "";
  let reasoning = "";
  for (const fragment of incoming) {
    state.fragments.push(fragment);
    const text = typeof fragment?.content === "string" ? fragment.content : "";
    if (!text) continue;
    if (fragment?.type === "THINK") {
      state.thinking += text;
      reasoning += text;
    } else {
      state.content += text;
      content += text;
    }
  }
  state.reasoning =
    state.fragments[state.fragments.length - 1]?.type === "THINK";

  // Bare frames that follow target the new fragment's text.
  state.path = CURRENT_FRAGMENT_CONTENT;
  state.op = "APPEND";

  return { content: state.hasTools ? "" : content, reasoning };
}

/** Resolve a fragment index (negative counts from the end) against the list. */
function fragmentAt(state: DSStreamState, index: number): DSFragment | undefined {
  if (index < 0) return state.fragments[state.fragments.length + index];
  return state.fragments[index];
}

/**
 * Append `v` to a buffer. APPEND frames carry new text only; SET/REPLACE
 * frames (and frames with no operation) may repeat everything seen so far.
 */
function accumulate(buffer: string, op: string | undefined, v: string): string {
  if (op === "APPEND") return v;
  if (buffer.length > 0 && v.startsWith(buffer)) return v.slice(buffer.length);
  return v;
}

function applyTextDelta(
  state: DSStreamState,
  v: unknown,
  op: string | undefined,
  reasoning: boolean
): StreamDelta {
  if (typeof v !== "string" || v.length === 0) return NO_DELTA;

  if (reasoning) {
    const delta = accumulate(state.thinking, op, v);
    state.thinking += delta;
    return { content: "", reasoning: delta };
  }

  const delta = accumulate(state.content, op, v);
  state.content += delta;
  // In tool mode, buffer content and emit it only at the end.
  return { content: state.hasTools ? "" : delta, reasoning: "" };
}

// ── Tool call parsing ───────────────────────────────────────────────

export interface ParsedToolCall {
  id: string;
  name: string;
  arguments: string;
  leadingText?: string;
}

/**
 * Parse a DeepSeek completion as a structured tool call.
 * The model is instructed to emit a single JSON object; tolerate
 * surrounding whitespace and markdown fences.
 */
export function parseToolCall(content: string): ParsedToolCall | null {
  const start = content.indexOf("{");
  if (start === -1) return null;

  const end = content.lastIndexOf("}");
  if (end === -1 || end < start) return null;

  try {
    const obj = JSON.parse(content.slice(start, end + 1)) as {
      name?: unknown;
      arguments?: unknown;
      params?: unknown;
      parameters?: unknown;
    };
    if (typeof obj.name !== "string" || obj.name.length === 0) return null;

    const args = obj.arguments ?? obj.params ?? obj.parameters ?? {};
    const argsStr =
      typeof args === "string" ? args : JSON.stringify(args ?? {});

    let leadingText = content.slice(0, start).trim();
    leadingText = leadingText.replace(/```(?:json)?\s*$/, "").trim();

    return {
      id: `call_${randomUUID().replace(/-/g, "").slice(0, 24)}`,
      name: obj.name,
      arguments: argsStr,
      leadingText: leadingText || undefined,
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
