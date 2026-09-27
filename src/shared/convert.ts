import { randomUUID } from "node:crypto";
import type {
  CompletionUsage,
  DSStreamEvent,
  OpenAIChunk,
  OpenAIChatRequest,
  OpenAITool,
  OpenAIToolFunction,
  OpenAIToolCall,
  OpenAIModelList,
  OpenAIModel,
  OpenAIMessage,
} from "./types.js";
import { MODELS, MODEL_MAP, type ModelId } from "./config.js";
import type { ContentPart, MessageContent } from "./types.js";

// ── Prompt builder ──────────────────────────────────────────────────

/**
 * Flatten message content to plain text for DeepSeek's web prompt.
 *
 * OpenAI clients send multimodal input as an array of content parts
 * (`[{type:"text",text:"…"}]`). Pushing that array into a string join yields
 * the literal text "[object Object]", so the model receives a request with no
 * actual task in it. Parts the web endpoint cannot carry (images, audio) are
 * named explicitly rather than dropped, so the model is never misled into
 * thinking it saw something it did not.
 */
export function contentToText(
  content: MessageContent | undefined
): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return String(content);

  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (!part || typeof part !== "object") return "";
      if (typeof part.text === "string") return part.text;
      const kind = typeof part.type === "string" ? part.type : "part";
      return `[${kind} omitted: this proxy does not forward attachments]`;
    })
    .filter((text) => text.length > 0)
    .join("\n");
}

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
    const text = contentToText(msg.content);
    switch (msg.role) {
      case "system":
        if (text) parts.push(`[System]: ${text}`);
        break;
      case "user":
        parts.push(text);
        break;
      case "assistant":
        if (text) parts.push(`[Assistant]: ${text}`);
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
        parts.push(`[Tool result for ${msg.tool_call_id ?? "tool"}]: ${text}`);
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

  const names = tools.map((t) => t.function.name);

  const lines = [
    "# Available Tools",
    "You can call functions by emitting one JSON object per line:",
    '{"name": "<function name>", "arguments": { ... }}',
    "",
    "Rules:",
    '- Every line MUST be a JSON object with BOTH a "name" and an "arguments" key.',
    `- "name" must be exactly one of: ${names.join(", ")}.`,
    "- The arguments object must match that function's JSON schema.",
    '- Never emit the arguments on their own without the "name" wrapper.',
    "- Do not wrap the JSON in markdown, add extra text, or explain.",
    "- Use one line per function call; use several lines for parallel calls.",
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

  const example = tools[0].function;
  const exampleArgs = Object.keys(
    (example.parameters?.properties as Record<string, unknown>) ?? {}
  )
    .slice(0, 1)
    .map((k) => `${JSON.stringify(k)}: "..."`)
    .join(", ");

  lines.push(
    "",
    "Functions:",
    JSON.stringify(tools.map((t) => t.function), null, 2),
    "",
    "Example of a correct call:",
    `{"name": ${JSON.stringify(example.name)}, "arguments": {${exampleArgs ? ` ${exampleArgs}` : ""} }}`
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
  /** The offered tool definitions, used to resolve wrapper-less calls. */
  tools: OpenAITool[];
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
  /** Token usage reported by DeepSeek, when it reports any. */
  usage: CompletionUsage | null;
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
export function createStreamState(
  hasTools = false,
  tools: OpenAITool[] = []
): DSStreamState {
  return {
    content: "",
    thinking: "",
    hasTools,
    tools,
    responseMessageId: null,
    finished: false,
    path: null,
    op: null,
    fragments: [],
    reasoning: false,
    seeded: false,
    usage: null,
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

  if (path === "response/accumulated_token_usage") {
    state.usage = readUsage(event.v, state.usage);
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
 * Read DeepSeek's `accumulated_token_usage`. The web endpoint sends a single
 * accumulator (not a prompt/completion split), so it is reported as
 * `total_tokens`; an object form is mapped field-by-field if one appears.
 */
function readUsage(v: unknown, prev: CompletionUsage | null): CompletionUsage | null {
  if (typeof v === "number" && Number.isFinite(v)) {
    return {
      prompt_tokens: prev?.prompt_tokens ?? 0,
      completion_tokens: prev?.completion_tokens ?? 0,
      total_tokens: Math.round(v),
    };
  }
  if (typeof v === "object" && v !== null) {
    const o = v as Record<string, unknown>;
    const num = (x: unknown) =>
      typeof x === "number" && Number.isFinite(x) ? Math.round(x) : 0;
    return {
      prompt_tokens: num(o.prompt_tokens),
      completion_tokens: num(o.completion_tokens),
      total_tokens: num(o.total_tokens),
    };
  }
  return prev;
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
  /** Position of this call in the turn, matching OpenAI's tool_call index. */
  index: number;
  id: string;
  name: string;
  arguments: string;
}

export interface ToolCallParseResult {
  calls: ParsedToolCall[];
  /** Prose emitted before the first tool call, if any. */
  leadingText?: string;
  /**
   * Set when the model clearly aimed for a tool call but the payload could not
   * be parsed. Callers must surface this instead of ending the turn with
   * `finish_reason: "stop"`, otherwise an agent loop reads a malformed call as
   * a completed step and silently stalls.
   */
  error?: string;
}

/** Marks text that was meant to be a tool call rather than prose. */
const TOOL_CALL_HINT = /"name"\s*:/;

/**
 * Locate top-level `{…}` spans, ignoring braces inside JSON strings.
 * Scanning for balanced objects — rather than slicing from the first `{` to the
 * last `}` — keeps prose such as "use {placeholder}" from swallowing the real
 * call that follows it, and lets several calls be found in one response.
 * `unterminated` flags an object left open at the end, which is a truncated
 * call rather than an absent one.
 */
function findJsonObjects(text: string): {
  spans: { start: number; end: number }[];
  unterminated: boolean;
} {
  const spans: { start: number; end: number }[] = [];
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') {
      inString = true;
    } else if (ch === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "}" && depth > 0) {
      depth--;
      if (depth === 0 && start >= 0) {
        spans.push({ start, end: i + 1 });
        start = -1;
      }
    }
  }
  return { spans, unterminated: depth > 0 };
}

/** Drop a trailing markdown fence from text preceding a tool call. */
function stripFence(text: string): string {
  return text.replace(/```(?:json)?\s*$/, "").trim();
}

function excerpt(text: string, max = 300): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
}

/** True when every key of `args` is a declared parameter of `fn`. */
function couldBeArgs(
  args: Record<string, unknown>,
  fn: OpenAIToolFunction
): boolean {
  const properties = fn.parameters?.properties;
  if (!properties || typeof properties !== "object") return false;

  const keys = Object.keys(args);
  if (keys.length === 0) return false;
  return keys.every((k) => k in (properties as Record<string, unknown>));
}

/**
 * Whether `args` is a complete argument object for `fn`: every key it uses is
 * declared and every required parameter is present.
 */
function matchesSchema(
  args: Record<string, unknown>,
  fn: OpenAIToolFunction
): boolean {
  if (!couldBeArgs(args, fn)) return false;
  const required = fn.parameters?.required;
  if (Array.isArray(required) && !required.every((k) => Object.hasOwn(args, k as string))) {
    return false;
  }
  return true;
}

/**
 * Parse a DeepSeek completion into zero or more structured tool calls.
 * Tolerates surrounding prose, markdown fences, and several calls per turn.
 */
export function parseToolCalls(
  content: string,
  tools: OpenAITool[] = []
): ToolCallParseResult {
  const calls: ParsedToolCall[] = [];
  let firstCallStart = -1;
  let malformed = false;
  let ambiguous = false;

  const { spans, unterminated } = findJsonObjects(content);
  if (unterminated) malformed = true;

  for (const span of spans) {
    let obj: Record<string, unknown>;
    try {
      obj = JSON.parse(content.slice(span.start, span.end)) as Record<
        string,
        unknown
      >;
    } catch {
      malformed = true;
      continue;
    }

    if (firstCallStart === -1) firstCallStart = span.start;

    const name =
      typeof obj.name === "string" && obj.name.length > 0
        ? obj.name
        : typeof obj.tool === "string" && obj.tool.length > 0
          ? obj.tool
          : typeof obj.function === "string" && obj.function.length > 0
            ? obj.function
            : undefined;

    const args = name
      ? ((obj.arguments ?? obj.args ?? obj.params ?? obj.parameters ?? {}) as
          | Record<string, unknown>
          | string)
      : ((obj.arguments ?? obj.args ?? obj.params ?? obj.parameters ?? obj) as
          | Record<string, unknown>
          | string);

    if (!name) {
      // The model often drops the wrapper and emits the bare arguments. Bind
      // them only when exactly one offered function could take them — guessing
      // between several risks running the wrong command. An object that fits
      // no function at all is just a JSON answer, not a broken call.
      const shaped =
        typeof args === "object" && args !== null
          ? tools.filter((t) =>
              couldBeArgs(args as Record<string, unknown>, t.function)
            )
          : [];
      if (shaped.length === 0) continue;

      const matches = shaped.filter((t) =>
        matchesSchema(args as Record<string, unknown>, t.function)
      );
      if (matches.length === 1) {
        calls.push(makeCall(calls.length, matches[0].function.name, args));
        continue;
      }
      ambiguous = true;
      continue;
    }

    calls.push(makeCall(calls.length, name, args));
  }

  const leadingText =
    firstCallStart === -1
      ? undefined
      : stripFence(content.slice(0, firstCallStart)) || undefined;

  if (calls.length === 0) {
    // Only complain when the text actually tried to be a tool call, so a prose
    // or JSON answer still ends as an ordinary turn.
    if (ambiguous) {
      const offered = tools.map((t) => `"${t.function.name}"`).join(", ");
      return {
        calls: [],
        leadingText,
        error: `Model emitted a tool call whose arguments did not satisfy exactly one function. Emit {"name": …, "arguments": {…}} using one of: ${offered}. Got: ${excerpt(content)}`,
      };
    }
    if (malformed && TOOL_CALL_HINT.test(content)) {
      return {
        calls: [],
        leadingText,
        error: `Model emitted an unparseable tool call: ${excerpt(content)}`,
      };
    }
    return { calls: [], leadingText };
  }

  return { calls, leadingText };
}

function makeCall(index: number, name: string, args: unknown): ParsedToolCall {
  return {
    index,
    id: `call_${randomUUID().replace(/-/g, "").slice(0, 24)}`,
    name,
    arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}),
  };
}

/**
 * Extract tool calls from buffered state content.
 * Only parsed when the request carried tool definitions, so a normal answer
 * that happens to contain JSON is left alone.
 */
export function extractToolCalls(state: DSStreamState): ToolCallParseResult {
  return state.hasTools ? parseToolCalls(state.content, state.tools) : { calls: [] };
}

/**
 * Create an OpenAI-compatible streaming chunk carrying tool calls.
 * `finish_reason` is left null: OpenAI sets it once, on the terminal chunk.
 */
export function makeToolCallChunk(
  completion: CompletionMeta,
  model: string,
  calls: ParsedToolCall[]
): OpenAIChunk {
  return makeChunk(completion, model, {
    tool_calls: calls.map(
      (tool) =>
        ({
          index: tool.index,
          id: tool.id,
          type: "function",
          function: { name: tool.name, arguments: tool.arguments },
        }) satisfies OpenAIToolCall
    ),
  });
}

// ── Completion identity ──────────────────────────────────────────────

/**
 * Identity shared by every chunk of one completion. OpenAI reuses a single
 * `id`/`created` pair across a stream; minting one per chunk breaks clients
 * that key caches, traces, or conversation state on the completion id.
 */
export interface CompletionMeta {
  id: string;
  created: number;
}

export function newCompletionMeta(): CompletionMeta {
  return {
    id: `chatcmpl-${randomUUID().replace(/-/g, "").slice(0, 24)}`,
    created: Math.floor(Date.now() / 1000),
  };
}

/**
 * Create an OpenAI-compatible chunk from a delta.
 */
export function makeChunk(
  completion: CompletionMeta,
  model: string,
  delta: Partial<OpenAIMessage>,
  finishReason: "stop" | "length" | "tool_calls" | null = null,
  index = 0
): OpenAIChunk {
  return {
    id: completion.id,
    object: "chat.completion.chunk",
    created: completion.created,
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

/**
 * Final usage-only chunk, emitted when the client sets
 * `stream_options.include_usage`. It carries no choices, matching OpenAI.
 */
export function makeUsageChunk(
  completion: CompletionMeta,
  model: string,
  usage: CompletionUsage
): OpenAIChunk {
  return {
    id: completion.id,
    object: "chat.completion.chunk",
    created: completion.created,
    model,
    choices: [],
    usage,
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
