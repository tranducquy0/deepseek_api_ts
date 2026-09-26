// ── Auth ────────────────────────────────────────────────────────────

export interface AuthData {
  /** Bearer token from localStorage; optional when cookie auth works */
  token?: string;
  cookies: Cookie[];
  /** ISO timestamp of last successful validation */
  lastValidated?: string;
}

export interface Cookie {
  name: string;
  value: string;
  domain: string;
}

// ── OpenAI types ────────────────────────────────────────────────────

export interface OpenAIMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

export interface OpenAIChatRequest {
  model: string;
  messages: OpenAIMessage[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  thinking?: boolean;
  tools?: OpenAITool[];
  tool_choice?: string | { type: "function"; function: { name: string } };
  /** Optional stable conversation identity used to key DeepSeek sessions */
  user?: string;
  /** Optional explicit DeepSeek chat session ID */
  chat_session_id?: string;
  chatSessionId?: string;
  session_id?: string;
}

export interface OpenAIChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: {
    index: number;
    delta: Partial<OpenAIMessage>;
    finish_reason: "stop" | "length" | "tool_calls" | null;
  }[];
}

export interface OpenAIModel {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
}

export interface OpenAIModelList {
  object: "list";
  data: OpenAIModel[];
}

// ── Tool calling ────────────────────────────────────────────────────

export interface OpenAIToolFunction {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

export interface OpenAITool {
  type: "function";
  function: OpenAIToolFunction;
}

export interface OpenAIToolCall {
  index?: number;
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

// ── DeepSeek internal types ─────────────────────────────────────────

export interface DSBreadcrumb {
  algorithm?: string;
  salt: string;
  expire_at: number;
  difficulty: number;
  challenge: string;
  signature: string;
  target_path?: string;
  expire_after?: number;
}

export interface DSPoWResponse {
  algorithm?: string;
  challenge: string;
  salt: string;
  answer: number;
  signature: string;
  target_path?: string;
}

export interface DSCreateSessionResp {
  data: {
    id: string;
  };
}

export interface DSChatCompletionReq {
  chat_session_id: string;
  parent_message_id: number | null;
  prompt: string;
  ref_file_ids: string[];
  thinking_enabled: boolean;
  search_enabled: boolean;
  model_type: string;
}

export interface DSStreamEvent {
  /** JSON path like "response/fragments/-1/content" */
  p?: string;
  /** Operation: APPEND or SET */
  o?: string;
  /** Value: string chunk or full object */
  v?: unknown;
  /** Full initial snapshot */
  request_message_id?: number;
  response_message_id?: number;
}

export interface DSMessage {
  id: number;
  role: string;
  content: string;
  thinking_content?: string;
  parent_id?: number;
  status?: string;
}

// ── Client config ───────────────────────────────────────────────────

export interface DeepSeekClientOpts {
  auth: AuthData;
  onAuthRefresh?: (newAuth: AuthData) => void;
}
