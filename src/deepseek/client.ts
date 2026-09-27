import { BASE_URL } from "../shared/config.js";
import type {
  AuthData,
  DSStreamEvent,
  DSCreateSessionResp,
  DSBreadcrumb,
} from "../shared/types.js";
import { solvePow, encodePow } from "./pow.js";

const HEADERS = {
  "content-type": "application/json",
  origin: BASE_URL,
  referer: `${BASE_URL}/`,
  "user-agent":
    "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
  "x-app-version": "20241129.1",
  "x-client-locale": "en_US",
  "x-client-platform": "web",
  "x-client-version": "2.0.2",
};

export class DeepSeekClient {
  private auth: AuthData;
  private onAuthRefresh?: (a: AuthData) => void;

  constructor(opts: { auth: AuthData; onAuthRefresh?: (a: AuthData) => void }) {
    this.auth = opts.auth;
    this.onAuthRefresh = opts.onAuthRefresh;
  }

  private headers(extra?: Record<string, string>): Record<string, string> {
    const tz = new Date().getTimezoneOffset();
    const headers: Record<string, string> = {
      ...HEADERS,
      "x-client-timezone-offset": String(tz),
      ...extra,
    };
    const cookieHeader = this.auth.cookies
      .map((c) => `${c.name}=${c.value}`)
      .join("; ");
    if (cookieHeader) headers.cookie = cookieHeader;
    if (this.auth.token) headers.authorization = `Bearer ${this.auth.token}`;
    return headers;
  }

  private async request(
    path: string,
    body?: unknown,
    extra?: Record<string, string>,
    signal?: AbortSignal
  ): Promise<Response> {
    const res = await fetch(`${BASE_URL}${path}`, {
      method: body ? "POST" : "GET",
      headers: this.headers(extra),
      body: body ? JSON.stringify(body) : undefined,
      signal,
    });

    if (res.status === 401) {
      throw new AuthExpiredError("DeepSeek auth expired (401)");
    }

    return res;
  }

  /**
   * Verify the current token.
   *
   * Throws `AuthExpiredError` when DeepSeek rejects it. Note that DeepSeek
   * answers a bad token with HTTP 200 and `{"code":40003}` in the body, so
   * `res.ok` is true and the status code alone proves nothing.
   */
  async validate(): Promise<void> {
    const res = await this.request("/api/v0/users/current");
    assertEnvelope(await res.json(), "/api/v0/users/current");
  }

  /** Create a new chat session, returns session ID. */
  async createSession(): Promise<string> {
    const res = await this.request("/api/v0/chat_session/create", {
      character_id: null,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Failed to create session: ${res.status} ${text}`);
    }
    const raw: unknown = await res.json();
    assertEnvelope(raw, "/api/v0/chat_session/create");
    const data = raw as {
      data?: {
        id?: string;
        chat_session_id?: string;
        chat_session?: {
          id?: string;
          chat_session_id?: string;
        };
        biz_data?: {
          id?: string;
          chat_session_id?: string;
          chat_session?: {
            id?: string;
            chat_session_id?: string;
          };
        };
      };
      id?: string;
      chat_session_id?: string;
      chat_session?: {
        id?: string;
        chat_session_id?: string;
      };
    };

    const sessionId =
      data?.data?.biz_data?.chat_session?.id ??
      data?.data?.biz_data?.chat_session?.chat_session_id ??
      data?.data?.biz_data?.id ??
      data?.data?.biz_data?.chat_session_id ??
      data?.data?.chat_session?.id ??
      data?.data?.chat_session?.chat_session_id ??
      data?.data?.id ??
      data?.data?.chat_session_id ??
      data?.id ??
      data?.chat_session_id ??
      data?.chat_session?.id ??
      data?.chat_session?.chat_session_id;

    if (!sessionId || typeof sessionId !== "string" || !sessionId.trim()) {
      throw new Error(
        `Failed to create session: invalid or missing session ID in response (${JSON.stringify(raw)})`
      );
    }

    return sessionId.trim();
  }

  /**
   * Send a chat completion request and yield SSE events.
   * The caller should parse these into OpenAI format.
   */
  async *chatCompletion(params: {
    chatSessionId: string;
    parentMessageId: number | string | null;
    prompt: string;
    thinkingEnabled: boolean;
    modelType: string;
    /** Aborted when the client disconnects, to stop reading the upstream stream. */
    signal?: AbortSignal;
  }): AsyncGenerator<DSStreamEvent> {
    if (!params.chatSessionId || typeof params.chatSessionId !== "string" || !params.chatSessionId.trim()) {
      throw new Error("chatSessionId is required and must be a non-empty string");
    }

    // 1. Get PoW challenge
    const powRes = await this.request("/api/v0/chat/create_pow_challenge", {
      target_path: "/api/v0/chat/completion",
    });
    if (!powRes.ok) {
      throw new Error(`PoW challenge failed: ${powRes.status}`);
    }
    const raw: unknown = await powRes.json();
    assertEnvelope(raw, "/api/v0/chat/create_pow_challenge");
    const wrapped = raw as { data?: { biz_data?: { challenge?: DSBreadcrumb } } };
    const challenge = (wrapped.data?.biz_data?.challenge ?? raw) as DSBreadcrumb;

    // 2. Solve PoW
    const powResp = await solvePow(challenge);
    const powHeader = encodePow(powResp);

    // 3. Send chat completion
    const body = {
      chat_session_id: params.chatSessionId,
      parent_message_id: parseParentMessageId(params.parentMessageId),
      prompt: params.prompt,
      ref_file_ids: [],
      thinking_enabled: params.thinkingEnabled,
      search_enabled: false,
      model_type: params.modelType,
    };

    const res = await this.request(
      "/api/v0/chat/completion",
      body,
      { "x-ds-pow-response": powHeader },
      params.signal
    );

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Chat completion failed: ${res.status} ${text}`);
    }

    // A rejected token makes DeepSeek answer this endpoint with a JSON error
    // envelope and HTTP 200 instead of an event stream. Reading that as SSE
    // yields no frames at all, which would look like a successful but empty
    // answer — so the content type is checked before parsing.
    const contentType = res.headers.get("content-type") ?? "";
    if (!contentType.includes("text/event-stream")) {
      const raw: unknown = await res.json().catch(() => null);
      assertEnvelope(raw, "/api/v0/chat/completion");
      throw new Error(
        `Chat completion returned ${contentType || "an unknown content type"} instead of an event stream`
      );
    }

    // 4. Parse SSE stream
    if (!res.body) throw new Error("No response body");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const parseLine = (line: string): DSStreamEvent | null => {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) return null;
      const json = trimmed.slice(5).trim();
      if (!json || json === "[DONE]") return null;
      try {
        return JSON.parse(json) as DSStreamEvent;
      } catch {
        return null;
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const event = parseLine(line);
        if (event) yield event;
      }
    }

    buffer += decoder.decode();
    if (buffer.length > 0) {
      const lines = buffer.split("\n");
      for (const line of lines) {
        const event = parseLine(line);
        if (event) yield event;
      }
    }
  }

  /** Update the stored auth and notify. */
  setAuth(auth: AuthData): void {
    this.auth = auth;
    this.onAuthRefresh?.(auth);
  }
}

export class AuthExpiredError extends Error {
  constructor(msg: string) {
    super(msg);
    this.name = "AuthExpiredError";
  }
}

/** A non-auth failure reported by DeepSeek in a response body. */
export class DeepSeekApiError extends Error {
  constructor(
    readonly code: number | string,
    message: string
  ) {
    super(`DeepSeek API error ${code}: ${message}`);
    this.name = "DeepSeekApiError";
  }
}

/** DeepSeek error codes that mean the token was rejected. */
const AUTH_ERROR_CODES = new Set([40003]);

/**
 * Throw if a DeepSeek response body reports a failure.
 *
 * DeepSeek signals API errors inside a HTTP 200 body as `{"code":…,"msg":…}`,
 * so `res.ok` and the status code are both useless for detecting them. Without
 * this, a rejected token surfaced as a missing session id, a generic 500, or —
 * worst of all — a successful-looking empty answer.
 */
function assertEnvelope(raw: unknown, what: string): void {
  const env = (raw ?? {}) as { code?: number | string; msg?: string };
  const code = env.code;
  if (code === undefined || code === null || code === 0 || code === "0") return;

  const message = typeof env.msg === "string" ? env.msg : "unknown error";
  if (AUTH_ERROR_CODES.has(Number(code)) || /token|authoriz/i.test(message)) {
    throw new AuthExpiredError(
      `DeepSeek rejected the token (${code}: ${message}). Run \`ds auth\` to re-authenticate.`
    );
  }
  throw new DeepSeekApiError(code, `${what}: ${message}`);
}

export function parseParentMessageId(id: number | string | null | undefined): number | null {
  if (id == null) return null;
  if (typeof id === "number") {
    return Number.isFinite(id) ? Math.floor(id) : null;
  }
  if (typeof id === "string") {
    const trimmed = id.trim();
    if (!trimmed) return null;
    const parsed = parseInt(trimmed, 10);
    return isNaN(parsed) ? null : parsed;
  }
  return null;
}
