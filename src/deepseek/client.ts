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
    extra?: Record<string, string>
  ): Promise<Response> {
    const res = await fetch(`${BASE_URL}${path}`, {
      method: body ? "POST" : "GET",
      headers: this.headers(extra),
      body: body ? JSON.stringify(body) : undefined,
    });

    if (res.status === 401) {
      throw new AuthExpiredError("DeepSeek auth expired (401)");
    }

    return res;
  }

  /** Validate the current token. */
  async validate(): Promise<boolean> {
    try {
      const res = await this.request("/api/v0/users/current");
      return res.ok;
    } catch {
      return false;
    }
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
    parentMessageId: string | null;
    prompt: string;
    thinkingEnabled: boolean;
    modelType: string;
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
    const wrapped = raw as { data?: { biz_data?: { challenge?: DSBreadcrumb } } };
    const challenge = (wrapped.data?.biz_data?.challenge ?? raw) as DSBreadcrumb;

    // 2. Solve PoW
    const powResp = await solvePow(challenge);
    const powHeader = encodePow(powResp);

    // 3. Send chat completion
    const body = {
      chat_session_id: params.chatSessionId,
      parent_message_id: params.parentMessageId,
      prompt: params.prompt,
      ref_file_ids: [],
      thinking_enabled: params.thinkingEnabled,
      search_enabled: false,
      model_type: params.modelType,
    };

    const res = await this.request(
      "/api/v0/chat/completion",
      body,
      { "x-ds-pow-response": powHeader }
    );

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Chat completion failed: ${res.status} ${text}`);
    }

    // 4. Parse SSE stream
    if (!res.body) throw new Error("No response body");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const json = line.slice(6).trim();
          if (!json || json === "[DONE]") continue;
          try {
            const event = JSON.parse(json) as DSStreamEvent;
            yield event;
          } catch {
            // skip malformed lines
          }
        }
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
