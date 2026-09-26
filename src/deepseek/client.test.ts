import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { DeepSeekClient, parseParentMessageId } from "./client.js";
import { deepSeekHash } from "./pow.js";

describe("parseParentMessageId", () => {
  it("converts string digits to numbers", () => {
    expect(parseParentMessageId("2")).toBe(2);
    expect(parseParentMessageId("12345")).toBe(12345);
    expect(parseParentMessageId(" 42 ")).toBe(42);
  });

  it("handles numbers directly", () => {
    expect(parseParentMessageId(2)).toBe(2);
    expect(parseParentMessageId(12345)).toBe(12345);
    expect(parseParentMessageId(0)).toBe(0);
  });

  it("returns null for null, undefined, empty, or non-numeric strings", () => {
    expect(parseParentMessageId(null)).toBeNull();
    expect(parseParentMessageId(undefined)).toBeNull();
    expect(parseParentMessageId("")).toBeNull();
    expect(parseParentMessageId("   ")).toBeNull();
    expect(parseParentMessageId("invalid")).toBeNull();
  });
});

describe("DeepSeekClient", () => {
  const auth = { token: "fake-token", cookies: [] };
  let client: DeepSeekClient;
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    client = new DeepSeekClient({ auth });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("extracts session ID from data.id format", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ data: { id: "sess-1" } }),
    } as Response);

    const sessionId = await client.createSession();
    expect(sessionId).toBe("sess-1");
  });

  it("extracts session ID from data.biz_data.id format", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ data: { biz_data: { id: "sess-biz-1" } } }),
    } as Response);

    const sessionId = await client.createSession();
    expect(sessionId).toBe("sess-biz-1");
  });

  it("extracts session ID from data.biz_data.chat_session_id format", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ data: { biz_data: { chat_session_id: "sess-biz-2" } } }),
    } as Response);

    const sessionId = await client.createSession();
    expect(sessionId).toBe("sess-biz-2");
  });

  it("extracts session ID from data.biz_data.chat_session.id format", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        code: 0,
        msg: "",
        data: {
          biz_code: 0,
          biz_msg: "",
          biz_data: {
            chat_session: {
              id: "fc5f10cf-acc8-47c0-aac5-755e7ebeb8b4",
              seq_id: 212560637,
              agent: "chat",
            },
            ttl_seconds: 259200,
          },
        },
      }),
    } as Response);

    const sessionId = await client.createSession();
    expect(sessionId).toBe("fc5f10cf-acc8-47c0-aac5-755e7ebeb8b4");
  });

  it("extracts session ID from data.biz_data.chat_session.chat_session_id format", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        data: {
          biz_data: {
            chat_session: {
              chat_session_id: "sess-chat-session-id-1",
            },
          },
        },
      }),
    } as Response);

    const sessionId = await client.createSession();
    expect(sessionId).toBe("sess-chat-session-id-1");
  });

  it("throws descriptive error when session ID is missing in response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ data: {} }),
    } as Response);

    await expect(client.createSession()).rejects.toThrow(
      "Failed to create session: invalid or missing session ID in response"
    );
  });

  it("throws error in chatCompletion when chatSessionId is empty", async () => {
    const generator = client.chatCompletion({
      chatSessionId: "",
      parentMessageId: null,
      prompt: "Hello",
      thinkingEnabled: false,
      modelType: "deepseek_chat",
    });

    await expect(generator.next()).rejects.toThrow(
      "chatSessionId is required and must be a non-empty string"
    );
  });

  it("converts string parentMessageId to numeric parent_message_id in request body", async () => {
    let capturedBody: any;
    const salt = "test-salt";
    const expireAt = 1234567890;
    const nonce = 42;
    const prefix = `${salt}_${expireAt}_`;
    const challenge = deepSeekHash(Buffer.from(`${prefix}${nonce}`)).toString("hex");

    globalThis.fetch = vi.fn().mockImplementation(async (url: string, opts?: any) => {
      if (url.endsWith("/create_pow_challenge")) {
        return {
          ok: true,
          json: async () => ({
            salt,
            expire_at: expireAt,
            challenge,
            difficulty: 100000,
            signature: "sig",
          }),
        } as Response;
      }
      if (url.endsWith("/completion")) {
        capturedBody = JSON.parse(opts.body);
        return {
          ok: true,
          body: {
            getReader: () => {
              let done = false;
              return {
                read: async () => {
                  if (done) return { done: true, value: undefined };
                  done = true;
                  const encoder = new TextEncoder();
                  return {
                    done: false,
                    value: encoder.encode('data: {"p":"response/status","v":"FINISHED"}\n\n'),
                  };
                },
              };
            },
          },
        } as Response;
      }
      return { ok: false, status: 404 } as Response;
    });

    const events = [];
    for await (const event of client.chatCompletion({
      chatSessionId: "sess-1",
      parentMessageId: "2",
      prompt: "Hello",
      thinkingEnabled: false,
      modelType: "deepseek_chat",
    })) {
      events.push(event);
    }

    expect(capturedBody).toBeDefined();
    expect(capturedBody.parent_message_id).toBe(2);
    expect(typeof capturedBody.parent_message_id).toBe("number");
  });

  it("parses trailing buffer data when stream closes without trailing newline", async () => {
    const salt = "test-salt";
    const expireAt = 1234567890;
    const nonce = 42;
    const prefix = `${salt}_${expireAt}_`;
    const challenge = deepSeekHash(Buffer.from(`${prefix}${nonce}`)).toString("hex");

    globalThis.fetch = vi.fn().mockImplementation(async (url: string, opts?: any) => {
      if (url.endsWith("/create_pow_challenge")) {
        return {
          ok: true,
          json: async () => ({
            salt,
            expire_at: expireAt,
            challenge,
            difficulty: 100000,
            signature: "sig",
          }),
        } as Response;
      }
      if (url.endsWith("/completion")) {
        return {
          ok: true,
          body: {
            getReader: () => {
              let done = false;
              return {
                read: async () => {
                  if (done) return { done: true, value: undefined };
                  done = true;
                  const encoder = new TextEncoder();
                  // Note: line does not end with \n before stream end
                  return {
                    done: false,
                    value: encoder.encode('data: {"p":"response/content","o":"SET","v":"Trailing line"}'),
                  };
                },
              };
            },
          },
        } as Response;
      }
      return { ok: false, status: 404 } as Response;
    });

    const events = [];
    for await (const event of client.chatCompletion({
      chatSessionId: "sess-1",
      parentMessageId: null,
      prompt: "Hello",
      thinkingEnabled: false,
      modelType: "deepseek_chat",
    })) {
      events.push(event);
    }

    expect(events).toHaveLength(1);
    expect(events[0]).toEqual({
      p: "response/content",
      o: "SET",
      v: "Trailing line",
    });
  });
});
