import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { DeepSeekClient } from "./client.js";

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
});
