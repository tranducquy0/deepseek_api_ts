import { describe, it, expect, vi, beforeEach } from "vitest";
import { SessionManager } from "./sessions.js";
import type { DeepSeekClient } from "../deepseek/client.js";

class FakeClient {
  sessionCalls = 0;
  async createSession(): Promise<string> {
    this.sessionCalls++;
    return `sess-${this.sessionCalls}`;
  }
}

describe("SessionManager", () => {
  let client: FakeClient;
  let sessions: SessionManager;

  beforeEach(() => {
    vi.useRealTimers();
    client = new FakeClient();
    sessions = new SessionManager();
  });

  it("creates a session on the first turn", async () => {
    const entry = await sessions.getOrCreate(client, "conv-1");
    expect(entry.sessionId).toBe("sess-1");
    expect(entry.parentMessageId).toBeNull();
    expect(entry.lastMessageCount).toBe(0);
  });

  it("reuses the live session and persists chaining state", async () => {
    const first = await sessions.getOrCreate(client, "conv-1");
    sessions.update("conv-1", { parentMessageId: "msg-1", messageCount: 2 });

    const second = await sessions.getOrCreate(client, "conv-1");
    expect(second.sessionId).toBe(first.sessionId);
    expect(second.parentMessageId).toBe("msg-1");
    expect(second.lastMessageCount).toBe(2);
    expect(client.sessionCalls).toBe(1);
  });

  it("keeps conversations isolated", async () => {
    const a = await sessions.getOrCreate(client, "conv-a");
    const b = await sessions.getOrCreate(client, "conv-b");
    expect(a.sessionId).not.toBe(b.sessionId);
    expect(client.sessionCalls).toBe(2);
  });

  it("expires entries after the TTL", async () => {
    vi.useFakeTimers();
    sessions = new SessionManager();
    await sessions.getOrCreate(client, "conv-1");
    expect(sessions.get("conv-1")).not.toBeNull();

    vi.setSystemTime(Date.now() + 61 * 60 * 1000);
    expect(sessions.get("conv-1")).toBeNull();
  });

  it("reset drops a conversation", async () => {
    await sessions.getOrCreate(client, "conv-1");
    sessions.reset("conv-1");
    expect(sessions.get("conv-1")).toBeNull();
  });
});