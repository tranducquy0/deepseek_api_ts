import type { DeepSeekClient } from "../deepseek/client.js";

export interface SessionEntry {
  sessionId: string;
  parentMessageId: string | null;
  lastMessageCount: number;
  updatedAt: number;
}

const SESSION_TTL = 60 * 60 * 1000; // 1 hour
const PRUNE_INTERVAL = 5 * 60 * 1000; // 5 minutes

/**
 * Tracks DeepSeek chat sessions per OpenAI conversation so follow-up
 * turns chain correctly via parent_message_id and only forward messages
 * the session has not seen yet.
 */
export class SessionManager {
  private entries = new Map<string, SessionEntry>();

  constructor() {
    setInterval(() => this.prune(), PRUNE_INTERVAL);
    this.prune();
  }

  /** Return a live session entry or create one via the client. */
  async getOrCreate(
    client: DeepSeekClient,
    key: string
  ): Promise<SessionEntry> {
    const existing = this.get(key);
    if (existing) return existing;

    const sessionId = await client.createSession();
    const entry: SessionEntry = {
      sessionId,
      parentMessageId: null,
      lastMessageCount: 0,
      updatedAt: Date.now(),
    };
    this.entries.set(key, entry);
    return entry;
  }

  /** Peek at a live entry without creating a new session. */
  get(key: string): SessionEntry | null {
    const entry = this.entries.get(key);
    if (!entry) return null;
    if (Date.now() - entry.updatedAt >= SESSION_TTL) {
      this.entries.delete(key);
      return null;
    }
    return entry;
  }

  /** Persist chaining/forwarding state after a completed turn. */
  update(
    key: string,
    state: { parentMessageId: string | null; messageCount: number }
  ): void {
    const entry = this.entries.get(key);
    if (!entry) return;
    entry.parentMessageId = state.parentMessageId;
    entry.lastMessageCount = state.messageCount;
    entry.updatedAt = Date.now();
  }

  /** Drop a conversation (forces a fresh DeepSeek session next turn). */
  reset(key: string): void {
    this.entries.delete(key);
  }

  private prune(): void {
    const now = Date.now();
    for (const [key, entry] of this.entries) {
      if (now - entry.updatedAt >= SESSION_TTL) this.entries.delete(key);
    }
  }
}