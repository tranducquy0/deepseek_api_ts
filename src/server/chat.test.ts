import { describe, it, expect, beforeEach, afterEach } from "vitest";
import express from "express";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";
import { chatRouter, forwardMessages } from "./chat.js";
import type { DSStreamEvent, OpenAIChatRequest } from "../shared/types.js";
import type { DeepSeekClient } from "../deepseek/client.js";

interface ChatCall {
  chatSessionId: string;
  parentMessageId: string | null;
  prompt: string;
  modelType: string;
}

class FakeClient {
  sessionCalls = 0;
  chatCalls: ChatCall[] = [];
  events: DSStreamEvent[] = [];

  async createSession(): Promise<string> {
    this.sessionCalls++;
    return `sess-${this.sessionCalls}`;
  }

  async *chatCompletion(params: {
    chatSessionId: string;
    parentMessageId: string | null;
    prompt: string;
    modelType: string;
  }): AsyncGenerator<DSStreamEvent> {
    this.chatCalls.push(params);
    for (const event of this.events) yield event;
  }
}

const STANDARD_EVENTS: DSStreamEvent[] = [
  { p: "response/message_id", v: 12345, response_message_id: 12345 },
  { o: "APPEND", p: "response/fragments/-1/content", v: "Hello there" },
  { p: "response/status", v: "FINISHED" },
];

describe("chatRouter", () => {
  let client: FakeClient;
  let server: Server;
  let base: string;

  beforeEach(async () => {
    client = new FakeClient();
    const app = express();
    app.use(express.json());
    app.use(chatRouter(() => client as unknown as DeepSeekClient));
    server = await new Promise<Server>((resolve) => {
      const srv = app.listen(0, () => resolve(srv));
    });
    base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });

  afterEach(() => {
    server.close();
  });

  const post = (body: OpenAIChatRequest & Record<string, unknown>) =>
    fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });

  it("rejects requests without messages", async () => {
    const res = await post({ model: "deepseek-chat", messages: [] });
    expect(res.status).toBe(400);
  });

  it("creates a session and passes a null parent on the first turn", async () => {
    client.events = STANDARD_EVENTS;
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "Hello from turn one" }],
      stream: false,
    });
    expect(res.status).toBe(200);
    const json = (await res.json()) as {
      choices: { message: { content: string }; finish_reason: string }[];
    };
    expect(json.choices[0].message.content).toBe("Hello there");
    expect(client.sessionCalls).toBe(1);
    expect(client.chatCalls[0].parentMessageId).toBeNull();
    expect(client.chatCalls[0].prompt).toContain("Hello from turn one");
  });

  it("chains a follow-up turn: same session, persisted parent, delta-only prompt", async () => {
    client.events = STANDARD_EVENTS;

    const first = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "Chain me please" }],
      stream: false,
    });
    expect(first.status).toBe(200);

    const second = await post({
      model: "deepseek-chat",
      messages: [
        { role: "user", content: "Chain me please" },
        { role: "assistant", content: "Hello there" },
        { role: "user", content: "Continue" },
      ],
      stream: false,
    });
    expect(second.status).toBe(200);

    expect(client.sessionCalls).toBe(1);
    expect(client.chatCalls).toHaveLength(2);
    expect(client.chatCalls[1].chatSessionId).toBe(client.chatCalls[0].chatSessionId);
    // Parent id persisted from turn one's response_message_id
    expect(client.chatCalls[1].parentMessageId).toBe("12345");
    // Previous assistant message is skipped when forwarded > 0 because parentMessageId holds it
    expect(client.chatCalls[1].prompt).not.toContain("Chain me please");
    expect(client.chatCalls[1].prompt).not.toContain("[Assistant]: Hello there");
    expect(client.chatCalls[1].prompt).toContain("Continue");
  });

  it("forwardMessages skips leading assistant messages when forwarded > 0", () => {
    const messages = [
      { role: "user", content: "Turn 1" },
      { role: "assistant", content: "Resp 1" },
      { role: "tool", content: "Tool output" },
      { role: "user", content: "Turn 2" },
    ] as const;

    // Turn 1 forwarded 1 message
    const delta = forwardMessages(messages as any, 1);
    expect(delta).toEqual([
      { role: "tool", content: "Tool output" },
      { role: "user", content: "Turn 2" },
    ]);
  });

  it("streams SSE events", async () => {
    client.events = STANDARD_EVENTS;
    const res = await post({
      model: "deepseek-reasoner",
      messages: [{ role: "user", content: "Say it" }],
      stream: true,
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain("Hello there");
    expect(text).toContain("data: [DONE]");
  });

  it("surfaces tool calls in non-streaming responses", async () => {
    client.events = [
      { p: "response/message_id", v: 1, response_message_id: 1 },
      {
        o: "APPEND",
        p: "response/fragments/-1/content",
        v: '{"name":"get_weather","arguments":{"city":"Hanoi"}}',
      },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "Weather tool please" }],
      tools: [
        {
          type: "function",
          function: { name: "get_weather", parameters: {} },
        },
      ],
      stream: false,
    });
    expect(res.status).toBe(200);
    const json = (await res.json()) as {
      choices: {
        finish_reason: string;
        message: { content: string | null; tool_calls: { function: { name: string } }[] };
      }[];
    };
    expect(json.choices[0].finish_reason).toBe("tool_calls");
    expect(json.choices[0].message.content).toBeNull();
    expect(json.choices[0].message.tool_calls[0].function.name).toBe("get_weather");
  });

  it("surfaces tool calls in streaming responses", async () => {
    client.events = [
      { p: "response/message_id", v: 2, response_message_id: 2 },
      {
        o: "APPEND",
        p: "response/fragments/-1/content",
        v: '{"name":"search","arguments":{"q":"test"}}',
      },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "Search please" }],
      tools: [{ type: "function", function: { name: "search", parameters: {} } }],
      stream: true,
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("tool_calls");
    expect(text).toContain('"name":"search"');
    expect(text).toContain('"finish_reason":"tool_calls"');
  });
});