import { describe, it, expect, beforeEach, afterEach } from "vitest";
import express from "express";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";
import { chatRouter, forwardMessages } from "./chat.js";
import type { DSStreamEvent, OpenAIChatRequest } from "../shared/types.js";
import type { DeepSeekClient } from "../deepseek/client.js";

interface ChatCall {
  chatSessionId: string;
  parentMessageId: number | string | null;
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
    parentMessageId: number | string | null;
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
    expect(client.chatCalls[1].parentMessageId).toBe(12345);
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

  it("streams responses with SET operations without truncation", async () => {
    client.events = [
      { p: "response/message_id", v: 100, response_message_id: 100 },
      { o: "SET", p: "response/fragments/0/content", v: "How do I check if an object" },
      { o: "APPEND", p: "response/fragments/0/content", v: " is an instance" },
      { o: "SET", p: "response/content", v: "How do I check if an object is an instance of a class?" },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "instance test" }],
      stream: true,
    });
    expect(res.status).toBe(200);
    const text = await res.text();

    // Parse the data lines to re-assemble the content deltas
    const lines = text.split("\n");
    let fullContent = "";
    for (const line of lines) {
      if (line.startsWith("data: ") && !line.includes("[DONE]")) {
        const json = JSON.parse(line.slice(6));
        const delta = json.choices?.[0]?.delta?.content;
        if (delta) fullContent += delta;
      }
    }
    expect(fullContent).toBe("How do I check if an object is an instance of a class?");
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

  it("replays a captured DeepSeek stream without dropping tokens", async () => {
    // Verbatim shape of a real chat.deepseek.com completion: the path is
    // declared once, then ~400 bare {"v":"…"} frames inherit it.
    client.events = [
      { request_message_id: 1, response_message_id: 2 } as DSStreamEvent,
      {
        v: {
          response: {
            message_id: 2,
            fragments: [{ id: 2, type: "RESPONSE", content: "#" }],
          },
        },
      },
      { o: "APPEND", p: "response/fragments/-1/content", v: " Checking" },
      { v: " if" },
      { v: " a" },
      { v: " Python" },
      { o: "SET", p: "response/fragments/-1/elapsed_secs", v: 0.42 },
      { p: "response", o: "BATCH", v: [{ p: "accumulated_token_usage", v: 12 }] },
      { o: "SET", p: "response/status", v: "FINISHED" },
    ];

    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "stream me" }],
      stream: true,
    });
    expect(res.status).toBe(200);
    const text = await res.text();

    let streamed = "";
    for (const line of text.split("\n")) {
      if (line.startsWith("data: ") && !line.includes("[DONE]")) {
        streamed += JSON.parse(line.slice(6)).choices?.[0]?.delta?.content ?? "";
      }
    }
    expect(streamed).toBe("# Checking if a Python");
  });

  it("streams reasoning_content separately from content", async () => {
    client.events = [
      { v: { response: { fragments: [{ id: 2, type: "THINK", content: "We" }] } } },
      { o: "APPEND", p: "response/fragments/-1/content", v: " need" },
      { v: " this" },
      {
        p: "response/fragments",
        o: "APPEND",
        v: [{ id: 3, type: "RESPONSE", content: "42" }],
      },
      { v: "." },
    ];

    const res = await post({
      model: "deepseek-reasoner",
      messages: [{ role: "user", content: "reason please" }],
      stream: true,
    });
    const text = await res.text();

    let content = "";
    let reasoning = "";
    for (const line of text.split("\n")) {
      if (line.startsWith("data: ") && !line.includes("[DONE]")) {
        const delta = JSON.parse(line.slice(6)).choices?.[0]?.delta ?? {};
        content += delta.content ?? "";
        reasoning += delta.reasoning_content ?? "";
      }
    }
    expect(reasoning).toBe("We need this");
    expect(content).toBe("42.");
  });

  it("defaults to a non-streaming JSON response when stream is omitted", async () => {
    client.events = STANDARD_EVENTS;
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "no stream field" }],
    } as OpenAIChatRequest);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("application/json");
    const json = (await res.json()) as {
      object: string;
      choices: { message: { content: string } }[];
    };
    expect(json.object).toBe("chat.completion");
    expect(json.choices[0].message.content).toBe("Hello there");
  });

  it("returns a loud 502 when the model emits an unparseable tool call", async () => {
    client.events = [
      { o: "APPEND", p: "response/fragments/-1/content", v: '{"name":"search","arguments":' },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "search please" }],
      tools: [{ type: "function", function: { name: "search", parameters: {} } }],
      stream: false,
    });
    // A clean 200 + finish_reason "stop" here would let an agent loop record a
    // completed step that never happened.
    expect(res.status).toBe(502);
    const json = (await res.json()) as {
      error: { code: string; message: string };
    };
    expect(json.error.code).toBe("tool_call_parse_failed");
    expect(json.error.message).toContain("unparseable tool call");
  });

  it("emits an SSE error frame instead of [DONE] on a bad tool call", async () => {
    client.events = [
      { o: "APPEND", p: "response/fragments/-1/content", v: '{"name":"search","arguments":' },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "search please" }],
      tools: [{ type: "function", function: { name: "search", parameters: {} } }],
      stream: true,
    });
    const text = await res.text();
    expect(text).toContain('"code":"tool_call_parse_failed"');
    // The SDK raises APIError on an error frame; [DONE] would imply success.
    expect(text).not.toContain("[DONE]");
    // No terminal finish_reason either — the turn never completed.
    const terminal = text
      .split("\n")
      .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
      .map((l) => JSON.parse(l.slice(6)).choices?.[0]?.finish_reason)
      .filter((r) => r !== null && r !== undefined);
    expect(terminal).toEqual([]);
  });

  it("returns several tool calls from one turn, indexed in order", async () => {
    client.events = [
      {
        o: "APPEND",
        p: "response/fragments/-1/content",
        v: '{"name":"read","arguments":{"p":"a"}}\n{"name":"read","arguments":{"p":"b"}}',
      },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "read both" }],
      tools: [{ type: "function", function: { name: "read", parameters: {} } }],
      stream: false,
    });
    expect(res.status).toBe(200);
    const json = (await res.json()) as {
      choices: {
        finish_reason: string;
        message: { tool_calls: { function: { name: string; arguments: string } }[] };
      }[];
    };
    expect(json.choices[0].finish_reason).toBe("tool_calls");
    expect(json.choices[0].message.tool_calls.map((c) => c.function.arguments)).toEqual([
      '{"p":"a"}',
      '{"p":"b"}',
    ]);
  });

  it("sets finish_reason exactly once per stream", async () => {
    client.events = [
      {
        o: "APPEND",
        p: "response/fragments/-1/content",
        v: '{"name":"search","arguments":{"q":"x"}}',
      },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "one tool" }],
      tools: [{ type: "function", function: { name: "search", parameters: {} } }],
      stream: true,
    });
    const text = await res.text();
    const reasons = text
      .split("\n")
      .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
      .map((l) => JSON.parse(l.slice(6)).choices[0].finish_reason)
      .filter((r) => r !== null);
    expect(reasons).toEqual(["tool_calls"]);
  });

  it("reuses one completion id across every chunk", async () => {
    client.events = [
      { o: "APPEND", p: "response/fragments/-1/content", v: "Hello" },
      { v: " there" },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "stable id" }],
      stream: true,
    });
    const text = await res.text();
    const ids = new Set(
      text
        .split("\n")
        .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
        .map((l) => JSON.parse(l.slice(6)).id)
    );
    expect(ids.size).toBe(1);
  });

  it("reports DeepSeek token usage and a usage chunk on request", async () => {
    client.events = [
      { o: "APPEND", p: "response/fragments/-1/content", v: "hi" },
      {
        p: "response",
        o: "BATCH",
        v: [{ p: "accumulated_token_usage", v: 465 }],
      },
      { p: "response/status", v: "FINISHED" },
    ];

    const streamed = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "usage please" }],
      stream: true,
      stream_options: { include_usage: true },
    } as OpenAIChatRequest);
    const usageChunk = (await streamed.text())
      .split("\n")
      .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
      .map((l) => JSON.parse(l.slice(6)))
      .find((c) => c.usage);
    expect(usageChunk?.choices).toEqual([]);
    expect(usageChunk?.usage.total_tokens).toBe(465);

    const json = (await (
      await post({
        model: "deepseek-chat",
        messages: [{ role: "user", content: "usage please" }],
        stream: false,
      } as OpenAIChatRequest)
    ).json()) as { usage: { total_tokens: number } };
    expect(json.usage.total_tokens).toBe(465);
  });

  it("reports per-request usage, not DeepSeek's session-cumulative total", async () => {
    const withUsage = (total: number): DSStreamEvent[] => [
      { o: "APPEND", p: "response/fragments/-1/content", v: "ok" },
      { p: "response", o: "BATCH", v: [{ p: "accumulated_token_usage", v: total }] },
      { p: "response/status", v: "FINISHED" },
    ];
    const ask = (content: string) =>
      post({
        model: "deepseek-chat",
        messages: [{ role: "user", content }],
        user: "usage-conv",
        stream: false,
      });

    client.events = withUsage(300);
    const first = (await (await ask("usage turn one")).json()) as {
      usage: { total_tokens: number };
    };
    client.events = withUsage(593);
    const second = (await (await ask("usage turn two")).json()) as {
      usage: { total_tokens: number };
    };

    expect(first.usage.total_tokens).toBe(300);
    // DeepSeek accumulates across the session; each turn reports its own cost.
    expect(second.usage.total_tokens).toBe(293);
  });

  it("forwards multimodal content parts as real prompt text", async () => {
    client.events = STANDARD_EVENTS;
    const res = await post({
      model: "deepseek-chat",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Summarise the repo." },
            { type: "image_url", image_url: { url: "data:image/png;base64,AAA" } },
          ],
        },
      ],
      stream: false,
    } as unknown as OpenAIChatRequest);

    expect(res.status).toBe(200);
    const prompt = client.chatCalls[0].prompt;
    // The bug this guards: an array joined into a string became "[object Object]",
    // leaving the model with a tool block and no task.
    expect(prompt).toContain("Summarise the repo.");
    expect(prompt).not.toContain("[object Object]");
    expect(prompt).toContain("image_url omitted");
  });

  it("keeps one conversation when content arrives as parts then as a string", async () => {
    client.events = STANDARD_EVENTS;
    await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: [{ type: "text", text: "same task" }] }],
      stream: false,
    } as unknown as OpenAIChatRequest);
    await post({
      model: "deepseek-chat",
      messages: [
        { role: "user", content: [{ type: "text", text: "same task" }] },
        { role: "assistant", content: "Hello there" },
        { role: "user", content: [{ type: "text", text: "and again" }] },
      ],
      stream: false,
    } as unknown as OpenAIChatRequest);

    // A key that changed shape would fork a second DeepSeek session and lose
    // the conversation.
    expect(client.sessionCalls).toBe(1);
    expect(client.chatCalls[1].prompt).toContain("and again");
  });

  it("uses explicit chat_session_id from body when provided", async () => {
    client.events = STANDARD_EVENTS;
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "Custom session test" }],
      chat_session_id: "my-custom-session-42",
      stream: false,
    });
    expect(res.status).toBe(200);
    expect(client.sessionCalls).toBe(0);
    expect(client.chatCalls[0].chatSessionId).toBe("my-custom-session-42");
  });
});