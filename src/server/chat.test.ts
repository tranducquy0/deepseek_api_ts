import { describe, it, expect, beforeEach, afterEach } from "vitest";
import express from "express";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";
import { chatRouter, planForward } from "./chat.js";
import { fingerprintMessages } from "../shared/convert.js";
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
  /** One entry per finished turn: how many events it yielded, and whether it
   *  was abandoned before yielding all of them. */
  yielded: number[] = [];
  abandoned: boolean[] = [];
  /** Resolves once a chatCompletion generator has finished. */
  aborted: Promise<void>;
  markAborted: () => void;
  /** Delay before the first event, to make overlap observable. */
  delayMs = 0;

  constructor() {
    this.aborted = new Promise((resolve) => {
      this.markAborted = resolve;
    });
  }

  async createSession(): Promise<string> {
    this.sessionCalls++;
    return `sess-${this.sessionCalls}`;
  }

  async *chatCompletion(params: {
    chatSessionId: string;
    parentMessageId: number | string | null;
    prompt: string;
    modelType: string;
    signal?: AbortSignal;
  }): AsyncGenerator<DSStreamEvent> {
    this.chatCalls.push(params);
    let yielded = 0;
    const total = this.events.length;
    try {
      if (this.delayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, this.delayMs));
      }
      if (params.signal?.aborted) return;
      for (const event of this.events) {
        if (params.signal?.aborted) return;
        yield event;
        yielded += 1;
        if (this.delayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, this.delayMs));
        }
      }
    } finally {
      this.yielded.push(yielded);
      // True when the consumer walked away mid-stream, which is what an
      // aborted upstream request looks like from here.
      this.abandoned.push(yielded < total);
      this.markAborted();
    }
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

  it("planForward sends only the appended suffix when history is intact", () => {
    const messages = [
      { role: "user", content: "turn 1" },
      { role: "assistant", content: "resp 1" },
      { role: "tool", content: "tool output" },
      { role: "user", content: "turn 2" },
    ] as any[];
    const entry = {
      lastMessageCount: 2,
      historyDigest: fingerprintMessages(messages.slice(0, 2)),
    };

    const plan = planForward(messages, entry);
    expect(plan.replay).toBe(false);
    // The leading assistant message is skipped: the DeepSeek session already
    // holds it as the parent node.
    expect(plan.delta.map((m: any) => m.role)).toEqual(["tool", "user"]);
  });

  it("planForward replays everything when the client rewrote history", () => {
    // A client that compacts history replaces old turns with a summary.
    const before = [
      { role: "user", content: "turn 1" },
      { role: "assistant", content: "resp 1" },
    ] as any[];
    const compacted = [
      { role: "user", content: "SUMMARY of turn 1" },
      { role: "user", content: "turn 2" },
    ] as any[];

    const plan = planForward(compacted, {
      lastMessageCount: 2,
      historyDigest: fingerprintMessages(before),
    });
    expect(plan.replay).toBe(true);
    expect(plan.delta).toEqual(compacted);
  });

  it("planForward replays when the client sends fewer messages than before", () => {
    const plan = planForward([{ role: "user", content: "only" }] as any[], {
      lastMessageCount: 5,
      historyDigest: "whatever",
    });
    expect(plan.replay).toBe(true);
  });

  it("planForward re-asks the last turn when the client appended nothing", () => {
    const messages = [{ role: "user", content: "turn 1" }] as any[];
    const plan = planForward(messages, {
      lastMessageCount: 1,
      historyDigest: fingerprintMessages(messages),
    });
    expect(plan.replay).toBe(false);
    expect(plan.delta).toEqual(messages);
  });

  it("keeps the conversation when the client echoes a response through its SDK", () => {
    // The Python SDK re-serialises the assistant message with null padding and
    // reordered keys; that must not read as rewritten history.
    const original = [
      { role: "user", content: "run it" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "bash", arguments: '{"command":"ls"}' } },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: "a.txt" },
    ] as any[];
    const roundTripped = [
      { role: "user", content: "run it" },
      {
        refusal: null,
        annotations: null,
        audio: null,
        function_call: null,
        role: "assistant",
        content: null,
        tool_calls: [
          { function: { arguments: '{"command":"ls"}', name: "bash" }, id: "call_1", type: "function" },
        ],
      },
      { tool_call_id: "call_1", content: "a.txt", role: "tool" },
    ] as any[];
    const next = [...roundTripped, { role: "user", content: "and again" }] as any[];

    const plan = planForward(next, {
      lastMessageCount: 3,
      historyDigest: fingerprintMessages(original),
    });
    expect(plan.replay).toBe(false);
    expect(plan.delta.map((m: any) => m.role)).toEqual(["user"]);
  });

  it("rebuilds the session and replays when the client compacts history", async () => {
    client.events = STANDARD_EVENTS;
    await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "original long question" }],
      stream: false,
    });
    expect(client.sessionCalls).toBe(1);

    // The client compacted the first turn away and appended a new one.
    client.events = STANDARD_EVENTS;
    const res = await post({
      model: "deepseek-chat",
      messages: [
        { role: "user", content: "SUMMARY: the original question" },
        { role: "user", content: "follow up" },
      ],
      stream: false,
    });
    expect(res.status).toBe(200);

    // A fresh DeepSeek session, with the whole rewritten history replayed into
    // it — not a misaligned slice of the old conversation.
    expect(client.sessionCalls).toBe(2);
    expect(client.chatCalls[1].chatSessionId).not.toBe(client.chatCalls[0].chatSessionId);
    expect(client.chatCalls[1].parentMessageId).toBeNull();
    const prompt = client.chatCalls[1].prompt;
    expect(prompt).toContain("SUMMARY: the original question");
    expect(prompt).toContain("follow up");
  });

  it("chains normally when the client only appends", async () => {
    client.events = STANDARD_EVENTS;
    await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "first" }],
      stream: false,
    });
    client.events = STANDARD_EVENTS;
    await post({
      model: "deepseek-chat",
      messages: [
        { role: "user", content: "first" },
        { role: "assistant", content: "Hello there" },
        { role: "user", content: "second" },
      ],
      stream: false,
    });

    expect(client.sessionCalls).toBe(1);
    expect(client.chatCalls[1].chatSessionId).toBe(client.chatCalls[0].chatSessionId);
    expect(client.chatCalls[1].parentMessageId).toBe(12345);
    expect(client.chatCalls[1].prompt).toContain("second");
    expect(client.chatCalls[1].prompt).not.toContain("Hello there");
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
    const ask = (messages: OpenAIMessage[]) =>
      post({ model: "deepseek-chat", messages, user: "usage-conv", stream: false });

    client.events = withUsage(300);
    const first = (await (await ask([{ role: "user", content: "turn one" }])).json()) as {
      usage: { total_tokens: number };
    };

    // Appended, not rewritten, so the session is reused and usage is diffed.
    client.events = withUsage(593);
    const second = (
      await (
        await ask([
          { role: "user", content: "turn one" },
          { role: "assistant", content: "ok" },
          { role: "user", content: "turn two" },
        ])
      ).json()
    ) as { usage: { total_tokens: number } };

    expect(first.usage.total_tokens).toBe(300);
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

  it("fails loudly when tool_choice required is not honoured", async () => {
    // The model answered in prose. A 200 + finish_reason "stop" would tell the
    // agent loop the model chose to finish, which is not what happened.
    client.events = [
      { o: "APPEND", p: "response/fragments/-1/content", v: "I cannot call tools." },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "use a tool" }],
      tools: [{ type: "function", function: { name: "search", parameters: {} } }],
      tool_choice: "required",
      stream: false,
    });
    expect(res.status).toBe(502);
    const json = (await res.json()) as { error: { code: string; message: string } };
    expect(json.error.code).toBe("tool_choice_violation");
    expect(json.error.message).toContain('"search"');
  });

  it("passes when tool_choice required is honoured", async () => {
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
      messages: [{ role: "user", content: "use a tool" }],
      tools: [{ type: "function", function: { name: "search", parameters: {} } }],
      tool_choice: "required",
      stream: false,
    });
    expect(res.status).toBe(200);
    const json = (await res.json()) as { choices: { finish_reason: string }[] };
    expect(json.choices[0].finish_reason).toBe("tool_calls");
  });

  it("rejects tool_choice required without tools", async () => {
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "no tools here" }],
      tool_choice: "required",
      stream: false,
    });
    expect(res.status).toBe(400);
    const json = (await res.json()) as { error: { code: string } };
    expect(json.error.code).toBe("tool_choice_without_tools");
    // No upstream call should have been made.
    expect(client.chatCalls).toHaveLength(0);
  });

  it("enforces a forced tool_choice function", async () => {
    client.events = [
      {
        o: "APPEND",
        p: "response/fragments/-1/content",
        v: '{"name":"search","arguments":{}}',
      },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "force one" }],
      tools: [
        { type: "function", function: { name: "search", parameters: {} } },
        { type: "function", function: { name: "write", parameters: {} } },
      ],
      tool_choice: { type: "function", function: { name: "write" } },
      stream: false,
    });
    expect(res.status).toBe(502);
    const json = (await res.json()) as { error: { code: string; message: string } };
    expect(json.error.code).toBe("tool_choice_violation");
    expect(json.error.message).toContain("write");
  });

  it("rejects a tool call when tool_choice is none", async () => {
    client.events = [
      {
        o: "APPEND",
        p: "response/fragments/-1/content",
        v: '{"name":"search","arguments":{}}',
      },
      { p: "response/status", v: "FINISHED" },
    ];
    const res = await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "no tools please" }],
      tools: [{ type: "function", function: { name: "search", parameters: {} } }],
      tool_choice: "none",
      stream: false,
    });
    expect(res.status).toBe(502);
    const json = (await res.json()) as { error: { code: string } };
    expect(json.error.code).toBe("tool_choice_violation");
  });

  it("serialises concurrent turns on the same conversation", async () => {
    client.events = STANDARD_EVENTS;
    client.delayMs = 20;

    // Without serialisation both turns would be in flight at once and the
    // second would overwrite the first's parent pointer.
    const results = await Promise.all([
      post({ model: "deepseek-chat", messages: [{ role: "user", content: "turn one" }], user: "shared" }),
      post({
        model: "deepseek-chat",
        messages: [
          { role: "user", content: "turn one" },
          { role: "assistant", content: "Hello there" },
          { role: "user", content: "turn two" },
        ],
        user: "shared",
      }),
    ]);
    for (const res of results) expect(res.status).toBe(200);

    // One DeepSeek session, and the second turn chained from the first's
    // response instead of racing it.
    expect(client.sessionCalls).toBe(1);
    expect(client.chatCalls).toHaveLength(2);
    expect(client.chatCalls[1].parentMessageId).toBe(12345);
    expect(client.chatCalls[1].prompt).toContain("turn two");
  });

  it("stops reading the upstream stream when the client disconnects", async () => {
    client.events = STANDARD_EVENTS;
    client.delayMs = 40;

    const controller = new AbortController();
    const pending = fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "deepseek-chat",
        messages: [{ role: "user", content: "abort me" }],
        stream: true,
      }),
      signal: controller.signal,
    });

    // Headers are sent immediately, so the response resolves; the body is
    // where the disconnect shows up.
    const res = await pending;
    expect(res.headers.get("content-type")).toContain("text/event-stream");

    // Wait until the upstream turn is genuinely underway (events arrive every
    // delayMs), then hang up mid-stream.
    await new Promise((resolve) => setTimeout(resolve, 60));
    await res.body?.cancel();
    controller.abort();
    await client.aborted;

    // Stopped partway through, which only happens if the abort signal reached
    // the upstream generator.
    expect(client.yielded[0]).toBeGreaterThan(0);
    expect(client.yielded[0]).toBeLessThan(client.events.length);
  });

  it("stays healthy and keeps context after a client disconnects", async () => {
    client.events = [
      { response_message_id: 777, request_message_id: 1 },
      { o: "APPEND", p: "response/fragments/-1/content", v: "partial" },
      { p: "response/status", v: "FINISHED" },
    ] as DSStreamEvent[];
    client.delayMs = 40;

    const controller = new AbortController();
    const pending = fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "deepseek-chat",
        messages: [{ role: "user", content: "interrupted" }],
        stream: true,
        user: "after-abort",
      }),
      signal: controller.signal,
    });
    const res = await pending;
    // Let the response id arrive so the early chaining has something to store.
    await new Promise((resolve) => setTimeout(resolve, 60));
    await res.body?.cancel();
    controller.abort();
    await client.aborted;

    // A disconnect must not discard the conversation: the response id was
    // chained when it arrived, so the next turn continues from it.
    client.delayMs = 0;
    client.events = STANDARD_EVENTS;
    const next = await post({
      model: "deepseek-chat",
      messages: [
        { role: "user", content: "interrupted" },
        { role: "assistant", content: "partial" },
        { role: "user", content: "carry on" },
      ],
      user: "after-abort",
    });
    expect(next.status).toBe(200);
    expect(client.chatCalls[1].parentMessageId).toBe(777);
    expect(client.sessionCalls).toBe(1);
  });

  it("chains the next turn from a response id learned before completion", async () => {
    // A turn cut short still created a message in the DeepSeek session, so the
    // response id must be persisted as soon as it arrives.
    client.events = [
      { response_message_id: 555, request_message_id: 1 },
      { o: "APPEND", p: "response/fragments/-1/content", v: "partial" },
      { p: "response/status", v: "FINISHED" },
    ] as DSStreamEvent[];

    await post({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "first" }],
      user: "recover",
    });
    client.events = STANDARD_EVENTS;
    await post({
      model: "deepseek-chat",
      messages: [
        { role: "user", content: "first" },
        { role: "assistant", content: "partial" },
        { role: "user", content: "second" },
      ],
      user: "recover",
    });

    expect(client.chatCalls[1].parentMessageId).toBe(555);
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