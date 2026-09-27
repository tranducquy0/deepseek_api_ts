import { describe, it, expect } from "vitest";
import {
  applyStreamEvent,
  buildPrompt,
  buildModelList,
  checkToolChoice,
  contentToText,
  createStreamState,
  extractToolCalls,
  fingerprintMessages,
  makeChunk,
  makeToolCallChunk,
  makeUsageChunk,
  mapModel,
  newCompletionMeta,
  parseToolCalls,
} from "./convert.js";
import type { DSStreamEvent } from "./types.js";

describe("contentToText", () => {
  it("passes plain strings through", () => {
    expect(contentToText("hello")).toBe("hello");
  });

  it("treats null and undefined as empty", () => {
    expect(contentToText(null)).toBe("");
    expect(contentToText(undefined)).toBe("");
  });

  it("flattens OpenAI content parts instead of stringifying them", () => {
    expect(
      contentToText([
        { type: "text", text: "first" },
        { type: "text", text: "second" },
      ])
    ).toBe("first\nsecond");
  });

  it("names parts it cannot forward rather than dropping them", () => {
    const text = contentToText([
      { type: "text", text: "what is this?" },
      { type: "image_url", image_url: { url: "data:image/png;base64,AAA" } },
    ]);
    expect(text).toContain("what is this?");
    expect(text).toContain("image_url omitted");
    expect(text).not.toContain("[object Object]");
  });
});

describe("buildPrompt", () => {
  it("renders a multimodal user turn as real text", () => {
    const prompt = buildPrompt([
      { role: "user", content: [{ type: "text", text: "Summarise the repo." }] },
    ]);
    expect(prompt).toBe("Summarise the repo.");
    expect(prompt).not.toContain("[object Object]");
  });

  it("flattens content parts in every role", () => {
    const prompt = buildPrompt([
      { role: "system", content: [{ type: "text", text: "be terse" }] },
      { role: "user", content: [{ type: "text", text: "hi" }] },
      { role: "assistant", content: [{ type: "text", text: "hello" }] },
      { role: "tool", tool_call_id: "c1", content: [{ type: "text", text: "42" }] },
    ]);
    expect(prompt).toContain("[System]: be terse");
    expect(prompt).toContain("[Assistant]: hello");
    expect(prompt).toContain("[Tool result for c1]: 42");
    expect(prompt).not.toContain("[object Object]");
  });

  it("flattens system/user/assistant messages", () => {
    const prompt = buildPrompt([
      { role: "system", content: "You are helpful" },
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello" },
    ]);
    expect(prompt).toContain("[System]: You are helpful");
    expect(prompt).toContain("Hi");
    expect(prompt).toContain("[Assistant]: Hello");
  });

  it("injects the tool block when tools are provided", () => {
    const prompt = buildPrompt(
      [{ role: "user", content: "weather?" }],
      [
        {
          type: "function",
          function: { name: "get_weather", description: "x", parameters: {} },
        },
      ]
    );
    expect(prompt).toContain("# Available Tools");
    expect(prompt).toContain("get_weather");
    expect(prompt).toContain("weather?");
  });

  it("names the callable functions and shows a worked example", () => {
    const prompt = buildPrompt(
      [{ role: "user", content: "weather?" }],
      [
        {
          type: "function",
          function: {
            name: "get_weather",
            parameters: { type: "object", properties: { city: { type: "string" } } },
          },
        },
        { type: "function", function: { name: "get_time", parameters: {} } },
      ]
    );
    expect(prompt).toContain('"name" must be exactly one of: get_weather, get_time');
    expect(prompt).toContain('{"name": "get_weather", "arguments": { "city": "..." }}');
  });

  it("honours tool_choice none", () => {
    const prompt = buildPrompt(
      [{ role: "user", content: "hi" }],
      [{ type: "function", function: { name: "f", parameters: {} } }],
      "none"
    );
    expect(prompt).toContain("Do NOT call any tool");
  });

  it("honours a forced tool_choice", () => {
    const prompt = buildPrompt(
      [{ role: "user", content: "hi" }],
      [{ type: "function", function: { name: "f", parameters: {} } }],
      { type: "function", function: { name: "f" } }
    );
    expect(prompt).toContain('MUST call the function "f"');
  });

  it("renders assistant tool_calls and tool results for continuation turns", () => {
    const prompt = buildPrompt([
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "get_weather", arguments: '{"city":"Hanoi"}' },
          },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: "Sunny" },
    ]);
    expect(prompt).toContain('{"name":"get_weather","arguments":{"city":"Hanoi"}}');
    expect(prompt).toContain("[Tool result for call_1]: Sunny");
  });

  it("skips null content instead of crashing", () => {
    expect(() =>
      buildPrompt([
        { role: "assistant", content: null },
        { role: "user", content: "go" },
      ])
    ).not.toThrow();
  });
});

describe("applyStreamEvent", () => {
  const append = (path: string, v: string): DSStreamEvent => ({
    p: path,
    o: "APPEND",
    v,
  });

  it("accumulates content deltas for APPEND, SET, REPLACE, and missing operation types", () => {
    const state = createStreamState();
    // SET event with initial fragment chunk
    expect(
      applyStreamEvent(state, {
        p: "response/fragments/0/content",
        o: "SET",
        v: "Hel",
      })
    ).toEqual({ content: "Hel", reasoning: "" });
    expect(state.content).toBe("Hel");

    // APPEND event
    expect(
      applyStreamEvent(state, {
        p: "response/fragments/0/content",
        o: "APPEND",
        v: "lo",
      })
    ).toEqual({ content: "lo", reasoning: "" });
    expect(state.content).toBe("Hello");

    // Cumulative SET event on response/content
    expect(
      applyStreamEvent(state, {
        p: "response/content",
        o: "SET",
        v: "Hello world",
      })
    ).toEqual({ content: " world", reasoning: "" });
    expect(state.content).toBe("Hello world");

    // Event with missing operation type
    expect(
      applyStreamEvent(state, {
        p: "response/content",
        v: "!",
      })
    ).toEqual({ content: "!", reasoning: "" });
    expect(state.content).toBe("Hello world!");
  });

  it("buffers without streaming content in tool mode", () => {
    const state = createStreamState(true);
    expect(applyStreamEvent(state, append("response/fragments/-1/content", "{}"))).toEqual({
      content: "",
      reasoning: "",
    });
    expect(state.content).toBe("{}");
  });

  it("collects thinking content", () => {
    const state = createStreamState();
    applyStreamEvent(state, append("response/fragments/-1/thinking_content", "hmm"));
    expect(state.thinking).toBe("hmm");
    expect(state.content).toBe("");
  });

  it("marks finished on FINISHED status", () => {
    const state = createStreamState();
    applyStreamEvent(state, { p: "response/status", v: "FINISHED" });
    expect(state.finished).toBe(true);
  });

  it("captures response_message_id", () => {
    const state = createStreamState();
    applyStreamEvent(state, { p: "response/message_id", v: 42, response_message_id: 42 });
    expect(state.responseMessageId).toBe(42);
  });

  // DeepSeek's real wire format: a path is declared once, then every token
  // arrives as a bare {"v":"…"} frame that inherits it.
  it("keeps every token of the live DeepSeek wire format", () => {
    const state = createStreamState();
    const frames: DSStreamEvent[] = [
      { request_message_id: 1, response_message_id: 2 },
      {
        v: {
          response: {
            message_id: 2,
            fragments: [{ id: 2, type: "RESPONSE", content: "#" }],
          },
        },
      },
      { p: "response/fragments/-1/content", o: "APPEND", v: " Checking" },
      { v: " if" },
      { v: " a" },
      { v: " Python" },
      { p: "response/fragments/-1/elapsed_secs", o: "SET", v: 0.42 },
      { p: "response", o: "BATCH", v: [{ p: "accumulated_token_usage", v: 12 }] },
      { p: "response/status", o: "SET", v: "FINISHED" },
    ];

    let out = "";
    for (const frame of frames) out += applyStreamEvent(state, frame).content;

    expect(out).toBe("# Checking if a Python");
    expect(state.content).toBe("# Checking if a Python");
    expect(state.finished).toBe(true);
    expect(state.responseMessageId).toBe(2);
  });

  it("seeds the opening token carried by the response snapshot", () => {
    const state = createStreamState();
    const delta = applyStreamEvent(state, {
      v: { response: { fragments: [{ id: 2, type: "RESPONSE", content: "Hel" }] } },
    });
    expect(delta.content).toBe("Hel");
  });

  it("emits text carried by newly announced fragments", () => {
    const state = createStreamState();
    applyStreamEvent(state, append("response/fragments/-1/content", "One"));
    const delta = applyStreamEvent(state, {
      p: "response/fragments",
      o: "APPEND",
      v: [{ id: 3, type: "RESPONSE", content: "Two" }],
    });
    expect(delta.content).toBe("Two");
    expect(state.content).toBe("OneTwo");
    // Bare frames after the announcement target the new fragment
    expect(applyStreamEvent(state, { v: "!" }).content).toBe("!");
  });

  it("routes THINK fragments to reasoning_content and RESPONSE to content", () => {
    const state = createStreamState();
    applyStreamEvent(state, {
      v: { response: { fragments: [{ id: 2, type: "THINK", content: "We" }] } },
    });
    expect(applyStreamEvent(state, { p: "response/fragments/-1/content", o: "APPEND", v: " need" })).toEqual({
      content: "",
      reasoning: " need",
    });

    // The answer fragment arrives as an announcement, then streams as bare frames
    applyStreamEvent(state, {
      p: "response/fragments",
      o: "APPEND",
      v: [{ id: 3, type: "RESPONSE", content: "Use" }],
    });
    expect(applyStreamEvent(state, { v: " `isinstance()`" }).content).toBe(" `isinstance()`");

    expect(state.thinking).toBe("We need");
    expect(state.content).toBe("Use `isinstance()`");
  });

  it("ignores unknown paths and bare frames before any path is declared", () => {
    const state = createStreamState();
    expect(applyStreamEvent(state, { v: "orphan token" })).toEqual({
      content: "",
      reasoning: "",
    });
    expect(applyStreamEvent(state, { p: "response/fragments/-1/elapsed_secs", o: "SET", v: 1 })).toEqual({
      content: "",
      reasoning: "",
    });
    expect(state.content).toBe("");
  });
});

describe("parseToolCall", () => {
  it("parses a plain JSON object", () => {
    const { calls, leadingText } = parseToolCalls(
      '{"name":"get_weather","arguments":{"city":"Hanoi"}}'
    );
    expect(calls[0].name).toBe("get_weather");
    expect(JSON.parse(calls[0].arguments)).toEqual({ city: "Hanoi" });
    expect(leadingText).toBeUndefined();
  });

  it("parses fenced JSON", () => {
    const { calls } = parseToolCalls('```json\n{"name":"search","params":{"q":"x"}}\n```');
    expect(calls[0].name).toBe("search");
    expect(JSON.parse(calls[0].arguments)).toEqual({ q: "x" });
  });

  it("parses JSON embedded in prose and captures leading text", () => {
    const { calls, leadingText } = parseToolCalls(
      'Let me check that.\n{"name":"f","arguments":{}}\nDone.'
    );
    expect(calls[0].name).toBe("f");
    expect(leadingText).toBe("Let me check that.");
  });

  it("parses several calls in one response, indexed in order", () => {
    const { calls } = parseToolCalls(
      '{"name":"read","arguments":{"p":"a"}}\n{"name":"read","arguments":{"p":"b"}}'
    );
    expect(calls.map((c) => [c.index, c.name, c.arguments])).toEqual([
      [0, "read", '{"p":"a"}'],
      [1, "read", '{"p":"b"}'],
    ]);
  });

  it("ignores a stray brace in prose instead of swallowing the call", () => {
    const { calls } = parseToolCalls(
      'Use {placeholder} then:\n{"name":"read","arguments":{}}'
    );
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toBe("read");
  });

  it("does not treat braces inside JSON strings as object boundaries", () => {
    const { calls } = parseToolCalls('{"name":"f","arguments":{"q":"a}b{c"}}');
    expect(calls[0].name).toBe("f");
    expect(JSON.parse(calls[0].arguments)).toEqual({ q: "a}b{c" });
  });

  it("returns no calls for non-JSON answers", () => {
    expect(parseToolCalls("just a normal answer").calls).toEqual([]);
    expect(parseToolCalls("{}").calls).toEqual([]);
    expect(parseToolCalls('{"name":"","arguments":{}}').calls).toEqual([]);
  });

  it("binds a wrapper-less argument object when only one function can take it", () => {
    // DeepSeek frequently drops the {"name":…,"arguments":…} wrapper.
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "bash",
          parameters: {
            type: "object",
            required: ["command"],
            properties: { command: { type: "string" }, timeout: { type: "number" } },
          },
        },
      },
    ];
    const { calls, error } = parseToolCalls('{"command": "ls -la"}', tools);
    expect(error).toBeUndefined();
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toBe("bash");
    expect(JSON.parse(calls[0].arguments)).toEqual({ command: "ls -la" });
  });

  it("refuses to guess when a wrapper-less object fits several functions", () => {
    // read and write both take a bare {path}; picking one could run the wrong
    // tool, so this must fail loudly instead.
    const tools = ["read", "write"].map((name) => ({
      type: "function" as const,
      function: {
        name,
        parameters: {
          type: "object",
          required: ["path"],
          properties: { path: { type: "string" } },
        },
      },
    }));
    const result = parseToolCalls('{"path": "a.py"}', tools);
    expect(result.calls).toEqual([]);
    expect(result.error).toContain("did not satisfy exactly one function");
    expect(result.error).toContain("read");
    expect(result.error).toContain("write");
  });

  it("does not hijack a JSON answer that no function accepts", () => {
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "bash",
          parameters: {
            type: "object",
            required: ["command"],
            properties: { command: { type: "string" } },
          },
        },
      },
    ];
    const result = parseToolCalls('{"answer": "42"}', tools);
    expect(result.calls).toEqual([]);
    expect(result.error).toBeUndefined();
  });

  it("accepts tool/function as aliases for the name field", () => {
    expect(parseToolCalls('{"tool":"read","args":{"path":"a"}}').calls[0].name).toBe("read");
    expect(
      parseToolCalls('{"function":"read","parameters":{"path":"a"}}').calls[0].name
    ).toBe("read");
  });

  it("reports a loud error when a tool call cannot be parsed", () => {
    const result = parseToolCalls('{"name": "read", "arguments": {"p": "a"');
    expect(result.calls).toEqual([]);
    expect(result.error).toMatch(/unparseable tool call/);
    expect(result.error).toContain('"name": "read"');
  });

  it("does not report an error for prose that merely mentions JSON", () => {
    expect(parseToolCalls("Here is a dict: {'a': 1}").error).toBeUndefined();
    expect(parseToolCalls('Use "name": value freely').error).toBeUndefined();
  });
});

describe("extractToolCalls", () => {
  it("extracts tool calls only in tool mode", () => {
    const state = createStreamState(true);
    state.content = '{"name":"f","arguments":{"a":1}}';
    expect(extractToolCalls(state).calls).toHaveLength(1);

    const plain = createStreamState(false);
    plain.content = '{"name":"f","arguments":{}}';
    expect(extractToolCalls(plain).calls).toEqual([]);
  });

  it("surfaces the parse error in tool mode only", () => {
    const state = createStreamState(true);
    state.content = '{"name": "f", "arguments":';
    expect(extractToolCalls(state).error).toBeDefined();

    const plain = createStreamState(false);
    plain.content = '{"name": "f", "arguments":';
    expect(extractToolCalls(plain).error).toBeUndefined();
  });
});

describe("checkToolChoice", () => {
  const tools = [
    { type: "function" as const, function: { name: "search", parameters: {} } },
    { type: "function" as const, function: { name: "write", parameters: {} } },
  ];
  const call = (name: string) => ({
    calls: [{ index: 0, id: "c1", name, arguments: "{}" }],
  });

  it("passes when required is satisfied", () => {
    expect(checkToolChoice("required", tools, call("search"))).toBeNull();
  });

  it("rejects a prose answer when a call was required", () => {
    const error = checkToolChoice("required", tools, { calls: [] });
    expect(error).toContain("required");
    expect(error).toContain('"search"');
    expect(error).toContain('"write"');
  });

  it("rejects a call when none was requested", () => {
    expect(checkToolChoice("none", tools, call("search"))).toContain("none");
    expect(checkToolChoice("none", tools, { calls: [] })).toBeNull();
  });

  it("rejects a call to a function other than the forced one", () => {
    const forced = { type: "function" as const, function: { name: "write" } };
    expect(checkToolChoice(forced, tools, call("write"))).toBeNull();
    expect(checkToolChoice(forced, tools, call("search"))).toContain("write");
    expect(checkToolChoice(forced, tools, { calls: [] })).toContain("write");
  });

  it("does not constrain auto", () => {
    expect(checkToolChoice("auto", tools, { calls: [] })).toBeNull();
    expect(checkToolChoice(undefined, tools, call("search"))).toBeNull();
  });
});

describe("fingerprintMessages", () => {
  it("is stable across client re-serialisation", () => {
    const original = [
      { role: "user", content: "go" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "c1", type: "function", function: { name: "bash", arguments: '{"command":"ls"}' } },
        ],
      },
    ] as any[];
    const roundTripped = [
      { role: "user", content: "go" },
      {
        role: "assistant",
        content: null,
        refusal: null,
        audio: null,
        annotations: null,
        tool_calls: [
          { function: { arguments: '{"command":"ls"}', name: "bash" }, id: "c1", type: "function" },
        ],
      },
    ] as any[];
    expect(fingerprintMessages(roundTripped)).toBe(fingerprintMessages(original));
  });

  it("ignores tool-call ids but not names or arguments", () => {
    const a = [
      { role: "assistant", content: null, tool_calls: [{ id: "x", type: "function", function: { name: "f", arguments: "{}" } }] },
    ] as any[];
    const b = [
      { role: "assistant", content: null, tool_calls: [{ id: "y", type: "function", function: { name: "f", arguments: "{}" } }] },
    ] as any[];
    const c = [
      { role: "assistant", content: null, tool_calls: [{ id: "x", type: "function", function: { name: "g", arguments: "{}" } }] },
    ] as any[];
    expect(fingerprintMessages(a)).toBe(fingerprintMessages(b));
    expect(fingerprintMessages(a)).not.toBe(fingerprintMessages(c));
  });

  it("treats string and content-part forms as the same text", () => {
    expect(
      fingerprintMessages([{ role: "user", content: [{ type: "text", text: "hi" }] }] as any[])
    ).toBe(fingerprintMessages([{ role: "user", content: "hi" }] as any[]));
  });

  it("changes when a message is edited", () => {
    const a = fingerprintMessages([{ role: "user", content: "one" }] as any[]);
    const b = fingerprintMessages([{ role: "user", content: "two" }] as any[]);
    expect(a).not.toBe(b);
  });
});

describe("chunk helpers", () => {
  it("builds content chunks", () => {
    const chunk = makeChunk(newCompletionMeta(), "deepseek-chat", { content: "hi" });
    expect(chunk.object).toBe("chat.completion.chunk");
    expect(chunk.choices[0].delta.content).toBe("hi");
    expect(chunk.choices[0].finish_reason).toBeNull();
  });

  it("reuses one id and created across every chunk of a completion", () => {
    const meta = newCompletionMeta();
    const a = makeChunk(meta, "deepseek-chat", { content: "a" });
    const b = makeChunk(meta, "deepseek-chat", { content: "b" }, "stop");
    expect(a.id).toBe(b.id);
    expect(a.created).toBe(b.created);
  });

  it("mints a distinct id per completion", () => {
    expect(newCompletionMeta().id).not.toBe(newCompletionMeta().id);
  });

  it("leaves finish_reason null on tool-call chunks", () => {
    const chunk = makeToolCallChunk(newCompletionMeta(), "deepseek-chat", [
      { index: 0, id: "call_1", name: "f", arguments: "{}" },
      { index: 1, id: "call_2", name: "g", arguments: "{}" },
    ]);
    // OpenAI sets finish_reason exactly once, on the terminal chunk.
    expect(chunk.choices[0].finish_reason).toBeNull();
    expect(chunk.choices[0].delta.tool_calls).toMatchObject([
      { index: 0, id: "call_1", type: "function", function: { name: "f" } },
      { index: 1, id: "call_2", type: "function", function: { name: "g" } },
    ]);
  });

  it("builds a usage-only chunk with no choices", () => {
    const meta = newCompletionMeta();
    const chunk = makeUsageChunk(meta, "deepseek-chat", {
      prompt_tokens: 0,
      completion_tokens: 465,
      total_tokens: 465,
    });
    expect(chunk.id).toBe(meta.id);
    expect(chunk.choices).toEqual([]);
    expect(chunk.usage?.total_tokens).toBe(465);
  });
});

describe("model mapping", () => {
  it("maps OpenAI ids to DeepSeek model types", () => {
    expect(mapModel("deepseek-chat")).toBe("default");
    expect(mapModel("deepseek-reasoner")).toBe("expert");
    expect(mapModel("unknown-model")).toBe("default");
  });

  it("lists the advertised models", () => {
    const list = buildModelList();
    expect(list.object).toBe("list");
    expect(list.data.map((m) => m.id)).toEqual(["deepseek-chat", "deepseek-reasoner"]);
  });
});