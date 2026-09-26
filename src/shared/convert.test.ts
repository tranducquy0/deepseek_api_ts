import { describe, it, expect } from "vitest";
import {
  applyStreamEvent,
  buildPrompt,
  buildModelList,
  createStreamState,
  extractToolCall,
  makeChunk,
  makeToolCallChunk,
  mapModel,
  parseToolCall,
} from "./convert.js";
import type { DSStreamEvent } from "./types.js";

describe("buildPrompt", () => {
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
    const call = parseToolCall('{"name":"get_weather","arguments":{"city":"Hanoi"}}');
    expect(call?.name).toBe("get_weather");
    expect(JSON.parse(call!.arguments)).toEqual({ city: "Hanoi" });
    expect(call?.leadingText).toBeUndefined();
  });

  it("parses fenced JSON", () => {
    const call = parseToolCall('```json\n{"name":"search","params":{"q":"x"}}\n```');
    expect(call?.name).toBe("search");
    expect(JSON.parse(call!.arguments)).toEqual({ q: "x" });
  });

  it("parses JSON embedded in prose and captures leading text", () => {
    const call = parseToolCall(
      'Let me check that.\n{"name":"f","arguments":{}}\nDone.'
    );
    expect(call?.name).toBe("f");
    expect(call?.leadingText).toBe("Let me check that.");
  });

  it("returns null for non-JSON answers", () => {
    expect(parseToolCall("just a normal answer")).toBeNull();
    expect(parseToolCall("{}")).toBeNull();
    expect(parseToolCall('{"name":"","arguments":{}}')).toBeNull();
  });
});


describe("extractToolCall", () => {
  it("extracts a tool call only in tool mode", () => {
    const state = createStreamState(true);
    state.content = '{"name":"f","arguments":{"a":1}}';
    expect(extractToolCall(state)).not.toBeNull();

    const plain = createStreamState(false);
    plain.content = '{"name":"f","arguments":{}}';
    expect(extractToolCall(plain)).toBeNull();
  });
});

describe("chunk helpers", () => {
  it("builds content chunks", () => {
    const chunk = makeChunk("deepseek-chat", { content: "hi" });
    expect(chunk.object).toBe("chat.completion.chunk");
    expect(chunk.choices[0].delta.content).toBe("hi");
    expect(chunk.choices[0].finish_reason).toBeNull();
  });

  it("builds tool-call chunks with tool_calls finish reason", () => {
    const chunk = makeToolCallChunk("deepseek-chat", {
      id: "call_1",
      name: "f",
      arguments: "{}",
    });
    expect(chunk.choices[0].finish_reason).toBe("tool_calls");
    expect(chunk.choices[0].delta.tool_calls).toMatchObject([
      { id: "call_1", type: "function", function: { name: "f", arguments: "{}" } },
    ]);
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