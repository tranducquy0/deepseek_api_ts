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
    ).toBe("Hel");
    expect(state.content).toBe("Hel");

    // APPEND event
    expect(
      applyStreamEvent(state, {
        p: "response/fragments/0/content",
        o: "APPEND",
        v: "lo",
      })
    ).toBe("lo");
    expect(state.content).toBe("Hello");

    // Cumulative SET event on response/content
    expect(
      applyStreamEvent(state, {
        p: "response/content",
        o: "SET",
        v: "Hello world",
      })
    ).toBe(" world");
    expect(state.content).toBe("Hello world");

    // Event with missing operation type
    expect(
      applyStreamEvent(state, {
        p: "response/content",
        v: "!",
      })
    ).toBe("!");
    expect(state.content).toBe("Hello world!");
  });

  it("buffers without streaming content in tool mode", () => {
    const state = createStreamState(true);
    expect(applyStreamEvent(state, append("response/fragments/-1/content", "{}"))).toBe("");
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