import { describe, it, expect } from "vitest";
import { KeyedSerialQueue } from "./queue.js";

const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));

describe("KeyedSerialQueue", () => {
  it("runs tasks for the same key one at a time", async () => {
    const queue = new KeyedSerialQueue();
    const order: string[] = [];

    const task = (name: string, delay: number) => async () => {
      order.push(`${name}:start`);
      await tick(delay);
      order.push(`${name}:end`);
      return name;
    };

    // The first task is the slowest, so any overlap would be obvious.
    const results = await Promise.all([
      queue.run("a", task("first", 30)),
      queue.run("a", task("second", 1)),
      queue.run("a", task("third", 1)),
    ]);

    expect(results).toEqual(["first", "second", "third"]);
    expect(order).toEqual([
      "first:start",
      "first:end",
      "second:start",
      "second:end",
      "third:start",
      "third:end",
    ]);
  });

  it("keeps different keys fully parallel", async () => {
    const queue = new KeyedSerialQueue();
    const order: string[] = [];

    const slow = queue.run("a", async () => {
      order.push("a:start");
      await tick(25);
      order.push("a:end");
    });
    const fast = queue.run("b", async () => {
      order.push("b:start");
      order.push("b:end");
    });

    await Promise.all([slow, fast]);
    // b finished while a was still running.
    expect(order).toEqual(["a:start", "b:start", "b:end", "a:end"]);
  });

  it("still runs the next task after one throws", async () => {
    const queue = new KeyedSerialQueue();
    const failing = queue.run("a", async () => {
      throw new Error("boom");
    });
    const following = queue.run("a", async () => "ok");

    await expect(failing).rejects.toThrow("boom");
    await expect(following).resolves.toBe("ok");
  });

  it("releases its slot once a key settles", async () => {
    const queue = new KeyedSerialQueue();
    await queue.run("a", async () => "done");
    expect(queue.size).toBe(0);
  });

  it("releases the slot after a failure too", async () => {
    const queue = new KeyedSerialQueue();
    await expect(
      queue.run("a", async () => {
        throw new Error("boom");
      })
    ).rejects.toThrow();
    expect(queue.size).toBe(0);
  });
});
