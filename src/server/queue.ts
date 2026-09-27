/**
 * Serialises work per key.
 *
 * A conversation is a single DeepSeek session with one `parent_message_id`
 * pointer, so two overlapping turns would race: whichever finished last would
 * overwrite the pointer and the forwarding state, and the other turn's messages
 * would be orphaned. Requests for the same conversation therefore run one at a
 * time; different conversations stay fully parallel.
 */
export class KeyedSerialQueue {
  private tails = new Map<string, Promise<unknown>>();

  /** Number of keys with work queued or running. */
  get size(): number {
    return this.tails.size;
  }

  /**
   * Run `fn` once every previously queued task for `key` has settled.
   * A failing task does not poison the queue: the next one still runs.
   */
  async run<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const previous = this.tails.get(key) ?? Promise.resolve();
    // Then(fn, fn) rather than then(fn): a rejected predecessor must not skip
    // this task.
    const result = previous.then(fn, fn);
    const tail = result.then(
      () => undefined,
      () => undefined
    );
    this.tails.set(key, tail);

    try {
      return await result;
    } finally {
      // Only clear the slot if no newer task has claimed it.
      if (this.tails.get(key) === tail) this.tails.delete(key);
    }
  }
}
