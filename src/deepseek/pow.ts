import { readFile } from "node:fs/promises";
import type { DSBreadcrumb, DSPoWResponse } from "../shared/types.js";

const WASM_URL = new URL("../vendor/sha3_wasm_bg.wasm", import.meta.url);

// ── Pure-TS DeepSeekHashV1 (fallback) ───────────────────────────────
// DeepSeekHashV1 is SHA3-256 whose Keccak-f[1600] permutation skips
// round 0, i.e. only round constants 1..23 are applied. This matches
// the official wasm_solve outputs byte-for-byte.

const MASK_64 = 0xffffffffffffffffn;

const RC = [
  0x0000000000000001n, 0x0000000000008082n, 0x800000000000808an,
  0x8000000080008000n, 0x000000000000808bn, 0x0000000080000001n,
  0x8000000080008081n, 0x8000000000008009n, 0x000000000000008an,
  0x0000000000000088n, 0x0000000080008009n, 0x000000008000000an,
  0x000000008000808bn, 0x800000000000008bn, 0x8000000000008089n,
  0x8000000000008003n, 0x8000000000008002n, 0x8000000000000080n,
  0x000000000000800an, 0x800000008000000an, 0x8000000080008081n,
  0x8000000000008080n, 0x0000000080000001n, 0x8000000080008008n,
];

function rotl(x: bigint, n: number): bigint {
  return ((x << BigInt(n)) | (x >> BigInt(64 - n))) & MASK_64;
}

function keccakF1600(state: bigint[], rounds: number, skipFirst: boolean): void {
  const c = new Array<bigint>(5);
  const row = new Array<bigint>(5);
  const start = skipFirst ? 1 : 0;

  for (let round = 0; round < rounds; round++) {
    // θ
    for (let x = 0; x < 5; x++)
      c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    for (let x = 0; x < 5; x++) {
      const d = c[(x + 4) % 5] ^ rotl(c[(x + 1) % 5], 1);
      for (let y = 0; y < 5; y++) state[x + 5 * y] ^= d;
    }

    // ρ + π (XKCP traversal: flat = x + 5*y)
    let x = 1;
    let y = 0;
    let current = state[x + 5 * y];
    for (let t = 0; t < 24; t++) {
      const r = Math.floor(((t + 1) * (t + 2)) / 2) % 64;
      const nextY = (2 * x + 3 * y) % 5;
      x = y;
      y = nextY;
      const tmp = state[x + 5 * y];
      state[x + 5 * y] = rotl(current, r);
      current = tmp;
    }

    // χ
    for (let y = 0; y < 5; y++) {
      for (let i = 0; i < 5; i++) row[i] = state[i + 5 * y];
      for (let i = 0; i < 5; i++)
        state[i + 5 * y] = row[i] ^ (~row[(i + 1) % 5] & row[(i + 2) % 5]);
    }

    // ι
    state[0] ^= RC[start + round];
  }
}

const RATE = 136; // bytes for 256-bit output

function absorb(state: bigint[], block: Buffer): void {
  for (let i = 0; i < RATE; i++) {
    const lane = Math.floor(i / 8);
    const shift = 8 * (i % 8);
    state[lane] = (state[lane] ^ (BigInt(block[i]) << BigInt(shift))) & MASK_64;
  }
}

function deepSeekHash(input: Buffer, rounds = 23, skipFirst = true): Buffer {
  const state = new Array<bigint>(25).fill(0n);

  // Absorb full blocks except the last (which needs padding)
  const last = input.length - (input.length % RATE);
  for (let off = 0; off < last; off += RATE) {
    absorb(state, input.subarray(off, off + RATE));
    keccakF1600(state, rounds, skipFirst);
  }

  // Multi-rate padding: 0x06 ... 0x80 in a single final block
  const block = Buffer.alloc(RATE, 0);
  input.copy(block, 0, last);
  block[input.length - last] = 0x06;
  block[RATE - 1] |= 0x80;
  absorb(state, block);
  keccakF1600(state, rounds, skipFirst);

  // Squeeze 32 bytes
  const out = Buffer.alloc(32);
  for (let i = 0; i < 32; i++) {
    const lane = Math.floor(i / 8);
    const shift = 8 * (i % 8);
    out[i] = Number((state[lane] >> BigInt(shift)) & 0xffn);
  }
  return out;
}

function solvePure(bc: DSBreadcrumb): number | null {
  const prefix = `${bc.salt}_${bc.expire_at}_`;
  for (let nonce = 0; nonce <= bc.difficulty; nonce++) {
    if (deepSeekHash(Buffer.from(prefix + nonce)).toString("hex") === bc.challenge)
      return nonce;
  }
  return null;
}

// ── WASM solver (official DeepSeek module) ──────────────────────────

interface WasmSolver {
  solve(challenge: string, prefix: string, difficulty: number): number | null;
}

let wasmPromise: Promise<WasmSolver> | null = null;

function loadWasm(): Promise<WasmSolver> {
  wasmPromise ??= (async () => {
    const buffer = await readFile(WASM_URL);
    const { instance } = await WebAssembly.instantiate(buffer);
    const e = instance.exports as unknown as Record<string, unknown>;
    const mem = e.memory as WebAssembly.Memory;
    const alloc = e.__wbindgen_export_0 as (len: number, align: number) => number;
    const stackPtr = e.__wbindgen_add_to_stack_pointer as (delta: number) => number;
    const wasmSolve = e.wasm_solve as (...args: unknown[]) => void;

    const writeStr = (s: string): [number, number] => {
      const bytes = Buffer.from(s, "utf8");
      const ptr = alloc(bytes.length, 1);
      // Re-read memory.buffer: it detaches whenever the alloc grows memory.
      new Uint8Array(mem.buffer, ptr, bytes.length).set(bytes);
      return [ptr, bytes.length];
    };

    return {
      solve(challenge: string, prefix: string, difficulty: number): number | null {
        const retptr = stackPtr(-16);
        try {
          const [challengePtr, challengeLen] = writeStr(challenge);
          const [prefixPtr, prefixLen] = writeStr(prefix);
          wasmSolve(retptr, challengePtr, challengeLen, prefixPtr, prefixLen, difficulty);
          const view = new DataView(mem.buffer);
          if (view.getInt32(retptr, true) === 0) return null;
          return view.getFloat64(retptr + 8, true);
        } finally {
          stackPtr(16);
        }
      },
    };
  })();
  return wasmPromise;
}

// ── Public API ──────────────────────────────────────────────────────

/**
 * Solve a DeepSeekHashV1 PoW challenge.
 * Uses the official WASM module, falling back to the pure-TS
 * 23-round Keccak implementation if it cannot be loaded.
 */
export async function solvePow(bc: DSBreadcrumb): Promise<DSPoWResponse> {
  let answer: number | null = null;

  try {
    const wasm = await loadWasm();
    answer = wasm.solve(bc.challenge, `${bc.salt}_${bc.expire_at}_`, bc.difficulty);
    if (answer == null) throw new Error("WASM solver returned no nonce");
  } catch (err) {
    console.warn(
      "⚠️  WASM PoW solver failed, using pure-TS fallback:",
      (err as Error).message
    );
    answer = solvePure(bc);
    if (answer == null)
      throw new Error(
        `PoW not solved (difficulty=${bc.difficulty}, challenge=${bc.challenge.slice(0, 16)}…)`
      );
  }

  return {
    algorithm: bc.algorithm ?? "DeepSeekHashV1",
    challenge: bc.challenge,
    salt: bc.salt,
    answer: Math.round(answer),
    signature: bc.signature,
    target_path: bc.target_path,
  };
}

/**
 * Encode PoW response as base64 for the x-ds-pow-response header.
 */
export function encodePow(pow: DSPoWResponse): string {
  return Buffer.from(JSON.stringify(pow)).toString("base64");
}