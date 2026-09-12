import { describe, it, expect } from "vitest";
import { deepSeekHash, encodePow, solvePure } from "./pow.js";
import type { DSBreadcrumb } from "../shared/types.js";

describe("deepSeekHash / solvePure", () => {
  it("finds the nonce for a self-consistent challenge", () => {
    const salt = "test-salt";
    const expireAt = 1234567890;
    const nonce = 42;
    const prefix = `${salt}_${expireAt}_`;
    const challenge = deepSeekHash(Buffer.from(`${prefix}${nonce}`)).toString("hex");

    const bc: DSBreadcrumb = {
      salt,
      expire_at: expireAt,
      challenge,
      difficulty: 100000,
      signature: "sig",
    };

    expect(solvePure(bc)).toBe(nonce);
  });

  it("returns null when the nonce is out of range", () => {
    const salt = "x";
    const expireAt = 1;
    const challenge = deepSeekHash(Buffer.from(`${salt}_${expireAt}_999`)).toString("hex");
    const bc: DSBreadcrumb = {
      salt,
      expire_at: expireAt,
      challenge,
      difficulty: 10,
      signature: "sig",
    };
    expect(solvePure(bc)).toBeNull();
  });

  it("is deterministic", () => {
    const input = Buffer.from("hello");
    expect(deepSeekHash(input).toString("hex")).toBe(
      deepSeekHash(Buffer.from("hello")).toString("hex")
    );
    expect(deepSeekHash(input).byteLength).toBe(32);
  });
});

describe("encodePow", () => {
  it("round-trips through base64", () => {
    const pow = { challenge: "c", salt: "s", answer: 7, signature: "sig" };
    const decoded = JSON.parse(Buffer.from(encodePow(pow), "base64").toString("utf-8"));
    expect(decoded).toMatchObject(pow);
  });
});