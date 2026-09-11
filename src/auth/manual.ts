import { createInterface } from "node:readline/promises";
import type { AuthData } from "../shared/types.js";
import { saveAuth } from "./store.js";

/**
 * Authenticate by pasting the token copied from the browser console:
 *   JSON.parse(localStorage.getItem("userToken")).value
 * @param token - optional pre-supplied token; prompts interactively if omitted.
 */
export async function authManual(token?: string): Promise<AuthData> {
  let value = token?.trim();
  if (!value) {
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    value = (
      await rl.question(
        'Paste your DeepSeek token (JSON.parse(localStorage.getItem("userToken")).value):\n'
      )
    ).trim();
    rl.close();
  }

  value = value.replace(/^["']|["']$/g, "");
  if (!value) throw new Error("No token provided.");

  const auth: AuthData = {
    token: value,
    cookies: [],
    lastValidated: new Date().toISOString(),
  };

  await saveAuth(auth);
  console.log(`✅ Auth saved to ~/.ds/auth.json`);
  return auth;
}