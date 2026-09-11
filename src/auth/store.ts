import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { AUTH_FILE } from "../shared/config.js";
import type { AuthData } from "../shared/types.js";

export async function loadAuth(): Promise<AuthData | null> {
  try {
    const raw = await readFile(AUTH_FILE, "utf-8");
    return JSON.parse(raw) as AuthData;
  } catch {
    return null;
  }
}

export async function saveAuth(data: AuthData): Promise<void> {
  await mkdir(dirname(AUTH_FILE), { recursive: true });
  await writeFile(AUTH_FILE, JSON.stringify(data, null, 2) + "\n", "utf-8");
}
