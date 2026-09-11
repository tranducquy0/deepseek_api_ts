import { homedir } from "node:os";
import { join } from "node:path";

export const DS_DIR = join(homedir(), ".ds");
export const AUTH_FILE = join(DS_DIR, "auth.json");
export const BASE_URL = "https://chat.deepseek.com";

export const DEFAULT_PORT = 3000;

export const MODELS = [
  "deepseek-chat",
  "deepseek-reasoner",
] as const;

export type ModelId = (typeof MODELS)[number];

export const MODEL_MAP: Record<ModelId, { ds: string; display: string }> = {
  "deepseek-chat": { ds: "default", display: "DeepSeek V3" },
  "deepseek-reasoner": { ds: "expert", display: "DeepSeek R1" },
};
