import express from "express";
import type { AuthData } from "../shared/types.js";
import { DeepSeekClient } from "../deepseek/client.js";
import { loadAuth, saveAuth } from "../auth/store.js";
import { chatRouter } from "./chat.js";
import { modelsRouter } from "./models.js";

export interface ServerOpts {
  port: number;
}

export async function startServer(opts: ServerOpts): Promise<void> {
  // Load auth
  const auth = await loadAuth();
  if (!auth) {
    console.error(
      "❌ No auth found. Run `ds auth` first to authenticate."
    );
    process.exit(1);
  }

  // Create client with auto-refresh callback
  let client = new DeepSeekClient({
    auth,
    onAuthRefresh: (newAuth: AuthData) => {
      saveAuth(newAuth);
    },
  });

  // Validate auth
  console.log("🔍 Validating auth…");
  const valid = await client.validate();
  if (!valid) {
    console.error(
      "❌ Auth is invalid or expired. Run `ds auth` to re-authenticate."
    );
    process.exit(1);
  }
  console.log("✅ Auth valid.");

  // Create Express app
  const app = express();
  app.use(express.json({ limit: "10mb" }));

  // CORS for local development
  app.use((_req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader(
      "Access-Control-Allow-Headers",
      "Content-Type, Authorization"
    );
    if (_req.method === "OPTIONS") {
      res.sendStatus(204);
      return;
    }
    next();
  });

  // Mount routes
  app.use(modelsRouter());
  app.use(chatRouter(() => client));

  // Health check
  app.get("/", (_req, res) => {
    res.json({
      status: "ok",
      service: "deepseek-api-proxy",
      endpoints: ["/v1/models", "/v1/chat/completions"],
    });
  });

  // Start
  app.listen(opts.port, () => {
    console.log(`\n🚀 DeepSeek API proxy running on http://localhost:${opts.port}`);
    console.log(`   POST http://localhost:${opts.port}/v1/chat/completions`);
    console.log(`   GET  http://localhost:${opts.port}/v1/models`);
    console.log(`\n   Press Ctrl+C to stop.\n`);
  });
}
