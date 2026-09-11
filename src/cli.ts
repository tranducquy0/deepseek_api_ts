import { Command } from "commander";
import { DEFAULT_PORT } from "./shared/config.js";

const program = new Command();

program
  .name("ds")
  .description("Unofficial OpenAI-compatible API for chat.deepseek.com")
  .version("0.1.0");

// ── ds auth ─────────────────────────────────────────────────────────

program
  .command("auth")
  .description("Authenticate with a manually copied token")
  .argument(
    "[token]",
    "DeepSeek token (JSON.parse(localStorage.getItem('userToken')).value); prompts if omitted"
  )
  .action(async (token?: string) => {
    const { authManual } = await import("./auth/manual.js");

    try {
      await authManual(token);
    } catch (err) {
      console.error("❌ Auth failed:", (err as Error).message);
      process.exit(1);
    }
  });

// ── ds api ──────────────────────────────────────────────────────────

program
  .command("api")
  .description("Start the OpenAI-compatible API server")
  .option("-p, --port <port>", "Port to listen on", String(DEFAULT_PORT))
  .action(async (opts: { port: string }) => {
    const { startServer } = await import("./server/app.js");
    const port = parseInt(opts.port, 10);

    if (isNaN(port) || port < 1 || port > 65535) {
      console.error("❌ Invalid port number:", opts.port);
      process.exit(1);
    }

    try {
      await startServer({ port });
    } catch (err) {
      console.error("❌ Server failed to start:", (err as Error).message);
      process.exit(1);
    }
  });

program.parse();