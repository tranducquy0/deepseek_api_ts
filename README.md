# DeepSeek API Proxy

Unofficial **OpenAI-compatible** API server for [chat.deepseek.com](https://chat.deepseek.com).
Run one command and point any OpenAI client at a local `http://localhost:3000/v1`
endpoint — compatible with ChatGPT-style apps, Claude Code, scripts, etc.

## Features

- `POST /v1/chat/completions` — streaming (`stream: true`) and non-streaming responses
- Tool / function calling (`tools`, `tool_choice`, `tool_calls`, `role: "tool"`)
- Thinking toggle for the reasoning model (`thinking: true`, or use `deepseek-reasoner`),
  streamed as OpenAI `reasoning_content`
- Multi-turn conversation chaining via DeepSeek session `parent_message_id`
- `GET /v1/models` and a JSON health endpoint
- PoW solver to satisfy DeepSeek's proof-of-work: official WASM module with a pure-TS
  Keccak fallback

## Requirements

- Node.js 18+ (tested on Node 22/26, including Termux)
- npm

## Install & build

```sh
npm install
npm run build
```

## Usage

### 1. Authenticate

```sh
node dist/cli.js auth
```

You will be asked to paste your DeepSeek token. To get it, open
`https://chat.deepseek.com` in a browser, open the dev console, and run:

```js
JSON.parse(localStorage.getItem("userToken")).value
```

Or pass it directly: `ds auth <TOKEN>`.

Credentials are stored in plaintext at `~/.ds/auth.json`. Keep it private.

### 2. Start the server

```sh
node dist/cli.js api          # http://localhost:3000
node dist/cli.js api -p 8080  # custom port
```

Container / Termux users can run it via `npm run dev` (`tsx src/cli.ts api`).

### 3. Call it like OpenAI

```sh
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{ "role": "user", "content": "Hello!" }],
    "stream": true
  }'
```

## API

### `POST /v1/chat/completions`

Body fields:

| Field              | Type                          | Notes                                              |
| ------------------ | ----------------------------- | -------------------------------------------------- |
| `model`            | `string`                      | `deepseek-chat` (V3) or `deepseek-reasoner` (R1)   |
| `messages`         | `OpenAIMessage[]`             | roles: `system`, `user`, `assistant`, `tool`       |
| `stream`           | `boolean`                     | SSE when `true`; JSON by default (as per OpenAI)      |
| `thinking`         | `boolean`                     | Force reasoning on/off (default: on for reasoner)  |
| `tools`            | `OpenAITool[]`                | Function calling                                   |
| `tool_choice`      | `string \| object`            | `"auto"`, `"none"`, `"required"`, or `{...}`       |
| `user`             | `string`                      | Optional stable conversation id for session reuse  |
| `temperature`      | `number`                      | Accepted, ignored                                  |
| `max_tokens`       | `number`                      | Accepted, ignored                                  |

DeepSeek's web API has no native tool support. Tools are injected into the prompt as
JSON schemas, and the model is instructed to emit a single JSON object
(`{"name": "...", "arguments": {...}}`). The proxy detects that object and converts it
into OpenAI `tool_calls` with `finish_reason: "tool_calls"` — so standard OpenAI
tool-calling clients work unchanged.

### `GET /v1/models`

Lists the supported model ids.

### `GET /`

Health/status, shows the available endpoints.

## Multi-turn behavior

The proxy keeps a DeepSeek session per conversation and remembers the parent message id,
so follow-up turns continue the same context instead of restarting. Sessions expire after
1 hour of inactivity.

A conversation is keyed by the `user` field when present; otherwise by a hash of the
first user message. Use a stable `user` id to keep related turns grouped, or omit it and
keep the first message identical across turns.

## Development

```sh
npm run build   # compile to dist/
npm test        # run vitest suite
npm run dev     # tsx hot-run src/cli.ts
```

## Security & limitations

- **Unofficial.** This reverse-engineers the web client; DeepSeek may change the API or
  terms without notice. Use at your own risk and don't rely on it for production.
- Tokens are stored in **plaintext** on disk.
- One DeepSeek session per conversation; concurrent requests to the same conversation are
  last-writer-wins.
- Changing `system` messages mid-conversation keeps the server-side one.
- `temperature` / `max_tokens` are accepted but not enforced.

## Roadmap

- Abort upstream DeepSeek stream when the client disconnects
- Account info endpoint and `ds status` CLI command
- Configurable bind host for LAN access
- Report real `usage` token counts (currently stubbed as `0`)