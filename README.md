# DeepSeek API Proxy

Unofficial **OpenAI-compatible** API server for [chat.deepseek.com](https://chat.deepseek.com).
Run one command and point any OpenAI client at a local `http://localhost:3000/v1`
endpoint — compatible with ChatGPT-style apps, Claude Code, scripts, etc.

## Features

- `POST /v1/chat/completions` — streaming (`stream: true`) and non-streaming responses
- Tool / function calling (`tools`, `tool_choice`, `tool_calls`, `role: "tool"`),
  including several calls per turn
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
| `stream`           | `boolean`                     | SSE when `true`; JSON by default (as per OpenAI)   |
| `stream_options`   | `{include_usage?}`            | Adds a final usage-only chunk when streaming       |
| `thinking`         | `boolean`                     | Force reasoning on/off (default: on for reasoner)  |
| `tools`            | `OpenAITool[]`                | Function calling                                   |
| `tool_choice`      | `string \| object`            | `"auto"`, `"none"`, `"required"`, or `{...}`; enforced |
| `user`             | `string`                      | Optional stable conversation id for session reuse  |
| `temperature`      | `number`                      | Accepted, ignored                                  |
| `max_tokens`       | `number`                      | Accepted, ignored                                  |

Every chunk of one completion shares a single `id`/`created` pair, and `finish_reason` is
set exactly once, on the terminal chunk — as OpenAI does.

`usage` is reported for both streaming and non-streaming responses. DeepSeek only exposes
a token total accumulated over the whole chat session, so the proxy diffs it against the
previous turn to report a per-request figure. The prompt/completion split is not available
from the web endpoint and stays `0`.

### Message content

`content` may be a plain string or an OpenAI-style array of content parts
(`[{"type": "text", "text": "…"}]`), which is what multimodal clients send. Parts are
flattened to text for DeepSeek's prompt.

DeepSeek's web endpoint carries no inline attachments, so `image_url` and other binary
parts **cannot be forwarded**. Rather than dropping them silently, the prompt names what
was omitted (e.g. `[image_url omitted: this proxy does not forward attachments]`) so the
model is never misled into thinking it saw an image.

### Tool calling

DeepSeek's web API has no native tool support. Tools are injected into the prompt as
JSON schemas, and the model is instructed to emit one JSON object per line
(`{"name": "...", "arguments": {...}}`). The proxy scans for those objects and converts
them into OpenAI `tool_calls` — several per turn are supported, indexed `0..n-1` and
reported with `finish_reason: "tool_calls"`, so standard OpenAI tool-calling clients work
unchanged.

Parsing tolerates surrounding prose, markdown fences, and stray braces. The model
frequently drops the `{"name": …, "arguments": …}` wrapper and emits the bare arguments
instead; those are bound to the offered function automatically **only when exactly one
function's schema accepts them**, since guessing between several could run the wrong
command. A `tool`/`function` key is accepted as an alias for `name`.

If the model aimed for a tool call but the result is malformed or ambiguous, the proxy
**fails loudly** rather than ending the turn as if nothing was called — a silent
`finish_reason: "stop"` would let an agent loop record a step that never happened:

| Mode         | Response                                                      |
| ------------ | ------------------------------------------------------------- |
| non-streaming| `502` with `error.code = "tool_call_parse_failed"`            |
| streaming    | an SSE `error` frame and **no** `[DONE]` (SDKs raise `APIError`) |

`tool_choice` is enforced, not merely suggested. Since it only reaches DeepSeek as a
prompt instruction, the model sometimes answers in prose anyway — so the turn is verified
against the request afterwards:

| `tool_choice`               | Violation                                                 |
| --------------------------- | --------------------------------------------------------- |
| `"required"`                | model answered without calling a function                |
| `"none"`                    | model called a function anyway                            |
| `{"function":{"name":"x"}}` | model called a function other than `x`, or none at all    |

A violation returns `502` with `error.code = "tool_choice_violation"` (or an SSE `error`
frame when streaming), so an agent loop cannot mistake it for a finished turn. Sending
`"required"` or a forced function without any `tools` is a `400`.

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

To avoid re-sending history, the proxy remembers a fingerprint of the messages it has
already forwarded. If the next request merely appends to them, only the new messages are
sent. If the client **rewrote** that history — compaction, trimming, or an edited
message — the old DeepSeek session no longer describes the conversation, so the proxy
starts a fresh one and replays the full history instead of sending a misaligned slice.

The fingerprint covers only what reaches the prompt, so a client that round-trips a
response through its own SDK (reordering keys, adding null padding, renaming fields) does
not trigger a replay. A client that mutates an *earlier* message every turn — a
timestamped system prompt, for instance — will replay each turn, which costs tokens but
still preserves context.

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
- Multimodal input is flattened to text; image and other attachment parts are **not**
  forwarded (see [Message content](#message-content)).
- One DeepSeek session per conversation; concurrent requests to the same conversation are
  last-writer-wins.
- `tool_choice` is verified after the turn, but a violation surfaces as an error rather
  than a retry.
- Rewritten client history is replayed into a new DeepSeek session, which costs tokens.
- Changing `system` messages mid-conversation keeps the server-side one.
- `temperature` / `max_tokens` are accepted but not enforced.

## Roadmap

- Abort upstream DeepSeek stream when the client disconnects
- Serialize turns per conversation, or reject concurrent ones with `409`
- Account info endpoint and `ds status` CLI command
- Configurable bind host for LAN access