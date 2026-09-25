# Contributing

Thanks for considering a contribution! This project welcomes small, focused improvements
from people at every experience level. You do not need a DeepSeek account, an API key, or
prior knowledge of the project to run the tests or make a first contribution.

If you are new to open source, a small documentation fix, a regression test, or a focused
bug fix is a perfectly good place to start. If you are unsure where to begin, open an
issue and describe what you would like to work on.

## What this project does

`deepseek-api-proxy` is an unofficial, reverse-engineered OpenAI-compatible API server
for [chat.deepseek.com](https://chat.deepseek.com). It provides chat completions,
streaming responses, tool calling, and a local `/v1` API that many OpenAI clients can
use.

DeepSeek's web API is unofficial and can change without notice, so contributions should
preserve the existing OpenAI-facing behavior unless a change is discussed first.

## Ways to help

- Try the server and report a reproducible bug.
- Fix an issue from the [roadmap](README.md#roadmap).
- Add or improve tests around a bug fix or new feature.
- Improve the README, setup instructions, or error messages.
- Improve developer tooling such as the build, test, or lint setup.

When reporting a problem, please include your Node.js and npm versions, your operating
system, the command or request you ran, what you expected to happen, and what happened
instead. Please remove tokens, cookies, and other private information first.

## Quick start for contributors

You will need Git, Node.js **22.12 or newer**, and npm. The current test tooling requires
this newer Node.js version.

From a checkout of the repository, install the exact dependencies recorded in
`package-lock.json` and run the baseline checks:

```sh
npm ci
npm run build
npm test
```

- `npm run build` type-checks the TypeScript source and copies the proof-of-work
  WebAssembly asset into `dist/vendor/`.
- `npm test` runs the Vitest suite. The tests are offline and do not need a real token.
- There is currently no separate lint, format, or typecheck script, so run the build
  before opening a pull request.

`dist/` and `node_modules/` are generated and ignored by Git. Do not commit them.

## Run it locally

To run the CLI directly from the TypeScript source:

```sh
npm run dev -- api -p 3000
```

To run the built version instead, build first and use:

```sh
node dist/cli.js auth
node dist/cli.js api -p 3000
```

The `api` command validates the credentials stored in `~/.ds/auth.json` at startup.
Authentication and account access are local operations: never put a real token in an
issue, test, log, screenshot, or pull request. See the [README usage examples](README.md#usage)
for a sample request.

## Project tour

| Path              | Purpose                                                        |
| ----------------- | -------------------------------------------------------------- |
| `src/cli.ts`      | The `ds` command-line interface.                                |
| `src/auth/`       | Token input and local credential storage.                       |
| `src/shared/`     | Shared types, configuration, and OpenAI/DeepSeek conversion.    |
| `src/deepseek/`   | DeepSeek API client and proof-of-work solver.                  |
| `src/server/`     | Express app, routes, and conversation session management.       |
| `src/vendor/`     | The proof-of-work WebAssembly asset used by the build.          |

A quick map like this can make a first change easier: start with the module you are
changing, its adjacent `*.test.ts` file, and the relevant section of the README.

## Make a change

1. Create a focused branch from `master`.
2. Make the smallest change that solves the problem or adds the feature.
3. Add or update a test when behavior changes.
4. Run `npm run build` and `npm test`.
5. Commit the change and open a pull request.

The project uses strict TypeScript with Node16 module resolution. When editing existing
code, follow the nearby style:

- Use two-space indentation, double quotes, and semicolons.
- Use `interface` for object shapes and `type` for unions or derived aliases.
- Use `import type` for type-only imports.
- Add `.js` to every relative import, even when the source file is a `.ts` file.
- Keep the CLI's lazy imports intact.
- Do not edit generated files in `dist/` by hand.

There is no configured formatter or linter yet, so match the surrounding code and avoid
unrelated formatting changes.

## Tests

Tests live next to the code they cover and use Vitest 5. Import test utilities explicitly,
for example:

```ts
import { describe, it, expect } from "vitest";
```

Helpful conventions already used in the repository:

- Prefer small in-memory fakes such as `FakeClient` over adding a mocking dependency.
- Keep tests offline and use clearly fake values such as `fake-token`.
- Restore replaced globals such as `globalThis.fetch` after each test.
- Use fake timers for session expiry and cleanup behavior.
- Add a regression test with every bug fix and cover error paths as well as success paths.

Some router tests start a temporary server on `127.0.0.1`; they do not contact DeepSeek,
but the environment must allow loopback networking.

## Issues and pull requests

Pull requests target the `master` branch. Conventional Commit-style subjects are used in
this repository, such as `fix: ...`, `feat: ...`, `test: ...`, `chore: ...`, or a scoped
subject such as `fix(server): ...`.

A helpful pull request description explains:

1. What changed and why.
2. Which behavior or API contract is affected.
3. How the change was tested, including `npm run build` and `npm test` results.
4. Any documentation, compatibility, or security considerations.

Keep pull requests small and reviewable. It is fine to open a draft pull request when you
want feedback before the implementation is complete. There is no repository CI
configuration, so reviewers rely on the results you report from your local environment.

## Security and compatibility

The project is unofficial and reverse-engineers the web client. Please keep changes
compatible with the existing routes, streaming format, session behavior, and
prompt/tool-call contract where possible. If a change intentionally alters one of those
contracts, explain it in the pull request and update the README.

Authentication data is stored in plaintext at `~/.ds/auth.json`. Treat it as a secret and
never commit it. For a possible vulnerability, contact the maintainer privately instead
of opening a public issue with exploit details or credentials.

## License

This project is released under the [MIT License](LICENSE). Contributions are accepted
under those terms.
