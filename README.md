# mimicord

Build Discord bots that talk like a real person. Feed it chat exports, it compiles a persona (style prompt, real example exchanges, measured writing stats, RAG memory over the full history) and runs a bot that replies in that person's voice.

Works with Anthropic Claude, OpenAI, DeepSeek, or a local Ollama model, picked per persona. All chat data stays on your machine; only prompt-sized excerpts go to the provider you choose.

## How it works

```
chat exports ──> ingest ──> corpus.db (sqlite)
                              │
                              ├─ stats    deterministic style numbers (free)
                              ├─ analyze  LLM style analysis, map-reduce over chunks
                              ├─ compile  persona.md + few-shot examples.json
                              └─ index    local embedding index of the history (free)
                                            │
                              chat (terminal REPL) / run (Discord bot)
```

The bot keeps a rolling context of the channel, retrieves a few real conversation snippets from the index, and asks the model for a reply in character. Replies get post-processed to match measured habits: lowercase starts, dropped periods, split into short message bursts, AI phrasing stripped.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/keksoslav/mimicord
cd mimicord
uv sync
cp .env.example .env       # fill in the keys you use
uv run mimicord new janez
```

Edit `personas/janez/persona.toml` (target author ids, provider, token env var), then:

```
uv run mimicord ingest janez --dce path/to/export.json --package path/to/package
uv run mimicord stats janez
uv run mimicord analyze janez
uv run mimicord compile janez
uv run mimicord index janez
uv run mimicord chat janez        # try the voice in your terminal first
uv run mimicord run janez --dry-run
uv run mimicord run janez
```

You can also skip the whole pipeline: `mimicord new` writes a starter `persona.md` you can edit by hand, and `chat`/`run` work with just that.

## GUI

Everything above also has a desktop app:

```
uv sync --extra gui
uv run mimicord gui
```

Persona list on the left, four tabs on the right: config (TOML editor with validation), pipeline (ingest with file pickers, stats, analyze, compile, index, with a live log), chat (talk to the persona), and bot (start/stop with a dry run toggle and the bot's log). Pipeline steps and the bot run in background threads so the window never freezes. The CLI keeps working the same either way.

## Getting chat history

Two routes that do not risk anyone's account:

- **Official data package.** Discord Settings > Privacy and Safety > Request all of my data. Takes a few days, contains only your own messages. Ingest with `--package path/to/package`.
- **DiscordChatExporter with a bot token.** Invite your own bot (below) to your server, then export channels with [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) using the bot's token and JSON format. This captures everyone's messages with full context, which makes for much better personas. Ingest with `--dce path/to/export.json` (repeatable, directories work too).

**Warning:** running DiscordChatExporter with your personal user token violates Discord's terms of service, and since early 2026 Discord actively enforces this (forced logouts, warnings, possible termination). mimicord never asks for a user token.

## Creating the bot account

1. [Discord Developer Portal](https://discord.com/developers/applications) > New Application, name it after the persona.
2. Bot page: enable **Message Content Intent** under Privileged Gateway Intents. If you skip this the bot fails at login with a privileged intents error. If you instead removed the intent from the code, every message would silently arrive empty.
3. Copy the bot token into `.env` under the name you set as `token_env`.
4. OAuth2 > URL Generator: scope `bot`, permissions View Channels, Send Messages, Read Message History. Open the URL and invite the bot to your server.

Test in a private server first. `--dry-run` connects and listens but prints replies to stdout instead of sending.

## Configuration

Everything lives in `personas/<name>/persona.toml`. The important knobs:

| key | what it does |
| --- | --- |
| `target.author_ids` | Discord user ids whose messages define the persona (most reliable) |
| `llm.provider` / `llm.model` | `anthropic`, `openai`, `deepseek`, `ollama` and the model id |
| `llm.analyze.model` | optional cheaper model for the per chunk analysis step |
| `discord.token_env` | env var holding the bot token |
| `discord.trigger_*` | mention, reply, keywords, random interject probability, always-on channels |
| `discord.cooldown_seconds` / `max_replies_per_hour` | the cost and spam safety net |
| `rag.enabled` | memory retrieval on or off |
| `style.max_burst` | max messages per reply |

## Providers and cost

| provider | notes |
| --- | --- |
| anthropic | default `claude-opus-5`; the persona prompt and few-shots are prompt-cached, which cuts input cost by roughly 90% on consecutive replies |
| claude-code | bills your claude.ai subscription instead of an API key, see below |
| openai | `gpt-4o` default, set any model you have access to |
| deepseek | cheap, OpenAI-compatible API |
| ollama | free and local; lower `context_messages` and example count for small context models |

### Using a claude.ai subscription (no API key)

Pro and Max plans include a monthly Agent SDK credit ($20 Pro, $100 Max 5x, $200 Max 20x) that covers third-party apps built on the Claude Agent SDK. Set `provider = "claude-code"` and `model = "sonnet"` (or `opus`/`haiku`) and mimicord routes replies through your local Claude Code login, drawing from that credit instead of API billing. Requirements: Claude Code installed and signed in on the machine running the bot.

Two properties make this the relaxed option: the credit hard-stops when spent (no surprise charges unless you explicitly enable extra usage credits on your account), and it is a separate pool from your chat and Claude Code limits. For an extra belt, set `max_replies_per_month` in persona.toml; the bot tracks its own reply count in `usage.json` and goes quiet when the budget is reached, resetting monthly. Replies are a bit slower than the raw API because each one runs through the Claude Code harness, which the typing simulation hides well.

Do not point the regular `anthropic` provider at a claude.ai OAuth token; that violates the consumer terms. The `claude-code` provider is the supported route.

Compiling a persona from a ~20k message corpus costs well under a dollar with a cheap analyze model (`claude-haiku-4-5`) and is a one-time cost. Replies are roughly a cent each on `claude-opus-5` with caching, a fifth of that on `claude-haiku-4-5`, free on Ollama. `mimicord inspect <name> --cost` prints an estimate for your actual persona, and the hourly reply cap bounds the worst case.

## Rules of the road

- The bot runs on a real bot account and is always labeled BOT. Self-bots (automating a user account) are against Discord ToS; mimicord never does that.
- Mimicking a friend? Ask them first, and tell the server. A persona bot is a party trick, not a disguise.
- Chat exports and everything compiled from them stay in `personas/`, which is gitignored. Prompt-sized excerpts (persona prompt, examples, retrieved memories, recent context) are sent to the LLM provider you configured, nothing else leaves the machine.

## Development

```
uv run pytest
```

Tests cover both export formats, dedup, trigger logic, prompt assembly stability, the JSON repair path, and retrieval, all against fixtures and fake providers. No network or API keys needed.

## License

MIT
