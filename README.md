# mimicord

Build Discord bots that talk like a real person. Feed it chat exports, it compiles a persona (style prompt, real example exchanges, message stats, RAG memory) and runs a bot that replies in that person's voice.

Work in progress. Full docs land with v0.1.

## How it will work

```
mimicord new <name>       # scaffold a persona
mimicord ingest <name>    # parse chat exports into a local corpus
mimicord stats <name>     # deterministic style stats
mimicord analyze <name>   # LLM style analysis over the corpus
mimicord compile <name>   # generate the persona prompt + few-shot examples
mimicord index <name>     # build the local RAG memory index
mimicord chat <name>      # talk to the persona in your terminal
mimicord run <name>       # run the Discord bot
```

Providers: Anthropic Claude, OpenAI, DeepSeek, or local Ollama. Picked per persona.

All chat data stays local (gitignored `personas/`). Only prompt-sized excerpts go to the LLM provider you pick.

## Rules of the road

- The bot runs on a real bot account from the Discord Developer Portal. Self-bots are against Discord ToS, mimicord never automates a user account.
- Export chats with the official data package (Settings > Privacy) or DiscordChatExporter using a bot token. User-token exports violate ToS and Discord actively enforces this.
- Mimicking a friend? Ask them first.
