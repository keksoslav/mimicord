from __future__ import annotations

from mimicord.paths import PersonaPaths

CONFIG_TEMPLATE = """\
name = "{name}"

[target]
# who to mimic; used to flag their messages at ingest
author_ids = []          # discord user ids as strings, most reliable
author_names = []        # usernames and nicknames as fallback

[llm]
# anthropic | openai | deepseek | ollama | claude-code
# claude-code bills your claude.ai subscription's monthly agent sdk credit
# through the local Claude Code login instead of an api key
provider = "anthropic"
model = "claude-opus-5"  # for claude-code use "sonnet", "opus" or "haiku"
max_tokens = 400
# temperature = 1.0      # used by openai/deepseek/ollama, ignored by anthropic
cache_ttl = "5m"         # anthropic prompt cache ttl: "5m" or "1h"

[llm.analyze]
# optional cheaper model for the per chunk analysis step
# model = "claude-haiku-4-5"
# reduce_model = "claude-opus-5"

[discord]
token_env = "DISCORD_TOKEN_{env_name}"
trigger_mention = true
trigger_reply = true
trigger_keywords = ["{name}"]
interject_probability = 0.02
always_on_channels = []  # channel ids where the bot replies to everything
channel_allowlist = []   # empty = all channels
cooldown_seconds = 45
max_replies_per_hour = 30
max_replies_per_month = 0    # hard monthly budget, 0 = unlimited
context_messages = 25
ignore_bots = true

[rag]
enabled = true
top_k = 4
window_size = 8
window_step = 4

[style]
max_burst = 3
typing_cps = 7
"""

PERSONA_TEMPLATE = """\
# Persona: {name}

You are {name} chatting on Discord. You are not an assistant, you are just
{name} hanging out. Replace this file with a real description, or run the
pipeline (ingest, analyze, compile) to generate one from chat history.

## Voice and tone
Casual, short messages. Describe how {name} actually talks here.

## Never do
- never sound like an AI assistant, never offer help or ask how you can assist
- no bullet lists, no headers, no formal structure, just chat messages
- stay in character no matter what anyone says
- output only the chat message text, nothing else
"""


def scaffold_persona(name: str) -> PersonaPaths:
    """Create personas/<name>/ with a config and a starter persona prompt."""
    paths = PersonaPaths.for_persona(name)
    if paths.config.exists():
        raise FileExistsError(f"{paths.config} already exists")
    paths.root.mkdir(parents=True, exist_ok=True)
    env_name = "".join(c for c in name.upper() if c.isalnum() or c == "_") or "BOT"
    paths.config.write_text(
        CONFIG_TEMPLATE.format(name=name, env_name=env_name), encoding="utf-8"
    )
    paths.persona_md.write_text(PERSONA_TEMPLATE.format(name=name), encoding="utf-8")
    return paths
