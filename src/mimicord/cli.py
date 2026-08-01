from __future__ import annotations

import logging

import typer
from dotenv import load_dotenv

from mimicord.paths import PersonaPaths

app = typer.Typer(no_args_is_help=True, add_completion=False)

CONFIG_TEMPLATE = """\
name = "{name}"

[target]
# who to mimic; used to flag their messages at ingest
author_ids = []          # discord user ids as strings, most reliable
author_names = []        # usernames and nicknames as fallback

[llm]
provider = "anthropic"   # anthropic | openai | deepseek | ollama
model = "claude-opus-5"
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


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="debug logging"),
) -> None:
    """Build Discord bots that talk like a real person."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def version() -> None:
    """Print the mimicord version."""
    from mimicord import __version__

    typer.echo(__version__)


@app.command()
def new(name: str) -> None:
    """Scaffold personas/<name>/ with a config and a starter persona prompt."""
    paths = PersonaPaths.for_persona(name)
    if paths.config.exists():
        typer.echo(f"{paths.config} already exists")
        raise typer.Exit(1)
    paths.root.mkdir(parents=True, exist_ok=True)
    env_name = "".join(c for c in name.upper() if c.isalnum() or c == "_") or "BOT"
    paths.config.write_text(
        CONFIG_TEMPLATE.format(name=name, env_name=env_name), encoding="utf-8"
    )
    paths.persona_md.write_text(PERSONA_TEMPLATE.format(name=name), encoding="utf-8")
    typer.echo(f"created {paths.root}")
    typer.echo("next: edit persona.toml (target authors, provider, token env) and try mimicord chat")


@app.command()
def chat(
    name: str,
    no_rag: bool = typer.Option(False, "--no-rag", help="skip memory retrieval"),
    show_prompt: bool = typer.Option(
        False, "--show-prompt", help="dump the assembled prompt every turn"
    ),
) -> None:
    """Talk to a persona in the terminal, no Discord needed."""
    from mimicord import repl

    repl.run(name, rag=not no_rag, show_prompt=show_prompt)


@app.command()
def ingest(
    name: str,
    dce: list[str] = typer.Option(
        None, "--dce", help="DiscordChatExporter json file or directory, repeatable"
    ),
    package: str = typer.Option(
        None, "--package", help="official data package directory (or its messages folder)"
    ),
    retag: bool = typer.Option(
        False, "--retag", help="only re-flag target messages after editing [target]"
    ),
) -> None:
    """Parse chat exports into the persona's local corpus."""
    from pathlib import Path

    from mimicord import ingest as ingest_mod
    from mimicord.config import load_config
    from mimicord.store import Store

    paths = PersonaPaths.for_persona(name)
    cfg = load_config(paths.config)
    with Store(paths.corpus) as store:
        if retag:
            changed = store.retag(cfg.target.author_ids, cfg.target.author_names)
            typer.echo(f"re-flagged target messages, {changed} rows now marked")
        else:
            if not dce and not package:
                typer.echo("nothing to do: pass --dce and/or --package")
                raise typer.Exit(1)
            parsed = 0
            if dce:
                parsed += ingest_mod.ingest_dce(store, [Path(p) for p in dce], cfg.target)
            if package:
                parsed += ingest_mod.ingest_package(store, Path(package), cfg.target)
            typer.echo(f"parsed {parsed} messages")
        counts = store.counts()
    typer.echo(
        f"corpus: {counts['total']} messages, {counts['target']} from target, "
        f"{counts['channels']} channels"
    )
    if counts["target"] == 0:
        typer.echo("warning: nothing matched [target], check author_ids/author_names")


@app.command()
def stats(name: str) -> None:
    """Compute deterministic style stats from the corpus."""
    import json

    from mimicord.analyze import stats as stats_mod
    from mimicord.store import Store

    paths = PersonaPaths.for_persona(name)
    if not paths.corpus.is_file():
        typer.echo("no corpus yet, run mimicord ingest first")
        raise typer.Exit(1)
    with Store(paths.corpus) as store:
        result = stats_mod.compute(store)
    paths.stats.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for line in stats_mod.summary_lines(result):
        typer.echo(line)
    typer.echo(f"\nwrote {paths.stats}")


@app.command()
def inspect(name: str) -> None:
    """Show what artifacts exist for a persona."""
    import json

    from mimicord.config import load_config
    from mimicord.store import Store

    paths = PersonaPaths.for_persona(name)
    cfg = load_config(paths.config)
    typer.echo(f"persona {cfg.name} ({cfg.llm.provider}/{cfg.llm.model})")

    if paths.corpus.is_file():
        with Store(paths.corpus) as store:
            counts = store.counts()
        typer.echo(
            f"  corpus       {counts['total']} messages, {counts['target']} target, "
            f"{counts['channels']} channels ({counts['first']} .. {counts['last']})"
        )
    else:
        typer.echo("  corpus       missing (mimicord ingest)")

    typer.echo(f"  stats        {'ok' if paths.stats.is_file() else 'missing (mimicord stats)'}")
    typer.echo(f"  profile      {'ok' if paths.profile.is_file() else 'missing (mimicord analyze)'}")
    typer.echo(f"  persona.md   {'ok' if paths.persona_md.is_file() else 'missing (mimicord compile)'}")
    if paths.examples.is_file():
        examples = json.loads(paths.examples.read_text(encoding="utf-8"))
        typer.echo(f"  examples     {len(examples.get('examples', []))} few-shots")
    else:
        typer.echo("  examples     missing (mimicord compile)")
    typer.echo(f"  memories     {'ok' if paths.chroma_dir.is_dir() else 'missing (mimicord index)'}")


@app.command()
def run(
    name: str,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="connect and listen but print replies instead of sending"
    ),
) -> None:
    """Run the Discord bot for a persona."""
    from mimicord import bot

    bot.run(name, dry_run=dry_run)


if __name__ == "__main__":
    app()
