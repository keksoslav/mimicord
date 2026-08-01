from __future__ import annotations

import logging

import typer
from dotenv import load_dotenv

from mimicord.paths import PersonaPaths

app = typer.Typer(no_args_is_help=True, add_completion=False)


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
    from mimicord.scaffold import scaffold_persona

    try:
        paths = scaffold_persona(name)
    except FileExistsError as error:
        typer.echo(str(error))
        raise typer.Exit(1)
    typer.echo(f"created {paths.root}")
    typer.echo("next: edit persona.toml (target authors, provider, token env) and try mimicord chat")


@app.command()
def gui() -> None:
    """Launch the desktop app (needs the gui extra: uv sync --extra gui)."""
    try:
        from mimicord.gui import main as gui_main
    except ImportError:
        typer.echo("GUI dependencies missing, install them with: uv sync --extra gui")
        raise typer.Exit(1)
    gui_main()


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


# rough per MTok prices for the cost estimate, input/output
PRICING_PER_MTOK = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "deepseek-chat": (0.27, 1.10),
    "gpt-4o": (2.50, 10.00),
}


def _cost_lines(paths: "PersonaPaths", cfg) -> list[str]:
    import json

    prefix_chars = 0
    if paths.persona_md.is_file():
        prefix_chars += len(paths.persona_md.read_text(encoding="utf-8"))
    if paths.examples.is_file():
        data = json.loads(paths.examples.read_text(encoding="utf-8"))
        prefix_chars += len(json.dumps(data, ensure_ascii=False))
    prefix_tokens = prefix_chars // 4  # rough chars-to-tokens
    live_tokens = 900  # context window + memories, ballpark
    out_tokens = min(cfg.llm.max_tokens, 150)

    lines = [
        "",
        f"cost estimate (rough) for {cfg.llm.provider}/{cfg.llm.model}",
        f"  cached prompt prefix  ~{prefix_tokens} tokens (persona.md + examples)",
        f"  per reply             ~{live_tokens} tokens in + ~{out_tokens} out",
    ]
    pricing = PRICING_PER_MTOK.get(cfg.llm.model)
    if cfg.llm.provider == "claude-code":
        lines.append("  billed to your claude.ai plan's monthly agent sdk credit")
        lines.append("  the credit hard-stops when spent, no surprise charges")
        if cfg.discord.max_replies_per_month:
            lines.append(
                f"  bot budget            {cfg.discord.max_replies_per_month} replies/month"
            )
    elif cfg.llm.provider == "ollama":
        lines.append("  local model, free")
    elif pricing is None:
        lines.append("  no built-in pricing for this model, check the provider's page")
    else:
        p_in, p_out = pricing
        cold = ((prefix_tokens + live_tokens) * p_in + out_tokens * p_out) / 1e6
        cached = (prefix_tokens * p_in * 0.1 + live_tokens * p_in + out_tokens * p_out) / 1e6
        lines.append(f"  first reply           ~${cold:.4f}")
        if cfg.llm.provider == "anthropic":
            lines.append(f"  cached replies        ~${cached:.4f} (prefix cache hit)")
            lines.append(
                f"  worst case per hour   ~${cached * cfg.discord.max_replies_per_hour:.2f} "
                f"(cap {cfg.discord.max_replies_per_hour}/h)"
            )
        else:
            lines.append(
                f"  worst case per hour   ~${cold * cfg.discord.max_replies_per_hour:.2f} "
                f"(cap {cfg.discord.max_replies_per_hour}/h)"
            )
    return lines


@app.command()
def inspect(
    name: str,
    cost: bool = typer.Option(False, "--cost", help="estimate per reply cost"),
) -> None:
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
    if cost:
        for line in _cost_lines(paths, cfg):
            typer.echo(line)


@app.command()
def index(
    name: str,
    rebuild: bool = typer.Option(False, "--rebuild", help="drop and rebuild the index"),
) -> None:
    """Build the persona's memory index from the corpus (local, free)."""
    from mimicord.config import load_config
    from mimicord.rag import build_index
    from mimicord.store import Store

    paths = PersonaPaths.for_persona(name)
    cfg = load_config(paths.config)
    if not paths.corpus.is_file():
        typer.echo("no corpus yet, run mimicord ingest first")
        raise typer.Exit(1)
    typer.echo("indexing (first run downloads a small local embedding model)...")
    with Store(paths.corpus) as store:
        total = build_index(
            paths,
            cfg.rag,
            store,
            rebuild=rebuild,
            progress=lambda done, all_: typer.echo(f"  {done}/{all_} windows"),
        )
    typer.echo(f"indexed {total} conversation windows into {paths.chroma_dir}")


@app.command()
def analyze(
    name: str,
    sample: int = typer.Option(50, "--sample", help="max chunks to analyze"),
    fresh: bool = typer.Option(
        False, "--fresh", help="ignore cached chunk analyses and redo everything"
    ),
) -> None:
    """LLM style analysis over the corpus (map per chunk, then one reduce)."""
    import json

    from mimicord.analyze.chunker import build_chunks, sample_chunks
    from mimicord.analyze.mapper import analyze_chunks
    from mimicord.analyze.reducer import reduce_profiles
    from mimicord.config import load_config
    from mimicord.llm.factory import get_provider
    from mimicord.store import Store

    paths = PersonaPaths.for_persona(name)
    cfg = load_config(paths.config)
    if not paths.corpus.is_file():
        typer.echo("no corpus yet, run mimicord ingest first")
        raise typer.Exit(1)

    with Store(paths.corpus) as store:
        chunks = sample_chunks(build_chunks(store), cap=sample)
        if not chunks:
            typer.echo("corpus has no target messages to analyze")
            raise typer.Exit(1)
        typer.echo(f"analyzing {len(chunks)} chunks with {cfg.llm.provider}")

        map_provider = get_provider(cfg.llm, role="map")

        def progress(chunk, cached):
            note = "cached" if cached else "done"
            typer.echo(
                f"  chunk {chunk.index + 1}/{len(chunks)} "
                f"({chunk.channel_name or chunk.channel_id}, "
                f"{chunk.target_count} target msgs) {note}"
            )

        results = analyze_chunks(
            chunks,
            map_provider,
            cfg.name,
            paths.chunks_dir,
            resume=not fresh,
            progress=progress,
        )

    typer.echo("merging into one profile...")
    reduce_provider = get_provider(cfg.llm, role="reduce")
    profile = reduce_profiles(results, reduce_provider, cfg.name)
    paths.profile.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"wrote {paths.profile}")


@app.command("compile")
def compile_cmd(
    name: str,
    examples: int = typer.Option(20, "--examples", help="few-shot examples to keep"),
) -> None:
    """Generate persona.md and few-shot examples from the analysis."""
    import json

    from mimicord.compile.examples import build_examples
    from mimicord.compile.persona import compile_persona
    from mimicord.config import load_config
    from mimicord.llm.factory import get_provider
    from mimicord.store import Store

    paths = PersonaPaths.for_persona(name)
    cfg = load_config(paths.config)
    if not paths.profile.is_file():
        typer.echo("no analysis profile yet, run mimicord analyze first")
        raise typer.Exit(1)
    if not paths.stats.is_file():
        typer.echo("no stats yet, run mimicord stats first")
        raise typer.Exit(1)

    profile = json.loads(paths.profile.read_text(encoding="utf-8"))
    stats_data = json.loads(paths.stats.read_text(encoding="utf-8"))
    provider = get_provider(cfg.llm, role="reduce")

    typer.echo("writing persona.md...")
    persona_text = compile_persona(profile, stats_data, provider, cfg.name)
    paths.persona_md.write_text(persona_text, encoding="utf-8")

    typer.echo("curating few-shot examples...")
    with Store(paths.corpus) as store:
        examples_data = build_examples(store, stats_data, provider, cfg.name, examples)
    paths.examples.write_text(
        json.dumps(examples_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    typer.echo(
        f"wrote {paths.persona_md} and {len(examples_data['examples'])} examples"
    )
    typer.echo("eyeball persona.md, then try: mimicord chat " + name)


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
