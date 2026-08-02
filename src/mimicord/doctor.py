from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from mimicord.config import ConfigError, PersonaConfig
from mimicord.paths import PersonaPaths

# view channels + send messages + read message history, nothing more
INVITE_PERMISSIONS = 68608


@dataclass
class Check:
    name: str
    ok: bool | None  # None means skipped, not failed
    detail: str

    @property
    def mark(self) -> str:
        return {True: "ok  ", False: "FAIL", None: "--  "}[self.ok]


def check_artifacts(paths: PersonaPaths, cfg: PersonaConfig) -> list[Check]:
    """Everything the bot needs on disk, in pipeline order."""
    checks: list[Check] = []

    if paths.corpus.is_file():
        from mimicord.store import Store

        with Store(paths.corpus) as store:
            counts = store.counts()
        ok = counts["target"] > 0
        detail = f"{counts['total']} messages, {counts['target']} from target"
        if not ok:
            detail += " (nothing matched [target], check author_ids)"
        checks.append(Check("corpus", ok, detail))
    else:
        checks.append(Check("corpus", None, "none yet, mimicord ingest"))

    checks.append(
        Check("stats", paths.stats.is_file(), "" if paths.stats.is_file() else "mimicord stats")
    )

    persona_ok = paths.persona_md.is_file()
    detail = ""
    if persona_ok:
        size = len(paths.persona_md.read_text(encoding="utf-8"))
        detail = f"{size} chars"
        if size < 400:
            detail += " (looks like the starter template, run mimicord compile)"
    else:
        detail = "missing, mimicord compile"
    checks.append(Check("persona.md", persona_ok, detail))

    if paths.extra.is_file():
        chars = len(paths.extra.read_text(encoding="utf-8").strip())
        checks.append(Check("extra.md", True, f"{chars} chars of hand-written facts"))
    else:
        checks.append(Check("extra.md", None, "none, optional"))

    if paths.examples.is_file():
        import json

        data = json.loads(paths.examples.read_text(encoding="utf-8"))
        count = len(data.get("examples", []))
        checks.append(Check("few-shots", count > 0, f"{count} examples"))
    else:
        checks.append(Check("few-shots", None, "none, mimicord compile"))

    for reaction in cfg.reactions:
        path = paths.media_dir / reaction.file
        checks.append(
            Check(
                f"reaction {reaction.name}",
                path.is_file(),
                str(path) if not path.is_file() else f"{path.stat().st_size // 1024} KB",
            )
        )

    if cfg.rag.enabled:
        checks.append(
            Check(
                "memories",
                paths.chroma_dir.is_dir(),
                "" if paths.chroma_dir.is_dir() else "missing, mimicord index",
            )
        )
    else:
        checks.append(Check("memories", None, "disabled in config"))

    return checks


def check_provider(cfg: PersonaConfig) -> Check:
    provider = cfg.llm.provider
    if provider == "claude-code":
        return Check(
            "llm", True, f"{provider}/{cfg.llm.model}, billed to your claude.ai plan"
        )
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }.get(provider)
    if env_var is None:
        return Check("llm", True, f"{provider}/{cfg.llm.model}, local")
    present = bool(os.environ.get(env_var))
    return Check(
        "llm",
        present,
        f"{provider}/{cfg.llm.model}" if present else f"{env_var} is not set",
    )


async def _probe_discord(token: str) -> dict:
    """Log in, note who we are and where, then disconnect immediately."""
    import discord

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    found: dict = {}

    @client.event
    async def on_ready() -> None:
        found["user"] = str(client.user)
        found["id"] = client.user.id if client.user else None
        found["guilds"] = [g.name for g in client.guilds]
        await client.close()

    try:
        await client.start(token)
    except discord.PrivilegedIntentsRequired:
        found["error"] = "intents"
    except discord.LoginFailure:
        found["error"] = "token"
    except Exception as error:  # network, gateway, anything unexpected
        found["error"] = f"connect: {error}"
    finally:
        if not client.is_closed():
            await client.close()
    return found


def check_discord(cfg: PersonaConfig) -> list[Check]:
    try:
        token = cfg.discord.token()
    except ConfigError as error:
        return [Check("discord token", False, str(error))]

    found = asyncio.run(_probe_discord(token))
    error = found.get("error")
    if error == "token":
        return [Check("discord token", False, "rejected, reset it in the dev portal")]
    if error == "intents":
        return [
            Check("discord token", True, "accepted"),
            Check(
                "message content intent",
                False,
                "not enabled on the bot page, the bot would see empty messages",
            ),
        ]
    if error:
        return [Check("discord login", False, str(error))]

    checks = [
        Check("discord token", True, f"logged in as {found.get('user')}"),
        Check("message content intent", True, "enabled"),
    ]
    guilds = found.get("guilds") or []
    if guilds:
        checks.append(Check("servers", True, ", ".join(guilds)))
    else:
        app_id = found.get("id")
        invite = (
            f"https://discord.com/oauth2/authorize?client_id={app_id}"
            f"&scope=bot&permissions={INVITE_PERMISSIONS}"
        )
        checks.append(Check("servers", False, f"not in any server yet: {invite}"))
    return checks
