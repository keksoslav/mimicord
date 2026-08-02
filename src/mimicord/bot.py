from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import discord

from mimicord.engine import ContextMessage, PersonaEngine
from mimicord.triggers import MessageFacts, TriggerState, should_reply
from mimicord.usage import UsageLedger

log = logging.getLogger(__name__)


class MimicClient(discord.Client):
    """One persona, one bot account, one process."""

    def __init__(self, engine: PersonaEngine, *, dry_run: bool = False) -> None:
        intents = discord.Intents.default()
        # privileged: must also be enabled on the bot page in the dev portal,
        # otherwise login fails; forgetting it here instead makes every
        # message.content arrive as an empty string
        intents.message_content = True
        super().__init__(
            intents=intents,
            allowed_mentions=discord.AllowedMentions(
                everyone=False, roles=False, users=True, replied_user=True
            ),
        )
        self.engine = engine
        self.cfg = engine.config.discord
        self.style = engine.config.style
        self.dry_run = dry_run
        self.state = TriggerState()
        self.ledger = UsageLedger(engine.paths.root / "usage.json")
        self.buffers: dict[int, deque[ContextMessage]] = defaultdict(
            lambda: deque(maxlen=self.cfg.context_messages)
        )
        self.seeded: set[int] = set()
        self.locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def on_ready(self) -> None:
        llm = self.engine.config.llm
        mode = "dry run, replies stay local" if self.dry_run else "live"
        log.info("logged in as %s (%s)", self.user, mode)
        log.info(
            "persona %s via %s/%s", self.engine.config.name, llm.provider, llm.model
        )

    async def on_message(self, message: discord.Message) -> None:
        if self.user is not None and message.author.id == self.user.id:
            return  # own sends are appended to the buffer at send time
        content = (message.content or "").strip()
        author = message.author.display_name
        if content:
            self.buffers[message.channel.id].append(ContextMessage(author, content))

        facts = MessageFacts(
            channel_id=str(message.channel.id),
            author_is_self=False,
            author_is_bot=message.author.bot,
            mentions_bot=self._mentions_me(message),
            replies_to_bot=self._replies_to_me(message),
            content=content,
        )
        decision, reason = should_reply(
            facts,
            self.cfg,
            self.state,
            time.monotonic(),
            random.random(),
            monthly_count=self.ledger.count(datetime.now(timezone.utc)),
        )
        if not decision:
            log.debug("skip #%s: %s", message.channel.id, reason)
            return

        lock = self.locks[message.channel.id]
        if lock.locked():
            log.debug("skip #%s: already generating", message.channel.id)
            return
        async with lock:
            self.state.note_reply(facts.channel_id, time.monotonic())
            # dry runs still call the llm, so they count against the budget too
            self.ledger.increment(datetime.now(timezone.utc))
            try:
                await self._respond(message.channel, reason)
            except Exception:
                log.exception("reply failed in #%s", message.channel.id)

    def _mentions_me(self, message: discord.Message) -> bool:
        if self.user is None:
            return False
        return any(m.id == self.user.id for m in message.mentions)

    def _replies_to_me(self, message: discord.Message) -> bool:
        if self.user is None or message.reference is None:
            return False
        resolved = message.reference.resolved
        return (
            isinstance(resolved, discord.Message)
            and resolved.author.id == self.user.id
        )

    async def _seed_buffer(self, channel) -> None:
        """First trigger in a channel backfills recent history for context."""
        if channel.id in self.seeded:
            return
        self.seeded.add(channel.id)
        try:
            history = [
                m
                async for m in channel.history(limit=self.cfg.context_messages)
            ]
        except discord.Forbidden:
            log.debug("no history permission in #%s", channel.id)
            return
        buffer = self.buffers[channel.id]
        buffer.clear()
        for m in reversed(history):  # history yields newest first
            text = (m.content or "").strip()
            if text:
                buffer.append(ContextMessage(m.author.display_name, text))

    async def _respond(self, channel, reason: str) -> None:
        await self._seed_buffer(channel)
        context = list(self.buffers[channel.id])
        if not context:
            return
        log.info("replying in #%s (%s)", channel.id, reason)
        bursts = await asyncio.to_thread(self.engine.reply, context)
        if not bursts:
            log.warning("nothing to say in #%s, staying quiet", channel.id)
            return
        persona = self.engine.config.name
        for burst in bursts:
            if self.dry_run:
                log.info("[dry run #%s] %s: %s", channel, persona, burst)
            else:
                async with channel.typing():
                    delay = min(len(burst) / self.style.typing_cps, 6.0)
                    await asyncio.sleep(delay + random.uniform(0.3, 0.9))
                await channel.send(burst)
                await asyncio.sleep(random.uniform(0.5, 1.5))
            self.buffers[channel.id].append(ContextMessage(persona, burst))


def run(name: str, *, dry_run: bool = False) -> None:
    # bot status lines are log records so both the cli and the gui can show
    # them; make sure they clear the default warning threshold
    logging.getLogger("mimicord").setLevel(logging.INFO)
    engine = PersonaEngine(name)
    token = engine.config.discord.token()
    client = MimicClient(engine, dry_run=dry_run)
    # our cli sets up logging, keep discord.py from installing its own handler
    client.run(token, log_handler=None)
