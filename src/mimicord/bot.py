from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import discord

from mimicord import postprocess, vision
from mimicord.config import Reaction
from mimicord.engine import ContextMessage, PersonaEngine
from mimicord.triggers import MessageFacts, TriggerState, should_poke, should_reply
from mimicord.usage import UsageLedger

log = logging.getLogger(__name__)

# how often to look for channels that have gone quiet
IDLE_CHECK_SECONDS = 300
# only recent talkers are worth poking, nobody wants a ping from three weeks ago
POKE_CANDIDATES = 6
# how far back to look for the same line before calling it a flood
REPEAT_WINDOW = 4
# how many recent messages he must not parrot back word for word
ECHO_LOOKBACK = 6


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
        # wall clock, so hours that passed while the bot was down still count
        self.last_seen: dict[int, datetime] = {}
        # channel -> {user id: display name}, oldest talker first
        self.people: dict[int, dict[int, str]] = defaultdict(dict)
        # kept per burst, not joined, so a reply that repeats only its first
        # line is still caught
        self.last_reply: dict[int, list[str]] = {}
        self.last_reaction_at: dict[int, float] = {}
        self.last_tag_at: dict[tuple[int, str], float] = {}
        self.idle_task: asyncio.Task | None = None

    async def setup_hook(self) -> None:
        if self.cfg.idle_hours > 0:
            self.idle_task = asyncio.create_task(self._idle_loop())

    async def close(self) -> None:
        if self.idle_task is not None:
            self.idle_task.cancel()
        await super().close()

    async def on_ready(self) -> None:
        llm = self.engine.config.llm
        mode = "dry run, replies stay local" if self.dry_run else "live"
        log.info("logged in as %s (%s)", self.user, mode)
        log.info(
            "persona %s via %s/%s", self.engine.config.name, llm.provider, llm.model
        )
        if self.cfg.idle_hours > 0:
            log.info(
                "speaks up after %gh of silence in %d channel(s)",
                self.cfg.idle_hours,
                len(self.cfg.poke_channels()),
            )

    async def on_message(self, message: discord.Message) -> None:
        if self.user is not None and message.author.id == self.user.id:
            return  # own sends are appended to the buffer at send time
        is_dm = message.guild is None
        if is_dm and not self.cfg.allow_dms:
            # bail before buffering or downloading anything. should_reply would
            # refuse this too, but by then we would already have pulled the
            # attachments off the cdn for someone we are never going to answer
            log.debug("ignoring a dm from %s", message.author)
            return
        content = (message.content or "").strip()
        author = message.author.display_name
        self.last_seen[message.channel.id] = datetime.now(timezone.utc)
        if not message.author.bot:
            self._note_person(message.channel.id, message.author)

        buffer = self.buffers[message.channel.id]
        # check before appending, otherwise the message always matches itself
        repeats = bool(content) and any(
            entry.author == author and entry.content == content
            for entry in list(buffer)[-REPEAT_WINDOW:]
        )
        images = await self._download_images(message)
        if content or images:
            buffer.append(ContextMessage(author, content, images))

        facts = MessageFacts(
            channel_id=str(message.channel.id),
            author_is_self=False,
            author_is_bot=message.author.bot,
            mentions_bot=self._mentions_me(message),
            replies_to_bot=self._replies_to_me(message),
            content=content,
            repeats_recent=repeats,
            is_dm=is_dm,
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
            if self.cfg.debounce_seconds:
                # hold the lock while the rest of their burst lands. those
                # messages get buffered and skipped, then answered as one
                await asyncio.sleep(self.cfg.debounce_seconds)
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

    async def _download_images(self, message: discord.Message) -> list:
        """Pull image attachments off the cdn and shrink them.

        Shrinking happens here rather than at send time so the buffer only
        ever holds the cheap version, however long it sits there.
        """
        cfg = self.engine.config.vision
        if not cfg.enabled or cfg.max_images <= 0 or not message.attachments:
            return []
        images = []
        for attachment in message.attachments:
            if len(images) >= cfg.max_images:
                break
            if not vision.is_image(attachment.content_type):
                continue
            if attachment.size > vision.MAX_SOURCE_BYTES:
                log.debug("skipping a %d byte attachment", attachment.size)
                continue
            try:
                raw = await attachment.read()
            except Exception as error:  # cdn hiccup, expired link, anything
                log.debug("could not fetch an attachment: %s", error)
                continue
            image = await asyncio.to_thread(vision.prepare, raw, cfg.max_edge)
            if image is not None:
                log.info(
                    "saw a %dx%d image in #%s, about %d tokens",
                    image.width,
                    image.height,
                    message.channel.id,
                    image.tokens,
                )
                images.append(image)
        return images

    def _note_person(self, channel_id: int, user) -> None:
        people = self.people[channel_id]
        people.pop(user.id, None)  # reinsert so the newest talker ends up last
        people[user.id] = user.display_name
        while len(people) > POKE_CANDIDATES:
            del people[next(iter(people))]

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
            if m.author.bot or (self.user is not None and m.author.id == self.user.id):
                continue
            self._note_person(channel.id, m.author)
        if history and channel.id not in self.last_seen:
            self.last_seen[channel.id] = history[0].created_at

    async def _respond(self, channel, reason: str) -> None:
        await self._seed_buffer(channel)
        context = list(self.buffers[channel.id])
        if not context:
            return
        log.info("replying in #%s (%s)", channel.id, reason)
        raw = await asyncio.to_thread(self.engine.reply, context)
        bursts = self._fresh(channel.id, raw, context)
        if raw and not bursts:
            # everything he came up with was already in the channel, either his
            # own last message or somebody else's. ask outright. one retry,
            # then give up: silence is cheaper than a loop
            log.info("only echoes in #%s, asking again", channel.id)
            self.ledger.increment(datetime.now(timezone.utc))
            raw = await asyncio.to_thread(
                self.engine.reply, context, self._say_something_else(channel.id)
            )
            bursts = self._fresh(channel.id, raw, context)
        if not bursts:
            log.info("nothing fresh to say in #%s, staying quiet", channel.id)
            return
        await self._send_bursts(channel, bursts)

    def _fresh(self, channel_id: int, bursts: list[str], context) -> list[str]:
        """Whatever is left once the lines already in the channel are removed."""
        persona = self.engine.config.name
        others = [
            m.content
            for m in context[-ECHO_LOOKBACK:]
            if m.content and m.author != persona
        ]
        return postprocess.drop_echoes(
            bursts,
            said_by_me=self.last_reply.get(channel_id, []),
            said_by_others=others,
        )

    def _say_something_else(self, channel_id: int) -> str:
        previous = " / ".join(self.last_reply.get(channel_id, []))
        return (
            f'You just said "{previous}" and you are about to say it again. '
            "Say something different. Do not rephrase that, react to what was "
            "actually said instead."
        )

    def _reaction_on_cooldown(self, channel_id: int, reaction) -> bool:
        """Hand him a folder of photos and he will reach for one every time.

        Asking him not to does not survive contact with a real conversation,
        so this is a wall rather than a request. Two of them: a global one so
        he is not permanently answering in pictures, and a per tag one so the
        same photo does not come out twice in an evening.
        """
        now = time.monotonic()
        gap = self.style.reaction_cooldown_seconds
        if gap > 0:
            last = self.last_reaction_at.get(channel_id)
            if last is not None and now - last < gap:
                return True
        if reaction.cooldown > 0:
            last = self.last_tag_at.get((channel_id, reaction.name))
            if last is not None and now - last < reaction.cooldown:
                return True
        return False

    async def _type_for(self, channel, seconds: float) -> None:
        """Show the typing indicator, then pause, before sending.

        The indicator is decoration. Discord drops the odd connection, and a
        blip here used to take the whole reply down with it, after we had
        already paid to generate it.
        """
        try:
            async with channel.typing():
                await asyncio.sleep(seconds)
                return
        except Exception as error:
            log.debug("typing indicator failed (%s), sending anyway", error)
        await asyncio.sleep(seconds)

    async def _send_bursts(self, channel, bursts: list[str], mention: str = "") -> None:
        """Post a reply as separate messages, turning [gif:x] tags into images.

        mention is prefixed to the first message that is actually text, so a
        ping never eats a gif tag and never arrives on its own line.
        """
        items: list[tuple[str, object, object]] = []  # text, reaction, file path
        for burst in bursts:
            wanted = postprocess.picture_query(burst)
            if wanted is not None:
                match = self.engine.find_picture(wanted)
                if match is None:
                    # asking for a photo that does not exist costs nothing and
                    # sends nothing, which is the right failure
                    log.info("no photo like %r in #%s", wanted, channel.id)
                    continue
                # keyed on the file so the same photo does not come round again
                found = Reaction(
                    name=f"pic:{match.path.name}",
                    cooldown=self.engine.config.pictures.repeat_cooldown_seconds,
                )
                if self._reaction_on_cooldown(channel.id, found):
                    log.info("holding a photo back in #%s, too soon", channel.id)
                    continue
                log.info(
                    "photo for %r -> %s (distance %.2f)",
                    wanted,
                    match.path.name,
                    match.distance,
                )
                items.append((burst, found, match.path))
                continue
            name = postprocess.reaction_name(burst)
            if name is None:
                items.append((burst, None, None))
                continue
            reaction = self.engine.find_reaction(name)
            path = None
            if reaction is not None and not reaction.url:
                path = self.engine.reaction_path(reaction)
                if path is None:
                    reaction = None
            if reaction is None:
                # never post a raw tag, it breaks the illusion worse than silence
                log.warning("reaction %r is unusable, dropping", name)
                continue
            if self._reaction_on_cooldown(channel.id, reaction):
                log.info("holding %r back in #%s, too soon after the last", name, channel.id)
                continue
            items.append((burst, reaction, path))
        if not items:
            return

        if mention:
            first_text = next(
                (i for i, item in enumerate(items) if item[1] is None), None
            )
            if first_text is None:
                items.insert(0, (mention, None, None))  # ping first, then the gif
            else:
                items[first_text] = (f"{mention} {items[first_text][0]}", None, None)

        persona = self.engine.config.name
        for text, reaction, path in items:
            if reaction is not None:
                now = time.monotonic()
                self.last_reaction_at[channel.id] = now
                self.last_tag_at[(channel.id, reaction.name)] = now
            if self.dry_run:
                if reaction is not None:
                    what = f"<sends {path.name if path else reaction.url}>"
                else:
                    what = text
                log.info("[dry run #%s] %s: %s", channel, persona, what)
            elif reaction is not None:
                await self._type_for(channel, random.uniform(0.8, 1.6))
                if path is not None:
                    await channel.send(file=discord.File(path))
                else:
                    await channel.send(reaction.url)
                await asyncio.sleep(random.uniform(0.5, 1.5))
            else:
                delay = min(len(text) / self.style.typing_cps, 6.0)
                await self._type_for(channel, delay + random.uniform(0.3, 0.9))
                await channel.send(text)
                await asyncio.sleep(random.uniform(0.5, 1.5))
            # the buffer keeps the tag so he can see he already reacted
            self.buffers[channel.id].append(ContextMessage(persona, text))
        self.last_seen[channel.id] = datetime.now(timezone.utc)
        # what was asked for, not what went out, so the check in _respond
        # compares like with like
        self.last_reply[channel.id] = list(bursts)

    async def _idle_loop(self) -> None:
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                await self._check_idle()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("idle check failed")
            await asyncio.sleep(IDLE_CHECK_SECONDS)

    async def _check_idle(self) -> None:
        now = datetime.now(timezone.utc)
        for raw_id in self.cfg.poke_channels():
            if not raw_id.isdigit():
                continue
            channel = self.get_channel(int(raw_id))
            if channel is None:
                log.debug("idle check: channel %s not visible", raw_id)
                continue
            await self._seed_buffer(channel)
            last = self.last_seen.get(channel.id)
            if last is None:
                continue  # nothing ever seen here, nothing to be quiet about
            decision, reason = should_poke(
                raw_id,
                self.cfg,
                self.state,
                time.monotonic(),
                (now - last).total_seconds(),
                monthly_count=self.ledger.count(now),
            )
            if not decision:
                log.debug("no poke in #%s: %s", channel.id, reason)
                continue
            lock = self.locks[channel.id]
            if lock.locked():
                continue
            async with lock:
                # claim the silence up front, so a send that fails does not
                # get retried every five minutes
                self.last_seen[channel.id] = now
                self.state.note_reply(raw_id, time.monotonic())
                self.ledger.increment(now)
                try:
                    await self._poke(channel)
                except Exception:
                    log.exception("idle poke failed in #%s", channel.id)

    async def _poke(self, channel) -> None:
        """Break a long silence by pinging someone who talks here."""
        people = self.people.get(channel.id) or {}
        if not people:
            log.info("#%s is quiet but there is nobody to poke", channel.id)
            return
        user_id = random.choice(list(people))
        name = people[user_id]
        log.info("poking %s in #%s after %gh", name, channel.id, self.cfg.idle_hours)
        direction = (
            f"Nobody has said anything here for about {self.cfg.idle_hours:g} hours. "
            f"Say one short thing to {name} to get the chat going again. Make it "
            "funny and make it sound like you, not like a greeting and not like "
            "an announcement. Do not start with their name, they get pinged "
            "anyway. Do not explain yourself, just say the thing."
        )
        context = list(self.buffers[channel.id])
        bursts = await asyncio.to_thread(self.engine.reply, context, direction)
        if not bursts:
            log.warning("nothing to poke #%s with, staying quiet", channel.id)
            return
        await self._send_bursts(channel, bursts, mention=f"<@{user_id}>")


def run(name: str, *, dry_run: bool = False) -> None:
    # bot status lines are log records so both the cli and the gui can show
    # them; make sure they clear the default warning threshold
    logging.getLogger("mimicord").setLevel(logging.INFO)
    engine = PersonaEngine(name)
    token = engine.config.discord.token()
    client = MimicClient(engine, dry_run=dry_run)
    # our cli sets up logging, keep discord.py from installing its own handler
    client.run(token, log_handler=None)
