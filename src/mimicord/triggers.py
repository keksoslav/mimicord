from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from mimicord.config import DiscordConfig


@dataclass
class MessageFacts:
    """Discord-free view of an incoming message so decisions stay unit-testable."""

    channel_id: str
    author_is_self: bool
    author_is_bot: bool
    mentions_bot: bool
    replies_to_bot: bool
    content: str


@dataclass
class TriggerState:
    """Per-run mutable state: channel cooldowns and the hourly reply budget."""

    last_reply_at: dict[str, float] = field(default_factory=dict)
    sent_at: deque = field(default_factory=deque)

    def note_reply(self, channel_id: str, now: float) -> None:
        self.last_reply_at[channel_id] = now
        self.sent_at.append(now)


def _keyword_hit(content: str, keywords: list[str]) -> bool:
    lowered = content.lower()
    return any(re.search(rf"\b{re.escape(k)}\b", lowered) for k in keywords)


def should_reply(
    facts: MessageFacts,
    cfg: DiscordConfig,
    state: TriggerState,
    now: float,
    roll: float,
    monthly_count: int = 0,
) -> tuple[bool, str]:
    """Decide whether to reply. roll is a pre-drawn random in [0, 1).

    Returns (decision, reason). Pure function of its inputs apart from
    pruning expired entries out of state.sent_at.
    """
    if facts.author_is_self:
        return False, "own message"
    if facts.author_is_bot and cfg.ignore_bots:
        return False, "bot author"
    if cfg.channel_allowlist and facts.channel_id not in cfg.channel_allowlist:
        return False, "channel not in allowlist"

    trigger = None
    if cfg.trigger_mention and facts.mentions_bot:
        trigger = "mention"
    elif cfg.trigger_reply and facts.replies_to_bot:
        trigger = "reply"
    elif cfg.trigger_keywords and _keyword_hit(facts.content, cfg.trigger_keywords):
        trigger = "keyword"
    elif facts.channel_id in cfg.always_on_channels:
        trigger = "always on"
    elif cfg.interject_probability > 0 and roll < cfg.interject_probability:
        trigger = "interject"
    if trigger is None:
        return False, "no trigger"

    if cfg.max_replies_per_month and monthly_count >= cfg.max_replies_per_month:
        return False, "monthly cap"
    while state.sent_at and now - state.sent_at[0] > 3600:
        state.sent_at.popleft()
    if len(state.sent_at) >= cfg.max_replies_per_hour:
        return False, "hourly cap"
    last = state.last_reply_at.get(facts.channel_id)
    if last is not None and now - last < cfg.cooldown_seconds:
        return False, "cooldown"
    return True, trigger
