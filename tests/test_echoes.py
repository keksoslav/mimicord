from __future__ import annotations

import asyncio

from mimicord.bot import MimicClient
from mimicord.engine import ContextMessage, PersonaEngine
from mimicord.postprocess import drop_echoes


def test_his_own_last_message_is_dropped():
    kept = drop_echoes(["nvem sam vprasi njo", "pa to"], said_by_me=["nvem sam vprasi njo"])
    assert kept == ["pa to"]


def test_the_bug_from_the_channel():
    """He repeated his own line and parroted Mike's, in one reply."""
    kept = drop_echoes(
        ["nvem sam vprasi njo", "Skibi skibi 5"],
        said_by_me=["nvem sam vprasi njo"],
        said_by_others=["Timi, kako kaj Juvan", "Skibi skibi 5"],
    )
    assert kept == []


def test_parroting_someone_is_dropped():
    kept = drop_echoes(["Skibi skibi 5"], said_by_others=["Skibi skibi 5"])
    assert kept == []


def test_short_agreement_is_not_parroting():
    """Two people saying kk is a conversation, not an echo."""
    for short in ("kk", "ja", "neja", "true", "xD"):
        assert drop_echoes([short], said_by_others=[short]) == [short]


def test_matching_ignores_case_and_padding():
    assert drop_echoes(["  SKIBI SKIBI 5 "], said_by_others=["skibi skibi 5"]) == []


def test_a_reply_that_repeats_itself_is_deduped():
    assert drop_echoes(["ma daj no", "ma daj no", "pa to"]) == ["ma daj no", "pa to"]


def test_a_normal_reply_survives():
    kept = drop_echoes(
        ["neja", "morm se ucit"],
        said_by_me=["kk"],
        said_by_others=["a gres jutri ven"],
    )
    assert kept == ["neja", "morm se ucit"]


class FakeChannel:
    id = 100

    def __str__(self) -> str:
        return "general"


def make_client(persona_dir, fake_provider) -> MimicClient:
    engine = PersonaEngine("testbot", rag_enabled=False)
    engine.provider = fake_provider
    client = MimicClient(engine, dry_run=True)
    client.seeded.add(FakeChannel.id)
    return client


def his(client) -> list[str]:
    return [m.content for m in client.buffers[FakeChannel.id] if m.author == "testbot"]


def test_a_partial_repeat_is_caught_and_retried(persona_dir, fake_provider):
    """The whole-reply check used to miss this: first burst repeats, second is new."""
    fake_provider.replies = ["nvem sam vprasi njo", "nvem sam vprasi njo\nSkibi skibi 5", "ma pridi"]
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel()
    client.buffers[channel.id].append(ContextMessage("Mike_", "Skibi skibi 5"))

    asyncio.run(client._respond(channel, "mention"))
    asyncio.run(client._respond(channel, "mention"))

    assert his(client) == ["nvem sam vprasi njo", "ma pridi"]


def test_only_the_echo_is_dropped_when_the_rest_is_new(persona_dir, fake_provider):
    fake_provider.replies = ["ne pridem", "ne pridem\nmam druge plane"]
    client = make_client(persona_dir, fake_provider)
    channel = FakeChannel()
    client.buffers[channel.id].append(ContextMessage("Mike_", "a gres ven"))

    asyncio.run(client._respond(channel, "mention"))
    asyncio.run(client._respond(channel, "mention"))

    # the repeated first line goes, the new line still gets sent, no retry
    assert his(client) == ["ne pridem", "mam druge plane"]
    assert len(fake_provider.calls) == 2
