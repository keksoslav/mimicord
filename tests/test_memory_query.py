from __future__ import annotations

from mimicord.engine import ContextMessage, memory_query


def ctx(*pairs) -> list[ContextMessage]:
    return [ContextMessage(a, t) for a, t in pairs]


def test_real_conversation_is_queryable():
    query = memory_query(ctx(("miha", "a gres jutri na pivo al ne")))
    assert "a gres jutri na pivo al ne" in query
    assert query.startswith("miha: ")


def test_only_pings_yields_no_query():
    """Someone typing just the bot's name is not a topic to look up."""
    assert memory_query(ctx(("kekSoslav", "@SaNdMaN"), ("kekSoslav", "<@123456>"))) == ""


def test_bare_name_summons_yield_no_query():
    """The exact shape that poisoned the first live test: the persona's name
    typed as plain text, which retrieved nothing but other people's pings."""
    context = ctx(
        ("kekSoslav", "timi?"), ("kekSoslav", "sandman"), ("kekSoslav", "@SaNdMaN")
    )
    assert memory_query(context, aliases={"sandman", "SaNdMaN", "timi"}) == ""


def test_alias_stripping_is_case_insensitive():
    assert memory_query(ctx(("miha", "SANDMAN")), aliases={"sandman"}) == ""


def test_alias_removed_from_the_query_itself():
    """The name must not reach the search: it dominates the embedding and
    retrieves every window where anyone was summoned."""
    query = memory_query(
        ctx(("miha", "sandman a si vidu kolk je ura zdele")), aliases={"sandman"}
    )
    assert "sandman" not in query.lower()
    assert "a si vidu kolk je ura zdele" in query


def test_alias_is_not_matched_inside_other_words():
    query = memory_query(ctx(("miha", "sandmanovo kolo je zunaj")), aliases={"sandman"})
    assert "sandmanovo" in query


def test_mentions_stripped_but_real_text_survives():
    query = memory_query(ctx(("kekSoslav", "<@99> a gres cs kasneje al kaj")))
    assert "<@99>" not in query
    assert "a gres cs kasneje al kaj" in query


def test_channel_mentions_stripped():
    query = memory_query(ctx(("miha", "poglej v <#555> je blo tam napisano vse")))
    assert "<#555>" not in query
    assert "je blo tam napisano vse" in query


def test_only_last_window_used():
    old = [("miha", f"stara sporocila {i} o necem drugem") for i in range(10)]
    query = memory_query(ctx(*old, ("miha", "novo vprasanje o cs2 danes")), window=2)
    assert "novo vprasanje" in query
    assert "stara sporocila 0" not in query


def test_empty_context():
    assert memory_query([]) == ""


def test_ping_spam_memories_are_dropped():
    """The exact windows that derailed the live bot: nothing but summons."""
    from mimicord.engine import is_useful_memory

    aliases = {"sandman", "timi"}
    spam = (
        "(2022-06-19, SaNdMaN) kekSoslav: Zdravo timi / kekSoslav: Timi? "
        "/ kekSoslav: Timi / kekSoslav: @SaNdMaN"
    )
    assert is_useful_memory(spam, aliases) is False


def test_substantive_memories_are_kept():
    from mimicord.engine import is_useful_memory

    real = (
        "(2024-02-11, SaNdMaN) SaNdMaN: ja sm bil na sihtu cel dan pa sm mrtu "
        "/ kekSoslav: kok dolg / SaNdMaN: od sestih do dveh"
    )
    assert is_useful_memory(real, {"sandman", "timi"}) is True


def test_memory_mentioning_the_name_in_passing_is_kept():
    from mimicord.engine import is_useful_memory

    text = (
        "(2024-05-02, general) kekSoslav: timi a si vidu da so dali nov update "
        "za cs2 / SaNdMaN: ja sm bral smoke so spet spremenili"
    )
    assert is_useful_memory(text, {"sandman", "timi"}) is True
