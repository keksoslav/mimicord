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


def test_alias_inside_a_real_sentence_still_queries():
    query = memory_query(
        ctx(("miha", "sandman a si vidu kolk je ura zdele")), aliases={"sandman"}
    )
    # the alias only gates the decision, the query keeps the original wording
    assert "sandman a si vidu kolk je ura zdele" in query


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
