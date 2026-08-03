from __future__ import annotations

from mimicord.postprocess import apply, clean

AUTHORS = {"kekSoslav", "Lev", "Mike_"}


def test_the_leak_from_the_channel():
    """Live: 'nikoli' then 'sandman: sam vprasi leva kdaj'. Only line one
    was ever checked, so the second went out with the prefix visible."""
    bursts = apply(
        "nikoli\nsandman: sam vprasi leva kdaj",
        persona_name="sandman",
        context_authors=AUTHORS,
    )
    assert bursts == ["nikoli", "sam vprasi leva kdaj"]


def test_his_own_name_on_the_first_line_still_goes():
    assert clean("sandman: ne morm", "sandman", AUTHORS) == "ne morm"


def test_somebody_elses_name_with_a_colon_goes_from_any_line():
    assert clean("ja\nLev: pridi", "sandman", AUTHORS) == "ja\npridi"


def test_every_line_is_cleaned_not_just_one():
    assert clean(
        "sandman: ena\nsandman: dve\nsandman: tri", "sandman", AUTHORS
    ) == "ena\ndve\ntri"


def test_a_bare_name_only_goes_at_the_very_start():
    """'lev pride jutri' is a real sentence, so a name with no colon is only
    stripped where it opens the reply and reads as answering someone."""
    assert clean("Lev pridi no", "sandman", AUTHORS) == "pridi no"
    assert clean("ja\nLev pride jutri", "sandman", AUTHORS) == "ja\nLev pride jutri"


def test_a_time_is_not_a_name():
    assert clean("cez 20:30 sm not", "sandman", AUTHORS) == "cez 20:30 sm not"


def test_ordinary_replies_are_untouched():
    for reply in ("neja nemorm", "jao ne\nkdaj mas popravni", "2h cs pol pa spat"):
        assert clean(reply, "sandman", AUTHORS) == reply


def test_it_survives_without_any_names():
    assert clean("sandman: ne morm") == "sandman: ne morm"  # nothing to match on
