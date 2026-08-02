from __future__ import annotations

from mimicord.postprocess import apply, clean, split_bursts


def test_strips_thinking_blocks():
    text = "<thinking>they want a greeting</thinking>hej"
    assert clean(text) == "hej"


def test_strips_stray_tags():
    assert clean("</thinking>ne vem") == "ne vem"


def test_strips_name_echo():
    assert clean("janez: ma daj", persona_name="janez") == "ma daj"


def test_unwraps_quotes():
    assert clean('"lol ne"') == "lol ne"


def test_strips_ai_isms():
    text = "sounds rough\nLet me know if you need anything else!"
    assert clean(text) == "sounds rough"


def test_split_bursts_caps_and_merges():
    text = "one\ntwo\nthree\nfour"
    assert split_bursts(text, max_burst=3) == ["one", "two", "three four"]


def test_split_bursts_empty():
    assert split_bursts("   \n  ") == []


def test_apply_lowercases_when_stats_say_so():
    stats = {"capitalization": {"starts_lowercase": 0.95}}
    assert apply("Ne vem", stats=stats) == ["ne vem"]


def test_apply_keeps_all_caps():
    stats = {"capitalization": {"starts_lowercase": 0.95}}
    assert apply("LOL ne", stats=stats) == ["LOL ne"]


def test_apply_drops_trailing_period():
    stats = {"punctuation": {"ends_with_period": 0.02}}
    assert apply("ma ja res.", stats=stats) == ["ma ja res"]


def test_apply_neutralizes_mass_pings():
    assert apply("@everyone wake up") == ["everyone wake up"]


def test_apply_no_stats_leaves_text_alone():
    assert apply("Ne vem.") == ["Ne vem."]


AUTHORS = {"kekSoslav", "Lev"}


def test_strips_leading_author_name_the_model_echoed():
    """Observed live: every other reply opened with whoever it answered."""
    assert clean("kekSoslav mislu da s ti", context_authors=AUTHORS) == "mislu da s ti"
    assert clean("kekSoslav kako si", context_authors=AUTHORS) == "kako si"


def test_strips_author_name_with_punctuation():
    assert clean("kekSoslav, ja sm tu", context_authors=AUTHORS) == "ja sm tu"
    assert clean("Lev: pridi", context_authors=AUTHORS) == "pridi"


def test_author_name_stripping_is_case_insensitive():
    assert clean("keksoslav ne vem", context_authors=AUTHORS) == "ne vem"


def test_only_the_first_occurrence_is_stripped():
    out = clean("kekSoslav a si vidu kaj je Lev naredil", context_authors=AUTHORS)
    assert out == "a si vidu kaj je Lev naredil"


def test_mention_form_is_left_alone():
    """@mentions are how he actually addresses people, keep them."""
    assert clean("@kekSoslav pridi", context_authors=AUTHORS) == "@kekSoslav pridi"


def test_name_alone_is_not_stripped_to_nothing():
    assert clean("kekSoslav", context_authors=AUTHORS) == "kekSoslav"


def test_name_inside_the_sentence_survives():
    out = clean("ja pa kekSoslav je reku da ne", context_authors=AUTHORS)
    assert out == "ja pa kekSoslav je reku da ne"
