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
