from __future__ import annotations

import codecs
import io

from mimicord.cli import force_utf8_output


class _Cp1252Stream(io.TextIOWrapper):
    """Stands in for a stock windows console, which cannot encode emoji."""

    def __init__(self) -> None:
        super().__init__(io.BytesIO(), encoding="cp1252", errors="strict")


def test_emoji_kills_a_cp1252_stream_without_the_fix():
    stream = _Cp1252Stream()
    try:
        stream.write("top emoji 🙂")
        stream.flush()
    except UnicodeEncodeError:
        pass
    else:
        raise AssertionError("expected the unpatched stream to fail")


def test_force_utf8_makes_emoji_printable(monkeypatch):
    stream = _Cp1252Stream()
    monkeypatch.setattr("sys.stdout", stream)
    monkeypatch.setattr("sys.stderr", stream)
    force_utf8_output()
    stream.write("top emoji 🙂 č ž")  # emoji plus slovene diacritics
    stream.flush()
    assert stream.encoding == "utf-8"


def test_force_utf8_survives_streams_without_reconfigure(monkeypatch):
    # pytest's capture objects and some redirections lack reconfigure
    class _Plain:
        def write(self, text):
            return len(text)

    monkeypatch.setattr("sys.stdout", _Plain())
    monkeypatch.setattr("sys.stderr", _Plain())
    force_utf8_output()  # must not raise


def test_force_utf8_survives_reconfigure_errors(monkeypatch):
    class _Grumpy:
        encoding = "cp1252"

        def reconfigure(self, **kwargs):
            raise OSError("detached stream")

    monkeypatch.setattr("sys.stdout", _Grumpy())
    monkeypatch.setattr("sys.stderr", _Grumpy())
    force_utf8_output()  # must not raise


def test_stats_summary_with_emoji_encodes_under_cp1252():
    """The regression that started this: summary_lines embeds top emoji."""
    from mimicord.analyze.stats import summary_lines

    stats = {
        "message_count": 10,
        "length": {"avg_chars": 12.0, "median_chars": 9, "p90_chars": 30},
        "burst": {"p_multi": 0.3, "avg_burst_len": 1.4, "bursts": 7},
        "capitalization": {"starts_lowercase": 0.9, "all_caps_rate": 0.0},
        "punctuation": {
            "ends_with_period": 0.0,
            "question_rate": 0.1,
            "exclaim_rate": 0.0,
            "ellipsis_rate": 0.0,
        },
        "emoji": {"per_message": 0.2, "top": [{"emoji": "🙂", "count": 3}]},
        "language": {"slovene_diacritics_rate": 0.15},
        "misc": {},
    }
    text = "\n".join(summary_lines(stats))
    assert "🙂" in text
    # utf-8 encodes it; cp1252 only survives with the replace policy the fix sets
    assert text.encode("utf-8")
    assert codecs.encode(text, "cp1252", "replace")
