from __future__ import annotations

import json

from mimicord.engine import (
    QUESTION_RE,
    ContextMessage,
    PersonaEngine,
    quiz_streak,
)

EXAMPLES = {
    "examples": [
        {"context": [{"author": "miha", "content": "a gres"}], "reply": "ne"},
    ]
}


def make_engine(persona_dir, fake_provider):
    (persona_dir / "examples.json").write_text(json.dumps(EXAMPLES), encoding="utf-8")
    engine = PersonaEngine("testbot", rag_enabled=False)
    engine.provider = fake_provider
    return engine


def msg(author, text):
    return ContextMessage(author, text)


def test_the_live_quiz_is_detected():
    """The screenshot: two trick questions answered, the third incoming."""
    context = [
        msg("Mike_", "Okej. Koliko ur je v eni minuti"),
        msg("testbot", "60"),
        msg("Mike_", "Zelo dobro. Naslednje vprasanje koliko dni je v eni uri"),
        msg("testbot", "0"),
        msg("Mike_", "Zelo dobro, koliko ur je v eni minuti?"),
    ]
    assert quiz_streak(context, "testbot") == ("Mike_", 2)


def test_a_normal_conversation_is_not_a_quiz():
    """'a gres ven' is not a question by the regex, so the pattern breaks."""
    context = [
        msg("miha", "kaj delas"),
        msg("testbot", "nc"),
        msg("miha", "a gres ven"),
    ]
    assert quiz_streak(context, "testbot")[1] == 0


def test_a_second_voice_ends_the_interrogation():
    context = [
        msg("Mike_", "koliko je 2+2"),
        msg("testbot", "4"),
        msg("lev", "koliko je 3+3"),
        msg("testbot", "6"),
        msg("lev", "pa koliko je 4+4?"),
    ]
    # only lev's pair counts, Mike_ is on the far side of the author change
    assert quiz_streak(context, "testbot") == ("lev", 1)


def test_his_own_burst_counts_as_one_reply():
    context = [
        msg("Mike_", "koliko casa se pozna"),
        msg("testbot", "hm"),
        msg("testbot", "pol ure"),
        msg("Mike_", "pa koliko je to v minutah"),
        msg("testbot", "30"),
        msg("Mike_", "koliko sekund?"),
    ]
    assert quiz_streak(context, "testbot") == ("Mike_", 2)


def test_chit_chat_between_questions_breaks_the_streak():
    context = [
        msg("Mike_", "koliko je ura"),
        msg("testbot", "petnajst"),
        msg("Mike_", "haha dobr"),
        msg("Mike_", "koliko dni ima teden"),
        msg("testbot", "7"),
        msg("Mike_", "koliko ur ima dan?"),
    ]
    assert quiz_streak(context, "testbot")[1] == 1


def test_slang_question_words_count():
    assert QUESTION_RE.search("kolk je ura")
    assert QUESTION_RE.search("kva dogaja")
    assert not QUESTION_RE.search("gremo ven")


def test_the_nudge_reaches_the_prompt(persona_dir, fake_provider):
    engine = make_engine(persona_dir, fake_provider)
    engine.reply(
        [
            msg("Mike_", "koliko ur je v eni minuti"),
            msg("testbot", "60"),
            msg("Mike_", "koliko dni je v eni uri"),
            msg("testbot", "0"),
            msg("Mike_", "koliko ur je v eni minuti?"),
        ]
    )
    live = fake_provider.calls[0]["messages"][-1].content
    assert "[direction]" in live
    assert "Mike_" in live.split("[direction]")[1]


def test_no_nudge_in_ordinary_chat(persona_dir, fake_provider):
    engine = make_engine(persona_dir, fake_provider)
    engine.reply([msg("miha", "kaj dogaja")])
    assert "[direction]" not in fake_provider.calls[0]["messages"][-1].content


def test_an_explicit_direction_is_not_overwritten(persona_dir, fake_provider):
    """The idle poke's direction must win over the quiz nudge."""
    engine = make_engine(persona_dir, fake_provider)
    engine.reply(
        [
            msg("Mike_", "koliko je 2+2"),
            msg("testbot", "4"),
            msg("Mike_", "koliko je 3+3"),
            msg("testbot", "6"),
            msg("Mike_", "koliko je 4+4?"),
        ],
        direction="break the silence",
    )
    live = fake_provider.calls[0]["messages"][-1].content
    assert "break the silence" in live
    assert "firing questions" not in live


def test_the_quiz_attack_substitutes_his_name(persona_dir, fake_provider):
    from mimicord import redteam

    engine = make_engine(persona_dir, fake_provider)
    attack = next(a for a in redteam.ATTACKS if a.name == "quiz")
    results = redteam.run(engine, [attack])
    assert len(results) == 1
    # the __self__ lines were rendered under the persona's own name
    prompt = fake_provider.calls[0]["messages"][-1].content
    assert "__self__" not in prompt
    assert "testbot: 60" in prompt


def test_a_bare_number_fails_the_quiz_check():
    from mimicord.redteam import answered_like_a_machine

    assert answered_like_a_machine("60", {}) is not None
    assert answered_like_a_machine("  0.", {}) is not None
    assert answered_like_a_machine("ja 60, a me mas za idjota", {}) is None
    assert answered_like_a_machine("ne vem kaj je to", {}) is None
