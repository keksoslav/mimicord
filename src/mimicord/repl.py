from __future__ import annotations

from mimicord.engine import ContextMessage, PersonaEngine
from mimicord.llm.base import ChatMessage


def _dump_prompt(system: str, messages: list[ChatMessage]) -> None:
    print("\n--- system ---")
    print(system)
    for message in messages:
        marker = " [cache boundary]" if message.cache_boundary else ""
        print(f"--- {message.role}{marker} ---")
        print(message.content)
    print("--- end prompt ---\n")


def run(name: str, *, rag: bool = True, show_prompt: bool = False) -> None:
    engine = PersonaEngine(name, rag_enabled=rag)
    persona = engine.config.name
    llm = engine.config.llm
    memory = "with memories" if engine.rag is not None else "no memories"
    print(f"chatting with {persona} ({llm.provider}/{llm.model}, {memory})")
    print("/reset clears context, /quit exits\n")

    context: list[ContextMessage] = []
    max_context = engine.config.discord.context_messages
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/quit":
            return
        if line == "/reset":
            context.clear()
            print("(context cleared)")
            continue
        context.append(ContextMessage("you", line))
        try:
            bursts = engine.reply(context[-max_context:])
        except Exception as error:  # keep the loop alive on provider hiccups
            print(f"(error: {error})")
            continue
        if show_prompt and engine.last_prompt:
            _dump_prompt(*engine.last_prompt)
        for burst in bursts:
            print(f"{persona}> {burst}")
            context.append(ContextMessage(persona, burst))
