from __future__ import annotations

import asyncio
import logging

from mimicord.llm.base import ChatMessage, ProviderError

log = logging.getLogger(__name__)


class ClaudeCodeProvider:
    """Claude through the Agent SDK, billed to a claude.ai subscription.

    Uses the machine's Claude Code login, so replies draw from the plan's own
    usage limits rather than from API credits. Anthropic announced a separate
    monthly Agent SDK credit and then paused it, so for now this shares the
    same session and weekly allowance as everything else on the plan. Hitting
    the limit stops the bot rather than billing you, unless usage credits were
    explicitly enabled on the account.
    """

    def __init__(self, model: str) -> None:
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as error:
            raise ProviderError("claude-agent-sdk is not installed, run uv sync") from error
        self._model = model

    def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float | None = None,
    ) -> str:
        # the agent sdk has no max_tokens/temperature surface; the persona
        # prompt and the post processor keep replies short instead
        prompt = self._flatten(messages)
        try:
            return asyncio.run(self._query(system, prompt))
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(f"agent sdk call failed: {error}") from error

    @staticmethod
    def _flatten(messages: list[ChatMessage]) -> str:
        """The sdk takes one prompt string, so the message list (few-shots,
        repair retries) becomes an inline conversation."""
        parts = []
        for message in messages:
            if message.role == "assistant":
                parts.append(f"you replied:\n{message.content}")
            else:
                parts.append(message.content)
        return "\n\n".join(parts)

    async def _query(self, system: str, prompt: str) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        options = ClaudeAgentOptions(
            system_prompt=system,
            model=self._model,
            # one turn is all a chat reply needs, but the sdk treats a turn
            # that did not finish cleanly as a hard error, so leave headroom
            max_turns=2,
            allowed_tools=[],  # chat only, the bot must never grow hands
            setting_sources=[],  # do not load this machine's CLAUDE.md files
        )
        parts: list[str] = []
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    cost = getattr(message, "total_cost_usd", None)
                    if cost is not None:
                        log.debug("agent sdk turn cost: $%.4f", cost)
        except Exception as error:
            # the sdk raises on an error result, which throws away whatever it
            # streamed first. if he already said his piece, send that
            if not parts:
                raise
            log.warning("agent sdk ended badly (%s), keeping what it said", error)
        text = "".join(parts).strip()
        if not text:
            raise ProviderError(
                "empty response from the agent sdk, is Claude Code logged in on this machine?"
            )
        return text
