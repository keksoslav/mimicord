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
        images = [image for message in messages for image in message.images]
        try:
            return asyncio.run(self._query(system, prompt, images))
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

    @staticmethod
    async def _stream(prompt: str, images: list):
        """The sdk takes either a plain string or a stream of message dicts.
        Pictures only fit in the second form."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        content += [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": image.data,
                },
            }
            for image in images
        ]
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
            "session_id": "mimicord",
        }

    async def _query(self, system: str, prompt: str, images: list | None = None) -> str:
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
            # one turn only. two used to be headroom against the sdk treating
            # an unfinished turn as a hard error, but the salvage below covers
            # that now, and a second turn gives the model somewhere to write
            # its plan out loud, which then ends up in the channel
            max_turns=1,
            allowed_tools=[],  # chat only, the bot must never grow hands
            setting_sources=[],  # do not load this machine's CLAUDE.md files
        )
        source = self._stream(prompt, images) if images else prompt
        # one entry per assistant turn, not one per text block. concatenating
        # every block a run produced is how a plan the model wrote to itself
        # ended up glued to the front of a real reply and posted to a channel
        turns: list[str] = []
        try:
            async for message in query(prompt=source, options=options):
                if isinstance(message, AssistantMessage):
                    said = "".join(
                        block.text
                        for block in message.content
                        if isinstance(block, TextBlock)
                    )
                    if said.strip():
                        turns.append(said)
                elif isinstance(message, ResultMessage):
                    cost = getattr(message, "total_cost_usd", None)
                    if cost is not None:
                        log.debug("agent sdk turn cost: $%.4f", cost)
        except Exception as error:
            # the sdk raises on an error result, which throws away whatever it
            # streamed first. if he already said his piece, send that
            if not turns:
                raise
            log.warning("agent sdk ended badly (%s), keeping what it said", error)
        if len(turns) > 1:
            log.warning(
                "agent sdk returned %d turns, keeping the last and dropping %r",
                len(turns),
                "".join(turns[:-1])[:120],
            )
        text = (turns[-1] if turns else "").strip()
        if not text:
            raise ProviderError(
                "empty response from the agent sdk, is Claude Code logged in on this machine?"
            )
        return text
