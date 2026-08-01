from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from mimicord.analyze.chunker import Chunk
from mimicord.llm.base import Provider, json_call

log = logging.getLogger(__name__)

MAP_SYSTEM = """\
You are analyzing how {name} writes in Discord logs. You get one chat \
transcript; study only the messages by {name} (context from others is there \
so you understand what they were reacting to).

Return only a JSON object with exactly these keys:
{{
  "tone": ["short descriptors"],
  "formality": "one phrase",
  "languages": [{{"lang": "iso code", "share": 0.0}}],
  "code_switching": "when and why they switch languages, or null",
  "verbal_tics": ["recurring words or fillers, exact spelling"],
  "typical_openers": ["how their messages tend to start"],
  "emoji_usage": "one phrase",
  "humor_style": "one phrase",
  "topics": ["what they talk about here"],
  "opinions": ["stated opinions, strongest first"],
  "relationships": [{{"person": "name", "dynamic": "one phrase"}}],
  "notable_quotes": ["verbatim short messages that are peak {name}, max 8"],
  "reply_style": "length, punctuation, pacing, one phrase"
}}

Base everything on what is actually in the transcript. Empty lists are fine. \
No commentary outside the JSON."""

MAP_MAX_TOKENS = 1500


def analyze_chunks(
    chunks: list[Chunk],
    provider: Provider,
    name: str,
    chunks_dir: Path,
    *,
    resume: bool = True,
    progress: Callable[[Chunk, bool], None] | None = None,
) -> list[dict]:
    """One style-analysis call per chunk, cached to disk so reruns are free."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    system = MAP_SYSTEM.format(name=name)
    results: list[dict] = []
    for chunk in chunks:
        out_path = chunks_dir / f"{chunk.index:03d}.json"
        if resume and out_path.is_file():
            results.append(json.loads(out_path.read_text(encoding="utf-8")))
            if progress:
                progress(chunk, True)
            continue
        data = json_call(
            provider, system=system, user=chunk.text, max_tokens=MAP_MAX_TOKENS
        )
        data["_chunk"] = {
            "index": chunk.index,
            "channel": chunk.channel_name or chunk.channel_id,
            "span": [chunk.start_ts, chunk.end_ts],
            "messages": chunk.message_count,
            "target_messages": chunk.target_count,
        }
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(data)
        if progress:
            progress(chunk, False)
    return results
