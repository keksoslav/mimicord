from __future__ import annotations

import base64
import io

import pytest

from mimicord import vision
from mimicord.config import ConfigError, VisionConfig, load_config
from mimicord.engine import ContextMessage, collect_images
from mimicord.llm.base import ChatMessage


def png(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 40, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_shrinks_a_big_photo(tmp_path):
    """A phone photo untouched is 16k tokens. This is the whole point."""
    image = vision.prepare(png(4000, 3000), max_edge=768)
    assert image is not None
    assert max(image.width, image.height) == 768
    assert image.tokens < 600
    assert image.media_type == "image/jpeg"
    base64.standard_b64decode(image.data)  # round trips


def test_small_images_are_not_upscaled():
    image = vision.prepare(png(100, 80), max_edge=768)
    assert (image.width, image.height) == (100, 80)


def test_max_edge_drives_the_cost():
    cheap = vision.prepare(png(2000, 2000), max_edge=384)
    dear = vision.prepare(png(2000, 2000), max_edge=1024)
    assert cheap.tokens * 6 < dear.tokens


def test_junk_bytes_are_not_an_image():
    assert vision.prepare(b"this is not a png", max_edge=768) is None


def test_content_type_filter():
    assert vision.is_image("image/png")
    assert vision.is_image("image/jpeg; charset=utf-8")
    assert not vision.is_image("video/mp4")
    assert not vision.is_image(None)


def fake_image(name: str):
    return vision.Image(media_type="image/jpeg", data=name, width=64, height=64)


def context_with_images(*positions: int, length: int = 8):
    messages = []
    for index in range(length):
        images = [fake_image(f"img{index}")] if index in positions else []
        messages.append(ContextMessage("mike", f"m{index}", images))
    return messages


def test_only_the_newest_image_is_sent():
    cfg = VisionConfig(enabled=True, max_images=1, lookback=4)
    picked = collect_images(context_with_images(5, 7), cfg)
    assert [i.data for i in picked] == ["img7"]


def test_old_images_are_not_paid_for_again():
    """The meme from 20 messages ago must not ride along on every reply."""
    cfg = VisionConfig(enabled=True, max_images=2, lookback=4)
    assert collect_images(context_with_images(0, 1), cfg) == []


def test_disabled_sends_nothing():
    cfg = VisionConfig(enabled=False, max_images=4, lookback=8)
    assert collect_images(context_with_images(6, 7), cfg) == []


def test_max_images_caps_it():
    cfg = VisionConfig(enabled=True, max_images=2, lookback=8)
    picked = collect_images(context_with_images(5, 6, 7), cfg)
    assert len(picked) == 2


def test_image_only_message_still_renders():
    assert ContextMessage("mike", "", [fake_image("x")]).render() == (
        "mike: (posted an image)"
    )
    assert ContextMessage("mike", "glej").render() == "mike: glej"


def write(tmp_path, text):
    path = tmp_path / "persona.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_vision_is_off_by_default(tmp_path):
    cfg = load_config(write(tmp_path, 'name = "x"\n')).vision
    assert cfg.enabled is False
    assert cfg.max_edge == 768


def test_absurd_max_edge_rejected(tmp_path):
    text = 'name = "x"\n[vision]\nenabled = true\nmax_edge = 4000\n'
    with pytest.raises(ConfigError, match="max_edge"):
        load_config(write(tmp_path, text))


def test_anthropic_wire_format_carries_the_image(monkeypatch):
    from mimicord.llm.anthropic_provider import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._cache_control = {"type": "ephemeral"}

    wire = provider._to_wire(
        ChatMessage("user", "kaj je to", images=[fake_image("BASE64")])
    )
    assert wire["content"][0] == {"type": "text", "text": "kaj je to"}
    assert wire["content"][1]["type"] == "image"
    assert wire["content"][1]["source"]["data"] == "BASE64"


def test_claude_code_stream_shape():
    """The sdk only accepts pictures through the message dict form."""
    import asyncio

    from mimicord.llm.claude_code import ClaudeCodeProvider

    async def collect():
        return [
            chunk
            async for chunk in ClaudeCodeProvider._stream(
                "kaj je to", [fake_image("BASE64")]
            )
        ]

    (chunk,) = asyncio.run(collect())
    content = chunk["message"]["content"]
    assert chunk["type"] == "user"
    assert content[0]["text"] == "kaj je to"
    assert content[1]["source"]["media_type"] == "image/jpeg"
