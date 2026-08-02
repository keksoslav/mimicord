from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# what discord marks an attachment as, and what claude will actually look at
IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")
# do not pull a 40 MB phone photo off the cdn just to shrink it to a thumbnail
MAX_SOURCE_BYTES = 8 * 1024 * 1024
# anthropic bills an image at roughly this many pixels per token
PIXELS_PER_TOKEN = 750

_warned = False


@dataclass
class Image:
    """A picture small enough to send without thinking about the bill."""

    media_type: str
    data: str  # base64
    width: int
    height: int

    @property
    def tokens(self) -> int:
        return round(self.width * self.height / PIXELS_PER_TOKEN)


def is_image(content_type: str | None) -> bool:
    return bool(content_type) and content_type.split(";")[0].strip() in IMAGE_TYPES


def prepare(raw: bytes, max_edge: int) -> Image | None:
    """Shrink and re-encode so one picture costs a few hundred tokens.

    Cost is pixels, not bytes, so this is the whole cost control: a phone
    photo is 16k tokens untouched and about 400 once it has been through
    here. Returns None when the bytes are not a readable image, or when
    Pillow is missing, because sending them unshrunk would be worse.
    """
    global _warned
    try:
        from PIL import Image as PILImage
    except ImportError:
        if not _warned:
            log.warning(
                "vision is on but Pillow is not installed, skipping images "
                "(uv sync --extra vision)"
            )
            _warned = True
        return None

    try:
        with PILImage.open(io.BytesIO(raw)) as source:
            source.seek(0)  # animated gifs: the first frame is the joke
            frame = source.convert("RGB")
            frame.thumbnail((max_edge, max_edge), PILImage.LANCZOS)
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=80, optimize=True)
            return Image(
                media_type="image/jpeg",
                data=base64.standard_b64encode(buffer.getvalue()).decode(),
                width=frame.width,
                height=frame.height,
            )
    except Exception as error:
        log.debug("could not read an attachment as an image: %s", error)
        return None
