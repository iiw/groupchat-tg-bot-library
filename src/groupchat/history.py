"""In-memory chat history: a per-chat window that grows from N to 2N and trims back to N.

The grow→trim cycle keeps the provider's prefix cache warm: the request prefix
stays identical while the window grows, so the cache misses once per cycle
instead of once per message. Nothing here ever touches the disk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Entry:
    """One message in a chat's history."""

    user_id: int | None
    sender: str
    text: str = ""
    image: str | None = None  # data: URI of a downscaled image
    has_image: bool = False   # stays True after `image` is dropped to save memory
    is_bot: bool = False


class ChatHistory:
    """Message window of a single chat."""

    def __init__(self, size: int, max_images: int = 0):
        self.size = size
        self.max_images = max_images
        self.entries: list[Entry] = []
        self.epoch = 0  # bumped by clear(): answers started before a reset are not appended

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(list(self.entries))

    def append(self, entry: Entry) -> None:
        if entry.image:
            entry.has_image = True
        if len(self.entries) >= 2 * self.size:
            self.entries = self.entries[-self.size:]
        self.entries.append(entry)
        if entry.image:
            self._forget_old_images()

    def _forget_old_images(self) -> None:
        """Only the last max_images images are ever sent, so older bytes are dropped."""
        with_images = [e for e in self.entries if e.image]
        for e in with_images[: max(len(with_images) - self.max_images, 0)]:
            e.image = None

    def resize(self, size: int) -> None:
        self.size = size
        if len(self.entries) > 2 * size:
            self.entries = self.entries[-size:]

    def clear(self) -> int:
        n = len(self.entries)
        self.entries = []
        self.epoch += 1
        return n
