"""Small JSON-backed key-value stores for settings and module data."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("groupchat")


class Store:
    """A dict that persists itself to `<data_dir>/<name>.json` on every change.

    Without a data_dir it lives in memory only. Values must be JSON-serializable.
    """

    def __init__(self, path: Path | None = None, *, private: bool = False):
        self.path = path
        self.private = private  # chmod 600 — for anything sensitive
        self.data: dict[str, Any] = {}
        if path is not None and path.exists():
            try:
                self.data = json.loads(path.read_text("utf-8"))
            except Exception as e:
                logger.warning("%s is unreadable, starting empty: %s", path, e)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Sets a value; None removes the key."""
        if value is None:
            self.data.pop(key, None)
        else:
            self.data[key] = value
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if self.private else 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.data, ensure_ascii=False, indent=2))
        if self.private:
            os.chmod(tmp, 0o600)  # an existing tmp file keeps its old mode
        tmp.replace(self.path)


class ChatSettings:
    """Per-chat overrides (system_prompt, model, context_size) on top of the bot defaults."""

    def __init__(self, store: Store):
        self.store = store

    def get(self, chat_id: int, key: str, default: Any = None) -> Any:
        return self.store.get(str(chat_id), {}).get(key, default)

    def set(self, chat_id: int, key: str, value: Any) -> None:
        chat = dict(self.store.get(str(chat_id), {}))
        if value is None:
            chat.pop(key, None)
        else:
            chat[key] = value
        self.store.set(str(chat_id), chat or None)
