"""Fakes for Telegram and the LLM API: the real handlers run on synthetic updates."""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import pytest
from telegram import Update

from groupchat import Bot

BOT_ID = 999
BOT_USERNAME = "test_chat_bot"
OWNER_ID = 1


class FakeTelegram:
    """Records everything the bot sends."""

    def __init__(self):
        self.sent: list[dict] = []
        self.actions: list[tuple] = []
        self.files: dict[str, bytes] = {}

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append({"chat_id": chat_id, "text": text, **kwargs})
        return SimpleNamespace(message_id=len(self.sent))

    async def send_chat_action(self, chat_id, action, **kwargs):
        self.actions.append((chat_id, action))

    async def get_file(self, file_id, **kwargs):
        data = self.files[file_id]

        async def download_as_bytearray():
            return bytearray(data)

        return SimpleNamespace(file_size=len(data), download_as_bytearray=download_as_bytearray)

    @property
    def texts(self) -> list[str]:
        return [m["text"] for m in self.sent]


class FakeLLM:
    """Stands in for AsyncOpenAI: replies come from a queue of strings / tool-call dicts."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.fail_on_images_once = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs, default=str)))
        if self.fail_on_images_once and any(isinstance(m.get("content"), list) for m in kwargs["messages"]):
            self.fail_on_images_once = False
            raise RuntimeError("bad image format")
        reply = self.replies.pop(0) if self.replies else "ok"
        if isinstance(reply, Exception):
            raise reply
        tool_calls = None
        content = reply
        if isinstance(reply, dict):  # {"tool": name, "args": {...}}
            content = None
            tool_calls = [SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name=reply["tool"], arguments=json.dumps(reply.get("args", {}))),
            )]
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
            usage=usage,
        )


_ids = itertools.count(1)


def make_update(tg: FakeTelegram, *, text=None, chat="group", user_id=2, first_name="Alice",
                username=None, reply_to=None, photo=None, caption=None, entities=None, **extra) -> Update:
    chat_obj = (
        {"id": user_id, "type": "private", "first_name": first_name}
        if chat == "private"
        else {"id": -100, "type": "supergroup", "title": "Test group"}
    )
    msg = {
        "message_id": next(_ids),
        "date": 0,
        "chat": chat_obj,
        "from": {"id": user_id, "is_bot": False, "first_name": first_name,
                 **({"username": username} if username else {})},
        **extra,
    }
    if text is not None:
        msg["text"] = text
        if text.startswith("/"):
            msg["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
    if entities:
        msg["entities"] = entities
    if photo:
        msg["photo"] = [{"file_id": photo, "file_unique_id": photo, "width": 10, "height": 10}]
    if caption:
        msg["caption"] = caption
    if reply_to:
        msg["reply_to_message"] = {"message_id": next(_ids), "date": 0, "chat": chat_obj, **reply_to}
    return Update.de_json({"update_id": next(_ids), "message": msg}, tg)


@pytest.fixture
def tg():
    return FakeTelegram()


@pytest.fixture
def make_bot():
    def factory(*replies, **kwargs) -> Bot:
        kwargs.setdefault("model", "test-model")
        kwargs.setdefault("owner_id", OWNER_ID)
        bot = Bot("123:token", "llm-key", **kwargs)
        bot.llm = FakeLLM(*replies)
        bot.bot_id, bot.username, bot.name = BOT_ID, BOT_USERNAME, "Testy"
        return bot

    return factory
