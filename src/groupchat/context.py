"""Context: everything a module needs about the current message, plus ways to answer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telegram import Chat, Message, Update, User
from telegram.constants import ChatType
from telegram.error import TelegramError

from .formatting import md_to_html, split_message

if TYPE_CHECKING:
    from .bot import Bot
    from .history import ChatHistory

logger = logging.getLogger("groupchat")


class Context:
    """Passed to every module hook, command and (optionally) tool."""

    def __init__(self, bot: "Bot", update: Update, *, text: str = "", args: str = ""):
        self.bot = bot
        self.update = update
        self.text = text          # message text or caption (placeholders for stickers, files, ...)
        self.args = args          # command arguments: "/cmd a b" → "a b"
        self.addressed = False    # whether the message is addressed to the bot

    # ---- telegram objects ----

    @property
    def message(self) -> Message:
        return self.update.effective_message

    @property
    def chat(self) -> Chat:
        return self.update.effective_chat

    @property
    def user(self) -> User | None:
        return self.update.effective_user

    @property
    def chat_id(self) -> int:
        return self.chat.id

    @property
    def user_id(self) -> int | None:
        return self.user.id if self.user else None

    @property
    def is_private(self) -> bool:
        return self.chat.type == ChatType.PRIVATE

    @property
    def is_owner(self) -> bool:
        return self.bot.owner_id is not None and self.user_id == self.bot.owner_id

    @property
    def history(self) -> "ChatHistory":
        return self.bot.history(self.chat_id)

    # ---- answering ----

    async def reply(self, text: str, *, markdown: bool = True) -> bool:
        """Replies to the current message; Markdown → Telegram HTML, long texts are split.

        Falls back to plain text when Telegram rejects the markup. Returns True if
        at least one chunk was delivered.
        """
        delivered = False
        for i, chunk in enumerate(split_message(text)):
            send = self._reply_chunk if i == 0 else self._send_plain_chunk
            try:
                if markdown:
                    await send(md_to_html(chunk), parse_mode="HTML")
                else:
                    await send(chunk)
                delivered = True
                continue
            except TelegramError as e:
                if not markdown:
                    logger.warning("failed to send a message: %s", e)
                    continue
                logger.info("markup rejected (%s), sending plain text", e)
            try:
                await send(chunk)
                delivered = True
            except TelegramError as e:
                logger.warning("failed to send a message: %s", e)
        return delivered

    async def _reply_chunk(self, text: str, **kwargs) -> Message:
        return await self.message.reply_text(text, allow_sending_without_reply=True, **kwargs)

    async def _send_plain_chunk(self, text: str, **kwargs) -> Message:
        return await self.message.get_bot().send_message(self.chat_id, text, **kwargs)

    async def answer(self, instruction: str | None = None) -> str | None:
        """Generates an AI answer from the chat history and replies with it.

        `instruction` is an extra one-off note for the model, e.g. "Summarize the chat".
        Returns the delivered answer, or None.
        """
        return await self.bot.respond(self, instruction)
