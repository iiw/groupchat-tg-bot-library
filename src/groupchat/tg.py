"""Telegram helpers: addressing, message content, reply quotes, images."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re

from telegram import Message, MessageEntity, User
from telegram.constants import ChatType

from . import texts

logger = logging.getLogger("groupchat")

MAX_MSG_CHARS = 1500             # very long messages are clipped before entering the history
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_SIDE = 1024


def mentions_bot(text: str, entities, bot_username: str, bot_id: int | None) -> bool:
    """Whether the entities mention the bot (@username or a text mention)."""
    for e in entities or ():
        if e.type == MessageEntity.MENTION:
            if text[e.offset : e.offset + e.length].lower() == f"@{bot_username}".lower():
                return True
        elif e.type == MessageEntity.TEXT_MENTION:
            if e.user is not None and bot_id is not None and e.user.id == bot_id:
                return True
    return False


def is_addressed(msg: Message, bot_username: str, bot_id: int | None, wake_words: list[str]) -> bool:
    """DMs: always. Groups: a reply to the bot, an @mention, or a wake word."""
    if msg.chat.type == ChatType.PRIVATE:
        return True
    reply = msg.reply_to_message
    if reply is not None and reply.from_user is not None and reply.from_user.id == bot_id:
        return True
    text = msg.text or msg.caption or ""
    if mentions_bot(text, msg.entities or msg.caption_entities, bot_username, bot_id):
        return True
    low = text.lower()
    # word-start match tolerates inflections: «Чонк» triggers on «Чонку», «Чонка»
    return any(w and re.search(rf"(?<![@\w]){re.escape(w)}", low) for w in wake_words)


def sender_name(user: User | None) -> str:
    if user is None:
        return texts.UNKNOWN_SENDER
    return user.full_name or (f"@{user.username}" if user.username else str(user.id))


def extract_content(msg: Message) -> tuple[str, str | None]:
    """(text, image file_id) for any message type."""
    if msg.text:
        return msg.text, None
    if msg.photo:
        return msg.caption or "", msg.photo[-1].file_id
    if msg.sticker:
        return texts.STICKER, None
    if msg.voice:
        return texts.VOICE, None
    if msg.video_note:
        return texts.VIDEO_NOTE, None
    if msg.video:
        return texts.VIDEO, None
    if msg.animation:
        return texts.ANIMATION, None
    if msg.audio:
        return texts.AUDIO, None
    if msg.document:
        if (msg.document.mime_type or "").startswith("image/"):
            return msg.caption or "", msg.document.file_id
        return texts.FILE.format(name=msg.document.file_name or "?"), None
    return "", None


def reply_quote(msg: Message, bot_id: int | None) -> str | None:
    """Quote of the message being replied to, unless it is the bot's own (already in history)."""
    r = msg.reply_to_message
    if r is None:
        return None
    author = r.from_user
    if author is not None and author.id == bot_id:
        return None
    text, file_id = extract_content(r)
    text = (text or "").strip()
    if file_id:
        text = f"{text} {texts.PHOTO}".strip()
    if not text:
        return None
    text = text[:MAX_MSG_CHARS]
    if author is None:
        return texts.REPLY_QUOTE_ANONYMOUS.format(text=text)
    return texts.REPLY_QUOTE.format(name=sender_name(author), text=text)


async def load_image(bot, file_id: str) -> str | None:
    """Downloads an image, downscales it to ≤1024 px JPEG; returns a data: URI."""
    try:
        tg_file = await bot.get_file(file_id)
        if tg_file.file_size and tg_file.file_size > MAX_IMAGE_BYTES:
            return None
        raw = bytes(await tg_file.download_as_bytearray())
        if len(raw) > MAX_IMAGE_BYTES:
            return None
        data, mime = await asyncio.to_thread(_shrink, raw)
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    except Exception as e:
        logger.warning("failed to load an image: %s", e)
        return None


def _shrink(raw: bytes) -> tuple[bytes, str]:
    try:
        from PIL import Image
    except ImportError:  # Pillow is optional: send the original (Telegram photos are ≤1280 px anyway)
        return raw, _sniff_mime(raw)
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        im.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=85)
        return out.getvalue(), "image/jpeg"


def _sniff_mime(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG"):
        return "image/png"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"
