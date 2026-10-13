"""OpenAI-compatible chat-completions layer (Mistral, OpenAI, OpenRouter, local servers, ...)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from openai import AsyncOpenAI

from .history import Entry

if TYPE_CHECKING:
    from .module import Tool

logger = logging.getLogger("groupchat")

MAX_TOOL_ROUNDS = 5


class LLMError(Exception):
    """An API call failed."""


def make_client(api_key: str, base_url: str | None = None) -> AsyncOpenAI:
    # a generous timeout leaves room for long answers; otherwise the SDK times out and retries them
    kwargs: dict = {"api_key": api_key, "timeout": 300.0, "max_retries": 2}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def history_to_messages(entries: list[Entry], max_images: int = 0, photo_marker: str = "[photo]") -> list[dict]:
    """Chat history → chat.completions messages.

    Other participants' lines become role=user as "Name: text" (consecutive ones are
    merged); the bot's own messages become role=assistant; the last `max_images`
    images are embedded as image_url parts, older ones collapse into a marker.
    """
    with_image = [i for i, e in enumerate(entries) if e.image]
    keep = set(with_image[-max_images:]) if max_images > 0 else set()

    messages: list[dict] = []
    for i, e in enumerate(entries):
        role = "assistant" if e.is_bot else "user"
        label = "" if e.is_bot else f"{e.sender}: "
        body = (e.text or "").strip()
        if e.has_image and i not in keep:
            body = f"{body} {photo_marker}".strip()

        if i in keep:
            parts = [
                {"type": "text", "text": (label + body).strip() or photo_marker},
                {"type": "image_url", "image_url": {"url": e.image}},
            ]
            messages.append({"role": role, "content": parts})
            continue
        content = (label + body).strip()
        if not content:
            continue
        prev = messages[-1] if messages else None
        if role == "user" and prev and prev["role"] == "user" and isinstance(prev["content"], str):
            prev["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})
    return messages


def has_images(messages: list[dict]) -> bool:
    return any(isinstance(m.get("content"), list) for m in messages)


def _flip_image_style(messages: list[dict]) -> list[dict]:
    """image_url as {"url": ...} <-> a bare string (some providers only accept one of them)."""
    out = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            out.append(m)
            continue
        parts = []
        for p in c:
            if p.get("type") == "image_url":
                iu = p.get("image_url")
                p = {"type": "image_url", "image_url": iu.get("url", "") if isinstance(iu, dict) else {"url": iu}}
            parts.append(p)
        out.append({**m, "content": parts})
    return out


def extract_text(content: Any) -> str:
    """Text of message.content; list-style content keeps only text parts (no "thinking")."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif getattr(p, "type", None) == "text" and isinstance(getattr(p, "text", None), str):
                parts.append(p.text)
        return "\n".join(parts)
    return "" if content is None else str(content)


async def complete(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    *,
    tools: list["Tool"] | None = None,
    run_tool: Callable[["Tool", dict], Awaitable[str]] | None = None,
    usage_out: dict | None = None,
    **params: Any,
) -> str:
    """One answer, running tool calls (up to MAX_TOOL_ROUNDS) along the way.

    On an image-format error, retries once with the other image_url shape.
    `usage_out` accumulates token counters (numbers only, no content).
    """
    messages = list(messages)
    by_name = {t.name: t for t in tools or []}
    flipped = False

    for round_no in range(MAX_TOOL_ROUNDS + 1):
        kwargs = dict(params)
        if by_name and round_no < MAX_TOOL_ROUNDS:
            kwargs["tools"] = [t.schema() for t in by_name.values()]
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as err:
            if has_images(messages) and not flipped:
                flipped = True
                messages = _flip_image_style(messages)
                try:
                    resp = await client.chat.completions.create(model=model, messages=messages, **kwargs)
                except Exception:
                    raise LLMError(str(err)) from err
            else:
                raise LLMError(str(err)) from err

        _add_usage(usage_out, getattr(resp, "usage", None))
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            return extract_text(msg.content).strip()

        messages.append({
            "role": "assistant",
            "content": extract_text(msg.content) or None,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                for c in calls
            ],
        })
        for c in calls:
            result = await _call_tool(by_name.get(c.function.name), c.function.arguments, run_tool)
            messages.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name, "content": result})
    return ""


async def _call_tool(tool: "Tool | None", raw_args: str | None, run_tool) -> str:
    if tool is None:
        return "Error: unknown tool."
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except ValueError as e:
        return f"Error: bad arguments ({e})."
    try:
        result = await run_tool(tool, args) if run_tool else await tool.handler(**args)
    except Exception as e:
        logger.warning("tool %s failed: %s", tool.name, e)
        return f"Error: {type(e).__name__}: {e}"
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def _add_usage(out: dict | None, usage: Any) -> None:
    if out is None or usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    for key, value in (
        ("prompt_tokens", getattr(usage, "prompt_tokens", None)),
        ("cached_tokens", getattr(details, "cached_tokens", None) if details else None),
        ("completion_tokens", getattr(usage, "completion_tokens", None)),
    ):
        if isinstance(value, int):
            out[key] = out.get(key, 0) + value
