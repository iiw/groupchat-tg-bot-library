"""Modules: the unit of extension.

A module is an instance of a `Module` subclass. Everything it brings is optional:

    class Weather(Module):
        prompt = "You can look up the weather."          # static system-prompt addition

        @command("weather", "Weather in a city")           # a Telegram command
        async def weather_cmd(self, ctx):
            await ctx.reply(await fetch(ctx.args))

        @tool("Current weather in a city")                 # a function the LLM may call
        async def get_weather(self, city: str) -> str:
            return await fetch(city)

        async def context(self, ctx):                       # fresh data for every request
            return "It is raining in Paris."

        async def on_message(self, ctx):                    # sees every message first
            ...

        async def allow(self, ctx):                         # gate before the LLM is called
            return True

        async def setup(self):                              # once, after the bot connects
            ...
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from .bot import Bot
    from .context import Context

CHATS = ("all", "private", "group")


@dataclass
class Command:
    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    chats: str = "all"          # "all" | "private" | "group"
    owner_only: bool = False


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    parameters: dict
    wants_ctx: bool = False

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def command(
    name: str | Callable | None = None,
    description: str = "",
    *,
    chats: str = "all",
    owner_only: bool = False,
):
    """Marks a method (or function) as a Telegram command handler: `async def f(ctx)`.

    Usable bare (`@command`) or with arguments (`@command("ping", "Check the bot")`).
    `chats` limits where the command works: "all", "private" or "group";
    `owner_only` hides it from everyone but the bot owner.
    """
    if chats not in CHATS:
        raise ValueError(f"chats must be one of {CHATS}")

    def mark(fn: Callable) -> Callable:
        cmd_name = (name if isinstance(name, str) else None) or fn.__name__
        if not re.fullmatch(r"[a-z0-9_]{1,32}", cmd_name):
            raise ValueError(f"invalid command name {cmd_name!r}: use 1–32 chars of a-z, 0-9, _")
        desc = description or _first_line(fn.__doc__) or cmd_name
        fn.__groupchat_command__ = Command(cmd_name, desc, fn, chats, owner_only)
        return fn

    if callable(name):
        return mark(name)
    return mark


def tool(description: str | Callable | None = None, *, name: str | None = None):
    """Marks a method (or function) as a tool the LLM can call.

    The JSON schema is derived from the signature: str / int / float / bool / list
    annotations; parameters with defaults are optional. A parameter named `ctx`
    receives the current `Context` and is hidden from the model.
    The return value is passed back to the model as text.
    """

    def mark(fn: Callable) -> Callable:
        desc = (description if isinstance(description, str) else None) or _first_line(fn.__doc__)
        if not desc:
            raise ValueError(f"tool {fn.__name__!r} needs a description (argument or docstring)")
        params, wants_ctx = _schema_from_signature(fn)
        fn.__groupchat_tool__ = Tool(name or fn.__name__, desc, fn, params, wants_ctx)
        return fn

    if callable(description):
        return mark(description)
    return mark


class Module:
    """Base class for bot modules. Override only what you need."""

    #: Static text appended to the system prompt. Keep it stable: it is part of the cached prefix.
    prompt: str | None = None

    #: Set by `Bot.add()`.
    bot: "Bot"

    @property
    def name(self) -> str:
        return type(self).__name__

    async def setup(self) -> None:
        """Called once after the bot has connected to Telegram (`self.bot.app` is ready)."""

    async def on_message(self, ctx: "Context") -> bool | None:
        """Called for every non-command message before it enters the history.

        Return True to consume the message: it is not stored and not answered.
        """
        return None

    async def allow(self, ctx: "Context") -> bool:
        """Called when the bot is addressed. Return False to skip the AI answer."""
        return True

    async def context(self, ctx: "Context") -> str | None:
        """Fresh data for this request (a data source). Goes after the chat history."""
        return None

    # ---- collected by the bot ----

    def commands(self) -> list[Command]:
        return [_bind(key, c, self) for key, c in _collect(self, "__groupchat_command__")]

    def tools(self) -> list[Tool]:
        return [_bind(key, t, self) for key, t in _collect(self, "__groupchat_tool__")]


def _collect(obj: object, attr: str) -> list[tuple[str, Any]]:
    found = []
    for klass in reversed(type(obj).__mro__):
        for key, value in vars(klass).items():
            spec = getattr(value, attr, None)
            if spec is not None:
                found = [s for s in found if s[0] != key] + [(key, spec)]
    return found


def _bind(key: str, spec, instance):
    bound = getattr(instance, key)
    return type(spec)(**{**spec.__dict__, "handler": bound})


def _first_line(doc: str | None) -> str:
    return (doc or "").strip().split("\n")[0].strip()


_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(annotation) -> dict:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _json_type(args[0]) if len(args) == 1 else {}
    if origin is typing.Literal:
        values = list(typing.get_args(annotation))
        return {"type": _JSON_TYPES.get(type(values[0]), "string"), "enum": values}
    if annotation in (list, tuple) or origin in (list, tuple):
        args = typing.get_args(annotation)
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    return {"type": "string"}


def _schema_from_signature(fn: Callable) -> tuple[dict, bool]:
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    props: dict[str, dict] = {}
    required: list[str] = []
    wants_ctx = False
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        if pname == "ctx":
            wants_ctx = True
            continue
        props[pname] = _json_type(hints.get(pname, str))
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": props, "required": required}, wants_ctx
