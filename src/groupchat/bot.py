"""Bot: wires Telegram, the chat history, the LLM and the modules together."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from telegram import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeChat,
    BotCommandScopeDefault,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import Application, ApplicationBuilder, ContextTypes, MessageHandler, filters

from . import texts
from .context import Context
from .history import ChatHistory, Entry
from .llm import LLMError, complete, has_images, history_to_messages, make_client
from .module import Command, Module, Tool, command as command_decorator, tool as tool_decorator
from .storage import ChatSettings, Store
from .tg import MAX_MSG_CHARS, extract_content, is_addressed, load_image, reply_quote, sender_name

logger = logging.getLogger("groupchat")


class Bot:
    """A Telegram group-chat AI bot. Extend it with modules:

        bot = Bot("TELEGRAM_TOKEN", "LLM_API_KEY", model="mistral-large-latest",
                  base_url="https://api.mistral.ai/v1")
        bot.add(MyModule())
        bot.run()
    """

    def __init__(
        self,
        token: str,
        api_key: str,
        *,
        model: str,
        base_url: str | None = None,
        vision_model: str | None = None,
        system_prompt: str | None = None,
        name: str | None = None,
        owner_id: int | None = None,
        wake_words: Iterable[str] = (),
        context_size: int = 25,
        max_images: int = 4,
        max_tokens: int = 2048,
        temperature: float = 0.4,
        timezone: str | tzinfo | None = None,
        data_dir: str | Path | None = None,
        builtins: bool = True,
        llm_params: dict[str, Any] | None = None,
    ):
        """
        token          — bot token from @BotFather.
        api_key        — key for the OpenAI-compatible LLM API.
        model          — model id; `vision_model` (optional) is used for requests with images.
        base_url       — API base URL (None = OpenAI), e.g. https://api.mistral.ai/v1.
        system_prompt  — the bot's persona; "{name}" is replaced with the bot's name
                         (default: groupchat.texts.DEFAULT_SYSTEM_PROMPT).
        name           — the bot's name (default: its Telegram first name).
        owner_id       — your Telegram user id: unlocks owner-only commands.
        wake_words     — words that address the bot in groups without an @mention.
        context_size   — N: each chat's history grows from N to 2N messages, then trims to N.
        max_images     — newest images sent to the model per request (0 = no vision).
        timezone       — for the current-time note, e.g. "Europe/Berlin" (default: server's).
        data_dir       — where settings and module data persist (None = memory only).
        builtins       — add the built-in Help, NewSession and Admin modules.
        llm_params     — extra arguments for chat.completions.create.
        """
        self.token = token
        self.model = model
        self.vision_model = vision_model
        self.system_prompt = system_prompt
        self.name = name
        self.owner_id = owner_id
        self.wake_words = [w.strip().lower() for w in wake_words if w.strip()]
        self.context_size = max(1, context_size)
        self.max_images = max(0, max_images)
        self.timezone = ZoneInfo(timezone) if isinstance(timezone, str) else timezone
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.llm_params = {"max_tokens": max_tokens, "temperature": temperature, **(llm_params or {})}
        self.llm = make_client(api_key, base_url)

        self.modules: list[Module] = []
        self._stores: dict[str, Store] = {}
        self.settings = ChatSettings(self.storage("chat_settings"))
        self.bot_id: int | None = None
        self.username: str = ""
        self._commands: dict[str, Command] = {}
        self._tools: dict[str, Tool] = {}
        self._histories: dict[int, ChatHistory] = {}
        self._locks: dict[tuple[int, int], asyncio.Lock] = {}
        self._app: Application | None = None

        if builtins:
            from .modules import Admin, Help, NewSession

            self.add(Help(), NewSession(), Admin())

    # ---------- extension ----------

    def add(self, *modules: Module) -> "Bot":
        """Adds modules. A command or tool with an existing name replaces the old one."""
        for m in modules:
            m.bot = self
            self.modules.append(m)
            for c in m.commands():
                self._register_command(c)
            for t in m.tools():
                self._tools[t.name] = t
        return self

    def command(self, name: str | Callable | None = None, description: str = "", **kwargs):
        """Registers a plain function as a command: `@bot.command("ping")`."""

        def register(fn):
            fn = command_decorator(name if isinstance(name, str) else None, description, **kwargs)(fn)
            self._register_command(fn.__groupchat_command__)
            return fn

        return register(name) if callable(name) else register

    def tool(self, description: str | Callable | None = None, **kwargs):
        """Registers a plain function as an LLM tool: `@bot.tool("What it does")`."""

        def register(fn):
            fn = tool_decorator(description if isinstance(description, str) else None, **kwargs)(fn)
            spec = fn.__groupchat_tool__
            self._tools[spec.name] = spec
            return fn

        return register(description) if callable(description) else register

    def _register_command(self, c: Command) -> None:
        if c.name in self._commands:
            logger.info("command /%s is overridden", c.name)
        self._commands[c.name] = c

    @property
    def commands(self) -> list[Command]:
        return list(self._commands.values())

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def storage(self, name: str, *, private: bool = False) -> Store:
        """A persistent key-value store `<data_dir>/<name>.json` (in memory without data_dir).

        The same name always returns the same Store object.
        """
        if name not in self._stores:
            path = self.data_dir / f"{name}.json" if self.data_dir else None
            self._stores[name] = Store(path, private=private)
        return self._stores[name]

    # ---------- per-chat state ----------

    def history(self, chat_id: int) -> ChatHistory:
        h = self._histories.get(chat_id)
        if h is None:
            h = self._histories[chat_id] = ChatHistory(self.context_size_for(chat_id), self.max_images)
        return h

    def context_size_for(self, chat_id: int) -> int:
        return int(self.settings.get(chat_id, "context_size") or self.context_size)

    def model_for(self, chat_id: int, with_images: bool = False) -> str:
        if with_images and self.vision_model:
            return self.vision_model
        return self.settings.get(chat_id, "model") or self.model

    def system_prompt_for(self, chat_id: int) -> str:
        return self.settings.get(chat_id, "system_prompt") or self.system_prompt or texts.DEFAULT_SYSTEM_PROMPT

    def can_use(self, cmd: Command, ctx: Context) -> bool:
        if cmd.owner_only and not ctx.is_owner:
            return False
        return cmd.chats == "all" or (cmd.chats == "private") == ctx.is_private

    # ---------- the AI answer ----------

    def build_messages(self, ctx: Context, notes: list[str]) -> tuple[str, list[dict]]:
        """(model, messages): system prompt + history + a trailing note with fast-changing data.

        Everything that changes per request (time, module data) goes to the END,
        so the prefix — system prompt and history — stays cacheable.
        """
        chat_id = ctx.chat_id
        convo = history_to_messages(list(self.history(chat_id)), self.max_images, texts.PHOTO)
        model = self.model_for(chat_id, has_images(convo))
        size = self.context_size_for(chat_id)
        parts = [self.system_prompt_for(chat_id).replace("{name}", self.name or "Assistant")]
        parts += [m.prompt.strip() for m in self.modules if m.prompt]
        parts.append(texts.CHAT_BLOCK.format(
            kind=texts.CHAT_KIND_PRIVATE if ctx.is_private else texts.CHAT_KIND_GROUP,
            title=ctx.chat.title or ctx.chat.full_name or "—",
            model=model,
            size=size,
            size2=2 * size,
            images=texts.IMAGES_ON.format(n=self.max_images) if self.max_images else texts.IMAGES_OFF,
        ))
        tail = "\n\n".join([texts.TIME_NOTE.format(now=datetime.now(self.timezone)), *notes])
        messages = [{"role": "system", "content": "\n\n".join(parts)}, *convo, {"role": "system", "content": tail}]
        return model, messages

    async def respond(self, ctx: Context, instruction: str | None = None) -> str | None:
        """Generates an AI answer for this chat and replies with it (see `Context.answer`)."""
        for m in self.modules:
            if not await self._safe(m.allow(ctx), m, "allow", default=False):
                return None
        hist = self.history(ctx.chat_id)
        epoch = hist.epoch
        notes = await asyncio.gather(*(self._safe(m.context(ctx), m, "context") for m in self.modules))
        notes = [n.strip() for n in notes if isinstance(n, str) and n.strip()]
        if instruction:
            notes.append(instruction)
        model, messages = self.build_messages(ctx, notes)

        async def run_tool(t: Tool, args: dict) -> Any:
            return await (t.handler(ctx=ctx, **args) if t.wants_ctx else t.handler(**args))

        usage: dict = {}
        stop = asyncio.Event()
        typing = asyncio.create_task(self._typing(ctx, stop))
        try:
            answer = await complete(
                self.llm, model, messages,
                tools=self.tools, run_tool=run_tool, usage_out=usage, **self.llm_params,
            )
        except LLMError as e:
            logger.error("LLM error (chat %s, model %s): %s", ctx.chat_id, model, e)
            await ctx.reply(texts.LLM_ERROR, markdown=False)
            return None
        finally:
            stop.set()
            await typing
        if usage:
            logger.info("usage chat=%s model=%s %s", ctx.chat_id, model, usage)

        if not answer:
            await ctx.reply(texts.EMPTY_ANSWER, markdown=False)
            return None
        if not await ctx.reply(answer):
            return None
        # the bot sees its own answers; skipped if the chat was reset while it was thinking
        if hist.epoch == epoch:
            hist.append(Entry(self.bot_id, self.name or "Assistant", answer, is_bot=True))
        return answer

    async def complete(self, prompt: str | list[dict], *, system: str | None = None, model: str | None = None) -> str:
        """One-off LLM call outside any chat: a prompt (or ready messages) → text."""
        if isinstance(prompt, str):
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        return await complete(self.llm, model or self.model, messages, **self.llm_params)

    @staticmethod
    async def _typing(ctx: Context, stop: asyncio.Event) -> None:
        bot = ctx.message.get_bot()
        while not stop.is_set():
            try:
                await bot.send_chat_action(ctx.chat_id, ChatAction.TYPING)
            except TelegramError:
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.5)
            except asyncio.TimeoutError:
                pass

    @staticmethod
    async def _safe(coro, module: Module, hook: str, default: Any = None) -> Any:
        try:
            return await coro
        except Exception:
            logger.exception("module %s failed in %s()", module.name, hook)
            return default

    # ---------- telegram handlers ----------

    async def handle_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        msg, chat, user = update.effective_message, update.effective_chat, update.effective_user
        if msg is None or chat is None or user is None or user.is_bot:
            return
        text, image_id = extract_content(msg)
        if not text and not image_id:
            return

        ctx = Context(self, update, text=text)
        ctx.addressed = is_addressed(msg, self.username, self.bot_id, self.wake_words)
        for m in self.modules:
            if await self._safe(m.on_message(ctx), m, "on_message"):
                return

        image = await load_image(msg.get_bot(), image_id) if image_id and self.max_images else None
        stored = text[:MAX_MSG_CHARS]
        if ctx.addressed and (quote := reply_quote(msg, self.bot_id)):
            # a reply to someone else's message: the quoted text may not be in the history
            stored = f"{quote}\n{stored}".rstrip()
        self.history(chat.id).append(
            Entry(user.id, sender_name(user), stored, image=image, has_image=bool(image_id))
        )
        if not ctx.addressed:
            return

        # one answer at a time per user per chat; different users are served in parallel
        lock = self._locks.setdefault((chat.id, user.id), asyncio.Lock())
        if lock.locked():
            await ctx.reply(texts.BUSY, markdown=False)
            return
        async with lock:
            await self.respond(ctx)

    async def handle_command(self, update: Update, _: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        msg = update.effective_message
        if msg is None or update.effective_chat is None or not msg.text:
            return
        head, *rest = msg.text.split(maxsplit=1)
        args = rest[0] if rest else ""
        name, _at, target = head[1:].partition("@")
        if target and self.username and target.lower() != self.username.lower():
            return  # "/cmd@other_bot"
        ctx = Context(self, update, text=msg.text, args=args.strip())
        cmd = self._commands.get(name.lower())
        if cmd is None or not self.can_use(cmd, ctx):
            if ctx.is_private:  # groups stay silent: no "unknown command" noise
                await ctx.reply(texts.UNKNOWN_COMMAND, markdown=False)
            return
        logger.info("/%s from %s in chat %s", cmd.name, ctx.user_id, ctx.chat_id)
        try:
            await cmd.handler(ctx)
        except Exception:
            logger.exception("command /%s failed", cmd.name)
            await ctx.reply(texts.COMMAND_ERROR, markdown=False)

    async def _post_init(self, app: Application) -> None:
        me = await app.bot.get_me()
        self.bot_id, self.username = me.id, me.username or ""
        self.name = self.name or me.first_name
        await self._set_menus(app)
        if getattr(me, "can_read_all_group_messages", None) is False:
            logger.warning(
                "Privacy mode is ON: in groups the bot sees only mentions and commands. "
                "Disable it in @BotFather → /setprivacy, or make the bot a group admin."
            )
        for m in self.modules:
            await m.setup()
        logger.info("@%s started: model %s, modules: %s", self.username, self.model,
                    ", ".join(m.name for m in self.modules) or "—")

    async def _set_menus(self, app: Application) -> None:
        def menu(*chats: str, owner: bool = False) -> list[BotCommand]:
            return [BotCommand(c.name, c.description[:256]) for c in self._commands.values()
                    if c.chats in chats and (owner or not c.owner_only)]

        private = menu("all", "private")
        try:
            await app.bot.set_my_commands(private, scope=BotCommandScopeDefault())
            await app.bot.set_my_commands(private, scope=BotCommandScopeAllPrivateChats())
            await app.bot.set_my_commands(menu("all", "group"), scope=BotCommandScopeAllGroupChats())
            if self.owner_id:
                await app.bot.set_my_commands(
                    menu("all", "private", owner=True), scope=BotCommandScopeChat(self.owner_id)
                )
        except TelegramError as e:
            logger.warning("failed to set the command menus: %s", e)

    # ---------- running ----------

    @property
    def app(self) -> Application:
        """The underlying python-telegram-bot Application (built on first access).

        Escape hatch for anything the modules don't cover: add your own handlers with
        `group=-1` so they run before the bot's catch-all message handler.
        """
        if self._app is None:
            self._app = (
                ApplicationBuilder().token(self.token).concurrent_updates(16).post_init(self._post_init).build()
            )
            new_messages = filters.UpdateType.MESSAGE
            self._app.add_handler(MessageHandler(new_messages & filters.COMMAND, self.handle_command))
            self._app.add_handler(MessageHandler(new_messages & ~filters.COMMAND, self.handle_message))
        return self._app

    def run(self, **polling_kwargs: Any) -> None:
        """Starts long polling (blocks until stopped)."""
        if not logging.getLogger().handlers:
            logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
            logging.getLogger("httpx").setLevel(logging.WARNING)
        polling_kwargs.setdefault("allowed_updates", Update.ALL_TYPES)
        polling_kwargs.setdefault("drop_pending_updates", True)
        self.app.run_polling(**polling_kwargs)
