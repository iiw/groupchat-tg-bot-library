# groupchat-tg-bot

A small Python library for building Telegram AI bots that live in group chats and DMs, on top of any OpenAI-compatible LLM API (Mistral, OpenAI, OpenRouter, local servers). You bring the bot token and the API key; the library brings the chat plumbing. Everything else you add as **modules**.

Extracted from [Le Chonk](https://github.com/iiw/le-chonk-bot-tg), minus its payments and sponsorship.

```python
from groupchat import Bot

Bot("TELEGRAM_TOKEN", "LLM_API_KEY", model="mistral-large-latest", base_url="https://api.mistral.ai/v1").run()
```

That's a working bot: in groups it answers when @mentioned, replied to, or called by a wake word; in DMs it answers everything; it keeps the recent conversation in mind and reads pictures.

## Install

```bash
pip install -e "path/to/groupchat-tg-bot[vision]"
```

Python 3.10+. Dependencies: `python-telegram-bot`, `openai`; `Pillow` is optional (extra `vision`, downscales images before sending them).

In @BotFather, turn privacy mode off (`/setprivacy` → Disable) or make the bot a group admin — otherwise it sees only mentions and commands in groups.

## Modules

A module is an instance of a `Module` subclass. You create it in code and pass it to `bot.add()`. Override only what you need:

```python
from groupchat import Bot, Module, command, tool

class Weather(Module):
    prompt = "You can look up the weather with the weather tool."   # → system prompt

    def __init__(self, api):
        self.api = api

    @command("weather", "Weather in a city: /weather <city>")           # → Telegram command
    async def weather_cmd(self, ctx):
        await ctx.reply(await self.api.get(ctx.args))

    @tool("Current weather in a city")                                   # → function the LLM can call
    async def weather(self, city: str, days: int = 1) -> str:
        return await self.api.get(city, days)

    async def context(self, ctx):                                         # → fresh data for each request
        return f"Weather alerts: {await self.api.alerts()}"

bot = Bot(TOKEN, API_KEY, model="mistral-large-latest", base_url="https://api.mistral.ai/v1")
bot.add(Weather(MyWeatherApi()))
bot.run()
```

| Hook | When | Use it for |
|---|---|---|
| `prompt` (attribute) | every request, in the system prompt | persona additions, rules, static knowledge |
| `async context(ctx) -> str \| None` | every request, after the history | data sources: anything fresh (DB rows, API data, state) |
| `@tool(description)` | when the model decides to call it | data the model fetches on demand, actions |
| `@command(name, description, chats=, owner_only=)` | `/name` | commands for people |
| `async on_message(ctx) -> bool \| None` | every non-command message, first | logging, moderation, keyword triggers; return `True` to swallow the message |
| `async allow(ctx) -> bool` | before every AI answer | access control, rate limits, quotas |
| `async setup()` | once, after the bot connects | init, background tasks, raw handlers via `self.bot.app` |

Every module gets `self.bot` when added. Modules run in the order they were added; a command or tool with an existing name replaces the earlier one (so built-ins can be overridden).

**Tools.** The JSON schema comes from the signature: `str`, `int`, `float`, `bool`, `list[...]`, `Literal[...]`, `X | None`; parameters with defaults are optional. A parameter named `ctx` receives the `Context` and is hidden from the model. Return anything — non-strings are JSON-encoded. An exception goes back to the model as an error text instead of breaking the answer. At most 5 tool rounds per answer.

**Commands.** `chats="all" | "private" | "group"`; `owner_only=True` makes a command visible and usable only for `owner_id`. Telegram command menus are set per scope automatically. Unknown or forbidden commands get a short note in DMs and silence in groups.

**Quick one-offs** without a class:

```python
@bot.command("ping", "Check the bot")
async def ping(ctx):
    await ctx.reply("pong")

@bot.tool("Adds two numbers")
async def add(a: int, b: int) -> int:
    return a + b
```

More in [examples/](examples/).

## Context

Passed to every hook, command and (if asked for) tool.

| | |
|---|---|
| `ctx.text`, `ctx.args` | message text (placeholders like `[sticker]` for non-text); command arguments |
| `ctx.message`, `ctx.chat`, `ctx.user`, `ctx.update` | python-telegram-bot objects |
| `ctx.chat_id`, `ctx.user_id`, `ctx.is_private`, `ctx.is_owner`, `ctx.addressed` | shortcuts |
| `ctx.history` | this chat's `ChatHistory` (`list(ctx.history)`, `.clear()`) |
| `await ctx.reply(text, markdown=True)` | reply; Markdown → Telegram HTML, long texts split, plain-text fallback |
| `await ctx.answer(instruction=None)` | make the AI answer now, optionally with a one-off instruction |
| `ctx.bot` | the `Bot` |

## Bot

```python
Bot(
    token, api_key, *,
    model,                     # model id
    base_url=None,             # None = OpenAI; e.g. https://api.mistral.ai/v1, https://openrouter.ai/api/v1
    vision_model=None,         # used for requests with images
    system_prompt=...,         # persona; "{name}" → the bot's name
    name=None,                 # default: the bot's Telegram first name
    owner_id=None,             # your Telegram id: unlocks owner-only commands
    wake_words=(),             # address the bot in groups without @mention (word-prefix, case-insensitive)
    context_size=25,           # history window N: grows to 2N, trims to N
    max_images=4,              # newest images sent per request (0 = no vision)
    max_tokens=2048, temperature=0.4,
    timezone=None,             # for the current-time note, e.g. "Europe/Berlin"
    data_dir=None,             # persist settings & module data as JSON here (None = memory only)
    builtins=True,             # add Help, NewSession, Admin
    llm_params=None,           # extra args for chat.completions.create
)
```

Methods: `add(*modules)`, `run(**polling_kwargs)`, `command(...)`, `tool(...)`, `storage(name, private=False)` → a persistent JSON key-value `Store`, `history(chat_id)`, `await complete(prompt, system=None)` → a one-off LLM call, `app` → the underlying `telegram.ext.Application` (add your own handlers with `group=-1` so they run first).

## Built-in modules

`groupchat.modules`:

- **Help** — `/start`, `/help` (lists the commands this user can run here), `/id`. `Help(intro="...")` for your own intro.
- **NewSession** — `/new` in DMs: forget the conversation so far.
- **Admin** (owner only) — `/status`, `/reset`, `/prompt <text> | clear`, `/model <id> | clear`, `/context <5–100> | clear`; per chat, persisted in `data_dir`.
- **Allowlist** (opt-in) — `Allowlist(users=[ids], usernames=["@nick"], message="...")`: only the owner and listed users get AI answers; `/allow`, `/deny`, `/allowed` for the owner; @username invites bind to the id on first contact.

Help, NewSession and Admin are added by default; `Bot(builtins=False)` gives a clean slate.

## How it works

- **Addressing.** DMs: always. Groups: a reply to the bot, an @mention, or a wake word. Everything else is quietly stored as context.
- **History.** Per chat, in RAM only — message texts never touch the disk. The window grows from N to 2N and trims back to N, so the provider's prefix cache misses once per cycle, not once per message. History from before the bot joined is invisible (a Telegram limitation); restarts begin empty.
- **Request layout.** System prompt (bot prompt + module `prompt`s + a "current chat" block) → history as `Name: text` lines → a trailing system note with the current time and module `context()` data. Fast-changing data goes last so the prefix stays cacheable.
- **Replies.** Addressing the bot in a reply to someone else's message quotes that message into the request. Non-text messages become placeholders (`[sticker]`, `[voice message]`, `[file: x.pdf]`, `[photo]`).
- **Images.** Downloaded (≤ 8 MB), downscaled to ≤ 1024 px JPEG (with Pillow), sent as `image_url` data URIs; only the newest `max_images` are kept in memory. If a provider rejects the image format, the request is retried once with the alternative shape.
- **Delivery.** "typing…" while thinking; one answer at a time per user per chat (a second trigger gets "one moment"); long answers are split at line/word boundaries; LLM errors become a friendly note. A failing module is logged and skipped — it never takes the bot down.

All bot-facing strings (`[photo]`, the busy line, error notes, the default prompt) live in `groupchat.texts` — reassign them to localize.

## Development

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

The tests run the real handlers on synthetic Telegram updates with a fake Telegram bot and a fake LLM client — no network, no keys.

## License

[GNU AGPL-3.0](LICENSE), inherited from Le Chonk, which this library is derived from.
