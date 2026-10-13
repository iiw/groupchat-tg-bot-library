"""The real Bot handlers on synthetic Telegram updates."""

from __future__ import annotations

import asyncio
import io

from groupchat import Module, command, tool
from groupchat.modules import Allowlist

from .conftest import BOT_ID, BOT_USERNAME, OWNER_ID, make_update


def system_prompt(call) -> str:
    return call["messages"][0]["content"]


def tail_note(call) -> str:
    return call["messages"][-1]["content"]


# ---------- addressing & history ----------

async def test_group_message_not_addressed_is_only_stored(tg, make_bot):
    bot = make_bot()
    await bot.handle_message(make_update(tg, text="just chatting"))
    assert bot.llm.calls == [] and tg.sent == []
    assert [e.text for e in bot.history(-100)] == ["just chatting"]


async def test_mention_triggers_answer_with_context(tg, make_bot):
    bot = make_bot("**Hello**, Alice!", wake_words=["testy"])
    await bot.handle_message(make_update(tg, text="earlier line", user_id=3, first_name="Bob"))
    text = f"@{BOT_USERNAME} hi"
    await bot.handle_message(make_update(tg, text=text, entities=[{"type": "mention", "offset": 0, "length": len(BOT_USERNAME) + 1}]))
    (call,) = bot.llm.calls
    assert call["model"] == "test-model" and call["max_tokens"] == 2048
    assert "You are Testy" in system_prompt(call) and "group chat: «Test group»" in system_prompt(call)
    assert call["messages"][1] == {"role": "user", "content": f"Bob: earlier line\nAlice: {text}"}
    assert tail_note(call).startswith("# Current time")
    assert tg.sent[-1]["text"] == "<b>Hello</b>, Alice!" and tg.sent[-1]["parse_mode"] == "HTML"
    last = list(bot.history(-100))[-1]
    assert last.is_bot and last.text == "**Hello**, Alice!"


async def test_wake_word_reply_and_private(tg, make_bot):
    bot = make_bot("a", "b", "c", wake_words=["Testy"])
    await bot.handle_message(make_update(tg, text="Привет, testyбот"))      # word-prefix match
    await bot.handle_message(make_update(tg, text="not me", reply_to={"text": "x", "from": {"id": BOT_ID, "is_bot": True, "first_name": "T"}}))
    await bot.handle_message(make_update(tg, text="dm", chat="private"))
    await bot.handle_message(make_update(tg, text="@othertesty nope"))    # no match after @
    assert len(bot.llm.calls) == 3


async def test_reply_to_someone_else_is_quoted(tg, make_bot):
    bot = make_bot("ok")
    await bot.handle_message(make_update(
        tg, text="testy what do you think?", chat="private",
        reply_to={"text": "pineapple pizza is great", "from": {"id": 5, "is_bot": False, "first_name": "Carl"}},
    ))
    stored = list(bot.history(2))[0].text
    assert stored.startswith("↪ In reply to Carl: «pineapple pizza is great»")


async def test_non_text_placeholders_and_images(tg, make_bot):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2000, 1000), "red").save(buf, "PNG")
    tg.files["ph1"] = buf.getvalue()
    bot = make_bot("I see red", max_images=1, vision_model="vision")
    await bot.handle_message(make_update(tg, sticker={"file_id": "s", "file_unique_id": "s", "width": 1, "height": 1,
                                                      "is_animated": False, "is_video": False, "type": "regular"}))
    await bot.handle_message(make_update(tg, photo="ph1", caption="testy look", chat="private"))
    assert list(bot.history(-100))[0].text == "[sticker]"
    entry = list(bot.history(2))[0]
    assert entry.image.startswith("data:image/jpeg;base64,") and entry.has_image
    (call,) = bot.llm.calls
    assert call["model"] == "vision"
    assert call["messages"][1]["content"][1]["type"] == "image_url"


async def test_bot_messages_are_ignored(tg, make_bot):
    bot = make_bot()
    upd = make_update(tg, text="hi", chat="private")
    upd.effective_user._unfreeze()
    upd.effective_user.is_bot = True
    await bot.handle_message(upd)
    assert len(bot.history(2)) == 0


async def test_llm_error_reports_and_skips_history(tg, make_bot):
    bot = make_bot(RuntimeError("down"), RuntimeError("down"), RuntimeError("down"))
    await bot.handle_message(make_update(tg, text="hi", chat="private"))
    assert "try again" in tg.texts[-1]
    assert [e.is_bot for e in bot.history(2)] == [False]


async def test_busy_user_gets_one_moment(tg, make_bot):
    bot = make_bot()
    gate = asyncio.Event()

    async def slow_create(**kwargs):
        await gate.wait()
        from types import SimpleNamespace as NS
        return NS(choices=[NS(message=NS(content="done", tool_calls=None))], usage=None)

    bot.llm.chat.completions.create = slow_create
    first = asyncio.create_task(bot.handle_message(make_update(tg, text="one", chat="private")))
    await asyncio.sleep(0.01)
    await bot.handle_message(make_update(tg, text="two", chat="private"))
    gate.set()
    await first
    assert "still answering" in tg.texts[0] and tg.texts[-1] == "done"


async def test_reset_during_answer_drops_it_from_history(tg, make_bot):
    bot = make_bot()

    async def create(**kwargs):
        bot.history(2).clear()
        from types import SimpleNamespace as NS
        return NS(choices=[NS(message=NS(content="late", tool_calls=None))], usage=None)

    bot.llm.chat.completions.create = create
    await bot.handle_message(make_update(tg, text="q", chat="private"))
    assert tg.texts == ["late"] and len(bot.history(2)) == 0


# ---------- modules ----------

class Weather(Module):
    prompt = "You know the weather."

    def __init__(self):
        self.seen = []

    async def on_message(self, ctx):
        self.seen.append((ctx.text, ctx.addressed))
        return ctx.text == "secret"

    async def context(self, ctx):
        return "Paris: 21°C"

    @tool("Weather for a city")
    async def weather(self, city: str, ctx) -> str:
        return f"{city} in chat {ctx.chat_id}: sunny"

    @command("summary", "Summarize", chats="group")
    async def summary(self, ctx):
        await ctx.answer("Summarize the conversation.")


async def test_module_hooks_prompt_context_and_tools(tg, make_bot):
    w = Weather()
    bot = make_bot({"tool": "weather", "args": {"city": "Rome"}}, "Rome is sunny").add(w)
    await bot.handle_message(make_update(tg, text="secret", chat="private"))
    await bot.handle_message(make_update(tg, text="weather in Rome?", chat="private"))
    assert w.seen == [("secret", True), ("weather in Rome?", True)]
    assert [e.text for e in bot.history(2)] == ["weather in Rome?", "Rome is sunny"]   # "secret" consumed
    first, second = bot.llm.calls
    assert "You know the weather." in system_prompt(first)
    assert "Paris: 21°C" in tail_note(first)
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "name": "weather",
                                      "content": "Rome in chat 2: sunny"}
    assert tg.texts == ["Rome is sunny"]


async def test_command_answer_with_instruction(tg, make_bot):
    bot = make_bot("Summary: nothing much").add(Weather())
    await bot.handle_message(make_update(tg, text="a"))
    await bot.handle_command(make_update(tg, text="/summary"))
    assert tail_note(bot.llm.calls[0]).endswith("Summarize the conversation.")
    assert tg.texts == ["Summary: nothing much"]


async def test_failing_module_does_not_break_the_bot(tg, make_bot):
    class Broken(Module):
        async def on_message(self, ctx):
            raise ValueError("oops")

        async def context(self, ctx):
            raise ValueError("oops")

    bot = make_bot("fine").add(Broken())
    await bot.handle_message(make_update(tg, text="hi", chat="private"))
    assert tg.texts == ["fine"]


async def test_plain_function_command_and_tool(tg, make_bot):
    bot = make_bot({"tool": "add", "args": {"a": 2, "b": 3}}, "5")

    @bot.command
    async def ping(ctx):
        """Check the bot"""
        await ctx.reply("pong")

    @bot.tool("Adds numbers")
    async def add(a: int, b: int) -> int:
        return a + b

    await bot.handle_command(make_update(tg, text="/ping", chat="private"))
    await bot.handle_message(make_update(tg, text="2+3?", chat="private"))
    assert tg.texts == ["pong", "5"]
    assert bot.llm.calls[1]["messages"][-1]["content"] == "5"
    assert bot._commands["ping"].description == "Check the bot"


# ---------- commands ----------

async def test_help_lists_what_the_user_can_run(tg, make_bot):
    bot = make_bot().add(Weather())
    await bot.handle_command(make_update(tg, text="/help", chat="private"))
    user_help = tg.texts[-1]
    assert "/new" in user_help and "/status" not in user_help and "/summary" not in user_help
    await bot.handle_command(make_update(tg, text="/help", chat="private", user_id=OWNER_ID))
    assert "/status" in tg.texts[-1]


async def test_command_scoping(tg, make_bot):
    bot = make_bot()
    await bot.handle_command(make_update(tg, text="/status"))                       # group, stranger: silent
    await bot.handle_command(make_update(tg, text="/nope"))                         # group, unknown: silent
    await bot.handle_command(make_update(tg, text="/new"))                          # private-only: silent in group
    await bot.handle_command(make_update(tg, text="/help@other_bot"))               # someone else's bot
    assert tg.sent == []
    await bot.handle_command(make_update(tg, text="/status", chat="private"))       # stranger in DM
    assert tg.texts == ["Unknown command. Try /help"]
    await bot.handle_command(make_update(tg, text=f"/status@{BOT_USERNAME}", user_id=OWNER_ID))
    assert "status" in tg.texts[-1]


async def test_command_error_is_reported(tg, make_bot):
    bot = make_bot()

    @bot.command("boom")
    async def boom(ctx):
        raise RuntimeError

    await bot.handle_command(make_update(tg, text="/boom", chat="private"))
    assert tg.texts == ["Something went wrong with this command."]


async def test_new_and_admin_commands(tg, make_bot, tmp_path):
    bot = make_bot(data_dir=tmp_path)
    await bot.handle_message(make_update(tg, text="x", user_id=OWNER_ID))
    owner = dict(user_id=OWNER_ID)
    await bot.handle_command(make_update(tg, text="/prompt Be a pirate.", **owner))
    await bot.handle_command(make_update(tg, text="/model big-model", **owner))
    await bot.handle_command(make_update(tg, text="/context 7", **owner))
    assert bot.system_prompt_for(-100) == "Be a pirate." and bot.model_for(-100) == "big-model"
    assert bot.history(-100).size == 7
    assert "big-model" in (tmp_path / "chat_settings.json").read_text()
    await bot.handle_command(make_update(tg, text="/reset", **owner))
    assert len(bot.history(-100)) == 0
    await bot.handle_command(make_update(tg, text="/model clear", **owner))
    assert bot.model_for(-100) == "test-model"

    await bot.handle_message(make_update(tg, text="dm", chat="private", user_id=7))
    await bot.handle_command(make_update(tg, text="/new", chat="private", user_id=7))
    assert len(bot.history(7)) == 0


async def test_builtins_can_be_disabled(make_bot):
    assert make_bot(builtins=False).commands == []


# ---------- allowlist ----------

async def test_allowlist(tg, make_bot, tmp_path):
    bot = make_bot("hi", "hi", "hi", "hi", data_dir=tmp_path).add(Allowlist(users=[10], usernames=["@Invited"]))
    await bot.handle_message(make_update(tg, text="hello", chat="private", user_id=99))   # stranger
    assert tg.texts == ["Sorry, this bot is private."] and bot.llm.calls == []
    await bot.handle_message(make_update(tg, text="testy?", user_id=99))                  # stranger, group: silent
    await bot.handle_message(make_update(tg, text="hello", chat="private", user_id=10))
    await bot.handle_message(make_update(tg, text="hello", chat="private", user_id=OWNER_ID))
    await bot.handle_message(make_update(tg, text="hello", chat="private", user_id=55, username="invited"))
    assert len(bot.llm.calls) == 3

    await bot.handle_command(make_update(tg, text="/allow 99", chat="private", user_id=OWNER_ID))
    await bot.handle_command(make_update(tg, text="/deny 10", chat="private", user_id=OWNER_ID))
    await bot.handle_command(make_update(tg, text="/allowed", chat="private", user_id=OWNER_ID))
    assert "• 55" in tg.texts[-1] and "• 99" in tg.texts[-1] and "• 10\n" not in tg.texts[-1] + "\n"

    # persisted: a fresh bot on the same data_dir sees the same list
    make_bot(data_dir=tmp_path).add(al := Allowlist())
    assert al.users == {55, 99} and al.invites == set()


# ---------- startup & delivery ----------

async def test_post_init_sets_menus_and_runs_setup(make_bot):
    from types import SimpleNamespace as NS

    class Hello(Module):
        started = False

        async def setup(self):
            Hello.started = True

    menus = {}

    async def set_my_commands(commands, scope=None):
        menus[type(scope).__name__ + str(getattr(scope, "chat_id", ""))] = [c.command for c in commands]

    async def get_me():
        return NS(id=42, username="real_bot", first_name="Realname", can_read_all_group_messages=True)

    bot = make_bot().add(Weather(), Hello())
    bot.name = None
    await bot._post_init(NS(bot=NS(get_me=get_me, set_my_commands=set_my_commands)))
    assert (bot.bot_id, bot.username, bot.name) == (42, "real_bot", "Realname") and Hello.started
    assert "summary" not in menus["BotCommandScopeAllPrivateChats"] and "new" in menus["BotCommandScopeAllPrivateChats"]
    assert menus["BotCommandScopeAllGroupChats"] == ["help", "id", "summary"]
    assert "status" in menus[f"BotCommandScopeChat{OWNER_ID}"]


def test_app_is_built_with_handlers(make_bot):
    app = make_bot().app
    assert len(app.handlers[0]) == 2


async def test_markup_rejected_falls_back_to_plain(tg, make_bot):
    from telegram.error import BadRequest

    original = tg.send_message

    async def picky(chat_id, text, **kwargs):
        if kwargs.get("parse_mode"):
            raise BadRequest("can't parse entities")
        return await original(chat_id, text, **kwargs)

    tg.send_message = picky
    bot = make_bot("**bold** " + "x" * 5000)
    await bot.handle_message(make_update(tg, text="hi", chat="private"))
    assert len(tg.sent) == 2 and tg.sent[0]["text"].startswith("**bold**")
    assert list(bot.history(2))[-1].is_bot


# ---------- review regressions ----------

async def test_multiline_command_args(tg, make_bot):
    bot = make_bot()
    await bot.handle_command(make_update(tg, text="/prompt\nBe a pirate.\nAlways.", user_id=OWNER_ID))
    assert bot.system_prompt_for(-100) == "Be a pirate.\nAlways."


def test_storage_is_shared_per_name(make_bot, tmp_path):
    bot = make_bot(data_dir=tmp_path)
    bot.storage("x").set("a", 1)
    bot.storage("x").set("b", 2)
    assert bot.storage("x") is bot.storage("x") and bot.storage("x").data == {"a": 1, "b": 2}


async def test_crashing_access_gate_denies(tg, make_bot):
    class BrokenGate(Module):
        async def allow(self, ctx):
            raise RuntimeError

    bot = make_bot().add(BrokenGate())
    await bot.handle_message(make_update(tg, text="hi", chat="private"))
    assert bot.llm.calls == [] and tg.sent == []


async def test_default_prompt_can_be_overridden_after_import(tg, make_bot, monkeypatch):
    from groupchat import texts

    monkeypatch.setattr(texts, "DEFAULT_SYSTEM_PROMPT", "Ты — {name}.")
    bot = make_bot("ok")
    await bot.handle_message(make_update(tg, text="hi", chat="private"))
    assert system_prompt(bot.llm.calls[0]).startswith("Ты — Testy.")


async def test_answer_survives_deleted_trigger(tg, make_bot):
    bot = make_bot("ok", wake_words=["testy"])
    await bot.handle_message(make_update(tg, text="testy hi"))
    assert tg.sent[0]["reply_parameters"].allow_sending_without_reply is True


def test_command_bound_under_its_attribute_name():
    import functools

    def logged(fn):
        @functools.wraps(fn)
        async def wrapper(*a, **kw):
            return await fn(*a, **kw)
        wrapper.__name__ = "something_else"
        return wrapper

    class M(Module):
        @command("go", "Go")
        @logged
        async def go(self, ctx): ...

    (c,) = M().commands()
    assert c.handler.__self__.__class__ is M
