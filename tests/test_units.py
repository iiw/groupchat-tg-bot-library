from __future__ import annotations

from typing import Literal

import pytest

from groupchat import Entry, Module, command, tool
from groupchat.formatting import md_to_html, split_message
from groupchat.history import ChatHistory
from groupchat.llm import complete, extract_text, history_to_messages
from groupchat.storage import ChatSettings, Store

from .conftest import FakeLLM


# ---------- formatting ----------

def test_md_to_html():
    assert md_to_html("**b** *i* ~~s~~ `x<y`") == "<b>b</b> <i>i</i> <s>s</s> <code>x&lt;y</code>"
    assert md_to_html("# Title") == "<b>Title</b>"
    assert md_to_html("[a](http://x?a=1&b=2)") == '<a href="http://x?a=1&amp;b=2">a</a>'
    assert md_to_html("```py\nif a<b:\n```") == '<pre><code class="language-py">if a&lt;b:</code></pre>'
    assert md_to_html("snake_case_name") == "snake_case_name"


def test_split_message():
    text = ("word " * 2000).strip()
    chunks = split_message(text, 1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert " ".join(chunks) == text
    assert split_message("short") == ["short"]


# ---------- history ----------

def test_history_grows_to_2n_then_trims_to_n():
    h = ChatHistory(size=3)
    for i in range(6):
        h.append(Entry(1, "A", str(i)))
    assert [e.text for e in h] == ["0", "1", "2", "3", "4", "5"]
    h.append(Entry(1, "A", "6"))
    assert [e.text for e in h] == ["3", "4", "5", "6"]


def test_history_keeps_only_newest_image_bytes():
    h = ChatHistory(size=10, max_images=2)
    for i in range(4):
        h.append(Entry(1, "A", str(i), image=f"data:{i}"))
    assert [e.image for e in h] == [None, None, "data:2", "data:3"]
    assert all(e.has_image for e in h)


def test_history_clear_bumps_epoch():
    h = ChatHistory(size=3)
    h.append(Entry(1, "A", "x"))
    assert h.clear() == 1 and h.epoch == 1 and len(h) == 0


# ---------- modules ----------

class Base(Module):
    @command("one", "First")
    async def one(self, ctx): ...

    @command("two", "Second")
    async def two(self, ctx): ...


class Child(Base):
    @command("one", "Overridden")
    async def one(self, ctx): ...

    @tool
    async def lookup(self, city: str, days: int = 1, unit: Literal["c", "f"] = "c",
                     tags: list[str] | None = None, ctx=None) -> str:
        """Weather lookup."""
        return f"{city}:{days}:{unit}:{ctx is not None}"


def test_module_collects_commands_and_tools():
    m = Child()
    cmds = {c.name: c.description for c in m.commands()}
    assert cmds == {"one": "Overridden", "two": "Second"}
    (t,) = m.tools()
    assert t.name == "lookup" and t.description == "Weather lookup." and t.wants_ctx
    assert t.parameters == {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "unit": {"type": "string", "enum": ["c", "f"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["city"],
    }


def test_command_validation():
    with pytest.raises(ValueError):
        command("Bad-Name")(lambda ctx: None)
    with pytest.raises(ValueError):
        command("x", chats="channel")
    with pytest.raises(ValueError):
        tool()(lambda: None)  # no description


# ---------- llm ----------

def test_history_to_messages():
    entries = [
        Entry(1, "Ann", "hi"),
        Entry(2, "Bob", "look", image="data:image/jpeg;base64,AAA", has_image=True),
        Entry(2, "Bob", "old pic", has_image=True),
        Entry(9, "Bot", "answer", is_bot=True),
        Entry(1, "Ann", "a"),
        Entry(2, "Bob", "b"),
    ]
    msgs = history_to_messages(entries, max_images=1)
    assert msgs[0] == {"role": "user", "content": "Ann: hi"}
    assert msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg")
    assert msgs[2] == {"role": "user", "content": "Bob: old pic [photo]"}
    assert msgs[3] == {"role": "assistant", "content": "answer"}
    assert msgs[4] == {"role": "user", "content": "Ann: a\nBob: b"}
    assert history_to_messages(entries, max_images=0)[0]["content"] == "Ann: hi\nBob: look [photo]\nBob: old pic [photo]"


def test_extract_text_drops_thinking():
    assert extract_text([{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "hi"}]) == "hi"
    assert extract_text(None) == ""


async def test_complete_runs_tools():
    m = Child()
    llm = FakeLLM({"tool": "lookup", "args": {"city": "Paris"}}, "It's sunny")
    seen = []

    async def run_tool(t, args):
        seen.append(args)
        return await t.handler(ctx="CTX", **args)

    usage = {}
    out = await complete(llm, "m", [{"role": "user", "content": "q"}], tools=m.tools(), run_tool=run_tool,
                         usage_out=usage)
    assert out == "It's sunny" and seen == [{"city": "Paris"}]
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"] == "Paris:1:c:True"
    assert llm.calls[0]["tools"][0]["function"]["name"] == "lookup"
    assert usage == {"prompt_tokens": 20, "completion_tokens": 10}


async def test_complete_reports_tool_errors_to_model():
    class Broken(Module):
        @tool("Breaks")
        async def boom(self) -> str:
            raise RuntimeError("nope")

    llm = FakeLLM({"tool": "boom"}, {"tool": "missing"}, "done")
    assert await complete(llm, "m", [], tools=Broken().tools()) == "done"
    assert "RuntimeError: nope" in llm.calls[1]["messages"][-1]["content"]
    assert "unknown tool" in llm.calls[2]["messages"][-1]["content"]


async def test_complete_stops_tool_loop():
    class Loop(Module):
        @tool("Loops")
        async def again(self) -> str:
            return "again"

    llm = FakeLLM(*([{"tool": "again"}] * 10))
    llm.replies.insert(5, "final")
    assert await complete(llm, "m", [], tools=Loop().tools()) == "final"
    assert "tools" not in llm.calls[-1]


async def test_complete_flips_image_style_on_error():
    llm = FakeLLM("seen")
    llm.fail_on_images_once = True
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]}]
    assert await complete(llm, "m", msgs) == "seen"
    assert llm.calls[-1]["messages"][0]["content"][0]["image_url"] == "data:x"


# ---------- storage ----------

def test_store_persists(tmp_path):
    s = Store(tmp_path / "x.json", private=True)
    s.set("a", 1)
    s.set("b", None)
    assert Store(tmp_path / "x.json").get("a") == 1
    assert (tmp_path / "x.json").stat().st_mode & 0o777 == 0o600
    cs = ChatSettings(Store(tmp_path / "s.json"))
    cs.set(5, "model", "m")
    cs.set(5, "model", None)
    assert cs.store.data == {}
