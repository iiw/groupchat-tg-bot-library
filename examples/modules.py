"""A bot extended with modules: a command, a tool, a data source, a prompt and access control.

    TELEGRAM_BOT_TOKEN=... LLM_API_KEY=... OWNER_ID=... python examples/modules.py
"""

import os
import random

from groupchat import Bot, Module, command, tool
from groupchat.modules import Allowlist


class Dice(Module):
    """A command for people and a tool for the model."""

    prompt = "When someone asks for randomness (dice, coin, lottery), use the roll tool — never make numbers up."

    @command("roll", "Roll a die: /roll [sides]")
    async def roll_cmd(self, ctx):
        sides = int(ctx.args) if ctx.args.isdigit() else 6
        await ctx.reply(f"🎲 {random.randint(1, sides)}")

    @tool("Roll an n-sided die")
    async def roll(self, sides: int = 6) -> int:
        return random.randint(1, sides)


class Office(Module):
    """A data source: fresh facts injected into every request."""

    def __init__(self, menu: list[str]):
        self.menu = menu

    async def context(self, ctx):
        return "Today's canteen menu: " + ", ".join(self.menu)


class Summary(Module):
    """A command that makes the AI answer with an extra instruction."""

    @command("tldr", "Summarize the recent conversation", chats="group")
    async def tldr(self, ctx):
        await ctx.answer("Summarize the conversation above in 3–5 bullet points.")


class Counter(Module):
    """Sees every message; keeps persistent per-chat counts in <data_dir>/counter.json."""

    async def setup(self):
        self.store = self.bot.storage("counter")

    async def on_message(self, ctx):
        key = str(ctx.chat_id)
        self.store.set(key, self.store.get(key, 0) + 1)

    @command("count", "How many messages I've seen here")
    async def count(self, ctx):
        await ctx.reply(f"I've seen {self.store.get(str(ctx.chat_id), 0)} messages in this chat.")


owner = int(os.environ["OWNER_ID"])
bot = Bot(
    os.environ["TELEGRAM_BOT_TOKEN"],
    os.environ["LLM_API_KEY"],
    base_url="https://api.mistral.ai/v1",
    model="mistral-large-latest",
    owner_id=owner,
    wake_words=["chonk"],
    system_prompt="You are {name}, a cheerful office assistant. Be brief.",
    timezone="Europe/Berlin",
    data_dir="data",
)
bot.add(
    Allowlist(users=[owner]),
    Dice(),
    Office(["borscht", "pelmeni", "syrniki"]),
    Summary(),
    Counter(),
)


@bot.command("ping", "Check that the bot is alive")
async def ping(ctx):
    await ctx.reply("pong")


bot.run()
