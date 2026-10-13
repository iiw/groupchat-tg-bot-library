"""groupchat — a small library for Telegram group-chat AI bots, extended with modules.

    from groupchat import Bot, Module, command, tool

    class Dice(Module):
        prompt = "You can roll dice with the roll tool."

        @command("roll", "Roll a die")
        async def roll_cmd(self, ctx):
            await ctx.reply(f"🎲 {random.randint(1, 6)}")

        @tool("Roll an n-sided die")
        async def roll(self, sides: int = 6) -> int:
            return random.randint(1, sides)

    Bot(TOKEN, API_KEY, model="mistral-large-latest", base_url="https://api.mistral.ai/v1").add(Dice()).run()
"""

from .bot import Bot
from .context import Context
from .history import ChatHistory, Entry
from .llm import LLMError
from .module import Command, Module, Tool, command, tool
from .storage import Store

__all__ = [
    "Bot", "Module", "Context", "command", "tool",
    "ChatHistory", "Entry", "Command", "Tool", "Store", "LLMError",
]
__version__ = "0.0.2"
