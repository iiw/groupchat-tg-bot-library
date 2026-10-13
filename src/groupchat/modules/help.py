from __future__ import annotations

from ..context import Context
from ..module import Module, command


class Help(Module):
    """/start and /help: an intro plus the commands this user can run here."""

    def __init__(self, intro: str | None = None):
        self.intro = intro

    def intro_text(self) -> str:
        if self.intro:
            return self.intro
        how = "mention me, reply to my message"
        if self.bot.wake_words:
            how += f" or call me «{self.bot.wake_words[0]}»"
        lines = [
            f"I'm {self.bot.name or 'an assistant bot'} — an AI assistant.",
            f"• In groups: {how} — I answer using the recent chat context.",
            "• In private messages: just write to me.",
        ]
        if self.bot.max_images:
            lines.append("• You can send pictures — I can see them.")
        return "\n".join(lines)

    @command("start", "Start", chats="private")
    async def start(self, ctx: Context) -> None:
        await self.help(ctx)

    @command("help", "Command list")
    async def help(self, ctx: Context) -> None:
        cmds = [c for c in self.bot.commands if c.name != "start" and self.bot.can_use(c, ctx)]
        text = self.intro_text()
        if cmds:
            text += "\n\nCommands:\n" + "\n".join(f"/{c.name} — {c.description}" for c in cmds)
        await ctx.reply(text, markdown=False)

    @command("id", "Your id and this chat's id")
    async def id(self, ctx: Context) -> None:
        await ctx.reply(f"🆔 Chat: {ctx.chat_id}\nYou: {ctx.user_id}", markdown=False)
