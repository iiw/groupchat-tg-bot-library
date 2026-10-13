from __future__ import annotations

from ..context import Context
from ..module import Module, command


class NewSession(Module):
    """/new in a private chat: forget the conversation so far."""

    @command("new", "Start a new conversation", chats="private")
    async def new(self, ctx: Context) -> None:
        ctx.history.clear()
        await ctx.reply("✅ New conversation — I'll only use messages from here on.", markdown=False)
