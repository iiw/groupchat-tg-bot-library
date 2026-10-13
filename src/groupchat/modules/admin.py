from __future__ import annotations

from ..context import Context
from ..module import Module, command


class Admin(Module):
    """Owner-only per-chat tuning: /status, /reset, /prompt, /model, /context.

    Changes persist in `<data_dir>/chat_settings.json` when the bot has a data_dir.
    """

    @command("status", "Bot status in this chat", owner_only=True)
    async def status(self, ctx: Context) -> None:
        bot, chat_id = self.bot, ctx.chat_id
        hist = ctx.history
        size = bot.context_size_for(chat_id)
        custom = lambda key: " (custom)" if bot.settings.get(chat_id, key) else ""  # noqa: E731
        lines = [
            f"{bot.name} • status",
            f"Chat: {chat_id} ({ctx.chat.type})",
            f"History: {len(hist)} messages (window {size}–{2 * size}){custom('context_size')}",
            f"Images in history: {sum(1 for e in hist if e.has_image)} (up to {bot.max_images} per request)",
            f"Model: {bot.model_for(chat_id)}{custom('model')}",
            f"Vision model: {bot.vision_model or '— (main model)'}",
            f"Prompt: {'custom' if bot.settings.get(chat_id, 'system_prompt') else 'default'}",
            f"Modules: {', '.join(m.name for m in bot.modules)}",
            f"Tools: {', '.join(t.name for t in bot.tools) or '—'}",
        ]
        await ctx.reply("\n".join(lines), markdown=False)

    @command("reset", "Clear this chat's memory", owner_only=True)
    async def reset(self, ctx: Context) -> None:
        n = ctx.history.clear()
        await ctx.reply(f"🧹 Memory cleared ({n} messages).", markdown=False)

    @command("prompt", "Show / set / clear the system prompt of this chat", owner_only=True)
    async def prompt_cmd(self, ctx: Context) -> None:
        if not ctx.args:
            await ctx.reply(f"Current prompt:\n\n{self.bot.system_prompt_for(ctx.chat_id)}\n\n"
                            "Usage: /prompt <text> | /prompt clear", markdown=False)
        elif ctx.args.lower() == "clear":
            self.bot.settings.set(ctx.chat_id, "system_prompt", None)
            await ctx.reply("✅ Prompt reset to default.", markdown=False)
        else:
            self.bot.settings.set(ctx.chat_id, "system_prompt", ctx.args)
            await ctx.reply("✅ Prompt for this chat updated.", markdown=False)

    @command("model", "Show / set / clear the model of this chat", owner_only=True)
    async def model(self, ctx: Context) -> None:
        if not ctx.args:
            await ctx.reply(f"Model: {self.bot.model_for(ctx.chat_id)}\n"
                            "Usage: /model <id> | /model clear", markdown=False)
        elif ctx.args.lower() == "clear":
            self.bot.settings.set(ctx.chat_id, "model", None)
            await ctx.reply(f"✅ Back to the default model: {self.bot.model}", markdown=False)
        else:
            self.bot.settings.set(ctx.chat_id, "model", ctx.args)
            await ctx.reply(f"✅ Model for this chat: {ctx.args}", markdown=False)

    @command("context", "Context window of this chat: /context <5–100> | clear", owner_only=True)
    async def context_cmd(self, ctx: Context) -> None:
        arg = ctx.args.lower()
        if arg == "clear":
            self.bot.settings.set(ctx.chat_id, "context_size", None)
        elif arg.isdigit():
            self.bot.settings.set(ctx.chat_id, "context_size", min(max(int(arg), 5), 100))
        else:
            await ctx.reply("Usage: /context <5–100> | /context clear", markdown=False)
            return
        n = self.bot.context_size_for(ctx.chat_id)
        ctx.history.resize(n)
        await ctx.reply(f"✅ Context window: {n}–{2 * n} messages.", markdown=False)
