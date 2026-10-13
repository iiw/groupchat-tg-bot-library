from __future__ import annotations

import logging
import re
from typing import Iterable

from ..context import Context
from ..module import Module, command
from ..storage import Store

logger = logging.getLogger("groupchat")


class Allowlist(Module):
    """Only the owner and listed users get AI answers.

    Users are given by id, or invited by @username — an invite binds to the id on
    first contact. The owner manages the list at runtime with /allow, /deny, /allowed;
    it persists in `<data_dir>/allowlist.json`. Strangers get `message` in private
    chats and silence in groups.
    """

    def __init__(
        self,
        users: Iterable[int] = (),
        usernames: Iterable[str] = (),
        message: str = "Sorry, this bot is private.",
    ):
        self.initial_users = list(users)
        self.initial_usernames = [_norm(u) for u in usernames]
        self.message = message
        self._store: Store | None = None

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = self.bot.storage("allowlist")
            users = set(self._store.get("users", [])) | set(self.initial_users)
            invites = set(self._store.get("invites", [])) | set(self.initial_usernames)
            self._save(users, invites)
        return self._store

    @property
    def users(self) -> set[int]:
        return set(self.store.get("users", []))

    @property
    def invites(self) -> set[str]:
        return set(self.store.get("invites", []))

    def _save(self, users: set[int], invites: set[str]) -> None:
        self._store.data.update(users=sorted(users), invites=sorted(invites))
        self._store.save()

    def is_allowed(self, user_id: int | None, username: str | None = None) -> bool:
        if user_id is None:
            return False
        if user_id == self.bot.owner_id or user_id in self.users:
            return True
        name = _norm(username or "")
        if name and name in self.invites:  # first contact of an invited user: bind the id
            self._save(self.users | {user_id}, self.invites - {name})
            logger.info("invite @%s bound to id %s", name, user_id)
            return True
        return False

    async def allow(self, ctx: Context) -> bool:
        user = ctx.user
        if self.is_allowed(ctx.user_id, user.username if user else None):
            return True
        logger.info("no access: id=%s chat %s", ctx.user_id, ctx.chat_id)
        if ctx.is_private and self.message:
            await ctx.reply(self.message, markdown=False)
        return False

    @command("allow", "Allow a user: /allow <id | @username>", owner_only=True)
    async def allow_cmd(self, ctx: Context) -> None:
        target = ctx.args.strip()
        if target.isdigit():
            self._save(self.users | {int(target)}, self.invites)
            await ctx.reply(f"✅ {target} allowed.", markdown=False)
        elif re.fullmatch(r"@?[A-Za-z0-9_]{4,32}", target):
            self._save(self.users, self.invites | {_norm(target)})
            await ctx.reply(f"✅ @{_norm(target)} invited — access turns on at first contact.", markdown=False)
        else:
            await ctx.reply("Usage: /allow <user_id | @username>  (ask them to send /id … or use @userinfobot)",
                            markdown=False)

    @command("deny", "Revoke access: /deny <id | @username>", owner_only=True)
    async def deny_cmd(self, ctx: Context) -> None:
        target = ctx.args.strip()
        if target.isdigit():
            self._save(self.users - {int(target)}, self.invites)
        elif target:
            self._save(self.users, self.invites - {_norm(target)})
        else:
            await ctx.reply("Usage: /deny <user_id | @username>", markdown=False)
            return
        await ctx.reply(f"✅ {target} removed.", markdown=False)

    @command("allowed", "Who has access", owner_only=True)
    async def allowed_cmd(self, ctx: Context) -> None:
        lines = ["👥 Access:"]
        if self.bot.owner_id:
            lines.append(f"• {self.bot.owner_id} — owner")
        lines += [f"• {uid}" for uid in sorted(self.users) if uid != self.bot.owner_id]
        lines += [f"• @{name} — invited, waiting for first contact" for name in sorted(self.invites)]
        await ctx.reply("\n".join(lines), markdown=False)


def _norm(username: str) -> str:
    return username.strip().lstrip("@").lower()
