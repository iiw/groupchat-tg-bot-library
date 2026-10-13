"""The smallest useful bot: an AI participant for group chats and DMs.

    TELEGRAM_BOT_TOKEN=... LLM_API_KEY=... python examples/minimal.py
"""

import os

from groupchat import Bot

Bot(
    os.environ["TELEGRAM_BOT_TOKEN"],
    os.environ["LLM_API_KEY"],
    base_url="https://api.mistral.ai/v1",
    model="mistral-large-latest",
).run()
