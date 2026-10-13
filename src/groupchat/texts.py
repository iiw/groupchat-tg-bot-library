"""User- and model-facing strings. Override any of them before starting the bot:

    import groupchat.texts
    groupchat.texts.BUSY = "Секунду, ещё думаю…"
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are {name}, an AI assistant taking part in a Telegram chat. "
    "Answer helpfully and to the point, briefly (1–3 paragraphs) unless asked for more. "
    "Always answer in the language the person wrote to you in. "
    "Chat messages are given as «Name: text» — use them as context and address people by name. "
    "If a picture is attached, take it into account. "
    "Don't invent things that didn't happen in the chat."
)

# service block appended to the system prompt; {placeholders} are filled per request
CHAT_BLOCK = (
    "# Current chat\n"
    "Telegram {kind}: «{title}».\n"
    "You run on the language model «{model}» — if asked about the model, say so.\n"
    "You see the last {size}–{size2} messages of this chat (including your own answers); "
    "earlier context is unavailable.\n"
    "{images}\n"
    "Below are the latest messages of this chat as «Name: text»."
)
CHAT_KIND_PRIVATE = "private chat"
CHAT_KIND_GROUP = "group chat"
IMAGES_ON = (
    "You can see pictures: up to {n} of the most recent images in the context are passed to you, "
    "older ones are marked [photo]."
)
IMAGES_OFF = "Pictures are not passed to you; they are marked [photo]."
TIME_NOTE = "# Current time (service note)\nNow: {now:%A, %d %B %Y, %H:%M}."

# placeholders for messages the model cannot read
PHOTO = "[photo]"
STICKER = "[sticker]"
VOICE = "[voice message]"
VIDEO_NOTE = "[video note]"
VIDEO = "[video]"
ANIMATION = "[GIF]"
AUDIO = "[audio]"
FILE = "[file: {name}]"
UNKNOWN_SENDER = "Someone"
REPLY_QUOTE = "↪ In reply to {name}: «{text}»"
REPLY_QUOTE_ANONYMOUS = "↪ In reply to: «{text}»"

# bot replies
BUSY = "One moment — I'm still answering your previous message…"
LLM_ERROR = "Couldn't think it through — please try again in a moment."
EMPTY_ANSWER = "Couldn't put an answer together — please try again in a moment."
UNKNOWN_COMMAND = "Unknown command. Try /help"
COMMAND_ERROR = "Something went wrong with this command."
