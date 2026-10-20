"""Turns the wellness chat messages into speakable sentences, dropping emojis and markdown"""

import re
import unicodedata

EMOJI_PARTS = re.compile("[‍⃣︎️\U0001f3fb-\U0001f3ff]")
MARKDOWN = re.compile(r"[*_`~#>|]+")


def speakable(text: str) -> str:
    """Drop emojis and markdown, and fold lines into one spoken paragraph."""
    text = EMOJI_PARTS.sub("", text)
    text = "".join(c for c in text if unicodedata.category(c) not in ("So", "Cs", "Co"))
    text = MARKDOWN.sub("", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # A line without end punctuation (a heading or list item) still ends as a sentence
    text = " ".join(line if line[-1] in ".!?:;," else line + "." for line in lines)
    return re.sub(r"\s+([.!?,;:])", r"\1", re.sub(r"\s+", " ", text)).strip()
