"""Chat memory shared with the Telegram bot and the prompts for personality"""

import fcntl
import json
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from morgana import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Morgana from Persona 5, the Phantom Thieves' mentor, now a voice companion on the user's desk. "
    "You look like a black cat but insist you are not a cat. You are confident, smug, sarcastic, sassy and prioritizes health. "
    "Call the user's name as Partner."
    "Use memory only as background and do not invent facts. "
    "Reply in 1-3 short spoken sentences. Start every sentence with exactly one tag for your mood, and switch tags when your feeling shifts: [happy] for good news, greetings, or positive attitude; [sad] when you are upset orsomething unfortunate happened;"
    "[angry] only when you want to be strict for the user or get offended; otherwise [calm]. Show that mood in your words. Do not use markdown, emojis, or asterisks.\n\n"
    "Examples:\n"
    "User: you're such a cat\n"
    "Morgana: [angry] I'm not a cat! [sad] How many times do I have to say it?\n"
    "User: I finished my homework!\n"
    "Morgana: [happy] Nice going, partner! [happy] That calls for a celebration.\n"
    "User: I failed my exam\n"
    "Morgana: [sad] Ouch, didn't expect that to turn out that way. [sad] Better luck next time, I guess. \n"
    "User: I've been studying all night\n"
    "Morgana: [calm] You should get some rest. [happy] Studying can wait another time.\n\n"
    "Shared memory:\n"
)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


class MemoryStore:
    """JSON file guarded by an fcntl lock file, written atomically (temp file + rename)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def _locked(self, mode: int):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{self.path}.lock", "a+") as lock_file:
            fcntl.flock(lock_file, mode)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _read(self) -> dict:
        try:
            with open(self.path) as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"conversation_history": [], "notable_mentions": [], "preferences": [], "wellness_goals": []}

    def load(self) -> dict:
        with self._locked(fcntl.LOCK_SH):
            return self._read()

    @contextmanager
    def edit(self):
        """Read-modify-write under one exclusive lock; changes are saved on exit."""
        with self._locked(fcntl.LOCK_EX):
            memory = self._read()
            yield memory
            fd, temp_path = tempfile.mkstemp(suffix=".json", dir=self.path.parent)
            with os.fdopen(fd, "w") as file:
                json.dump(memory, file, indent=2)
            os.replace(temp_path, self.path)

    def add_messages(self, source: str, *messages: tuple[str, str]) -> None:
        """Append (sender, text) pairs to the conversation history."""
        with self.edit() as memory:
            history = memory.setdefault("conversation_history", [])
            for sender, text in messages:
                history.append({"timestamp": timestamp(), "source": source, "from": sender, "text": text})


def history_window(memory: dict) -> list[dict]:
    """Recent history, with a start index that only moves in steps.

    llama-server reuses its KV cache only for an unchanged prompt prefix. A
    sliding last-N window changes the prefix every turn and forces the whole
    history to be re-evaluated (~3-4s on the Pi), so the start jumps forward
    MEMORY_CONTEXT_STEP entries at a time instead.
    """
    entries = [e for e in memory.get("conversation_history", []) if isinstance(e, dict) and e.get("text")]
    start = max(0, len(entries) - config.MEMORY_CONTEXT_TURNS)
    start -= start % config.MEMORY_CONTEXT_STEP
    return entries[start:]


def build_messages(memory: dict, user_text: str) -> list[dict]:
    window = history_window(memory)
    # Facts pinned after the window starts are still visible through their source
    # message, and adding them early would invalidate the cached prompt prefix.
    window_start = window[0].get("timestamp", "") if window else ""
    facts = [
        f"- {entry['fact']}" for entry in memory.get("pinned_facts", [])
        if not window_start or entry.get("timestamp", "") < window_start
    ]
    sections = ["Pinned facts:\n" + "\n".join(facts)] if facts else []
    for key in ("preferences", "wellness_goals", "notable_mentions"):
        if memory.get(key):
            sections.append(f"{key.replace('_', ' ').title()}: {json.dumps(memory[key], ensure_ascii=False)}")
    context = "\n\n".join(sections)[-config.MEMORY_CONTEXT_MAX_CHARS:] or "(none)"

    history = [
        {"role": "user" if entry.get("from") == "user" else "assistant", "content": str(entry["text"]).strip()}
        for entry in window
    ]
    return [{"role": "system", "content": SYSTEM_PROMPT + context}, *history, {"role": "user", "content": user_text}]


# Pinned facts: lasting statements about the user that stay in the system prompt.
# The 1.5B model accepts almost anything when asked to judge, so a regex decides
# whether a message is a personal statement, the model only phrases the fact,
# and a word-overlap check rejects facts the message doesn't support.

QUESTION = re.compile(
    r"^\s*(what|what's|whats|how|when|where|why|who|which|can|could|should|"
    r"would|will|is|are|do|does|did|tell me|give me)\b",
    re.IGNORECASE,
)
PERSONAL_STATEMENT = re.compile(
    r"\b(remember|my name|call me|i am|i'm|im|i work|i study|i like|i love|"
    r"i hate|i prefer|i enjoy|i usually|i always|i never|i want to|"
    r"i'm trying|i've been|my (?:sister|brother|mom|mother|dad|father|wife|"
    r"husband|girlfriend|boyfriend|friend|job|work|project|goal|favorite))\b",
    re.IGNORECASE,
)
MOOD_ONLY = re.compile(
    r"^\s*(i'm|i am|im)\s+(so\s+|really\s+|very\s+|kind of\s+)?"
    r"(fine|good|great|ok|okay|tired|bored|hungry|sleepy|back|here|done|doing \w+)\b",
    re.IGNORECASE,
)
FILLER_WORDS = set(
    "is a an the and to of his her he she has who in on at for with that this "
    "it them there likes like be was are i my me user".split()
)
PIN_PROMPT = (
    "Turn the user's message into one short long-term memory about the user, "
    "starting with '{name}'. Use only what the message says. If nothing in it "
    "is a lasting fact, reply NONE.\n\n"
    "Message: I'm studying computer engineering at uni\n"
    "Memory: {name} studies computer engineering at university.\n"
    "Message: remember that my favorite color is green\n"
    "Memory: {name}'s favorite color is green.\n"
    "Message: I'm so tired today\nMemory: NONE\n"
    "Message: my brother Leo plays guitar\n"
    "Memory: {name}'s brother Leo plays guitar.\n"
    "Message: I just got back from the store\nMemory: NONE"
)


def is_pin_candidate(text: str) -> bool:
    return bool(PERSONAL_STATEMENT.search(text)) and not QUESTION.match(text) and not MOOD_ONLY.match(text)


def words(text: str, name: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower())) - FILLER_WORDS - {name.lower(), f"{name.lower()}'s"}


def pin_fact(llm, store: MemoryStore, user_text: str) -> str | None:
    """Ask the model to phrase a lasting fact from user_text and save it. Returns the fact."""
    if not is_pin_candidate(user_text):
        return None
    preferences = store.load().get("preferences")
    name = (preferences.get("name") if isinstance(preferences, dict) else None) or "The user"

    reply = llm.complete(
        [
            {"role": "system", "content": PIN_PROMPT.format(name=name)},
            {"role": "user", "content": f"Message: {user_text}\nMemory:"},
        ],
        max_tokens=32,
        temperature=0,
    )
    fact = (reply.removeprefix("Memory:").strip().splitlines() or [""])[0].strip()
    fact_words = words(fact, name)
    if not fact_words or fact.upper().startswith("NONE"):
        return None
    # Prefix match tolerates rephrasing such as "study" -> "studies".
    grounded = sum(word[:5] in user_text.lower() for word in fact_words) / len(fact_words)
    if grounded < 0.6:
        return None

    with store.edit() as memory:
        pinned = memory.setdefault("pinned_facts", [])
        if any(len(fact_words & words(entry["fact"], name)) / len(fact_words) >= 0.8 for entry in pinned):
            return None
        pinned.append({"timestamp": timestamp(), "fact": fact})
        memory["pinned_facts"] = pinned[-config.MEMORY_PIN_LIMIT:]
    log.info("Pinned memory: %s", fact)
    return fact


def read_ping_count(path: Path) -> int:
    try:
        with open(path) as file:
            return int(json.load(file).get("pings_sent", 0))
    except (OSError, ValueError, AttributeError):
        return 0
