"""Talks to the local model and turns its streamed reply into speakable sentences"""

import logging
import re
import threading

from openai import OpenAI, OpenAIError

from morgana import config
from morgana.memory import MemoryStore, build_messages, pin_fact
from morgana.state import EMOTIONS, Stopwatch
from morgana.text import speakable

log = logging.getLogger(__name__)

FALLBACK_REPLY = "I am currently unable to connect to my local processing core."
EMOTION_TAG = re.compile(rf"^\s*\[({'|'.join(EMOTIONS)})\]\s*", re.IGNORECASE)
SENTENCE_END = re.compile(r"^(.+?[.!?])(?:\s+|$)", re.DOTALL)
SUMMARY_PROMPT = (
    "Rewrite the wellness reminder as a few sentences."
    "Do not use emojis, markdown or quotes.\n"
)


class LocalLLM:
    """llama-server through its OpenAI-compatible API."""

    def __init__(self):
        self.client = OpenAI(
            base_url=config.LOCAL_LLM_BASE_URL, api_key="local", timeout=config.LOCAL_LLM_TIMEOUT, max_retries=0
        )

    def complete(self, messages: list[dict], **options) -> str:
        response = self.client.chat.completions.create(model=config.LOCAL_LLM_MODEL, messages=messages, **options)
        return (response.choices[0].message.content or "").strip()

    def stream(self, messages: list[dict], **options):
        """Yield text pieces as the model generates them."""
        chunks = self.client.chat.completions.create(
            model=config.LOCAL_LLM_MODEL, messages=messages, stream=True, **options
        )
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def mood_grammar(user_text: str) -> str:
    """GBNF grammar that makes the reply start with exactly one emotion tag."""
    choices = " | ".join(f'"{mood}"' for mood in EMOTIONS)
    return f'root ::= "[" ({choices}) "] " [^[\\x00]*'


class SentenceSplitter:
    """Buffers streamed text and returns each sentence once it is complete. 
    While want_clauses() is true (little audio left to play), long sentences are also cut at commas or conjunctions so speech starts sooner. """

    def __init__(self, want_clauses=lambda: False):
        self.emotion = "calm"
        self.buffer = ""
        self.tag_checked = False
        self.want_clauses = want_clauses
        clause_words = "|".join(w for w in config.CLAUSE_SPLIT_WORDS if w) or "(?!)"
        self.clause_end = re.compile(
            rf"^(.{{{config.FIRST_CLAUSE_MIN_CHARS},}}?(?:[,;:](?=\s)|(?=\s(?:{clause_words})\s)))\s+", re.DOTALL
        )

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        if not self.tag_checked:
            if match := EMOTION_TAG.match(self.buffer):
                self.emotion = match.group(1).lower()
                self.buffer = self.buffer[match.end():]
            elif any(f"[{e}]".startswith(self.buffer.lstrip().lower()) for e in EMOTIONS):
                return [] 
            self.tag_checked = True

        sentences = []
        while True:
            match = SENTENCE_END.search(self.buffer)
            if not match and config.FIRST_CLAUSE_MIN_CHARS > 0 and self.want_clauses():
                match = self.clause_end.search(self.buffer)
            if not match:
                return sentences
            self.buffer = self.buffer[match.end():]
            if sentence := match.group(1).strip():
                sentences.append(sentence)

    def flush(self) -> str:
        rest, self.buffer = self.buffer.strip(), ""
        return rest


class Conversation:
    def __init__(self, llm: LocalLLM, store: MemoryStore):
        self.llm = llm
        self.store = store

    def stream_reply(self, user_text, source, on_sentence, want_clauses, stopwatch: Stopwatch) -> tuple[str, str]:
        """Reply by calling on_sentence(emotion, text) for each sentence."""
        log.info("Prompting local LLM (%s): %r", source, user_text)
        splitter = SentenceSplitter(want_clauses)
        reply = ""
        spoke = False
        try:
            stopwatch.mark("llm_request_started")
            messages = build_messages(self.store.load(), user_text)
            pieces = self.llm.stream(
                messages,
                max_tokens=config.LOCAL_LLM_MAX_TOKENS,
                temperature=0.7,
                extra_body={"grammar": mood_grammar(user_text)},  # llama-server GBNF
            )
            for piece in pieces:
                if not reply:
                    stopwatch.mark("llm_first_token")
                reply += piece
                for sentence in splitter.feed(piece):
                    on_sentence(splitter.emotion, sentence)
                    spoke = True
            rest = splitter.flush()
            if spoke and rest and not rest.endswith((".", "!", "?")):
                # The token limit ended the reply mid-sentence; stop at the last full one.
                log.info("Dropped unfinished sentence: %r", rest)
                reply = reply.rstrip()[:-len(rest)]
            elif rest:
                on_sentence(splitter.emotion, rest)
                spoke = True
            stopwatch.mark("llm_done")
        except OpenAIError as error:
            log.error("Local LLM error: %s", error)

        match = EMOTION_TAG.match(reply)
        text = reply[match.end():].strip() if match else reply.strip()
        if not text:
            if not spoke:
                on_sentence("calm", FALLBACK_REPLY)
            return "calm", FALLBACK_REPLY

        self.store.add_messages(source, ("user", user_text[:200]), ("bot", text[:300]))
        return splitter.emotion, text

    def summarize(self, text: str) -> str:
        text = speakable(text)
        if len(text) <= config.SUMMARY_MIN_CHARS:
            return text
        try:
            summary = self.llm.complete(
                [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": f"Reminder: {text}\nSentence:"}],
                max_tokens=48,
                temperature=0.3,
            )
        except OpenAIError as error:
            log.warning("Ping summary failed: %s", error)
            return text
        summary = EMOTION_TAG.sub("", summary.removeprefix("Sentence:").strip().removeprefix("Reminder:")).strip().strip('"')
        log.info("Ping summary: %r", summary)
        # A summary without end punctuation ran out of tokens mid-sentence.
        return speakable(summary) if summary.endswith((".", "!", "?")) else text

    def warm_cache(self, user_text: str | None = None) -> None:
        """In the background, pin any fact from user_text, then pre-evaluate the next prompt."""
        threading.Thread(target=self._warm, args=(user_text,), daemon=True).start()

    def _warm(self, user_text: str | None) -> None:
        try:
            if config.MEMORY_PINNING and user_text:
                pin_fact(self.llm, self.store, user_text)
            if config.LLM_CACHE_WARMUP:
                self.llm.complete(build_messages(self.store.load(), ""), max_tokens=1)
        except OpenAIError as error:
            log.warning("LLM cache warmup failed: %s", error)
