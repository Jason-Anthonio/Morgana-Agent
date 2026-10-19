"""Talks to the local model and turns its streamed reply into speakable sentences."""

import logging
import re
import threading

from openai import OpenAI, OpenAIError

from morgana import config
from morgana.memory import MemoryStore, build_messages, pin_fact
from morgana.state import EMOTIONS, Stopwatch

log = logging.getLogger(__name__)

FALLBACK_REPLY = "I am currently unable to connect to my local processing core."
EMOTION_TAG = re.compile(rf"^\s*\[({'|'.join(EMOTIONS)})\]\s*", re.IGNORECASE)
SENTENCE_END = re.compile(r"^(.+?[.!?])(?:\s+|$)", re.DOTALL)


class LocalLLM:
    """llama-server through its OpenAI-compatible API."""

    def __init__(self):
        # llama-server ignores the key, but the client requires one.
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


class SentenceSplitter:
    """Buffers streamed text and returns each sentence once it is complete.

    The reply starts with an [emotion] tag, which is stripped and kept in
    `emotion`. While `want_clauses()` is true (little audio left to play), long
    sentences are also cut at commas or conjunctions so speech starts sooner.
    """

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
                return []  # might still become a tag
            # The small model often skips the tag; waiting for one would hold
            # back every sentence until generation ends.
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
        """Stream a reply, calling on_sentence(emotion, text) for each sentence.

        Returns (emotion, full reply). Both turns are saved to memory.
        """
        log.info("Prompting local LLM (%s): %r", source, user_text)
        splitter = SentenceSplitter(want_clauses)
        reply = ""
        spoke = False
        try:
            stopwatch.mark("llm_request_started")
            messages = build_messages(self.store.load(), user_text)
            for piece in self.llm.stream(messages, max_tokens=config.LOCAL_LLM_MAX_TOKENS, temperature=0.7):
                if not reply:
                    stopwatch.mark("llm_first_token")
                reply += piece
                for sentence in splitter.feed(piece):
                    on_sentence(splitter.emotion, sentence)
                    spoke = True
            if rest := splitter.flush():
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

    def warm_cache(self, user_text: str | None = None) -> None:
        """In the background, pin any fact from user_text, then pre-evaluate the next prompt.

        The next query then reuses llama-server's KV cache, so cold starts,
        history window jumps and new pinned facts are paid for now instead of
        during a reply. Both run in one thread because the server has a single
        slot and the pin request evicts the cached prefix.
        """
        threading.Thread(target=self._warm, args=(user_text,), daemon=True).start()

    def _warm(self, user_text: str | None) -> None:
        try:
            if config.MEMORY_PINNING and user_text:
                pin_fact(self.llm, self.store, user_text)
            if config.LLM_CACHE_WARMUP:
                self.llm.complete(build_messages(self.store.load(), ""), max_tokens=1)
        except OpenAIError as error:
            log.warning("LLM cache warmup failed: %s", error)
