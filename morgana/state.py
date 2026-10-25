"""State shared by the audio, HTTP and display threads."""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)

EMOTIONS = ("calm", "happy", "sad", "angry")


class Mode(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"


@dataclass
class AgentStatus:
    mode: Mode = Mode.IDLE
    emotion: str = "calm"
    # perf_counter() (start, end) of speech in the playing and next audio chunk.
    voiced: list[tuple[float, float]] = field(default_factory=list)
    shutdown: threading.Event = field(default_factory=threading.Event)
    # Held for a whole interaction so voice, Telegram and pings never talk over each other.
    turn_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def running(self) -> bool:
        return not self.shutdown.is_set()


class Stopwatch:
    """Logs how long each stage of an interaction takes."""

    def __init__(self, label: str):
        self.label = label
        self.started = self.last = time.perf_counter()
        self.first_audio_played = False

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        log.info("[%s] +%.3fs %s (%.3fs)", self.label, now - self.started, stage, now - self.last)
        self.last = now
