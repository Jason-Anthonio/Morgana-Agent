"""Runs wakeword, listen, think, and speak"""

import logging
import time
from contextlib import contextmanager

import numpy as np
import pyaudio
import pygame
from openwakeword.model import Model as WakewordModel

from morgana import config
from morgana.audio import Microphone, Speaker, load_voice
from morgana.llm import Conversation
from morgana.state import AgentStatus, Mode, Stopwatch

log = logging.getLogger(__name__)


def clear_wakeword(wakeword: WakewordModel) -> None:
    silence = np.zeros(config.INPUT_FRAMES_PER_BUFFER, dtype=np.int16)
    for _ in range(40):  # 3.2s, one frame at a time like the microphone
        wakeword.predict(silence)
    wakeword.reset()


class Assistant:
    def __init__(self, conversation: Conversation, status: AgentStatus):
        self.conversation = conversation
        self.status = status
        self.voice = load_voice()
        pygame.mixer.init(buffer=config.CUE_MIXER_BUFFER)
        self.cue = pygame.mixer.Sound(config.CUE_SOUND_PATH) if config.CUE_SOUND_PATH.exists() else None

    @contextmanager
    def turn(self, label: str):
        """Wait for any other interaction to finish, then time this one."""
        with self.status.turn_lock:
            stopwatch = Stopwatch(label)
            yield stopwatch
            stopwatch.mark("complete")

    def reply(self, text: str, source: str, stopwatch: Stopwatch) -> tuple[str, str]:
        """Answer out loud, speaking each sentence as soon as it is generated."""
        with Speaker(self.voice, self.status, stopwatch) as speaker:
            result = self.conversation.stream_reply(
                text,
                source,
                on_sentence=speaker.say,
                want_clauses=lambda: speaker.seconds_queued() < config.CLAUSE_SPLIT_AHEAD_SECONDS,
                stopwatch=stopwatch,
            )
        self.conversation.warm_cache(text)
        return result

    def announce(self, text: str, emotion: str, stopwatch: Stopwatch) -> None:
        """Speak fixed text, such as a wellness reminder from the Telegram bot."""
        with Speaker(self.voice, self.status, stopwatch) as speaker:
            speaker.say(emotion, text)
        self.conversation.warm_cache()

    def run_voice_loop(self) -> None:
        """Wait for the wakeword, then listen and reply, until shutdown."""
        wakeword = WakewordModel(wakeword_model_paths=[str(config.WAKEWORD_MODEL_PATH)])
        audio = pyaudio.PyAudio()
        microphone = Microphone(audio)
        microphone.open()
        self.conversation.warm_cache()
        log.info("Ready. Waiting for the wakeword...")
        ignore_until = 0.0
        try:
            while self.status.running:
                scores = wakeword.predict(microphone.read_frame())
                if self.status.turn_lock.locked():
                    ignore_until = time.monotonic() + config.WAKEWORD_COOLDOWN_SECONDS
                if time.monotonic() < ignore_until:
                    continue
                if max(scores.values(), default=0) > config.WAKEWORD_THRESHOLD:
                    log.info("Wakeword detected")
                    microphone.close()
                    self.handle_voice_query(microphone)
                    clear_wakeword(wakeword)
                    microphone.open()
                    ignore_until = time.monotonic() + config.WAKEWORD_COOLDOWN_SECONDS
        except Exception:
            log.exception("Voice loop stopped")
        finally:
            microphone.close()
            audio.terminate()

    def handle_voice_query(self, microphone: Microphone) -> None:
        with self.turn("voice") as stopwatch:
            self.status.mode = Mode.LISTENING
            if self.cue:
                self.cue.play()
                time.sleep(self.cue.get_length())
            query = microphone.transcribe_query(stopwatch)
            self.status.mode = Mode.IDLE
            if query:
                self.reply(query, "voice", stopwatch)
