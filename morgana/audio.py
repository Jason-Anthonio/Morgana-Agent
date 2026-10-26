"""Microphone input and Piper text-to-speech output."""

import logging
import queue
import threading
import time

import numpy as np
import onnxruntime
import pyaudio
import speech_recognition as sr
from piper.voice import PiperVoice, SynthesisConfig

from morgana import config
from morgana.state import AgentStatus, Mode, Stopwatch
from morgana.text import speakable

log = logging.getLogger(__name__)


class Microphone:
    """Reads frames for the wakeword model and records spoken queries.

    The frame stream must be closed while a query is recorded, because
    speech_recognition opens the same device.
    """

    def __init__(self, audio: pyaudio.PyAudio):
        self.audio = audio
        self.stream = None
        devices = [audio.get_device_info_by_index(i) for i in range(audio.get_device_count())]
        inputs = [d for d in devices if d["maxInputChannels"] > 0]
        wanted = config.INPUT_DEVICE
        matches = [
            d for d in inputs
            if (int(wanted) == d["index"] if wanted.isdigit() else wanted.casefold() in d["name"].casefold())
        ]
        if not matches:
            available = ", ".join(f"{d['index']}: {d['name']}" for d in inputs) or "none"
            raise RuntimeError(f"Microphone {wanted!r} not found. Available inputs: {available}")
        self.device_index = matches[0]["index"]
        log.info("Microphone: %s (index %s)", matches[0]["name"], self.device_index)

    def open(self) -> None:
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=config.INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=config.INPUT_FRAMES_PER_BUFFER,
            input_device_index=self.device_index,
        )
        # Drop audio buffered while the stream was opening.
        if stale := self.stream.get_read_available():
            self.stream.read(stale, exception_on_overflow=False)

    def close(self) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def read_frame(self) -> np.ndarray:
        pcm = self.stream.read(config.INPUT_FRAMES_PER_BUFFER, exception_on_overflow=False)
        return np.frombuffer(pcm, dtype=np.int16)

    def transcribe_query(self, stopwatch: Stopwatch) -> str:
        """Record one spoken query and return its transcript, or "" on failure."""
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 2.0
        try:
            with sr.Microphone(self.device_index, config.INPUT_SAMPLE_RATE, config.INPUT_FRAMES_PER_BUFFER) as source:
                log.info("Listening for a query...")
                recording = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            stopwatch.mark("transcription_started")
            text = recognizer.recognize_google(recording)
            stopwatch.mark("transcription_done")
            return text
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            log.info("No query heard")
        except (sr.RequestError, OSError) as error:
            log.error("Transcription failed: %s", error)
        return ""


def load_voice() -> PiperVoice:
    """Piper limited to TTS_THREADS cores, leaving the rest to llama-server.

    Both default to every core, and synthesizing one sentence while the model
    writes the next slowed generation from ~6 to ~1 token/s.
    """
    voice = PiperVoice.load(str(config.PIPER_MODEL_PATH))
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = config.TTS_THREADS
    options.inter_op_num_threads = 1
    voice.session = onnxruntime.InferenceSession(
        str(config.PIPER_MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return voice


def synthesize(voice: PiperVoice, text: str):
    """Yield int16 PCM chunks shaped for gapless, click-free playback.

    Piper's per-chunk normalization drives every clause to full volume, and each
    chunk starts with ~260ms of silence and can end mid-waveform. So use one
    fixed gain, trim the silence, and fade the edges instead.
    """
    rate = voice.config.sample_rate
    keep_lead = int(rate * config.TTS_KEEP_LEAD_SECONDS)
    keep_tail = int(rate * config.TTS_KEEP_TAIL_SECONDS)
    fade_in = np.linspace(0, 1, int(rate * 0.005), dtype=np.float32)
    fade_out = np.linspace(1, 0, int(rate * 0.015), dtype=np.float32)
    settings = SynthesisConfig(length_scale=config.TTS_LENGTH_SCALE, normalize_audio=False, volume=config.TTS_VOLUME)

    for chunk in voice.synthesize(text, syn_config=settings):
        audio = chunk.audio_float_array
        voiced = np.flatnonzero(np.abs(audio) > 0.01 * config.TTS_VOLUME)
        if voiced.size == 0:
            continue
        audio = audio[max(0, voiced[0] - keep_lead):voiced[-1] + 1 + keep_tail].copy()
        if len(audio) > len(fade_in) + len(fade_out):
            audio[:len(fade_in)] *= fade_in
            audio[-len(fade_out):] *= fade_out
        yield (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


class Speaker:
    """Speaks queued sentences, synthesizing the next one while the current one plays.

    Use as a context manager: say() sentences inside the block, and leaving the
    block waits until everything has been spoken.
    """

    def __init__(self, voice: PiperVoice, status: AgentStatus, stopwatch: Stopwatch):
        self.voice = voice
        self.status = status
        self.stopwatch = stopwatch
        self.rate = voice.config.sample_rate
        self.sentences = queue.Queue()
        self.chunks = queue.Queue()
        self.playing_until = 0.0
        self.queued_seconds = 0.0
        self.synthesizing_chars = 0
        self.threads = [threading.Thread(target=self._synthesize_loop), threading.Thread(target=self._play_loop)]

    def __enter__(self):
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *_):
        self.sentences.put(None)
        for thread in self.threads:
            thread.join()

    def say(self, emotion: str, text: str) -> None:
        if text := speakable(text):
            self.sentences.put((emotion, text))

    def seconds_queued(self) -> float:
        """Estimated speech still ahead of the speaker, counting text not yet synthesized."""
        pending_chars = self.synthesizing_chars + sum(len(item[1]) for item in list(self.sentences.queue) if item)
        return (
            max(0.0, self.playing_until - time.perf_counter())
            + self.queued_seconds
            + pending_chars * config.TTS_SECONDS_PER_CHAR
        )

    def _synthesize_loop(self) -> None:
        while (item := self.sentences.get()) is not None:
            emotion, text = item
            log.info("Morgana (%s): %s", emotion, text)
            self.synthesizing_chars = len(text)
            try:
                for pcm in synthesize(self.voice, text):
                    self.queued_seconds += len(pcm) / 2 / self.rate
                    self.chunks.put((emotion, pcm))
            except Exception:
                log.exception("TTS failed")
            self.synthesizing_chars = 0
        self.chunks.put(None)

    def _play_loop(self) -> None:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=self.rate, output=True, frames_per_buffer=1024)
        try:
            while self.status.running and (item := self.chunks.get()) is not None:
                emotion, pcm = item
                duration = len(pcm) / 2 / self.rate
                self.queued_seconds -= duration
                start = max(time.perf_counter(), self.playing_until)
                self.playing_until = start + duration
                speech = (start + config.TTS_KEEP_LEAD_SECONDS, self.playing_until - config.TTS_KEEP_TAIL_SECONDS)
                self.status.voiced = [*self.status.voiced[-1:], speech]
                self.status.emotion, self.status.mode = emotion, Mode.SPEAKING
                if not self.stopwatch.first_audio_played:
                    self.stopwatch.first_audio_played = True
                    self.stopwatch.mark("first_audio")  # before write(), which blocks until played
                stream.write(pcm)
            stream.write(bytes(2 * int(self.rate * config.TTS_END_PADDING_SECONDS)))
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            self.status.mode = Mode.IDLE
