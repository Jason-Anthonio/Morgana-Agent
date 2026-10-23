"""Runtime settings and personal .env file"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {"1", "true", "yes"}


def _path(name: str, default: Path) -> Path:
    path = Path(os.path.expanduser(os.environ.get(name, str(default))))
    return path if path.is_absolute() else PROJECT_ROOT / path


LOG_LEVEL = _str("LOG_LEVEL", "INFO")

# Local LLM (llama-server, OpenAI-compatible API)
LOCAL_LLM_HOST = _str("LOCAL_LLM_HOST", "127.0.0.1")
LOCAL_LLM_PORT = _int("LOCAL_LLM_PORT", 8080)
LOCAL_LLM_BASE_URL = _str("LOCAL_LLM_BASE_URL", f"http://{LOCAL_LLM_HOST}:{LOCAL_LLM_PORT}/v1")
LOCAL_LLM_MODEL = _str("LOCAL_LLM_MODEL", "morgana-local")
LOCAL_LLM_TIMEOUT = _float("LOCAL_LLM_TIMEOUT", 30)
LOCAL_LLM_MAX_TOKENS = _int("LOCAL_LLM_MAX_TOKENS", 64)
LLM_CACHE_WARMUP = _bool("LLM_CACHE_WARMUP", True)

# Conversation memory
WELLNESS_MEMORY_PATH = _path("WELLNESS_MEMORY_PATH", Path("~/.hermes/scripts/wellness_memory/memory.json"))
WELLNESS_STATE_PATH = _path("WELLNESS_STATE_PATH", Path("~/.hermes/scripts/wellness_state.json"))
MEMORY_CONTEXT_TURNS = _int("MEMORY_CONTEXT_TURNS", 8)
MEMORY_CONTEXT_STEP = max(1, _int("MEMORY_CONTEXT_STEP", 8))
MEMORY_CONTEXT_MAX_CHARS = _int("MEMORY_CONTEXT_MAX_CHARS", 6000)
MEMORY_PINNING = _bool("MEMORY_PINNING", True)
MEMORY_PIN_LIMIT = _int("MEMORY_PIN_LIMIT", 20)

# Microphone and wakeword
INPUT_DEVICE = _str("MORGANA_INPUT_DEVICE", "C270 HD WEBCAM")
INPUT_SAMPLE_RATE = 16000
INPUT_FRAMES_PER_BUFFER = 1280
WAKEWORD_MODEL_PATH = _path("WAKEWORD_MODEL_PATH", Path("models/wakeword.onnx"))
WAKEWORD_THRESHOLD = _float("WAKEWORD_THRESHOLD", 0.9)
WAKEWORD_COOLDOWN_SECONDS = _float("WAKEWORD_COOLDOWN_SECONDS", 1.5)

# TTS
PIPER_MODEL_PATH = _path("PIPER_MODEL_PATH", Path("models/v2.onnx"))
CUE_SOUND_PATH = _path("CUE_SOUND_PATH", Path("assets/sounds/got_it.wav"))
TTS_LENGTH_SCALE = _float("TTS_LENGTH_SCALE", 1.12)
TTS_VOLUME = _float("TTS_VOLUME", 1.5)
TTS_KEEP_LEAD_SECONDS = _float("TTS_KEEP_LEAD_SECONDS", 0.04)
TTS_KEEP_TAIL_SECONDS = _float("TTS_KEEP_TAIL_SECONDS", 0.12)
TTS_SECONDS_PER_CHAR = _float("TTS_SECONDS_PER_CHAR", 0.05)
CUE_MIXER_BUFFER = _int("CUE_MIXER_BUFFER", 4096)

# Split sentences into clauses
FIRST_CLAUSE_MIN_CHARS = _int("FIRST_CLAUSE_MIN_CHARS", 12)  # 0 disables clause splitting
CLAUSE_SPLIT_AHEAD_SECONDS = _float("CLAUSE_SPLIT_AHEAD_SECONDS", 2.0)
CLAUSE_SPLIT_WORDS = tuple(
    _str("CLAUSE_SPLIT_WORDS", "and|but|or|so|because|which|while|then|when|if").split("|")
)

# Wellness pings longer than this are summarized by the LLM before speaking
SUMMARY_MIN_CHARS = _int("SUMMARY_MIN_CHARS", 100)

# HTTP gateway for the Telegram wellness bot
API_HOST = _str("AGENT_API_HOST", "127.0.0.1")
API_PORT = _int("AGENT_API_PORT", 8643)

# Display
FACES_DIR = PROJECT_ROOT / "assets" / "faces"
SCREEN_SIZE = (800, 480)
FPS = 30
