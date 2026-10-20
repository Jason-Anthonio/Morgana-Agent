from contextlib import contextmanager
from types import SimpleNamespace

import requests

from morgana.api import start_api
from morgana.llm import Conversation, SentenceSplitter
from morgana.memory import MemoryStore, history_window, is_pin_candidate, pin_fact
from morgana.state import AgentStatus, Stopwatch
from morgana.text import speakable


def test_sentence_splitter_reads_emotion_tag_across_chunks():
    splitter = SentenceSplitter()
    sentences = []
    for chunk in ["[ha", "ppy] Hello", " there. How", " are you?"]:
        sentences += splitter.feed(chunk)
    assert splitter.emotion == "happy"
    assert sentences == ["Hello there.", "How are you?"]


def test_sentence_splitter_cuts_clauses_when_audio_runs_low():
    text = "I went to the market, and then I came home "
    assert SentenceSplitter(lambda: False).feed(text) == []
    assert SentenceSplitter(lambda: True).feed(text) == ["I went to the market,"]


def test_history_window_moves_in_steps():
    memory = {"conversation_history": [{"text": f"m{i}"} for i in range(13)]}
    assert history_window(memory)[0]["text"] == "m0"
    memory["conversation_history"].append({"text": "m13"})
    memory["conversation_history"].append({"text": "m14"})
    memory["conversation_history"].append({"text": "m15"})
    assert history_window(memory)[0]["text"] == "m8"


def test_pin_fact_only_pins_grounded_personal_facts(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    llm = SimpleNamespace(complete=lambda *args, **kwargs: "The user studies computer engineering.")
    assert not is_pin_candidate("what is my name")
    assert not is_pin_candidate("I'm so tired")
    assert pin_fact(llm, store, "I study computer engineering") == "The user studies computer engineering."
    assert pin_fact(llm, store, "I study computer engineering") is None  # duplicate
    assert len(store.load()["pinned_facts"]) == 1


PING = (
    "😎 Looking good, Phantom Thief! Quick check-in, partner!\n\n"
    "💧 Hydration check! Have you had water in the last hour?\n\n"
    "_Reply with how you're doing, or just 'skip' to move on!_"
)


def test_speakable_drops_emojis_and_markdown():
    assert speakable("🧘‍♀️ Stretch *report*!") == "Stretch report!"
    assert speakable("👍🏽") == ""
    assert speakable(PING) == (
        "Looking good, Phantom Thief! Quick check-in, partner! Hydration check! "
        "Have you had water in the last hour? Reply with how you're doing, or just 'skip' to move on!"
    )


def test_summarize_shortens_pings(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    llm = SimpleNamespace(complete=lambda *args, **kwargs: "[calm] 💧 Drink some water, partner!")
    assert Conversation(llm, store).summarize(PING) == "Drink some water, partner!"
    assert Conversation(llm, store).summarize("🚶 Take a walk!") == "Take a walk!"  # short: no LLM
    cut_off = SimpleNamespace(complete=lambda *args, **kwargs: "Drink some water and")
    assert Conversation(cut_off, store).summarize(PING) == speakable(PING)


def test_api(tmp_path):
    @contextmanager
    def turn(label):
        yield Stopwatch(label)

    assistant = SimpleNamespace(
        status=AgentStatus(),
        conversation=SimpleNamespace(store=MemoryStore(tmp_path / "memory.json"), summarize=lambda text: "Drink."),
        turn=turn,
        reply=lambda text, source, stopwatch: ("happy", f"You said {text}"),
        announce=lambda text, emotion, stopwatch: None,
    )
    server = start_api(assistant, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_port}/api"
    try:
        assert requests.get(f"{url}/status").json()["agent_state"] == "IDLE"
        assert requests.post(f"{url}/chat", json={"text": "hi"}).json() == {"reply": "You said hi", "emotion": "happy"}
        assert requests.post(f"{url}/tts", json={"text": "💧 Drink water"}).json()["text"] == "Drink."
        assert requests.post(f"{url}/chat", json={"text": ""}).status_code == 400
        assert requests.get(f"{url}/nope").status_code == 404
    finally:
        server.shutdown()
