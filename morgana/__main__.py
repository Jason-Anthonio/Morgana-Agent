"""Start Morgana: python -m morgana"""

import logging
import threading

from morgana import config
from morgana.api import start_api
from morgana.assistant import Assistant
from morgana.face import run_display
from morgana.llm import Conversation, LocalLLM
from morgana.memory import MemoryStore
from morgana.state import AgentStatus


def main():
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    status = AgentStatus()
    assistant = Assistant(Conversation(LocalLLM(), MemoryStore(config.WELLNESS_MEMORY_PATH)), status)
    api = start_api(assistant)
    threading.Thread(target=assistant.run_voice_loop, daemon=True).start()
    try:
        run_display(status)
    except KeyboardInterrupt:
        pass
    finally:
        status.shutdown.set()
        api.shutdown()


if __name__ == "__main__":
    main()
