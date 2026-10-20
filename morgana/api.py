"""Local HTTP API so the Telegram wellness bot can use the voice assistant.

    GET  /api/health  liveness check
    GET  /api/status  agent mode, emotion and wellness ping count
    POST /api/tts     speak {"text", "emotion"} out loud
    POST /api/chat    answer {"text"} out loud; returns {"reply", "emotion"}
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from morgana import config
from morgana.memory import read_ping_count
from morgana.state import EMOTIONS

log = logging.getLogger(__name__)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            self.send_json(200, {"status": "ok", "port": self.server.server_port})
        elif self.path == "/api/status":
            status = self.server.assistant.status
            self.send_json(200, {
                "agent_state": status.mode.value,
                "current_emotion": status.emotion,
                "pings_sent": read_ping_count(config.WELLNESS_STATE_PATH),
                "ok": True,
            })
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/api/tts", "/api/chat"):
            self.send_json(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            text = body["text"].strip()
            if not text:
                raise ValueError("empty text")
        except (ValueError, KeyError, TypeError, AttributeError):
            self.send_json(400, {"error": "JSON body with a non-empty text field required"})
            return

        assistant = self.server.assistant
        try:
            if self.path == "/api/tts":
                emotion = body.get("emotion") if body.get("emotion") in EMOTIONS else "calm"
                with assistant.turn("wellness_ping") as stopwatch:
                    spoken = assistant.conversation.summarize(text)
                    stopwatch.mark("summarized")
                    assistant.conversation.store.add_messages("wellness_ping", ("bot", spoken[:300]))
                    assistant.announce(spoken, emotion, stopwatch)
                self.send_json(200, {"spoken": True, "emotion": emotion, "text": spoken})
            else:
                with assistant.turn("telegram") as stopwatch:
                    emotion, reply = assistant.reply(text, "telegram", stopwatch)
                self.send_json(200, {"reply": reply, "emotion": emotion})
        except Exception as error:
            log.exception("API request failed")
            self.send_json(500, {"error": str(error)})

    def send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # keep the console for the assistant's own logs


def start_api(assistant, host=config.API_HOST, port=config.API_PORT) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.assistant = assistant
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("HTTP API listening on %s:%s", host, port)
    return server
