"""Animated face shown on the display, driven by the agent's mode and emotion."""

import random
import time
from pathlib import Path

import pygame

from morgana import config
from morgana.state import AgentStatus, Mode

BLINK_INTERVAL = (5.0, 10.0)
BLINK_SECONDS = 0.15
MOUTH_FRAME_SECONDS = 0.1
SPEAKING_DIRS = {"calm": "speaking", "happy": "happy", "sad": "sad", "angry": "angry"}


def _load_image(path: Path) -> pygame.Surface:
    return pygame.transform.scale(pygame.image.load(path).convert_alpha(), config.SCREEN_SIZE)


def _load_frames(directory: Path) -> list[pygame.Surface]:
    return [_load_image(path) for path in sorted(directory.glob("*.png"))]


class Face:
    def __init__(self, faces_dir: Path = config.FACES_DIR):
        self.idle_open = _load_image(faces_dir / "idle" / "open.png")
        self.idle_closed = _load_image(faces_dir / "idle" / "closed.png")
        self.listening = _load_frames(faces_dir / "wakeword") or [self.idle_open]
        calm = _load_frames(faces_dir / SPEAKING_DIRS["calm"]) or [self.idle_open]
        self.speaking = {
            emotion: _load_frames(faces_dir / folder) or calm for emotion, folder in SPEAKING_DIRS.items()
        }
        self._listening_index = 0
        self._next_blink = time.monotonic() + random.uniform(*BLINK_INTERVAL)
        self._mouth_frame = self.idle_open
        self._mouth_changed_at = 0.0

    def frame(self, status: AgentStatus) -> pygame.Surface:
        now = time.monotonic()
        if status.mode is Mode.LISTENING:
            self._listening_index = (self._listening_index + 1) % len(self.listening)
            return self.listening[self._listening_index]
        if status.mode is Mode.SPEAKING:
            frames = self.speaking.get(status.emotion, self.speaking["calm"])
            clock = time.perf_counter()
            if not any(start <= clock < end for start, end in status.voiced):
                return frames[0] 
            if now - self._mouth_changed_at > MOUTH_FRAME_SECONDS:
                self._mouth_frame = random.choice(frames)
                self._mouth_changed_at = now
            return self._mouth_frame
        if now >= self._next_blink + BLINK_SECONDS:
            self._next_blink = now + random.uniform(*BLINK_INTERVAL)
        return self.idle_closed if now >= self._next_blink else self.idle_open


def run_display(status: AgentStatus) -> None:
    """Draw the face until the window is closed. Ctrl+` toggles fullscreen."""
    pygame.display.init()
    pygame.display.set_caption("Morgana")
    screen = pygame.display.set_mode(config.SCREEN_SIZE)
    face = Face()
    clock = pygame.time.Clock()
    fullscreen = False

    while status.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                status.shutdown.set()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKQUOTE and event.mod & pygame.KMOD_CTRL:
                fullscreen = not fullscreen
                screen = pygame.display.set_mode(config.SCREEN_SIZE, pygame.FULLSCREEN if fullscreen else 0)
        screen.blit(face.frame(status), (0, 0))
        pygame.display.flip()
        clock.tick(config.FPS)
    pygame.quit()
