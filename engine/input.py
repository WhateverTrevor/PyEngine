"""Keyboard and mouse state.

Edge events (key/button presses, releases, wheel) accumulate until a fixed
update step consumes them, so behaviors see each press exactly once even when
the render rate and the fixed update rate diverge.
"""
from __future__ import annotations

import pygame


class InputManager:
    def __init__(self):
        self._pressed: set[int] = set()
        self._mouse_pressed: set[int] = set()
        self._mouse_released: set[int] = set()
        self.wheel = 0.0
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.captured = False

    def process(self, events) -> None:
        for e in events:
            if e.type == pygame.KEYDOWN:
                self._pressed.add(e.key)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                self._mouse_pressed.add(e.button)
            elif e.type == pygame.MOUSEBUTTONUP:
                self._mouse_released.add(e.button)
            elif e.type == pygame.MOUSEWHEEL:
                self.wheel += e.y
        dx, dy = pygame.mouse.get_rel()
        if self.captured:
            self.mouse_dx += dx
            self.mouse_dy += dy

    def consume_edges(self) -> None:
        """Called by the engine after the first fixed step of a frame."""
        self._pressed.clear()
        self._mouse_pressed.clear()
        self._mouse_released.clear()
        self.wheel = 0.0
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0

    def held(self, key: int) -> bool:
        return bool(pygame.key.get_pressed()[key])

    def pressed(self, key: int) -> bool:
        """True once per press."""
        return key in self._pressed

    def mouse_button_pressed(self, button: int = 1) -> bool:
        return button in self._mouse_pressed

    def mouse_button_released(self, button: int = 1) -> bool:
        return button in self._mouse_released

    def mouse_held(self, button: int = 1) -> bool:
        buttons = pygame.mouse.get_pressed(num_buttons=3)
        return bool(buttons[button - 1]) if 1 <= button <= 3 else False

    @property
    def mouse_pos(self) -> tuple[int, int]:
        return pygame.mouse.get_pos()

    def set_captured(self, flag: bool) -> None:
        if flag == self.captured:
            return
        self.captured = flag
        try:
            pygame.event.set_grab(flag)
            pygame.mouse.set_visible(not flag)
            pygame.mouse.get_rel()  # discard the warp jump
            self.mouse_dx = self.mouse_dy = 0.0
        except pygame.error:
            pass
