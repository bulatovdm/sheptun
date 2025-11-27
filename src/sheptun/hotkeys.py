"""Global hotkey management for Sheptun."""

import logging
from collections.abc import Callable
from typing import Any

from pynput import keyboard

logger = logging.getLogger("sheptun.hotkeys")


def parse_hotkey(hotkey_str: str) -> set[Any] | None:
    """Parse hotkey string like '<ctrl>+<alt>+s' into set of keys."""
    if not hotkey_str:
        return None

    try:
        keys: set[Any] = set()
        for part in hotkey_str.split("+"):
            part = part.strip().lower()
            if part == "<cmd>":
                keys.add(keyboard.Key.cmd)
            elif part == "<shift>":
                keys.add(keyboard.Key.shift)
            elif part == "<ctrl>":
                keys.add(keyboard.Key.ctrl)
            elif part == "<alt>":
                keys.add(keyboard.Key.alt)
            elif part == "<space>":
                keys.add(keyboard.Key.space)
            elif len(part) == 1:
                keys.add(keyboard.KeyCode.from_char(part))
            else:
                logger.warning(f"Unknown hotkey part: {part}")
                return None
        return keys if keys else None
    except Exception as e:
        logger.warning(f"Failed to parse hotkey '{hotkey_str}': {e}")
        return None


def format_hotkey_display(hotkey_str: str) -> str:
    """Format hotkey string for display (e.g., '<ctrl>+<alt>+s' -> '⌃⌥S')."""
    return (
        hotkey_str.replace("<cmd>", "⌘")
        .replace("<shift>", "⇧")
        .replace("<ctrl>", "⌃")
        .replace("<alt>", "⌥")
        .replace("<space>", "Space")
        .upper()
    )


class HotkeyManager:
    """Manages global hotkeys for toggle and push-to-talk modes."""

    def __init__(
        self,
        toggle_hotkey: str | None = None,
        ptt_hotkey: str | None = None,
    ) -> None:
        self._toggle_keys = parse_hotkey(toggle_hotkey) if toggle_hotkey else None
        self._ptt_keys = parse_hotkey(ptt_hotkey) if ptt_hotkey else None

        self._toggle_hotkey_str = toggle_hotkey or ""
        self._ptt_hotkey_str = ptt_hotkey or ""

        self._pressed_keys: set[Any] = set()
        self._listener: keyboard.Listener | None = None

        self._toggle_active = False
        self._ptt_active = False

        self._on_toggle: Callable[[], None] | None = None
        self._on_ptt_start: Callable[[], None] | None = None
        self._on_ptt_stop: Callable[[], None] | None = None

    @property
    def toggle_hotkey_display(self) -> str:
        """Get formatted toggle hotkey for display."""
        return format_hotkey_display(self._toggle_hotkey_str) if self._toggle_hotkey_str else ""

    @property
    def ptt_hotkey_display(self) -> str:
        """Get formatted PTT hotkey for display."""
        return format_hotkey_display(self._ptt_hotkey_str) if self._ptt_hotkey_str else ""

    def set_callbacks(
        self,
        on_toggle: Callable[[], None] | None = None,
        on_ptt_start: Callable[[], None] | None = None,
        on_ptt_stop: Callable[[], None] | None = None,
    ) -> None:
        """Set callback functions for hotkey events."""
        self._on_toggle = on_toggle
        self._on_ptt_start = on_ptt_start
        self._on_ptt_stop = on_ptt_stop

    def start(self) -> None:
        """Start listening for hotkeys."""
        if self._listener is not None:
            return

        if not self._toggle_keys and not self._ptt_keys:
            logger.warning("No hotkeys configured, skipping listener")
            return

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

        hotkeys = []
        if self._toggle_keys:
            hotkeys.append(f"toggle={self._toggle_hotkey_str}")
        if self._ptt_keys:
            hotkeys.append(f"ptt={self._ptt_hotkey_str}")
        logger.info(f"Hotkey listener started: {', '.join(hotkeys)}")

    def stop(self) -> None:
        """Stop listening for hotkeys."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            logger.info("Hotkey listener stopped")

    def _on_press(self, key: Any) -> None:
        """Handle key press event."""
        self._pressed_keys.add(key)

        # Check toggle hotkey
        if (
            self._toggle_keys
            and self._toggle_keys.issubset(self._pressed_keys)
            and not self._toggle_active
        ):
            self._toggle_active = True
            logger.debug("Toggle hotkey pressed")
            if self._on_toggle:
                self._on_toggle()

        # Check PTT hotkey
        if (
            self._ptt_keys
            and self._ptt_keys.issubset(self._pressed_keys)
            and not self._ptt_active
        ):
            self._ptt_active = True
            logger.debug("PTT hotkey pressed")
            if self._on_ptt_start:
                self._on_ptt_start()

    def _on_release(self, key: Any) -> None:
        """Handle key release event."""
        self._pressed_keys.discard(key)

        # Reset toggle state when keys released
        if self._toggle_active and self._toggle_keys and not self._toggle_keys.issubset(self._pressed_keys):
            self._toggle_active = False

        # Handle PTT release
        if self._ptt_active and self._ptt_keys and not self._ptt_keys.issubset(self._pressed_keys):
            self._ptt_active = False
            logger.debug("PTT hotkey released")
            if self._on_ptt_stop:
                self._on_ptt_stop()
