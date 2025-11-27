import subprocess
import time
from dataclasses import dataclass
from typing import Any

import Quartz

CGEventCreateKeyboardEvent: Any = getattr(Quartz, "CGEventCreateKeyboardEvent")  # noqa: B009
CGEventPost: Any = getattr(Quartz, "CGEventPost")  # noqa: B009
CGEventSetFlags: Any = getattr(Quartz, "CGEventSetFlags")  # noqa: B009
kCGEventFlagMaskAlternate: int = getattr(Quartz, "kCGEventFlagMaskAlternate")  # noqa: B009
kCGEventFlagMaskCommand: int = getattr(Quartz, "kCGEventFlagMaskCommand")  # noqa: B009
kCGEventFlagMaskControl: int = getattr(Quartz, "kCGEventFlagMaskControl")  # noqa: B009
kCGEventFlagMaskShift: int = getattr(Quartz, "kCGEventFlagMaskShift")  # noqa: B009
kCGHIDEventTap: Any = getattr(Quartz, "kCGHIDEventTap")  # noqa: B009


@dataclass(frozen=True, slots=True)
class KeyCode:
    code: int
    needs_shift: bool = False


KEY_CODES: dict[str, KeyCode] = {
    "a": KeyCode(0),
    "b": KeyCode(11),
    "c": KeyCode(8),
    "d": KeyCode(2),
    "e": KeyCode(14),
    "f": KeyCode(3),
    "g": KeyCode(5),
    "h": KeyCode(4),
    "i": KeyCode(34),
    "j": KeyCode(38),
    "k": KeyCode(40),
    "l": KeyCode(37),
    "m": KeyCode(46),
    "n": KeyCode(45),
    "o": KeyCode(31),
    "p": KeyCode(35),
    "q": KeyCode(12),
    "r": KeyCode(15),
    "s": KeyCode(1),
    "t": KeyCode(17),
    "u": KeyCode(32),
    "v": KeyCode(9),
    "w": KeyCode(13),
    "x": KeyCode(7),
    "y": KeyCode(16),
    "z": KeyCode(6),
    "0": KeyCode(29),
    "1": KeyCode(18),
    "2": KeyCode(19),
    "3": KeyCode(20),
    "4": KeyCode(21),
    "5": KeyCode(23),
    "6": KeyCode(22),
    "7": KeyCode(26),
    "8": KeyCode(28),
    "9": KeyCode(25),
    "space": KeyCode(49),
    " ": KeyCode(49),
    "return": KeyCode(36),
    "enter": KeyCode(36),
    "tab": KeyCode(48),
    "delete": KeyCode(51),
    "backspace": KeyCode(51),
    "escape": KeyCode(53),
    "up": KeyCode(126),
    "down": KeyCode(125),
    "left": KeyCode(123),
    "right": KeyCode(124),
    "home": KeyCode(115),
    "end": KeyCode(119),
    "pageup": KeyCode(116),
    "pagedown": KeyCode(121),
    "f1": KeyCode(122),
    "f2": KeyCode(120),
    "f3": KeyCode(99),
    "f4": KeyCode(118),
    "f5": KeyCode(96),
    "f6": KeyCode(97),
    "f7": KeyCode(98),
    "f8": KeyCode(100),
    "f9": KeyCode(101),
    "f10": KeyCode(109),
    "f11": KeyCode(103),
    "f12": KeyCode(111),
    "-": KeyCode(27),
    "=": KeyCode(24),
    "[": KeyCode(33),
    "]": KeyCode(30),
    "\\": KeyCode(42),
    ";": KeyCode(41),
    "'": KeyCode(39),
    ",": KeyCode(43),
    ".": KeyCode(47),
    "/": KeyCode(44),
    "`": KeyCode(50),
    "!": KeyCode(18, needs_shift=True),
    "@": KeyCode(19, needs_shift=True),
    "#": KeyCode(20, needs_shift=True),
    "$": KeyCode(21, needs_shift=True),
    "%": KeyCode(23, needs_shift=True),
    "^": KeyCode(22, needs_shift=True),
    "&": KeyCode(26, needs_shift=True),
    "*": KeyCode(28, needs_shift=True),
    "(": KeyCode(25, needs_shift=True),
    ")": KeyCode(29, needs_shift=True),
    "_": KeyCode(27, needs_shift=True),
    "+": KeyCode(24, needs_shift=True),
    "{": KeyCode(33, needs_shift=True),
    "}": KeyCode(30, needs_shift=True),
    "|": KeyCode(42, needs_shift=True),
    ":": KeyCode(41, needs_shift=True),
    '"': KeyCode(39, needs_shift=True),
    "<": KeyCode(43, needs_shift=True),
    ">": KeyCode(47, needs_shift=True),
    "?": KeyCode(44, needs_shift=True),
    "~": KeyCode(50, needs_shift=True),
}

MODIFIER_FLAGS: dict[str, int] = {
    "shift": kCGEventFlagMaskShift,
    "control": kCGEventFlagMaskControl,
    "ctrl": kCGEventFlagMaskControl,
    "command": kCGEventFlagMaskCommand,
    "cmd": kCGEventFlagMaskCommand,
    "option": kCGEventFlagMaskAlternate,
    "alt": kCGEventFlagMaskAlternate,
}


class MacOSKeyboardSender:
    def __init__(self, key_delay: float = 0.01) -> None:
        self._key_delay = key_delay

    def send_text(self, text: str) -> None:
        old_clipboard = self._get_clipboard()
        self._set_clipboard(text)
        self.send_hotkey(["command", "v"])
        time.sleep(0.05)
        if old_clipboard:
            self._set_clipboard(old_clipboard)

    def _get_clipboard(self) -> str:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout

    def _set_clipboard(self, text: str) -> None:
        subprocess.run(
            ["pbcopy"],
            input=text,
            text=True,
            check=False,
        )

    def send_key(self, key: str) -> None:
        key_lower = key.lower()
        key_code = KEY_CODES.get(key_lower)

        if key_code is None:
            return

        flags = kCGEventFlagMaskShift if key_code.needs_shift else 0
        self._send_key_event(key_code.code, flags)

    def send_hotkey(self, keys: list[str]) -> None:
        if not keys:
            return

        modifier_flags = 0
        target_key: str | None = None

        for key in keys:
            key_lower = key.lower()
            if key_lower in MODIFIER_FLAGS:
                modifier_flags |= MODIFIER_FLAGS[key_lower]
            else:
                target_key = key_lower

        if target_key is None:
            return

        key_code = KEY_CODES.get(target_key)
        if key_code is None:
            return

        self._send_key_event(key_code.code, modifier_flags)

    def _send_key_event(self, key_code: int, flags: int = 0) -> None:
        key_down = CGEventCreateKeyboardEvent(None, key_code, True)
        key_up = CGEventCreateKeyboardEvent(None, key_code, False)

        if flags:
            CGEventSetFlags(key_down, flags)
            CGEventSetFlags(key_up, flags)

        CGEventPost(kCGHIDEventTap, key_down)
        CGEventPost(kCGHIDEventTap, key_up)


class FocusAwareKeyboardSender:
    """Keyboard sender wrapper that tracks focus and waits for it to return."""

    def __init__(
        self,
        keyboard_sender: MacOSKeyboardSender | None = None,
        focus_timeout: float = 10.0,
    ) -> None:
        from sheptun.focus import FocusAwareTextBuffer, FocusTracker

        self._keyboard = keyboard_sender or MacOSKeyboardSender()
        self._focus_tracker = FocusTracker()
        self._text_buffer = FocusAwareTextBuffer(
            send_text_callback=self._keyboard.send_text,
            focus_tracker=self._focus_tracker,
            focus_timeout=focus_timeout,
        )

    def start_capture(self) -> None:
        """Start tracking the current focused app."""
        self._text_buffer.start_capture()

    def end_capture(self) -> None:
        """Stop tracking focus."""
        self._text_buffer.end_capture()

    def send_text(self, text: str) -> None:
        """Send text, waiting for focus if needed."""
        self._text_buffer.send_text(text)

    def send_key(self, key: str) -> None:
        """Send a key press."""
        self._keyboard.send_key(key)

    def send_hotkey(self, keys: list[str]) -> None:
        """Send a hotkey combination."""
        self._keyboard.send_hotkey(keys)
