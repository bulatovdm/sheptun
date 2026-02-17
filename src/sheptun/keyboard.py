from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sheptun.remote import RemoteClient

import AppKit
import Quartz

logger = logging.getLogger("sheptun.keyboard")

NSData: Any = getattr(AppKit, "NSData")  # noqa: B009
NSPasteboard: Any = getattr(AppKit, "NSPasteboard")  # noqa: B009
NSPasteboardType: Any = getattr(AppKit, "NSPasteboardType")  # noqa: B009
NSPasteboardTypeString: Any = getattr(AppKit, "NSPasteboardTypeString")  # noqa: B009

TRANSIENT_PASTEBOARD_TYPE = "org.nspasteboard.TransientType"

CGEventCreateKeyboardEvent: Any = getattr(Quartz, "CGEventCreateKeyboardEvent")  # noqa: B009
CGEventKeyboardSetUnicodeString: Any = getattr(  # noqa: B009
    Quartz, "CGEventKeyboardSetUnicodeString"
)
CGEventPost: Any = getattr(Quartz, "CGEventPost")  # noqa: B009
CGEventSetFlags: Any = getattr(Quartz, "CGEventSetFlags")  # noqa: B009
CGEventSetIntegerValueField: Any = getattr(Quartz, "CGEventSetIntegerValueField")  # noqa: B009
CGEventSourceCreate: Any = getattr(Quartz, "CGEventSourceCreate")  # noqa: B009
CGEventSourceSetLocalEventsSuppressionInterval: Any = getattr(  # noqa: B009
    Quartz, "CGEventSourceSetLocalEventsSuppressionInterval"
)
kCGEventSourceStatePrivate: int = getattr(Quartz, "kCGEventSourceStatePrivate")  # noqa: B009
kCGEventSourceUserData: int = 42  # CGEventField for user data
kCGEventFlagMaskAlternate: int = getattr(Quartz, "kCGEventFlagMaskAlternate")  # noqa: B009
kCGEventFlagMaskCommand: int = getattr(Quartz, "kCGEventFlagMaskCommand")  # noqa: B009
kCGEventFlagMaskControl: int = getattr(Quartz, "kCGEventFlagMaskControl")  # noqa: B009
kCGEventFlagMaskShift: int = getattr(Quartz, "kCGEventFlagMaskShift")  # noqa: B009
kCGEventFlagMaskNonCoalesced: int = getattr(Quartz, "kCGEventFlagMaskNonCoalesced")  # noqa: B009
kCGAnnotatedSessionEventTap: Any = getattr(Quartz, "kCGAnnotatedSessionEventTap")  # noqa: B009
kCGHIDEventTap: Any = getattr(Quartz, "kCGHIDEventTap")  # noqa: B009

MAX_UNICODE_STRING_LENGTH = 20
SHEPTUN_EVENT_MARKER = 0x534850  # "SHP" — unique marker for our events


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
    def __init__(self, key_delay: float | None = None, use_clipboard: bool = False) -> None:
        from sheptun.settings import settings

        self._key_delay = key_delay if key_delay is not None else settings.key_delay
        self._use_clipboard = use_clipboard
        self._pasteboard = NSPasteboard.generalPasteboard()
        self._event_source = CGEventSourceCreate(kCGEventSourceStatePrivate)
        CGEventSourceSetLocalEventsSuppressionInterval(self._event_source, 0.0)

    def start_capture(self) -> None:
        pass

    def end_capture(self) -> None:
        pass

    def send_text(self, text: str) -> None:
        if self._use_clipboard:
            self._send_via_clipboard(text)
        else:
            self._send_via_events(text)

    def _send_via_clipboard(self, text: str) -> None:
        logger.debug(f"Sending text via clipboard: '{text}' (len={len(text)})")
        old_contents = self._get_clipboard()
        old_change_count = self._pasteboard.changeCount()

        self._set_clipboard(text)
        time.sleep(0.05)
        self._paste()
        time.sleep(0.05)

        if old_contents is not None:
            self._restore_clipboard(old_contents)
        elif self._pasteboard.changeCount() != old_change_count:
            self._pasteboard.clearContents()
        logger.debug("Clipboard send complete")

    def _get_clipboard(self) -> str | None:
        result: str | None = self._pasteboard.stringForType_(NSPasteboardTypeString)
        return result

    def _set_clipboard(self, text: str) -> None:
        self._pasteboard.clearContents()
        self._pasteboard.setString_forType_(text, NSPasteboardTypeString)
        self._pasteboard.setData_forType_(
            NSData.data(), NSPasteboardType(TRANSIENT_PASTEBOARD_TYPE)
        )

    def _restore_clipboard(self, text: str) -> None:
        self._pasteboard.clearContents()
        self._pasteboard.setString_forType_(text, NSPasteboardTypeString)

    def _paste(self) -> None:
        self._send_key_event(KEY_CODES["v"].code, kCGEventFlagMaskCommand)

    def has_text_before_cursor(self) -> bool:
        return _get_cursor_position() > 0

    def get_cursor_position(self) -> int:
        return _get_cursor_position()

    def _send_via_events(self, text: str) -> None:
        logger.debug(f"Sending text via events: '{text}' (len={len(text)})")
        chunks: list[str] = []
        for i in range(0, len(text), MAX_UNICODE_STRING_LENGTH):
            chunk = text[i : i + MAX_UNICODE_STRING_LENGTH]
            chunks.append(chunk)
            self._send_unicode_string(chunk)
            if i + MAX_UNICODE_STRING_LENGTH < len(text):
                time.sleep(self._key_delay)
        logger.debug(f"Events send complete: {len(chunks)} chunks: {chunks}")

    def _send_unicode_string(self, text: str) -> None:
        key_down = CGEventCreateKeyboardEvent(self._event_source, 0, True)
        CGEventKeyboardSetUnicodeString(key_down, len(text), text)
        CGEventSetFlags(key_down, kCGEventFlagMaskNonCoalesced)
        CGEventSetIntegerValueField(key_down, kCGEventSourceUserData, SHEPTUN_EVENT_MARKER)
        CGEventPost(kCGAnnotatedSessionEventTap, key_down)

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
    ) -> None:
        from sheptun.focus import FocusAwareTextBuffer, FocusTracker

        self._keyboard = keyboard_sender or MacOSKeyboardSender()
        self._focus_tracker = FocusTracker()
        self._text_buffer = FocusAwareTextBuffer(
            send_text_callback=self._keyboard.send_text,
            focus_tracker=self._focus_tracker,
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

    def has_text_before_cursor(self) -> bool:
        return self._keyboard.has_text_before_cursor()

    def get_cursor_position(self) -> int:
        return self._keyboard.get_cursor_position()


class RemoteAwareKeyboardSender:
    """Routes keyboard actions to local or remote machine based on cursor position."""

    def __init__(
        self,
        local_sender: MacOSKeyboardSender,
        remote_client: RemoteClient,
        auto_detect: bool = True,
    ) -> None:
        from sheptun.remote import is_cursor_on_local_screen

        self._local = local_sender
        self._remote = remote_client
        self._auto_detect = auto_detect
        self._is_cursor_on_local_screen = is_cursor_on_local_screen
        self._force_remote = False

    @property
    def force_remote(self) -> bool:
        return self._force_remote

    @force_remote.setter
    def force_remote(self, value: bool) -> None:
        self._force_remote = value

    def _is_remote(self) -> bool:
        if self._force_remote:
            return True
        if self._auto_detect:
            return not self._is_cursor_on_local_screen()
        return False

    def send_text(self, text: str) -> None:
        if self._is_remote() and self._remote.send_text(text):
            return
        self._local.send_text(text)

    def send_key(self, key: str) -> None:
        if self._is_remote() and self._remote.send_key(key):
            return
        self._local.send_key(key)

    def send_hotkey(self, keys: list[str]) -> None:
        if self._is_remote() and self._remote.send_hotkey(keys):
            return
        self._local.send_hotkey(keys)

    def start_capture(self) -> None:
        self._local.start_capture()

    def end_capture(self) -> None:
        self._local.end_capture()

    def has_text_before_cursor(self) -> bool:
        return self._local.has_text_before_cursor()

    def get_cursor_position(self) -> int:
        return self._local.get_cursor_position()


def _get_cursor_position() -> int:
    try:
        import ApplicationServices as AS  # type: ignore[import-not-found,import-untyped]
        import CoreFoundation as CF  # type: ignore[import-not-found,import-untyped]

        AXUIElementCopyAttributeValue: Any = getattr(AS, "AXUIElementCopyAttributeValue")  # noqa: B009
        AXUIElementCreateSystemWide: Any = getattr(AS, "AXUIElementCreateSystemWide")  # noqa: B009
        kAXFocusedUIElementAttribute: Any = getattr(AS, "kAXFocusedUIElementAttribute")  # noqa: B009
        kAXSelectedTextRangeAttribute: Any = getattr(AS, "kAXSelectedTextRangeAttribute")  # noqa: B009
        CFRange: Any = getattr(CF, "CFRange")  # noqa: B009

        system_wide: Any = AXUIElementCreateSystemWide()
        err: Any
        focused_element: Any
        err, focused_element = AXUIElementCopyAttributeValue(
            system_wide, kAXFocusedUIElementAttribute, None
        )
        if err != 0 or focused_element is None:
            logger.debug(f"_get_cursor_position: no focused element (err={err})")
            return -1

        selection_range: Any
        err, selection_range = AXUIElementCopyAttributeValue(
            focused_element, kAXSelectedTextRangeAttribute, None
        )
        if err != 0 or selection_range is None:
            logger.debug(f"_get_cursor_position: no selection range (err={err})")
            return -1

        location: int
        if isinstance(selection_range, CFRange):
            location = int(selection_range.location)
        elif hasattr(selection_range, "rangeValue"):
            location = int(selection_range.rangeValue().location)
        elif hasattr(selection_range, "location"):
            location = int(selection_range.location)
        else:
            logger.debug(f"_get_cursor_position: unknown range type {type(selection_range)}")
            return -1

        logger.debug(f"_get_cursor_position: location={location}")
        return location
    except Exception as e:
        logger.debug(f"_get_cursor_position: exception {e}")
        return -1
