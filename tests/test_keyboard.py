import threading
from unittest.mock import MagicMock, patch

import pytest

from sheptun.keyboard import (
    CONCEALED_PASTEBOARD_TYPE,
    KEY_CODES,
    MODIFIER_FLAGS,
    TRANSIENT_PASTEBOARD_TYPE,
    KeyCode,
    MacOSKeyboardSender,
    _run_on_main_sync,
)


class TestKeyCodes:
    def test_alphabet_keys_exist(self) -> None:
        for char in "abcdefghijklmnopqrstuvwxyz":
            assert char in KEY_CODES, f"Missing key: {char}"

    def test_digit_keys_exist(self) -> None:
        for digit in "0123456789":
            assert digit in KEY_CODES, f"Missing key: {digit}"

    def test_special_keys_exist(self) -> None:
        special_keys = [
            "space",
            "return",
            "enter",
            "tab",
            "delete",
            "backspace",
            "escape",
            "up",
            "down",
            "left",
            "right",
        ]
        for key in special_keys:
            assert key in KEY_CODES, f"Missing key: {key}"

    def test_function_keys_exist(self) -> None:
        for i in range(1, 13):
            key = f"f{i}"
            assert key in KEY_CODES, f"Missing key: {key}"

    def test_keycode_structure(self) -> None:
        for keycode in KEY_CODES.values():
            assert isinstance(keycode, KeyCode)
            assert isinstance(keycode.code, int)
            assert isinstance(keycode.needs_shift, bool)

    def test_shifted_keys_have_needs_shift(self) -> None:
        shifted_chars = '!@#$%^&*()_+{}|:"<>?~'
        for char in shifted_chars:
            if char in KEY_CODES:
                assert KEY_CODES[char].needs_shift, f"Key '{char}' should need shift"

    def test_lowercase_keys_no_shift(self) -> None:
        for char in "abcdefghijklmnopqrstuvwxyz":
            assert not KEY_CODES[char].needs_shift, f"Key '{char}' should not need shift"


class TestModifierFlags:
    def test_modifier_aliases(self) -> None:
        assert MODIFIER_FLAGS["ctrl"] == MODIFIER_FLAGS["control"]
        assert MODIFIER_FLAGS["cmd"] == MODIFIER_FLAGS["command"]
        assert MODIFIER_FLAGS["alt"] == MODIFIER_FLAGS["option"]

    def test_all_modifiers_exist(self) -> None:
        required = ["shift", "control", "ctrl", "command", "cmd", "option", "alt"]
        for mod in required:
            assert mod in MODIFIER_FLAGS, f"Missing modifier: {mod}"

    def test_modifier_flags_are_integers(self) -> None:
        for mod, flag in MODIFIER_FLAGS.items():
            assert isinstance(flag, int), f"Modifier '{mod}' flag should be int"


class TestRunOnMainSync:
    def test_calls_directly_on_main_thread(self) -> None:
        called = False

        def func() -> None:
            nonlocal called
            called = True

        assert threading.current_thread() is threading.main_thread()
        _run_on_main_sync(func)
        assert called

    def test_propagates_exception(self) -> None:
        def func() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            _run_on_main_sync(func)

    def test_fallback_without_runloop(self) -> None:
        called_on_thread: str | None = None

        def func() -> None:
            nonlocal called_on_thread
            called_on_thread = threading.current_thread().name

        with patch("sheptun.keyboard._has_running_runloop", return_value=False):
            t = threading.Thread(target=_run_on_main_sync, args=(func,), name="test-bg")
            t.start()
            t.join(timeout=5.0)

        assert called_on_thread == "test-bg"


class TestClipboardMarkers:
    def _make_sender_with_mock_pasteboard(self) -> tuple[MacOSKeyboardSender, MagicMock]:
        sender = MacOSKeyboardSender(use_clipboard=True)
        pasteboard = MagicMock()
        sender._pasteboard = pasteboard
        return sender, pasteboard

    def test_set_clipboard_marks_transient(self) -> None:
        sender, pasteboard = self._make_sender_with_mock_pasteboard()
        sender._set_clipboard("привет")
        marked_types = [call.args[1] for call in pasteboard.setData_forType_.call_args_list]
        assert any(str(t) == TRANSIENT_PASTEBOARD_TYPE for t in marked_types)

    def test_set_clipboard_marks_concealed(self) -> None:
        sender, pasteboard = self._make_sender_with_mock_pasteboard()
        sender._set_clipboard("привет")
        marked_types = [call.args[1] for call in pasteboard.setData_forType_.call_args_list]
        assert any(str(t) == CONCEALED_PASTEBOARD_TYPE for t in marked_types)

    def test_set_clipboard_writes_text(self) -> None:
        sender, pasteboard = self._make_sender_with_mock_pasteboard()
        sender._set_clipboard("привет")
        pasteboard.setString_forType_.assert_called_once()
        assert pasteboard.setString_forType_.call_args.args[0] == "привет"


class TestClipboardSnapshotRestore:
    def _make_sender(self) -> tuple[MacOSKeyboardSender, MagicMock]:
        sender = MacOSKeyboardSender(use_clipboard=True)
        pasteboard = MagicMock()
        sender._pasteboard = pasteboard
        return sender, pasteboard

    def _fake_item(self, type_to_data: dict[str, bytes]) -> MagicMock:
        item = MagicMock()
        item.types.return_value = list(type_to_data.keys())
        item.dataForType_.side_effect = lambda t: type_to_data.get(t)
        return item

    def test_snapshot_captures_all_types_of_all_items(self) -> None:
        sender, pasteboard = self._make_sender()
        pasteboard.pasteboardItems.return_value = [
            self._fake_item({"public.utf8-plain-text": b"hi", "public.html": b"<b>hi</b>"}),
            self._fake_item({"public.png": b"\x89PNG"}),
        ]
        snapshot = sender._snapshot_clipboard()
        assert snapshot == [
            [("public.utf8-plain-text", b"hi"), ("public.html", b"<b>hi</b>")],
            [("public.png", b"\x89PNG")],
        ]

    def test_snapshot_empty_when_no_items(self) -> None:
        sender, pasteboard = self._make_sender()
        pasteboard.pasteboardItems.return_value = None
        assert sender._snapshot_clipboard() == []

    def test_restore_writes_all_items_back(self) -> None:
        sender, pasteboard = self._make_sender()
        snapshot = [
            [("public.png", b"\x89PNG")],
            [("public.utf8-plain-text", b"hi")],
        ]
        sender._restore_clipboard(snapshot)
        pasteboard.clearContents.assert_called_once()
        pasteboard.writeObjects_.assert_called_once()
        written_items = pasteboard.writeObjects_.call_args.args[0]
        assert len(written_items) == 2

    def test_restore_noop_on_empty_snapshot(self) -> None:
        sender, pasteboard = self._make_sender()
        sender._restore_clipboard([])
        pasteboard.writeObjects_.assert_not_called()

    def test_image_in_clipboard_survives_send(self) -> None:
        sender, pasteboard = self._make_sender()
        pasteboard.pasteboardItems.return_value = [self._fake_item({"public.png": b"\x89PNG"})]
        pasteboard.changeCount.return_value = 1
        with (
            patch("sheptun.keyboard._run_on_main_sync", side_effect=lambda f: f()),
            patch.object(sender, "_paste"),
        ):
            sender._send_via_clipboard("голосовой текст")
        pasteboard.writeObjects_.assert_called_once()
        written_items = pasteboard.writeObjects_.call_args.args[0]
        assert len(written_items) == 1
