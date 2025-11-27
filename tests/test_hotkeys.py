from pynput import keyboard

from sheptun.hotkeys import format_hotkey_display, parse_hotkey


class TestParseHotkey:
    def test_parse_single_key(self) -> None:
        result = parse_hotkey("s")
        assert result is not None
        assert keyboard.KeyCode.from_char("s") in result

    def test_parse_cmd_modifier(self) -> None:
        result = parse_hotkey("<cmd>+s")
        assert result is not None
        assert keyboard.Key.cmd in result
        assert keyboard.KeyCode.from_char("s") in result

    def test_parse_ctrl_modifier(self) -> None:
        result = parse_hotkey("<ctrl>+s")
        assert result is not None
        assert keyboard.Key.ctrl in result

    def test_parse_alt_modifier(self) -> None:
        result = parse_hotkey("<alt>+s")
        assert result is not None
        assert keyboard.Key.alt in result

    def test_parse_shift_modifier(self) -> None:
        result = parse_hotkey("<shift>+s")
        assert result is not None
        assert keyboard.Key.shift in result

    def test_parse_space_key(self) -> None:
        result = parse_hotkey("<space>")
        assert result is not None
        assert keyboard.Key.space in result

    def test_parse_multiple_modifiers(self) -> None:
        result = parse_hotkey("<ctrl>+<alt>+s")
        assert result is not None
        assert keyboard.Key.ctrl in result
        assert keyboard.Key.alt in result
        assert keyboard.KeyCode.from_char("s") in result

    def test_parse_empty_string(self) -> None:
        result = parse_hotkey("")
        assert result is None

    def test_parse_unknown_key(self) -> None:
        result = parse_hotkey("<unknown>")
        assert result is None

    def test_parse_case_insensitive(self) -> None:
        result = parse_hotkey("<CMD>+S")
        assert result is not None
        assert keyboard.Key.cmd in result


class TestFormatHotkeyDisplay:
    def test_format_cmd(self) -> None:
        assert "⌘" in format_hotkey_display("<cmd>+s")

    def test_format_shift(self) -> None:
        assert "⇧" in format_hotkey_display("<shift>+s")

    def test_format_ctrl(self) -> None:
        assert "⌃" in format_hotkey_display("<ctrl>+s")

    def test_format_alt(self) -> None:
        assert "⌥" in format_hotkey_display("<alt>+s")

    def test_format_space(self) -> None:
        assert "SPACE" in format_hotkey_display("<space>")

    def test_format_uppercase(self) -> None:
        result = format_hotkey_display("<ctrl>+<alt>+s")
        assert "S" in result

    def test_format_complex_hotkey(self) -> None:
        result = format_hotkey_display("<ctrl>+<alt>+s")
        assert "⌃" in result
        assert "⌥" in result
        assert "S" in result
