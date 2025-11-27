from sheptun.keyboard import KEY_CODES, MODIFIER_FLAGS, KeyCode


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
        shifted_chars = "!@#$%^&*()_+{}|:\"<>?~"
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
