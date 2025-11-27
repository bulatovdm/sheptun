from sheptun.i18n import MESSAGES, t


class TestI18n:
    def test_t_returns_translation(self) -> None:
        assert t("menu_toggle") == "Вкл/Выкл"

    def test_t_returns_key_if_not_found(self) -> None:
        assert t("unknown_key") == "unknown_key"

    def test_all_messages_are_strings(self) -> None:
        for key, value in MESSAGES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_required_keys_exist(self) -> None:
        required_keys = [
            "menu_toggle",
            "menu_toggle_stop",
            "menu_ptt",
            "menu_restart",
            "menu_quit",
            "notification_loading",
            "notification_error",
            "help_commands",
            "help_title",
        ]
        for key in required_keys:
            assert key in MESSAGES, f"Missing required key: {key}"
