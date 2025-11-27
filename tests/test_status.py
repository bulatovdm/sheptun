# pyright: reportPrivateUsage=false
from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from sheptun.status import (
    STATUS_LABELS,
    STATUS_STYLES,
    ConsoleStatusIndicator,
    SimpleStatusIndicator,
    Status,
)


class TestStatusEnum:
    def test_all_statuses_have_styles(self) -> None:
        for status in Status:
            assert status in STATUS_STYLES

    def test_all_statuses_have_labels(self) -> None:
        for status in Status:
            assert status in STATUS_LABELS

    def test_styles_are_tuples(self) -> None:
        for style in STATUS_STYLES.values():
            assert isinstance(style, tuple)
            assert len(style) == 2
            assert isinstance(style[0], str)  # style
            assert isinstance(style[1], str)  # icon


class TestSimpleStatusIndicator:
    def test_start_prints_message(self) -> None:
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        indicator = SimpleStatusIndicator(console=console)

        indicator.start()

        assert "Sheptun запущен" in output.getvalue()

    def test_stop_prints_message(self) -> None:
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        indicator = SimpleStatusIndicator(console=console)

        indicator.stop()

        assert "Sheptun остановлен" in output.getvalue()

    def test_listening_prints_message(self) -> None:
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        indicator = SimpleStatusIndicator(console=console)

        indicator.listening()

        assert "Слушаю" in output.getvalue()

    def test_processing_prints_message(self) -> None:
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        indicator = SimpleStatusIndicator(console=console)

        indicator.processing()

        assert "Обработка" in output.getvalue()

    def test_error_prints_message(self) -> None:
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        indicator = SimpleStatusIndicator(console=console)

        indicator.error("Test error")

        output_text = output.getvalue()
        assert "Ошибка" in output_text
        assert "Test error" in output_text

    def test_idle_does_nothing(self) -> None:
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        indicator = SimpleStatusIndicator(console=console)

        indicator.idle()

        assert output.getvalue() == ""

    def test_show_recognized_prints_text(self) -> None:
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        indicator = SimpleStatusIndicator(console=console)

        indicator.show_recognized("привет")

        output_text = output.getvalue()
        assert "Распознано" in output_text
        assert "привет" in output_text

    def test_show_action_prints_description(self) -> None:
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        indicator = SimpleStatusIndicator(console=console)

        indicator.show_action("Ввод текста")

        output_text = output.getvalue()
        assert "Действие" in output_text
        assert "Ввод текста" in output_text

    @patch("sheptun.status._send_notification")
    def test_show_help_prints_commands(self, mock_notify: MagicMock) -> None:
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        indicator = SimpleStatusIndicator(console=console)

        indicator.show_help()

        assert "Доступные команды" in output.getvalue()
        mock_notify.assert_called_once()


class TestConsoleStatusIndicator:
    def test_initial_status_is_idle(self) -> None:
        indicator = ConsoleStatusIndicator()
        assert indicator._status == Status.IDLE

    def test_listening_changes_status(self) -> None:
        indicator = ConsoleStatusIndicator()
        indicator.listening()
        assert indicator._status == Status.LISTENING

    def test_processing_changes_status(self) -> None:
        indicator = ConsoleStatusIndicator()
        indicator.processing()
        assert indicator._status == Status.PROCESSING

    def test_error_changes_status_and_message(self) -> None:
        indicator = ConsoleStatusIndicator()
        indicator.error("Test error")
        assert indicator._status == Status.ERROR
        assert indicator._message == "Test error"

    def test_idle_clears_message(self) -> None:
        indicator = ConsoleStatusIndicator()
        indicator.error("Test error")
        indicator.idle()
        assert indicator._status == Status.IDLE
        assert indicator._message == ""

    def test_render_returns_panel(self) -> None:
        from rich.panel import Panel

        indicator = ConsoleStatusIndicator()
        panel = indicator._render()
        assert isinstance(panel, Panel)

    def test_stop_clears_live(self) -> None:
        indicator = ConsoleStatusIndicator()
        indicator._live = MagicMock()
        indicator.stop()
        assert indicator._live is None
