# pyright: reportPrivateUsage=false
from unittest.mock import MagicMock, patch

from sheptun.focus import BufferedText, FocusAwareTextBuffer, FocusTracker


class TestBufferedText:
    def test_buffered_text_creation(self) -> None:
        text = BufferedText(text="hello")
        assert text.text == "hello"
        assert text.timestamp > 0


class TestFocusTracker:
    def test_capture_current_app(self) -> None:
        tracker = FocusTracker()
        with patch.object(tracker, "get_frontmost_app", return_value="com.example.app"):
            result = tracker.capture_current_app()
            assert result == "com.example.app"

    def test_is_same_app_focused_returns_true_when_same(self) -> None:
        tracker = FocusTracker()
        with patch.object(tracker, "get_frontmost_app", return_value="com.example.app"):
            tracker.capture_current_app()
            assert tracker.is_same_app_focused()

    def test_is_same_app_focused_returns_false_when_different(self) -> None:
        tracker = FocusTracker()
        tracker._current_app = "com.example.app"
        with patch.object(tracker, "get_frontmost_app", return_value="com.other.app"):
            assert not tracker.is_same_app_focused()

    def test_is_same_app_focused_returns_true_when_no_capture(self) -> None:
        tracker = FocusTracker()
        assert tracker.is_same_app_focused()

    def test_wait_for_app_focus_immediate_match(self) -> None:
        tracker = FocusTracker()
        with patch.object(tracker, "get_frontmost_app", return_value="com.example.app"):
            result = tracker.wait_for_app_focus("com.example.app", timeout=1.0)
            assert result

    def test_wait_for_app_focus_none_app_id(self) -> None:
        tracker = FocusTracker()
        result = tracker.wait_for_app_focus(None, timeout=1.0)
        assert result


class TestFocusAwareTextBuffer:
    def test_send_text_directly_when_no_capture(self) -> None:
        callback = MagicMock()
        buffer = FocusAwareTextBuffer(send_text_callback=callback)
        buffer.send_text("hello")
        callback.assert_called_once_with("hello")

    def test_send_text_directly_when_same_app_focused(self) -> None:
        callback = MagicMock()
        tracker = MagicMock()
        tracker.capture_current_app.return_value = "com.example.app"
        tracker.get_frontmost_app.return_value = "com.example.app"

        buffer = FocusAwareTextBuffer(
            send_text_callback=callback, focus_tracker=tracker
        )
        buffer.start_capture()
        buffer.send_text("hello")
        callback.assert_called_once_with("hello")

    def test_waits_when_focus_changed(self) -> None:
        callback = MagicMock()
        tracker = MagicMock()
        tracker.capture_current_app.return_value = "com.example.app"
        tracker.get_frontmost_app.return_value = "com.other.app"
        tracker.wait_for_app_focus.return_value = True

        buffer = FocusAwareTextBuffer(
            send_text_callback=callback, focus_tracker=tracker, focus_timeout=1.0
        )
        buffer.start_capture()
        buffer.send_text("hello")

        tracker.wait_for_app_focus.assert_called_once()
        callback.assert_called_once_with("hello")

    def test_no_send_on_timeout(self) -> None:
        callback = MagicMock()
        tracker = MagicMock()
        tracker.capture_current_app.return_value = "com.example.app"
        tracker.get_frontmost_app.return_value = "com.other.app"
        tracker.wait_for_app_focus.return_value = False

        buffer = FocusAwareTextBuffer(
            send_text_callback=callback, focus_tracker=tracker, focus_timeout=0.1
        )
        buffer.start_capture()
        buffer.send_text("hello")

        callback.assert_not_called()

    def test_end_capture_clears_target(self) -> None:
        callback = MagicMock()
        tracker = MagicMock()
        tracker.capture_current_app.return_value = "com.example.app"
        tracker.get_frontmost_app.return_value = "com.other.app"

        buffer = FocusAwareTextBuffer(
            send_text_callback=callback, focus_tracker=tracker
        )
        buffer.start_capture()
        buffer.end_capture()

        # After ending capture, should send directly
        buffer.send_text("hello")
        callback.assert_called_once_with("hello")
