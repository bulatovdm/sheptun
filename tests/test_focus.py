# pyright: reportPrivateUsage=false
from unittest.mock import MagicMock, patch

from sheptun.focus import (
    FocusAwareTextBuffer,
    FocusState,
    FocusTracker,
)


class TestFocusState:
    def test_default_values(self) -> None:
        state = FocusState()
        assert state.app_bundle_id is None
        assert state.window_title is None

    def test_with_values(self) -> None:
        state = FocusState(app_bundle_id="com.example.app", window_title="My Window")
        assert state.app_bundle_id == "com.example.app"
        assert state.window_title == "My Window"


class TestFocusTracker:
    def test_capture_current_app(self) -> None:
        tracker = FocusTracker()
        state = FocusState(app_bundle_id="com.example.app", window_title="Test")
        with patch.object(tracker, "get_current_state", return_value=state):
            result = tracker.capture_current_app()
            assert result == "com.example.app"
            assert tracker._current_state.window_title == "Test"

    def test_is_same_focus_returns_true_when_same(self) -> None:
        tracker = FocusTracker()
        state = FocusState(app_bundle_id="com.example.app", window_title="Test")
        with patch.object(tracker, "get_current_state", return_value=state):
            tracker.capture_current_app()
            assert tracker.is_same_focus()

    def test_is_same_focus_returns_true_when_only_window_changed(self) -> None:
        tracker = FocusTracker()
        tracker._current_state = FocusState(
            app_bundle_id="com.example.app", window_title="Window 1"
        )
        new_state = FocusState(app_bundle_id="com.example.app", window_title="Window 2")
        with patch.object(tracker, "get_current_state", return_value=new_state):
            assert tracker.is_same_focus()

    def test_is_same_app_focused_returns_true_when_same(self) -> None:
        tracker = FocusTracker()
        tracker._current_state = FocusState(app_bundle_id="com.example.app")
        with patch.object(tracker, "get_frontmost_app", return_value="com.example.app"):
            assert tracker.is_same_app_focused()

    def test_is_same_app_focused_returns_false_when_different(self) -> None:
        tracker = FocusTracker()
        tracker._current_state = FocusState(app_bundle_id="com.example.app")
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

    def test_send_text_to_current_window(self) -> None:
        callback = MagicMock()
        tracker = MagicMock()
        tracker.get_current_state.return_value = FocusState(
            app_bundle_id="com.example.app", window_title="Window"
        )

        buffer = FocusAwareTextBuffer(
            send_text_callback=callback, focus_tracker=tracker
        )
        buffer.send_text("hello")
        callback.assert_called_once_with("hello")

    def test_send_text_after_app_changed(self) -> None:
        callback = MagicMock()
        tracker = MagicMock()
        tracker.get_current_state.return_value = FocusState(
            app_bundle_id="com.other.app", window_title="Different Window"
        )

        buffer = FocusAwareTextBuffer(
            send_text_callback=callback, focus_tracker=tracker
        )
        buffer.start_capture()
        buffer.send_text("hello")
        callback.assert_called_once_with("hello")
