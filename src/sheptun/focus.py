"""Focus tracking and text buffering for macOS applications."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger("sheptun.focus")


@dataclass
class BufferedText:
    """Text waiting to be inserted when focus returns."""

    text: str
    timestamp: float = field(default_factory=time.time)


class FocusTracker:
    """Tracks the frontmost application on macOS using NSWorkspace."""

    def __init__(self) -> None:
        self._current_app: str | None = None
        self._lock = threading.Lock()

    def get_frontmost_app(self) -> str | None:
        """Get the bundle identifier of the frontmost application."""
        try:
            import AppKit

            NSWorkspace = getattr(AppKit, "NSWorkspace")  # noqa: B009
            workspace = NSWorkspace.sharedWorkspace()
            frontmost = workspace.frontmostApplication()
            if frontmost:
                return str(frontmost.bundleIdentifier())
            return None
        except ImportError:
            logger.warning("AppKit not available, cannot track focus")
            return None
        except Exception as e:
            logger.warning(f"Failed to get frontmost app: {e}")
            return None

    def capture_current_app(self) -> str | None:
        """Capture and store the current frontmost application."""
        with self._lock:
            self._current_app = self.get_frontmost_app()
            logger.debug(f"Captured current app: {self._current_app}")
            return self._current_app

    def is_same_app_focused(self) -> bool:
        """Check if the same app that was captured is still focused."""
        with self._lock:
            if self._current_app is None:
                return True
            current = self.get_frontmost_app()
            return current == self._current_app

    def wait_for_app_focus(
        self,
        app_bundle_id: str | None,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> bool:
        """Wait until the specified app gains focus or timeout occurs."""
        if app_bundle_id is None:
            return True

        start_time = time.time()
        while time.time() - start_time < timeout:
            current = self.get_frontmost_app()
            if current == app_bundle_id:
                return True
            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for {app_bundle_id} to gain focus")
        return False


class FocusAwareTextBuffer:
    """Buffers text and inserts it when the target app regains focus."""

    def __init__(
        self,
        send_text_callback: Callable[[str], None],
        focus_tracker: FocusTracker | None = None,
        focus_timeout: float = 10.0,
    ) -> None:
        self._send_text = send_text_callback
        self._focus_tracker = focus_tracker or FocusTracker()
        self._focus_timeout = focus_timeout
        self._buffer: list[BufferedText] = []
        self._lock = threading.Lock()
        self._target_app: str | None = None

    def start_capture(self) -> None:
        """Start a capture session - records the current focused app."""
        self._target_app = self._focus_tracker.capture_current_app()
        logger.debug(f"Started capture, target app: {self._target_app}")

    def send_text(self, text: str) -> None:
        """Send text, buffering if focus has changed."""
        if self._target_app is None:
            # No capture session, send directly
            self._send_text(text)
            return

        current_app = self._focus_tracker.get_frontmost_app()

        if current_app == self._target_app:
            # Same app focused, send directly
            logger.debug("Target app focused, sending text directly")
            self._send_text(text)
        else:
            # Focus changed, wait for it to return then send
            logger.info(
                f"Focus changed from {self._target_app} to {current_app}, "
                f"waiting for focus to return before sending text"
            )
            self._wait_and_send(text)

    def _wait_and_send(self, text: str) -> None:
        """Wait for focus to return and then send text."""
        if self._focus_tracker.wait_for_app_focus(
            self._target_app, timeout=self._focus_timeout
        ):
            # Small delay to ensure the app is ready to receive input
            time.sleep(0.1)
            logger.debug("Focus returned, sending buffered text")
            self._send_text(text)
        else:
            logger.warning(
                f"Timeout waiting for focus, text not sent: {text[:50]}..."
                if len(text) > 50
                else f"Timeout waiting for focus, text not sent: {text}"
            )

    def end_capture(self) -> None:
        """End the capture session."""
        self._target_app = None
        logger.debug("Ended capture session")
