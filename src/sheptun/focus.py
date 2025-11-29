import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("sheptun.focus")


@dataclass
class FocusState:
    app_bundle_id: str | None = None
    window_title: str | None = None


class FocusTracker:
    def __init__(self) -> None:
        self._current_state: FocusState = FocusState()
        self._lock = threading.Lock()

    def get_frontmost_app(self) -> str | None:
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

    def get_main_window_title(self, pid: int) -> str | None:
        try:
            import ApplicationServices  # type: ignore[import-untyped]

            AXUIElementCreateApplication = getattr(  # noqa: B009
                ApplicationServices, "AXUIElementCreateApplication"
            )
            AXUIElementCopyAttributeValue = getattr(  # noqa: B009
                ApplicationServices, "AXUIElementCopyAttributeValue"
            )
            kAXTitleAttribute = getattr(  # noqa: B009
                ApplicationServices, "kAXTitleAttribute"
            )

            app_ref = AXUIElementCreateApplication(pid)
            err, window = AXUIElementCopyAttributeValue(
                app_ref, "AXFocusedWindow", None
            )
            if err != 0 or window is None:
                return None

            err, title = AXUIElementCopyAttributeValue(window, kAXTitleAttribute, None)
            if err != 0:
                return None

            return str(title) if title else None
        except Exception as e:
            logger.debug(f"Failed to get window title: {e}")
            return None

    def get_current_state(self) -> FocusState:
        try:
            import AppKit

            NSWorkspace = getattr(AppKit, "NSWorkspace")  # noqa: B009
            workspace = NSWorkspace.sharedWorkspace()
            frontmost = workspace.frontmostApplication()
            if not frontmost:
                return FocusState()

            app_bundle_id = str(frontmost.bundleIdentifier())
            pid = frontmost.processIdentifier()
            window_title = self.get_main_window_title(pid)

            return FocusState(app_bundle_id=app_bundle_id, window_title=window_title)
        except Exception as e:
            logger.warning(f"Failed to get focus state: {e}")
            return FocusState()

    def capture_current_app(self) -> str | None:
        with self._lock:
            self._current_state = self.get_current_state()
            logger.debug(
                f"Captured focus: app={self._current_state.app_bundle_id}, "
                f"window='{self._current_state.window_title}'"
            )
            return self._current_state.app_bundle_id

    def is_same_focus(self) -> bool:
        with self._lock:
            if self._current_state.app_bundle_id is None:
                return True
            current = self.get_current_state()
            return (
                current.app_bundle_id == self._current_state.app_bundle_id
                and current.window_title == self._current_state.window_title
            )

    def is_same_app_focused(self) -> bool:
        with self._lock:
            if self._current_state.app_bundle_id is None:
                return True
            current = self.get_frontmost_app()
            return current == self._current_state.app_bundle_id

    def wait_for_focus(
        self,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> bool:
        if self._current_state.app_bundle_id is None:
            return True

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_same_focus():
                return True
            time.sleep(poll_interval)

        logger.warning(
            f"Timeout waiting for focus: app={self._current_state.app_bundle_id}, "
            f"window='{self._current_state.window_title}'"
        )
        return False

    def wait_for_app_focus(
        self,
        app_bundle_id: str | None,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> bool:
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
    def __init__(
        self,
        send_text_callback: Callable[[str], None],
        focus_tracker: FocusTracker | None = None,
        focus_timeout: float = 10.0,
    ) -> None:
        self._send_text = send_text_callback
        self._focus_tracker = focus_tracker or FocusTracker()
        self._focus_timeout = focus_timeout
        self._target_app: str | None = None

    def start_capture(self) -> None:
        self._target_app = self._focus_tracker.capture_current_app()
        logger.debug(f"Started capture, target app: {self._target_app}")

    def send_text(self, text: str) -> None:
        if self._target_app is None:
            self._send_text(text)
            return

        if self._focus_tracker.is_same_focus():
            logger.debug("Same focus, sending text directly")
            self._send_text(text)
        else:
            logger.info("Focus changed, waiting for focus to return")
            self._wait_and_send(text)

    def _wait_and_send(self, text: str) -> None:
        if self._focus_tracker.wait_for_focus(timeout=self._focus_timeout):
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
        self._target_app = None
        logger.debug("Ended capture session")
