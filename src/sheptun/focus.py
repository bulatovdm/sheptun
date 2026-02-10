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
            err, window = AXUIElementCopyAttributeValue(app_ref, "AXFocusedWindow", None)
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
            is_same = self._check_focus_match(current)

            if not is_same:
                logger.debug(
                    f"Focus changed: expected app={self._current_state.app_bundle_id}, "
                    f"window='{self._current_state.window_title}' | "
                    f"actual app={current.app_bundle_id}, window='{current.window_title}'"
                )
            return is_same

    def _check_focus_match(self, current: FocusState) -> bool:
        return current.app_bundle_id == self._current_state.app_bundle_id

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
    ) -> None:
        self._send_text = send_text_callback
        self._focus_tracker = focus_tracker or FocusTracker()

    def start_capture(self) -> None:
        logger.debug("Started capture")

    def send_text(self, text: str) -> None:
        current_state = self._focus_tracker.get_current_state()
        logger.debug(f"Sending text to current window: {current_state.window_title}")
        self._send_text(text)

    def end_capture(self) -> None:
        logger.debug("Ended capture session")
