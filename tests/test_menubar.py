# pyright: reportPrivateUsage=false
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import rumps

from sheptun.types import AppState

if TYPE_CHECKING:
    from sheptun.menubar import SheptunMenubar

# Simple property to replace rumps icon (avoids NSImage creation in tests)
_test_icon_prop = property(
    lambda self: self.__dict__.get("_test_icon", ""),
    lambda self, v: self.__dict__.__setitem__("_test_icon", v),
)


def _create_menubar() -> SheptunMenubar:
    from sheptun.menubar import SheptunMenubar

    def _noop_init(_self: Any, *_a: Any, **_kw: Any) -> None:
        pass

    with patch.object(SheptunMenubar, "__init__", _noop_init):
        app = SheptunMenubar.__new__(SheptunMenubar)
        app._engine = MagicMock()  # type: ignore[assignment]
        app._state = AppState.IDLE
        app._state_lock = threading.Lock()
        app._uc_active = False
        app._uc_deactivate_timer = None
        app._receive_timer = None
        app._icon_idle = "/icons/mic_idle.png"
        app._icon_listening = "/icons/mic_active.png"
        app._icon_processing = "/icons/mic_processing.png"
        app._icon_remote_idle = "/icons/mic_remote_idle.png"
        app._icon_remote_listening = "/icons/mic_remote_active.png"
        app._icon_receive_idle = "/icons/mic_receive.png"
        app._icon_receive_listening = "/icons/mic_receive_active.png"
        return app


class TestOnWake:
    def test_wake_restarts_engine_when_listening(self) -> None:
        app = _create_menubar()
        app._state = AppState.RECORDING_TOGGLE

        app.onWake_(None)

        app._engine.stop.assert_called_once()  # type: ignore[union-attr]
        app._engine.start.assert_called_once()  # type: ignore[union-attr]

    def test_wake_does_nothing_when_idle(self) -> None:
        app = _create_menubar()
        app._state = AppState.IDLE

        app.onWake_(None)

        app._engine.stop.assert_not_called()  # type: ignore[union-attr]
        app._engine.start.assert_not_called()  # type: ignore[union-attr]

    def test_wake_does_nothing_when_no_engine(self) -> None:
        app = _create_menubar()
        app._state = AppState.RECORDING_TOGGLE
        app._engine = None

        app.onWake_(None)

    def test_wake_does_nothing_during_ptt(self) -> None:
        app = _create_menubar()
        app._state = AppState.RECORDING_PTT

        app.onWake_(None)

        app._engine.stop.assert_not_called()  # type: ignore[union-attr]
        app._engine.start.assert_not_called()  # type: ignore[union-attr]


class TestResolveIcon:
    def test_idle_icon_resolves_to_remote_when_uc_active(self) -> None:
        app = _create_menubar()
        app._uc_active = True

        with patch("sheptun.menubar.settings", remote_enabled=True):
            result = app._resolve_icon(app._icon_idle)
            assert result == app._icon_remote_idle

    def test_listening_icon_resolves_to_remote_when_uc_active(self) -> None:
        app = _create_menubar()
        app._uc_active = True

        with patch("sheptun.menubar.settings", remote_enabled=True):
            result = app._resolve_icon(app._icon_listening)
            assert result == app._icon_remote_listening

    def test_processing_icon_unchanged_when_uc_active(self) -> None:
        app = _create_menubar()
        app._uc_active = True

        with patch("sheptun.menubar.settings", remote_enabled=True):
            result = app._resolve_icon(app._icon_processing)
            assert result == app._icon_processing

    def test_icons_unchanged_when_uc_inactive(self) -> None:
        app = _create_menubar()
        app._uc_active = False

        with patch("sheptun.menubar.settings", remote_enabled=True):
            assert app._resolve_icon(app._icon_idle) == app._icon_idle
            assert app._resolve_icon(app._icon_listening) == app._icon_listening

    def test_icons_unchanged_when_remote_disabled(self) -> None:
        app = _create_menubar()
        app._uc_active = True

        with patch("sheptun.menubar.settings", remote_enabled=False):
            assert app._resolve_icon(app._icon_idle) == app._icon_idle
            assert app._resolve_icon(app._icon_listening) == app._icon_listening


class TestRefreshIcon:
    def _setup_app(self) -> SheptunMenubar:
        app = _create_menubar()
        app._run_on_main_thread = lambda f: f()  # type: ignore[assignment]
        return app

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_refresh_sets_idle_icon_when_idle(self) -> None:
        app = self._setup_app()
        app._state = AppState.IDLE

        with patch("sheptun.menubar.settings", remote_enabled=False):
            app._refresh_icon()
            assert app.icon == app._icon_idle

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_refresh_sets_listening_icon_when_recording(self) -> None:
        app = self._setup_app()
        app._state = AppState.RECORDING_TOGGLE

        with patch("sheptun.menubar.settings", remote_enabled=False):
            app._refresh_icon()
            assert app.icon == app._icon_listening

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_refresh_sets_remote_icon_when_uc_active(self) -> None:
        app = self._setup_app()
        app._state = AppState.IDLE
        app._uc_active = True

        with patch("sheptun.menubar.settings", remote_enabled=True):
            app._refresh_icon()
            assert app.icon == app._icon_remote_idle

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_refresh_sets_remote_listening_when_uc_and_recording(self) -> None:
        app = self._setup_app()
        app._state = AppState.RECORDING_TOGGLE
        app._uc_active = True

        with patch("sheptun.menubar.settings", remote_enabled=True):
            app._refresh_icon()
            assert app.icon == app._icon_remote_listening


class TestUCNotificationDetection:
    def _setup_app(self) -> SheptunMenubar:
        app = _create_menubar()
        app._run_on_main_thread = lambda f: f()  # type: ignore[assignment]
        return app

    def _make_notification(self, bundle_id: str) -> MagicMock:
        notification = MagicMock()
        app_info = MagicMock()
        app_info.bundleIdentifier.return_value = bundle_id
        notification.userInfo.return_value = {"NSWorkspaceApplicationKey": app_info}
        return notification

    def test_uc_detected_when_uc_becomes_frontmost(self) -> None:
        app = self._setup_app()
        notification = self._make_notification("com.apple.universalcontrol")
        app.onActiveAppChanged_(notification)
        assert app._uc_active is True

    def test_uc_cleared_after_debounce(self) -> None:
        app = self._setup_app()
        app._uc_active = True
        notification = self._make_notification("com.apple.terminal")

        app.onActiveAppChanged_(notification)
        assert app._uc_active is True  # still True, debounce pending

        time.sleep(0.4)
        assert app._uc_active is False

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_icon_updates_to_remote_when_uc_detected(self) -> None:
        app = self._setup_app()
        app._state = AppState.IDLE
        notification = self._make_notification("com.apple.universalcontrol")

        with patch("sheptun.menubar.settings", remote_enabled=True):
            app.onActiveAppChanged_(notification)
            assert app.icon == app._icon_remote_idle

    def test_icon_reverts_after_debounce(self) -> None:
        app = self._setup_app()
        app._state = AppState.IDLE
        app._uc_active = True
        refresh_calls: list[int] = []
        app._refresh_icon = lambda: refresh_calls.append(1)  # type: ignore[assignment]
        notification = self._make_notification("com.apple.terminal")

        app.onActiveAppChanged_(notification)
        assert not refresh_calls  # not yet

        time.sleep(0.4)
        assert refresh_calls  # fired after debounce

    def test_no_refresh_when_already_inactive(self) -> None:
        app = self._setup_app()
        app._uc_active = False
        refresh_calls: list[int] = []
        app._refresh_icon = lambda: refresh_calls.append(1)  # type: ignore[assignment]
        notification = self._make_notification("com.apple.terminal")

        app.onActiveAppChanged_(notification)

        assert not refresh_calls

    def test_deactivation_is_debounced(self) -> None:
        app = self._setup_app()
        app._uc_active = True
        notification = self._make_notification("com.apple.terminal")

        app.onActiveAppChanged_(notification)

        assert app._uc_active is True
        assert app._uc_deactivate_timer is not None
        app._uc_deactivate_timer.cancel()

    def test_deactivation_fires_after_debounce(self) -> None:
        app = self._setup_app()
        app._uc_active = True
        refresh_calls: list[int] = []
        app._refresh_icon = lambda: refresh_calls.append(1)  # type: ignore[assignment]
        notification = self._make_notification("com.apple.terminal")

        app.onActiveAppChanged_(notification)
        time.sleep(0.4)

        assert app._uc_active is False
        assert refresh_calls

    def test_no_refresh_during_ptt_on_deactivate(self) -> None:
        app = self._setup_app()
        app._uc_active = True
        app._state = AppState.RECORDING_PTT
        refresh_calls: list[int] = []
        app._refresh_icon = lambda: refresh_calls.append(1)  # type: ignore[assignment]

        app._deactivate_uc()

        assert app._uc_active is False
        assert not refresh_calls

    def test_activation_cancels_pending_deactivation(self) -> None:
        app = self._setup_app()
        app._uc_active = True

        app.onActiveAppChanged_(self._make_notification("com.apple.terminal"))
        assert app._uc_deactivate_timer is not None

        app._uc_active = False
        app.onActiveAppChanged_(self._make_notification("com.apple.universalcontrol"))
        assert app._uc_deactivate_timer is None
        assert app._uc_active is True

    def test_handles_none_user_info(self) -> None:
        app = self._setup_app()
        notification = MagicMock()
        notification.userInfo.return_value = None

        app.onActiveAppChanged_(notification)  # should not raise

        assert app._uc_active is False

    def test_handles_missing_app_key(self) -> None:
        app = self._setup_app()
        notification = MagicMock()
        notification.userInfo.return_value = {}

        app.onActiveAppChanged_(notification)

        assert app._uc_active is False


class TestOnRemoteReceive:
    def _setup_app(self) -> SheptunMenubar:
        app = _create_menubar()
        app._run_on_main_thread = lambda f: f()  # type: ignore[assignment]
        return app

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_receive_shows_receive_idle_icon(self) -> None:
        app = self._setup_app()
        app._state = AppState.IDLE

        with patch("sheptun.menubar.settings", remote_enabled=False):
            app._on_remote_receive()
            assert app.icon == app._icon_receive_idle

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_receive_shows_receive_listening_icon_during_recording(self) -> None:
        app = self._setup_app()
        app._state = AppState.RECORDING_TOGGLE

        with patch("sheptun.menubar.settings", remote_enabled=False):
            app._on_remote_receive()
            assert app.icon == app._icon_receive_listening

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_receive_reverts_icon_after_timer(self) -> None:
        app = self._setup_app()
        app._state = AppState.IDLE

        with patch("sheptun.menubar.settings", remote_enabled=False):
            app._on_remote_receive()
            assert app.icon == app._icon_receive_idle

            # Timer should revert to idle
            time.sleep(0.7)
            assert app.icon == app._icon_idle

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_receive_starts_revert_timer(self) -> None:
        app = self._setup_app()
        app._state = AppState.IDLE

        with patch("sheptun.menubar.settings", remote_enabled=False):
            app._on_remote_receive()
            assert app._receive_timer is not None
            assert app._receive_timer.is_alive()
            app._receive_timer.cancel()


class TestPTTIcon:
    def _setup_app(self) -> SheptunMenubar:
        app = _create_menubar()
        app._run_on_main_thread = lambda f: f()  # type: ignore[assignment]
        app._ptt_recorder = MagicMock()
        app._ptt_keyboard = MagicMock()
        app._engine = MagicMock()
        return app

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_ptt_shows_remote_listening_when_uc_active(self) -> None:
        app = self._setup_app()
        app._state = AppState.IDLE
        app._uc_active = True

        with patch("sheptun.menubar.settings", remote_enabled=True, use_clipboard=False):
            app._on_ptt_start()
            assert app.icon == app._icon_remote_listening

    @patch.object(rumps.App, "icon", _test_icon_prop)
    def test_ptt_shows_regular_listening_when_uc_inactive(self) -> None:
        app = self._setup_app()
        app._state = AppState.IDLE
        app._uc_active = False

        with patch("sheptun.menubar.settings", remote_enabled=True, use_clipboard=False):
            app._on_ptt_start()
            assert app.icon == app._icon_listening
