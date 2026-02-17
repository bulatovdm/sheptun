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
