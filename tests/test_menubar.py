# pyright: reportPrivateUsage=false
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from sheptun.types import AppState

if TYPE_CHECKING:
    from sheptun.menubar import SheptunMenubar


def _create_menubar() -> SheptunMenubar:
    from sheptun.menubar import SheptunMenubar

    def _noop_init(_self: Any, *_a: Any, **_kw: Any) -> None:
        pass

    with patch.object(SheptunMenubar, "__init__", _noop_init):
        app = SheptunMenubar.__new__(SheptunMenubar)
        app._engine = MagicMock()  # type: ignore[assignment]
        app._state = AppState.IDLE
        app._state_lock = threading.Lock()
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
