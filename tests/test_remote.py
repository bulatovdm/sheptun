import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

from sheptun.remote import (
    DEFAULT_PORT,
    RemoteClient,
    RemoteHTTPServer,
    RemoteServer,
)


class TestRemoteClient:
    def test_ping_success(self) -> None:
        server, port = _start_test_server(token="")
        try:
            client = RemoteClient("127.0.0.1", port)
            result = client.ping()
            assert result is not None
            assert result["status"] == "ok"
            assert "hostname" in result
        finally:
            server.shutdown()

    def test_ping_failure(self) -> None:
        client = RemoteClient("127.0.0.1", 1)  # Invalid port
        result = client.ping()
        assert result is None

    def test_send_text(self) -> None:
        keyboard = MagicMock()
        server, port = _start_test_server(token="", keyboard=keyboard)
        try:
            client = RemoteClient("127.0.0.1", port)
            assert client.send_text("hello") is True
            keyboard.send_text.assert_called_once_with("hello")
        finally:
            server.shutdown()

    def test_send_key(self) -> None:
        keyboard = MagicMock()
        server, port = _start_test_server(token="", keyboard=keyboard)
        try:
            client = RemoteClient("127.0.0.1", port)
            assert client.send_key("enter") is True
            keyboard.send_key.assert_called_once_with("enter")
        finally:
            server.shutdown()

    def test_send_hotkey(self) -> None:
        keyboard = MagicMock()
        server, port = _start_test_server(token="", keyboard=keyboard)
        try:
            client = RemoteClient("127.0.0.1", port)
            assert client.send_hotkey(["cmd", "c"]) is True
            keyboard.send_hotkey.assert_called_once_with(["cmd", "c"])
        finally:
            server.shutdown()

    def test_auth_required(self) -> None:
        keyboard = MagicMock()
        server, port = _start_test_server(token="secret", keyboard=keyboard)
        try:
            client_no_token = RemoteClient("127.0.0.1", port)
            assert client_no_token.send_text("hello") is False
            keyboard.send_text.assert_not_called()

            client_with_token = RemoteClient("127.0.0.1", port, token="secret")
            assert client_with_token.send_text("hello") is True
            keyboard.send_text.assert_called_once_with("hello")
        finally:
            server.shutdown()

    def test_busy_server_returns_false(self) -> None:
        keyboard = MagicMock()
        server, port = _start_test_server(token="", keyboard=keyboard, is_busy=lambda: True)
        try:
            client = RemoteClient("127.0.0.1", port)
            assert client.send_text("hello") is False
            keyboard.send_text.assert_not_called()
        finally:
            server.shutdown()

    def test_unicode_text(self) -> None:
        keyboard = MagicMock()
        server, port = _start_test_server(token="", keyboard=keyboard)
        try:
            client = RemoteClient("127.0.0.1", port)
            assert client.send_text("Привет мир! 🌍") is True
            keyboard.send_text.assert_called_once_with("Привет мир! 🌍")
        finally:
            server.shutdown()


class TestRemoteServer:
    def test_start_stop(self) -> None:
        keyboard = MagicMock()
        server = RemoteServer(keyboard_sender=keyboard, port=0)
        server.start()
        assert server.port > 0
        server.stop()

    def test_default_port(self) -> None:
        assert DEFAULT_PORT == 7849


class TestRemoteAwareKeyboardSender:
    def test_local_when_cursor_on_screen(self) -> None:
        from sheptun.keyboard import RemoteAwareKeyboardSender

        local = MagicMock()
        remote = MagicMock()

        with patch("sheptun.remote.is_cursor_on_local_screen", return_value=True):
            sender = RemoteAwareKeyboardSender(local, remote, auto_detect=True)
            sender.send_text("hello")
            local.send_text.assert_called_once_with("hello")
            remote.send_text.assert_not_called()

    def test_remote_when_cursor_off_screen(self) -> None:
        from sheptun.keyboard import RemoteAwareKeyboardSender

        local = MagicMock()
        remote = MagicMock()
        remote.send_text.return_value = True

        with patch("sheptun.remote.is_cursor_on_local_screen", return_value=False):
            sender = RemoteAwareKeyboardSender(local, remote, auto_detect=True)
            sender.send_text("hello")
            remote.send_text.assert_called_once_with("hello")
            local.send_text.assert_not_called()

    def test_fallback_to_local_on_remote_failure(self) -> None:
        from sheptun.keyboard import RemoteAwareKeyboardSender

        local = MagicMock()
        remote = MagicMock()
        remote.send_text.return_value = False

        with patch("sheptun.remote.is_cursor_on_local_screen", return_value=False):
            sender = RemoteAwareKeyboardSender(local, remote, auto_detect=True)
            sender.send_text("hello")
            remote.send_text.assert_called_once_with("hello")
            local.send_text.assert_called_once_with("hello")

    def test_force_remote(self) -> None:
        from sheptun.keyboard import RemoteAwareKeyboardSender

        local = MagicMock()
        remote = MagicMock()
        remote.send_key.return_value = True

        with patch("sheptun.remote.is_cursor_on_local_screen", return_value=True):
            sender = RemoteAwareKeyboardSender(local, remote, auto_detect=True)
            sender.force_remote = True
            sender.send_key("enter")
            remote.send_key.assert_called_once_with("enter")
            local.send_key.assert_not_called()

    def test_auto_detect_disabled(self) -> None:
        from sheptun.keyboard import RemoteAwareKeyboardSender

        local = MagicMock()
        remote = MagicMock()

        with patch("sheptun.remote.is_cursor_on_local_screen", return_value=False):
            sender = RemoteAwareKeyboardSender(local, remote, auto_detect=False)
            sender.send_text("hello")
            local.send_text.assert_called_once_with("hello")
            remote.send_text.assert_not_called()

    def test_start_end_capture_delegated(self) -> None:
        from sheptun.keyboard import RemoteAwareKeyboardSender

        local = MagicMock()
        remote = MagicMock()

        with patch("sheptun.remote.is_cursor_on_local_screen", return_value=True):
            sender = RemoteAwareKeyboardSender(local, remote)
            sender.start_capture()
            sender.end_capture()
            local.start_capture.assert_called_once()
            local.end_capture.assert_called_once()


def _start_test_server(
    token: str = "",
    keyboard: Any = None,
    is_busy: Any = None,
) -> tuple[RemoteHTTPServer, int]:
    if keyboard is None:
        keyboard = MagicMock()
    server = RemoteHTTPServer(0, keyboard, token, is_busy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    return server, server.server_address[1]
