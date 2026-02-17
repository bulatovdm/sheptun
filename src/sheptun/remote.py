import json
import logging
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger("sheptun.remote")

DEFAULT_PORT = 7849
CONNECT_TIMEOUT = 2.0
READ_TIMEOUT = 5.0
BONJOUR_SERVICE_TYPE = "_sheptun._tcp."
BONJOUR_SERVICE_NAME = "Sheptun"


def is_cursor_on_local_screen() -> bool:
    """Check if the mouse cursor is within any local display bounds."""
    try:
        import AppKit

        NSEvent: Any = getattr(AppKit, "NSEvent")  # noqa: B009
        NSScreen: Any = getattr(AppKit, "NSScreen")  # noqa: B009
        NSPointInRect: Any = getattr(AppKit, "NSPointInRect")  # noqa: B009

        mouse_location = NSEvent.mouseLocation()
        screens = NSScreen.screens()

        for screen in screens:
            frame = screen.frame()
            if NSPointInRect(mouse_location, frame):
                return True

        return False
    except Exception as e:
        logger.debug(f"Cursor detection failed: {e}")
        return True  # Safe default: assume local


# ---------------------------------------------------------------------------
# Remote Server
# ---------------------------------------------------------------------------


class _RemoteRequestHandler(BaseHTTPRequestHandler):
    server: "RemoteHTTPServer"  # pyright: ignore[reportIncompatibleVariableOverride]

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(format, *args)

    def _check_auth(self) -> bool:
        token = self.server.token
        if not token:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {token}":
            return True
        self._send_json(401, {"error": "unauthorized"})
        return False

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(400, {"error": "empty body"})
            return None
        try:
            body = self.rfile.read(length)
            return json.loads(body)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            self._send_json(400, {"error": "invalid json"})
            return None

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_busy(self) -> bool:
        if self.server.is_busy and self.server.is_busy():
            self._send_json(409, {"error": "busy", "message": "local recording active"})
            return True
        return False

    def do_GET(self) -> None:
        if not self._check_auth():
            return

        if self.path == "/api/ping":
            hostname = socket.gethostname()
            self._send_json(200, {"status": "ok", "hostname": hostname})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._check_auth():
            return

        if self._check_busy():
            return

        data = self._read_json()
        if data is None:
            return

        keyboard = self.server.keyboard_sender
        if keyboard is None:
            self._send_json(500, {"error": "no keyboard sender"})
            return

        if self.path == "/api/text":
            text = data.get("text", "")
            if not text:
                self._send_json(400, {"error": "missing text"})
                return
            keyboard.send_text(text)
            self._send_json(200, {"status": "ok"})

        elif self.path == "/api/key":
            key = data.get("key", "")
            if not key:
                self._send_json(400, {"error": "missing key"})
                return
            keyboard.send_key(key)
            self._send_json(200, {"status": "ok"})

        elif self.path == "/api/hotkey":
            keys = data.get("keys", [])
            if not keys:
                self._send_json(400, {"error": "missing keys"})
                return
            keyboard.send_hotkey(keys)
            self._send_json(200, {"status": "ok"})

        else:
            self._send_json(404, {"error": "not found"})


class RemoteHTTPServer(HTTPServer):
    def __init__(
        self,
        port: int,
        keyboard_sender: Any,
        token: str = "",
        is_busy: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(("0.0.0.0", port), _RemoteRequestHandler)
        self.keyboard_sender = keyboard_sender
        self.token = token
        self.is_busy = is_busy


class RemoteServer:
    def __init__(
        self,
        keyboard_sender: Any,
        port: int = DEFAULT_PORT,
        token: str = "",
        is_busy: Callable[[], bool] | None = None,
    ) -> None:
        self._server = RemoteHTTPServer(port, keyboard_sender, token, is_busy)
        self._thread: threading.Thread | None = None
        self._bonjour_service: Any = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._start_bonjour()
        logger.info(f"Remote server started on port {self.port}")

    def stop(self) -> None:
        self._stop_bonjour()
        self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Remote server stopped")

    def start_blocking(self) -> None:
        self._start_bonjour()
        logger.info(f"Remote server listening on port {self.port}")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_bonjour()
            self._server.server_close()

    def _start_bonjour(self) -> None:
        try:
            from Foundation import NSNetService  # type: ignore[import-untyped]

            service = NSNetService.alloc().initWithDomain_type_name_port_(
                "", BONJOUR_SERVICE_TYPE, BONJOUR_SERVICE_NAME, self.port
            )
            service.publish()
            self._bonjour_service = service
            logger.info(f"Bonjour service published: {BONJOUR_SERVICE_TYPE}")
        except Exception as e:
            logger.warning(f"Failed to publish Bonjour service: {e}")

    def _stop_bonjour(self) -> None:
        if self._bonjour_service is not None:
            self._bonjour_service.stop()
            self._bonjour_service = None


# ---------------------------------------------------------------------------
# Remote Client
# ---------------------------------------------------------------------------


class RemoteClient:
    def __init__(self, host: str, port: int = DEFAULT_PORT, token: str = "") -> None:
        self._base_url = f"http://{host}:{port}"
        self._token = token

    @property
    def host(self) -> str:
        return self._base_url

    def send_text(self, text: str) -> bool:
        return self._post("/api/text", {"text": text})

    def send_key(self, key: str) -> bool:
        return self._post("/api/key", {"key": key})

    def send_hotkey(self, keys: list[str]) -> bool:
        return self._post("/api/hotkey", {"keys": keys})

    def ping(self) -> dict[str, Any] | None:
        try:
            req = self._make_request("/api/ping", method="GET")
            with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as resp:
                return json.loads(resp.read())  # type: ignore[no-any-return]
        except Exception as e:
            logger.debug(f"Ping failed: {e}")
            return None

    def _post(self, path: str, data: dict[str, Any]) -> bool:
        try:
            body = json.dumps(data).encode("utf-8")
            req = self._make_request(path, method="POST", body=body)
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as resp:
                status = resp.status
                if status == 200:
                    return True
                logger.warning(f"Remote server returned {status} for {path}")
                return False
        except urllib.error.HTTPError as e:
            if e.code == 409:
                logger.info("Remote server busy (local recording active)")
            else:
                logger.warning(f"Remote request failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Remote request failed: {e}")
            return False

    def _make_request(
        self,
        path: str,
        method: str = "GET",
        body: bytes | None = None,
    ) -> urllib.request.Request:
        url = f"{self._base_url}{path}"
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Content-Type", "application/json")
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        return req


# ---------------------------------------------------------------------------
# Bonjour Discovery
# ---------------------------------------------------------------------------


class RemoteDiscovery:
    """Discover Sheptun remote servers via Bonjour/mDNS."""

    def __init__(self) -> None:
        self._hosts: dict[str, tuple[str, int]] = {}
        self._lock = threading.Lock()
        self._browser: Any = None
        self._delegate: Any = None
        self._thread: threading.Thread | None = None
        self._run_loop: Any = None

    @property
    def hosts(self) -> dict[str, tuple[str, int]]:
        with self._lock:
            return dict(self._hosts)

    @property
    def first_host(self) -> tuple[str, int] | None:
        with self._lock:
            if self._hosts:
                return next(iter(self._hosts.values()))
            return None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            import Foundation  # type: ignore[import-untyped]

            NSNetServiceBrowser: Any = getattr(Foundation, "NSNetServiceBrowser")  # noqa: B009
            NSObject: Any = getattr(Foundation, "NSObject")  # noqa: B009
            NSRunLoop: Any = getattr(Foundation, "NSRunLoop")  # noqa: B009
            NSDate: Any = getattr(Foundation, "NSDate")  # noqa: B009

            discovery = self
            # Strong refs to prevent garbage collection of ObjC objects
            service_refs: list[tuple[Any, Any]] = []

            class ServiceDelegate(NSObject):  # type: ignore[misc]
                def netServiceDidResolveAddress_(self, service: Any) -> None:  # type: ignore[override]
                    name = str(service.name())
                    hostname = str(service.hostName())
                    port = int(service.port())
                    with discovery._lock:
                        discovery._hosts[name] = (hostname, port)
                    logger.info(f"Discovered remote: {name} at {hostname}:{port}")

                def netService_didNotResolve_(  # type: ignore[override]
                    self, service: Any, _error: Any
                ) -> None:
                    logger.warning(f"Failed to resolve: {service.name()}")

            class BrowserDelegate(NSObject):  # type: ignore[misc]
                def netServiceBrowser_didFindService_moreComing_(  # type: ignore[override]
                    self, _browser: Any, service: Any, _more_coming: bool
                ) -> None:
                    sd = ServiceDelegate.alloc().init()
                    service_refs.append((service, sd))
                    service.setDelegate_(sd)
                    service.resolveWithTimeout_(5.0)

                def netServiceBrowser_didRemoveService_moreComing_(  # type: ignore[override]
                    self, _browser: Any, service: Any, _more_coming: bool
                ) -> None:
                    name = str(service.name())
                    with discovery._lock:
                        discovery._hosts.pop(name, None)
                    logger.info(f"Remote disappeared: {name}")

            self._delegate = BrowserDelegate.alloc().init()
            self._browser = NSNetServiceBrowser.alloc().init()
            self._browser.setDelegate_(self._delegate)
            self._browser.searchForServicesOfType_inDomain_(BONJOUR_SERVICE_TYPE, "")

            self._run_loop = NSRunLoop.currentRunLoop()
            logger.info("Bonjour discovery started")

            while self._browser is not None:
                self._run_loop.runMode_beforeDate_(
                    "NSDefaultRunLoopMode", NSDate.dateWithTimeIntervalSinceNow_(1.0)
                )
        except Exception as e:
            logger.warning(f"Failed to start Bonjour discovery: {e}")

    def stop(self) -> None:
        if self._browser is not None:
            self._browser.stop()
            self._browser = None
            self._delegate = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._hosts.clear()
