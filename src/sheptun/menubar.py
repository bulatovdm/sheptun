import importlib.resources
import logging
import threading
from collections.abc import Callable
from typing import Any

import rumps

from sheptun.audio import AudioConfig, AudioRecorder, VoiceActivityConfig
from sheptun.commands import CommandConfigLoader, CommandParser
from sheptun.config import get_config_path
from sheptun.engine import BaseVoiceEngine
from sheptun.hotkeys import HotkeyManager
from sheptun.i18n import t
from sheptun.keyboard import (
    FocusAwareKeyboardSender,
    MacOSKeyboardSender,
    RemoteAwareKeyboardSender,
)
from sheptun.recognition import WhisperRecognizer
from sheptun.settings import settings, setup_logging
from sheptun.types import AppState, KeyboardSender, SpeechRecognizer

setup_logging()
logger = logging.getLogger("sheptun.menubar")


def _get_icon_path(name: str) -> str:
    with importlib.resources.as_file(
        importlib.resources.files("sheptun.resources").joinpath(name)
    ) as path:
        return str(path)


class MenubarStatusIndicator:
    def __init__(self, app: "SheptunMenubar") -> None:
        self._app = app

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def listening(self) -> None:
        self._app.set_listening(True)

    def processing(self) -> None:
        self._app.set_processing()

    def error(self, message: str) -> None:
        self._app.show_notification(t("notification_error"), message)

    def idle(self) -> None:
        self._app.set_listening(False)

    def show_recognized(self, text: str) -> None:
        pass

    def show_action(self, action_description: str) -> None:
        pass

    def show_help(self) -> None:
        import subprocess

        logger.info("Showing help notification")
        message = t("help_commands")
        title = t("help_title")
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True)


class MenubarVoiceEngine(BaseVoiceEngine):
    def __init__(
        self,
        recognizer: SpeechRecognizer,
        command_parser: CommandParser,
        keyboard_sender: KeyboardSender,
        status_indicator: MenubarStatusIndicator,
        audio_config: AudioConfig | None = None,
        vad_config: VoiceActivityConfig | None = None,
        record_dataset: bool = False,
    ) -> None:
        super().__init__(
            recognizer=recognizer,
            command_parser=command_parser,
            keyboard_sender=keyboard_sender,
            status_indicator=status_indicator,
            audio_config=audio_config,
            vad_config=vad_config,
            record_dataset=record_dataset,
        )

    def _log(self, message: str) -> None:
        logger.info(message)


class SheptunMenubar(rumps.App):  # type: ignore[misc]
    def __init__(self, model_name: str = "base") -> None:
        super().__init__("Sheptun", quit_button=None, template=True)  # type: ignore[arg-type]
        self._model_name = model_name
        self._engine: MenubarVoiceEngine | None = None
        self._state = AppState.IDLE
        self._state_lock = threading.Lock()
        self._ptt_recorder: AudioRecorder | None = None
        self._ptt_keyboard: KeyboardSender | None = None
        self._remote_server: Any = None
        self._remote_discovery: Any = None
        self._remote_client: Any = None

        self._icon_idle = _get_icon_path("mic_idle.png")
        self._icon_listening = _get_icon_path("mic_active.png")
        self._icon_processing = _get_icon_path("mic_processing.png")
        self._icon_remote_idle = _get_icon_path("mic_remote_idle.png")
        self._icon_remote_listening = _get_icon_path("mic_remote_active.png")
        self._icon_receive_idle = _get_icon_path("mic_receive.png")
        self._icon_receive_listening = _get_icon_path("mic_receive_active.png")
        self._uc_active = False
        self._receive_timer: threading.Timer | None = None
        self.icon = self._icon_idle

        self._hotkey_manager = HotkeyManager(
            toggle_hotkey=settings.hotkey_toggle,
            ptt_hotkey=settings.hotkey_ptt,
        )
        self._hotkey_manager.set_callbacks(
            on_toggle=self._on_toggle_hotkey,
            on_ptt_start=self._on_ptt_start,
            on_ptt_stop=self._on_ptt_stop,
        )

        toggle_display = self._hotkey_manager.toggle_hotkey_display
        ptt_display = self._hotkey_manager.ptt_hotkey_display
        self._toggle_menu_item = rumps.MenuItem(
            f"{t('menu_toggle')} ({toggle_display})", callback=self._toggle_listening
        )
        self._ptt_menu_item = rumps.MenuItem(f"{t('menu_ptt')} ({ptt_display})")

        menu_items: list[Any] = [
            self._toggle_menu_item,
            self._ptt_menu_item,
            None,
        ]

        if settings.remote_enabled:
            self._remote_status_item = rumps.MenuItem(t("menu_remote_status"))
            menu_items.append(self._remote_status_item)
            menu_items.append(None)

        menu_items.extend(
            [
                rumps.MenuItem(t("menu_restart"), callback=self._restart),
                rumps.MenuItem(t("menu_quit"), callback=self._quit),
            ]
        )
        self.menu = menu_items

        self._hotkey_manager.start()
        self._subscribe_to_wake_notifications()
        if settings.remote_enabled:
            self._subscribe_to_app_activation()
        self._start_remote_services()

    def _start_remote_services(self) -> None:
        if settings.remote_serve:
            from sheptun.remote import RemoteServer

            keyboard = MacOSKeyboardSender(use_clipboard=settings.use_clipboard)
            self._remote_server = RemoteServer(
                keyboard_sender=keyboard,
                port=settings.remote_port,
                token=settings.remote_token,
                is_busy=lambda: self._state != AppState.IDLE,
                on_receive=self._on_remote_receive,
            )
            self._remote_server.start()
            logger.info(f"Remote server started on port {settings.remote_port}")

        if settings.remote_enabled:
            self._init_remote_client()

    def _init_remote_client(self) -> None:
        from sheptun.remote import RemoteClient

        if settings.remote_host:
            self._remote_client = RemoteClient(
                host=settings.remote_host,
                port=settings.remote_port,
                token=settings.remote_token,
            )
            logger.info(f"Remote client configured: {settings.remote_host}:{settings.remote_port}")
        else:
            from sheptun.remote import RemoteDiscovery

            self._remote_discovery = RemoteDiscovery()
            self._remote_discovery.start()
            logger.info("Bonjour discovery started, waiting for remote servers")

    def _get_remote_client(self) -> Any:
        if self._remote_client is not None:
            return self._remote_client
        if self._remote_discovery is not None:
            host_info = self._remote_discovery.first_host
            if host_info is not None:
                from sheptun.remote import RemoteClient

                hostname, port = host_info
                self._remote_client = RemoteClient(
                    host=hostname, port=port, token=settings.remote_token
                )
                logger.info(f"Auto-discovered remote: {hostname}:{port}")
                return self._remote_client
        return None

    def _run_on_main_thread(self, func: Callable[[], None]) -> None:
        """Schedule a function to run on the main thread via NSOperationQueue."""
        try:
            from Foundation import NSOperationQueue  # type: ignore[import-untyped]

            NSOperationQueue.mainQueue().addOperationWithBlock_(func)
        except ImportError:
            func()

    def _resolve_icon(self, icon_path: str) -> str:
        """Substitute remote icon variant when UC is active."""
        if self._uc_active and settings.remote_enabled:
            if icon_path == self._icon_idle:
                return self._icon_remote_idle
            if icon_path == self._icon_listening:
                return self._icon_remote_listening
        return icon_path

    def _set_icon(self, icon_path: str) -> None:
        """Thread-safe icon update."""
        resolved = self._resolve_icon(icon_path)
        self._run_on_main_thread(lambda: setattr(self, "icon", resolved))

    def _on_toggle_hotkey(self) -> None:
        logger.info("Toggle hotkey pressed")
        self._toggle_listening(self._toggle_menu_item)

    def _on_ptt_start(self) -> None:
        logger.info("PTT hotkey pressed")

        with self._state_lock:
            if self._state != AppState.IDLE:
                logger.info(f"PTT ignored, current state: {self._state}")
                return
            self._state = AppState.RECORDING_PTT

        self._ensure_engine_initialized()

        if self._ptt_recorder is None:
            self._ptt_recorder = AudioRecorder()

        if self._ptt_keyboard is None:
            base_keyboard = MacOSKeyboardSender(use_clipboard=settings.use_clipboard)
            if settings.remote_enabled:
                self._ptt_keyboard = RemoteAwareKeyboardSender(
                    local_sender=base_keyboard,
                    remote_client=self._remote_client,
                    auto_detect=settings.remote_auto_detect,
                    remote_client_factory=self._get_remote_client,
                )
            else:
                self._ptt_keyboard = FocusAwareKeyboardSender(keyboard_sender=base_keyboard)

        self._ptt_keyboard.start_capture()
        self._set_icon(self._icon_listening)
        self._ptt_recorder.start()

    def _on_ptt_stop(self) -> None:
        logger.info("PTT hotkey released")

        with self._state_lock:
            if self._state != AppState.RECORDING_PTT:
                return

        if self._ptt_recorder is None or not self._ptt_recorder.is_recording():
            self._set_state(AppState.IDLE)
            return

        audio_data = self._ptt_recorder.stop()
        if len(audio_data) < 1000:
            self._set_state(AppState.IDLE)
            self._set_icon(self._icon_idle)
            return

        self._set_state(AppState.PROCESSING)
        self._set_icon(self._icon_processing)
        threading.Thread(target=self._process_ptt_audio, args=(audio_data,), daemon=True).start()

    def _process_ptt_audio(self, audio_data: bytes) -> None:
        self._ensure_engine_initialized()

        if self._engine is None:
            self._set_state(AppState.IDLE)
            self._set_icon(self._icon_idle)
            self._end_ptt_capture()
            return

        try:
            if self._ptt_keyboard is not None:
                self._engine.set_keyboard_sender(self._ptt_keyboard)
            self._engine.recognize_and_execute(audio_data)
        except Exception as e:
            logger.exception(f"PTT error: {e}")
        finally:
            self._set_state(AppState.IDLE)
            self._set_icon(self._icon_idle)
            self._end_ptt_capture()
            self._restore_engine_keyboard()

    def _restore_engine_keyboard(self) -> None:
        if self._engine is None:
            return
        base_keyboard = MacOSKeyboardSender(use_clipboard=settings.use_clipboard)
        if settings.remote_enabled:
            self._engine.set_keyboard_sender(
                RemoteAwareKeyboardSender(
                    local_sender=base_keyboard,
                    remote_client=self._remote_client,
                    auto_detect=settings.remote_auto_detect,
                    remote_client_factory=self._get_remote_client,
                )
            )
        else:
            self._engine.set_keyboard_sender(base_keyboard)

    def _end_ptt_capture(self) -> None:
        if self._ptt_keyboard is not None:
            self._ptt_keyboard.end_capture()

    def _set_state(self, state: AppState) -> None:
        with self._state_lock:
            self._state = state
        logger.debug(f"State changed to: {state}")

    def _ensure_engine_initialized(self) -> None:
        if self._engine is None:
            self._set_icon(self._icon_processing)
            self.show_notification("Sheptun", t("notification_loading"))
            self._init_engine()

    def _init_engine(self) -> None:
        if self._engine is not None:
            return

        config_path = get_config_path()
        config = CommandConfigLoader.load(config_path)

        recognizer: SpeechRecognizer
        if settings.recognizer == "apple":
            from sheptun.apple_speech import AppleSpeechRecognizer

            recognizer = AppleSpeechRecognizer()
        elif settings.recognizer == "mlx":
            from sheptun.recognition import MLXWhisperRecognizer

            recognizer = MLXWhisperRecognizer(model_name=self._model_name)
            if not recognizer.is_model_cached():
                logger.info("MLX model not cached, downloading...")
                self.title = t("notification_downloading")
                recognizer.download_model(
                    on_progress=lambda pct: setattr(self, "title", f"{pct}%"),
                )
            self.title = t("notification_loading")
            recognizer.warmup()
            self.title = ""
        else:
            recognizer = WhisperRecognizer(model_name=self._model_name)

        command_parser = CommandParser(config)
        base_keyboard = MacOSKeyboardSender(use_clipboard=settings.use_clipboard)
        keyboard_sender: FocusAwareKeyboardSender | RemoteAwareKeyboardSender
        if settings.remote_enabled:
            keyboard_sender = RemoteAwareKeyboardSender(
                local_sender=base_keyboard,
                remote_client=self._remote_client,
                auto_detect=settings.remote_auto_detect,
                remote_client_factory=self._get_remote_client,
            )
        else:
            keyboard_sender = FocusAwareKeyboardSender(keyboard_sender=base_keyboard)
        status_indicator = MenubarStatusIndicator(self)

        self._engine = MenubarVoiceEngine(
            recognizer=recognizer,
            command_parser=command_parser,
            keyboard_sender=keyboard_sender,
            status_indicator=status_indicator,
            record_dataset=settings.record_dataset,
        )

    def _toggle_listening(self, _sender: rumps.MenuItem) -> None:
        with self._state_lock:
            if self._state == AppState.RECORDING_PTT:
                logger.info("Toggle ignored, PTT is active")
                return
            is_toggle_active = self._state == AppState.RECORDING_TOGGLE

        if self._engine is None:
            self._set_icon(self._icon_processing)
            self.show_notification("Sheptun", t("notification_loading"))
            threading.Thread(target=self._init_and_start, daemon=True).start()
        elif is_toggle_active:
            self._engine.stop()
            self._update_toggle_menu_title()
        else:
            self._engine.start()
            self._update_toggle_menu_title()

    def _init_and_start(self) -> None:
        self._init_engine()
        if self._engine is not None:
            self._engine.start()
            self._update_toggle_menu_title()

    def _update_toggle_menu_title(self) -> None:
        toggle_display = self._hotkey_manager.toggle_hotkey_display
        is_toggle_active = self._state == AppState.RECORDING_TOGGLE
        if is_toggle_active:
            self._toggle_menu_item.title = f"{t('menu_toggle_stop')} ({toggle_display})"
        else:
            self._toggle_menu_item.title = f"{t('menu_toggle')} ({toggle_display})"

    def set_listening(self, listening: bool) -> None:
        if listening:
            self._set_state(AppState.RECORDING_TOGGLE)
            self._set_icon(self._icon_listening)
        else:
            self._set_state(AppState.IDLE)
            self._set_icon(self._icon_idle)

    def set_processing(self) -> None:
        self._set_icon(self._icon_processing)

    def show_notification(self, title: str, message: str) -> None:
        rumps.notification(title, "", message)  # type: ignore[no-untyped-call]

    def _subscribe_to_wake_notifications(self) -> None:
        try:
            import AppKit

            NSWorkspace = getattr(AppKit, "NSWorkspace")  # noqa: B009
            NSWorkspaceDidWakeNotification = getattr(  # noqa: B009
                AppKit, "NSWorkspaceDidWakeNotification"
            )
            center = NSWorkspace.sharedWorkspace().notificationCenter()
            center.addObserver_selector_name_object_(
                self,
                "onWake:",
                NSWorkspaceDidWakeNotification,
                None,
            )
            logger.info("Subscribed to wake notifications")
        except Exception as e:
            logger.warning(f"Failed to subscribe to wake notifications: {e}")

    def onWake_(self, _notification: Any) -> None:
        logger.info("System woke from sleep")
        with self._state_lock:
            was_listening = self._state == AppState.RECORDING_TOGGLE

        if was_listening and self._engine is not None:
            logger.info("Restarting engine after wake")
            self._engine.stop()
            self._engine.start()

    def _subscribe_to_app_activation(self) -> None:
        try:
            import AppKit

            NSWorkspace = getattr(AppKit, "NSWorkspace")  # noqa: B009
            NSWorkspaceDidActivateApplicationNotification = getattr(  # noqa: B009
                AppKit, "NSWorkspaceDidActivateApplicationNotification"
            )
            center = NSWorkspace.sharedWorkspace().notificationCenter()
            center.addObserver_selector_name_object_(
                self,
                "onAppActivated:",
                NSWorkspaceDidActivateApplicationNotification,
                None,
            )
            logger.info("Subscribed to app activation notifications")
        except Exception as e:
            logger.warning(f"Failed to subscribe to app activation: {e}")

    def onAppActivated_(self, notification: Any) -> None:
        from sheptun.remote import UC_BUNDLE_ID

        try:
            app = notification.userInfo().get("NSWorkspaceApplicationKey")
            bundle_id = str(app.bundleIdentifier() or "") if app else ""
            was_uc = self._uc_active
            self._uc_active = bundle_id == UC_BUNDLE_ID

            if self._uc_active != was_uc:
                logger.info(f"UC active: {self._uc_active}")
                self._refresh_icon()
        except Exception as e:
            logger.debug(f"App activation handler error: {e}")

    def _refresh_icon(self) -> None:
        """Re-apply current icon with UC state taken into account."""
        with self._state_lock:
            state = self._state

        if state == AppState.IDLE:
            self._set_icon(self._icon_idle)
        elif state in (AppState.RECORDING_TOGGLE, AppState.RECORDING_PTT):
            self._set_icon(self._icon_listening)
        else:
            self._set_icon(self._icon_processing)

    def _on_remote_receive(self) -> None:
        """Called by RemoteServer when text is received from remote."""
        with self._state_lock:
            state = self._state

        if state in (AppState.RECORDING_TOGGLE, AppState.RECORDING_PTT):
            receive_icon = self._icon_receive_listening
        else:
            receive_icon = self._icon_receive_idle

        self._run_on_main_thread(lambda: setattr(self, "icon", receive_icon))

        if self._receive_timer is not None:
            self._receive_timer.cancel()
        self._receive_timer = threading.Timer(0.5, self._refresh_icon)
        self._receive_timer.daemon = True
        self._receive_timer.start()

    def _restart(self, _: Any) -> None:
        import os
        import subprocess

        app_path = settings.app_path
        subprocess.Popen(
            ["open", "-n", str(app_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        os._exit(0)

    def _quit(self, _: Any) -> None:
        if self._engine is not None:
            self._engine.stop()
        if self._remote_server is not None:
            self._remote_server.stop()
        if self._remote_discovery is not None:
            self._remote_discovery.stop()
        self._hotkey_manager.stop()
        rumps.quit_application()  # type: ignore[no-untyped-call]


def _hide_dock_icon() -> None:
    try:
        import AppKit  # type: ignore[import-untyped]

        NSApplication = getattr(AppKit, "NSApplication")  # noqa: B009
        policy = getattr(AppKit, "NSApplicationActivationPolicyAccessory")  # noqa: B009
        NSApplication.sharedApplication().setActivationPolicy_(policy)
    except ImportError:
        logger.warning("AppKit not available, cannot hide dock icon")


def run_menubar(model_name: str = "base") -> None:
    _hide_dock_icon()
    app = SheptunMenubar(model_name=model_name)
    app.run()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    run_menubar(model_name=settings.model)
