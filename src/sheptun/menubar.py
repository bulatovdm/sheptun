import importlib.resources
import logging
import threading
from typing import Any

import rumps

from sheptun.audio import AudioConfig, AudioRecorder, VoiceActivityConfig
from sheptun.commands import CommandConfigLoader, CommandParser
from sheptun.config import get_config_path
from sheptun.engine import BaseVoiceEngine
from sheptun.hotkeys import HotkeyManager
from sheptun.i18n import t
from sheptun.keyboard import FocusAwareKeyboardSender, MacOSKeyboardSender
from sheptun.recognition import WhisperRecognizer
from sheptun.settings import settings, setup_logging
from sheptun.types import AppState

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
        recognizer: WhisperRecognizer,
        command_parser: CommandParser,
        keyboard_sender: FocusAwareKeyboardSender | MacOSKeyboardSender,
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
        self._ptt_keyboard: FocusAwareKeyboardSender | None = None

        self._icon_idle = _get_icon_path("mic_idle.png")
        self._icon_listening = _get_icon_path("mic_active.png")
        self._icon_processing = _get_icon_path("mic_processing.png")
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

        self.menu = [
            self._toggle_menu_item,
            self._ptt_menu_item,
            None,
            rumps.MenuItem(t("menu_restart"), callback=self._restart),
            rumps.MenuItem(t("menu_quit"), callback=self._quit),
        ]
        self._hotkey_manager.start()

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
            self._ptt_keyboard = FocusAwareKeyboardSender()

        self._ptt_keyboard.start_capture()
        self.icon = self._icon_listening
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
            self.icon = self._icon_idle
            return

        self._set_state(AppState.PROCESSING)
        self.icon = self._icon_processing
        threading.Thread(target=self._process_ptt_audio, args=(audio_data,), daemon=True).start()

    def _process_ptt_audio(self, audio_data: bytes) -> None:
        self._ensure_engine_initialized()

        if self._engine is None:
            self._set_state(AppState.IDLE)
            self.icon = self._icon_idle
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
            self.icon = self._icon_idle
            self._end_ptt_capture()
            self._engine.set_keyboard_sender(MacOSKeyboardSender(use_clipboard=settings.use_clipboard))

    def _end_ptt_capture(self) -> None:
        if self._ptt_keyboard is not None:
            self._ptt_keyboard.end_capture()

    def _set_state(self, state: AppState) -> None:
        with self._state_lock:
            self._state = state
        logger.debug(f"State changed to: {state}")

    def _ensure_engine_initialized(self) -> None:
        if self._engine is None:
            self.icon = self._icon_processing
            self.show_notification("Sheptun", t("notification_loading"))
            self._init_engine()

    def _init_engine(self) -> None:
        if self._engine is not None:
            return

        config_path = get_config_path()
        config = CommandConfigLoader.load(config_path)
        recognizer = WhisperRecognizer(model_name=self._model_name)
        command_parser = CommandParser(config)
        base_keyboard = MacOSKeyboardSender(use_clipboard=settings.use_clipboard)
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
            self.icon = self._icon_processing
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
            self.icon = self._icon_listening
        else:
            self._set_state(AppState.IDLE)
            self.icon = self._icon_idle

    def set_processing(self) -> None:
        self.icon = self._icon_processing

    def show_notification(self, title: str, message: str) -> None:
        rumps.notification(title, "", message)  # type: ignore[no-untyped-call]

    def _restart(self, _: Any) -> None:
        import os
        import subprocess

        if self._engine is not None:
            self._engine.stop()
        self._hotkey_manager.stop()

        app_path = settings.app_path
        subprocess.Popen(
            ["open", str(app_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        os._exit(0)

    def _quit(self, _: Any) -> None:
        if self._engine is not None:
            self._engine.stop()
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
    run_menubar()
