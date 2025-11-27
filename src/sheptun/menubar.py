import importlib.resources
import logging
import threading
from typing import Any

import rumps

from sheptun.audio import AudioConfig, AudioRecorder, VoiceActivityConfig
from sheptun.commands import CommandConfigLoader, CommandParser
from sheptun.config import get_config_path
from sheptun.hotkeys import HotkeyManager
from sheptun.i18n import t
from sheptun.keyboard import MacOSKeyboardSender
from sheptun.recognition import WhisperRecognizer
from sheptun.settings import settings, setup_logging
from sheptun.types import Action, ActionType

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

    def show_recognized(self, _text: str) -> None:
        pass

    def show_action(self, _action_description: str) -> None:
        pass

    def show_help(self) -> None:
        import subprocess

        logger.info("Showing help notification")
        message = t("help_commands")
        title = t("help_title")
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True)


class MenubarVoiceEngine:
    def __init__(
        self,
        recognizer: WhisperRecognizer,
        command_parser: CommandParser,
        keyboard_sender: MacOSKeyboardSender,
        status_indicator: MenubarStatusIndicator,
        audio_config: AudioConfig | None = None,
        vad_config: VoiceActivityConfig | None = None,
    ) -> None:
        from sheptun.audio import ContinuousAudioRecorder

        self._recognizer = recognizer
        self._command_parser = command_parser
        self._keyboard = keyboard_sender
        self._status = status_indicator
        self._recorder = ContinuousAudioRecorder(audio_config, vad_config)
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        self._status.listening()
        self._recorder.set_callback(self._on_speech_detected)
        self._recorder.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._recorder.stop()
        self._status.idle()

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def recognize_and_execute(self, audio_data: bytes) -> None:
        try:
            result = self._recognizer.recognize(audio_data, self._recorder.sample_rate)
            if result and result.text:
                logger.info(f"Recognized: '{result.text}'")
                action = self._command_parser.parse(result.text)
                if action:
                    self._execute_action(action)
        except Exception as e:
            logger.exception(f"Recognition error: {e}")

    def _on_speech_detected(self, audio_data: bytes) -> None:
        if not self._running:
            return

        logger.debug(f"Speech detected: {len(audio_data)} bytes")
        self._status.processing()

        try:
            result = self._recognizer.recognize(audio_data, self._recorder.sample_rate)

            if result is None or not result.text:
                logger.debug("Recognition returned empty result")
                self._status.listening()
                return

            logger.info(f"Recognized: '{result.text}'")
            action = self._command_parser.parse(result.text)
            logger.debug(f"Parsed action: {action}")

            if action is not None:
                self._execute_action(action)

        except Exception as e:
            logger.exception(f"Error processing speech: {e}")

        if self._running:
            self._status.listening()

    def _execute_action(self, action: Action) -> None:
        logger.info(f"Executing action: {action.action_type.name} = {action.value}")
        match action.action_type:
            case ActionType.STOP:
                self.stop()

            case ActionType.TEXT:
                if isinstance(action.value, str):
                    self._keyboard.send_text(action.value)

            case ActionType.KEY:
                if isinstance(action.value, str):
                    self._keyboard.send_key(action.value)

            case ActionType.HOTKEY:
                if isinstance(action.value, list):
                    self._keyboard.send_hotkey(action.value)

            case ActionType.SLASH:
                if isinstance(action.value, str):
                    self._keyboard.send_text(action.value)
                    self._keyboard.send_key("return")

            case ActionType.HELP:
                self._status.show_help()


class SheptunMenubar(rumps.App):  # type: ignore[misc]
    def __init__(self, model_name: str = "base") -> None:
        super().__init__("Sheptun", quit_button=None, template=True)  # type: ignore[arg-type]
        self._model_name = model_name
        self._engine: MenubarVoiceEngine | None = None
        self._is_listening = False
        self._ptt_recorder: AudioRecorder | None = None

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
        self._toggle_menu_item = rumps.MenuItem(f"{t('menu_toggle')} ({toggle_display})", callback=self._toggle_listening)
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
        self._ensure_engine_initialized()

        if self._ptt_recorder is None:
            self._ptt_recorder = AudioRecorder()

        self.icon = self._icon_listening
        self._ptt_recorder.start()

    def _on_ptt_stop(self) -> None:
        logger.info("PTT hotkey released")
        if self._ptt_recorder is None:
            return

        audio_data = self._ptt_recorder.stop()
        if len(audio_data) < 1000:
            self.icon = self._icon_idle
            return

        self.icon = self._icon_processing
        threading.Thread(target=self._process_ptt_audio, args=(audio_data,), daemon=True).start()

    def _process_ptt_audio(self, audio_data: bytes) -> None:
        self._ensure_engine_initialized()

        if self._engine is None:
            self.icon = self._icon_idle
            return

        try:
            self._engine.recognize_and_execute(audio_data)
        except Exception as e:
            logger.exception(f"PTT error: {e}")
        finally:
            self.icon = self._icon_idle

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
        keyboard_sender = MacOSKeyboardSender()
        status_indicator = MenubarStatusIndicator(self)

        self._engine = MenubarVoiceEngine(
            recognizer=recognizer,
            command_parser=command_parser,
            keyboard_sender=keyboard_sender,
            status_indicator=status_indicator,
        )

    def _toggle_listening(self, _sender: rumps.MenuItem) -> None:
        if self._engine is None:
            self.icon = self._icon_processing
            self.show_notification("Sheptun", t("notification_loading"))
            threading.Thread(target=self._init_and_start, daemon=True).start()
        elif self._is_listening:
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
        if self._is_listening:
            self._toggle_menu_item.title = f"{t('menu_toggle_stop')} ({toggle_display})"
        else:
            self._toggle_menu_item.title = f"{t('menu_toggle')} ({toggle_display})"

    def set_listening(self, listening: bool) -> None:
        self._is_listening = listening
        self.icon = self._icon_listening if listening else self._icon_idle

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
        subprocess.Popen(["open", str(app_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
