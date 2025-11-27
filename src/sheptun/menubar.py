import importlib.resources
import json
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Any

import rumps
from pynput import keyboard

from sheptun.audio import AudioConfig, VoiceActivityConfig
from sheptun.commands import CommandConfigLoader, CommandParser
from sheptun.config import get_config_path
from sheptun.keyboard import MacOSKeyboardSender
from sheptun.recognition import WhisperRecognizer
from sheptun.settings import settings, setup_logging
from sheptun.types import Action, ActionType

setup_logging()
logger = logging.getLogger("sheptun.menubar")

PREFS_FILE = Path.home() / ".config" / "sheptun" / "menubar_prefs.json"


class HotkeyMode(Enum):
    TOGGLE = "toggle"
    PUSH_TO_TALK = "push_to_talk"


def _load_prefs() -> dict[str, Any]:
    """Load preferences from file."""
    if PREFS_FILE.exists():
        try:
            return json.loads(PREFS_FILE.read_text())  # type: ignore[no-any-return]
        except Exception:
            pass
    return {}


def _save_prefs(prefs: dict[str, Any]) -> None:
    """Save preferences to file."""
    PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREFS_FILE.write_text(json.dumps(prefs))


def _parse_hotkey(hotkey_str: str) -> set[Any] | None:
    """Parse hotkey string like '<cmd>+<shift>+m' into set of keys."""
    try:
        keys: set[Any] = set()
        for part in hotkey_str.split("+"):
            part = part.strip().lower()
            if part == "<cmd>":
                keys.add(keyboard.Key.cmd)
            elif part == "<shift>":
                keys.add(keyboard.Key.shift)
            elif part == "<ctrl>":
                keys.add(keyboard.Key.ctrl)
            elif part == "<alt>":
                keys.add(keyboard.Key.alt)
            elif len(part) == 1:
                keys.add(keyboard.KeyCode.from_char(part))
            else:
                logger.warning(f"Unknown hotkey part: {part}")
                return None
        return keys
    except Exception as e:
        logger.warning(f"Failed to parse hotkey '{hotkey_str}': {e}")
        return None


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
        self._app.show_notification("Ошибка", message)

    def idle(self) -> None:
        self._app.set_listening(False)

    def show_recognized(self, _text: str) -> None:
        pass

    def show_action(self, _action_description: str) -> None:
        pass

    def show_help(self) -> None:
        import subprocess

        logger.info("Showing help notification")
        message = "энтер, таб, эскейп, пробел, вверх, вниз, влево, вправо, удали, клир, стоп"
        script = f'display notification "{message}" with title "Sheptun - Команды"'
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
        """Recognize audio and execute action (for push-to-talk)."""
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
        self._hotkey_listener: keyboard.Listener | None = None
        self._hotkey_keys: set[Any] = _parse_hotkey(settings.hotkey) or set()
        self._pressed_keys: set[Any] = set()
        self._hotkey_active = False

        prefs = _load_prefs()
        mode_str = prefs.get("hotkey_mode", HotkeyMode.TOGGLE.value)
        self._hotkey_mode = HotkeyMode(mode_str) if mode_str in [m.value for m in HotkeyMode] else HotkeyMode.TOGGLE

        self._icon_idle = _get_icon_path("mic_idle.png")
        self._icon_listening = _get_icon_path("mic_active.png")
        self._icon_processing = _get_icon_path("mic_processing.png")

        self.icon = self._icon_idle
        hotkey_display = settings.hotkey.replace("<cmd>", "⌘").replace("<shift>", "⇧").replace("<ctrl>", "⌃").replace("<alt>", "⌥").upper()
        self._start_menu_item = rumps.MenuItem(f"Начать слушать ({hotkey_display})", callback=self._toggle_listening)

        self._mode_toggle = rumps.MenuItem("Toggle (вкл/выкл)", callback=self._set_mode_toggle)
        self._mode_ptt = rumps.MenuItem("Push-to-talk (удерживать)", callback=self._set_mode_ptt)
        self._update_mode_menu()

        self.menu = [
            self._start_menu_item,
            None,
            ["Режим хоткея", [self._mode_toggle, self._mode_ptt]],
            None,
            rumps.MenuItem("Перезапустить", callback=self._restart),
            rumps.MenuItem("Выход", callback=self._quit),
        ]
        self._start_hotkey_listener()

    def _update_mode_menu(self) -> None:
        self._mode_toggle.state = self._hotkey_mode == HotkeyMode.TOGGLE
        self._mode_ptt.state = self._hotkey_mode == HotkeyMode.PUSH_TO_TALK

    def _set_mode_toggle(self, _: Any) -> None:
        self._hotkey_mode = HotkeyMode.TOGGLE
        self._update_mode_menu()
        _save_prefs({"hotkey_mode": self._hotkey_mode.value})
        logger.info("Switched to toggle mode")

    def _set_mode_ptt(self, _: Any) -> None:
        self._hotkey_mode = HotkeyMode.PUSH_TO_TALK
        self._update_mode_menu()
        _save_prefs({"hotkey_mode": self._hotkey_mode.value})
        logger.info("Switched to push-to-talk mode")

    def _start_hotkey_listener(self) -> None:
        if not self._hotkey_keys:
            logger.warning("No hotkey configured, skipping listener")
            return

        def on_press(key: Any) -> None:
            self._pressed_keys.add(key)
            if self._hotkey_keys.issubset(self._pressed_keys) and not self._hotkey_active:
                self._hotkey_active = True
                if self._hotkey_mode == HotkeyMode.TOGGLE:
                    logger.info("Hotkey pressed, toggling listening")
                    self._toggle_listening(self._start_menu_item)
                else:
                    logger.info("Hotkey pressed, starting push-to-talk")
                    self._start_ptt()

        def on_release(key: Any) -> None:
            self._pressed_keys.discard(key)
            if self._hotkey_active and not self._hotkey_keys.issubset(self._pressed_keys):
                self._hotkey_active = False
                if self._hotkey_mode == HotkeyMode.PUSH_TO_TALK:
                    logger.info("Hotkey released, stopping push-to-talk")
                    self._stop_ptt()

        self._hotkey_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._hotkey_listener.start()
        logger.info(f"Hotkey listener started for {settings.hotkey}")

    def _start_ptt(self) -> None:
        """Start push-to-talk recording."""
        from sheptun.audio import AudioRecorder

        if self._engine is None:
            self.icon = self._icon_processing
            self.show_notification("Sheptun", "Загрузка модели...")
            self._init_engine()

        if not hasattr(self, "_ptt_recorder"):
            self._ptt_recorder = AudioRecorder()

        self.icon = self._icon_listening
        self._ptt_recorder.start()

    def _stop_ptt(self) -> None:
        """Stop push-to-talk and process audio."""
        if not hasattr(self, "_ptt_recorder"):
            return

        audio_data = self._ptt_recorder.stop()
        if len(audio_data) < 1000:
            self.icon = self._icon_idle
            return

        self.icon = self._icon_processing
        threading.Thread(target=self._process_ptt_audio, args=(audio_data,), daemon=True).start()

    def _process_ptt_audio(self, audio_data: bytes) -> None:
        """Process push-to-talk audio in background."""
        if self._engine is None:
            self._init_engine()

        if self._engine is None:
            self.icon = self._icon_idle
            return

        try:
            self._engine.recognize_and_execute(audio_data)
        except Exception as e:
            logger.exception(f"PTT error: {e}")
        finally:
            self.icon = self._icon_idle

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

    def _toggle_listening(self, sender: rumps.MenuItem) -> None:
        if self._engine is None:
            self.icon = self._icon_processing
            self.show_notification("Sheptun", "Загрузка модели...")
            threading.Thread(target=self._init_and_start, daemon=True).start()
        elif self._is_listening:
            self._engine.stop()
            sender.title = "Начать слушать"
        else:
            self._engine.start()
            sender.title = "Остановить"

    def _init_and_start(self) -> None:
        self._init_engine()
        if self._engine is not None:
            self._engine.start()
            self._start_menu_item.title = "Остановить"

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
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()

        app_path = settings.app_path
        subprocess.Popen(["open", str(app_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os._exit(0)

    def _quit(self, _: Any) -> None:
        if self._engine is not None:
            self._engine.stop()
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
        rumps.quit_application()  # type: ignore[no-untyped-call]


def _hide_dock_icon() -> None:
    """Hide the app icon from the Dock."""
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
