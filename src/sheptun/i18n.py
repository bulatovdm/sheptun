"""Локализация интерфейса."""

MESSAGES = {
    "menu_toggle": "Вкл/Выкл",
    "menu_toggle_stop": "Остановить",
    "menu_ptt": "Нажми и говори",
    "menu_restart": "Перезапустить",
    "menu_quit": "Выход",
    "notification_loading": "Загрузка модели...",
    "notification_error": "Ошибка",
    "help_commands": "энтер, таб, эскейп, пробел, вверх, вниз, влево, вправо, удали, клир, стоп",
    "help_title": "Sheptun - Команды",
}


def t(key: str) -> str:
    return MESSAGES.get(key, key)
