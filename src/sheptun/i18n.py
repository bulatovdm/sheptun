MESSAGES = {
    "menu_toggle": "Вкл/Выкл",
    "menu_toggle_stop": "Остановить",
    "menu_ptt": "Нажми и говори",
    "menu_restart": "Перезапустить",
    "menu_quit": "Выход",
    "notification_loading": "Загрузка модели...",
    "notification_downloading": "Скачивание модели...",
    "notification_error": "Ошибка",
    "help_commands": "энтер, таб, эскейп, пробел, вверх, вниз, влево, вправо, удали, клир, стоп",
    "help_title": "Sheptun - Команды",
    "menu_remote_status": "Remote: поиск...",
    "menu_remote_connected": "Remote: {host}",
    "menu_remote_disconnected": "Remote: не подключён",
}


def t(key: str) -> str:
    return MESSAGES.get(key, key)
