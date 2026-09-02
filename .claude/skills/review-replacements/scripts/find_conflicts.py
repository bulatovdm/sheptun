#!/usr/bin/env python3
"""Найти конфликты МЕЖДУ правилами replacements.yaml.

Обычное ревью смотрит на правило по отдельности и такие дефекты пропускает: каждое
правило само по себе разумно, ломает их только сочетание. Реальный пример — анализатор
в разных батчах породил `"commit": "коммит"` (freq=420) и `"коммит": "commit"`
(freq=1771); пара стоила 3 п.п. CER и 4 п.п. EPI.

Проверки:
  1. Обратные пары — A→B и B→A. Текст зависит от того, какое правило совпало первым.
     Регистровые нормализации (`api`→`API`) безвредны и не считаются конфликтом.
  2. Мёртвые цепочки — значение правила является ключом другого. Замена однопроходная
     (`commands.py:apply_replacements`), поэтому вторая ступень никогда не сработает:
     автор правила ждёт одного результата, получает другой.
  3. Порча терминов — правило переводит в кириллицу слово, которое другие правила
     считают эталонной латиницей. Такое слово ещё и подсказывается декодеру
     (`gigaam.load_hotwords`), то есть правило гасит работу биасинга.

Usage:
    python3 find_conflicts.py [REPLACEMENTS.yaml]

Код возврата 1, если найден хотя бы один конфликт.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

DEFAULT_SRC = "src/sheptun/config/replacements.yaml"
LATIN_TERM = re.compile(r"[A-Za-z][\w.+#/-]*")
CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def _load(path: Path) -> dict[str, str]:
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict):
        raise SystemExit(f"{path}: ожидался словарь правил")
    return rules


def find_reverse_pairs(rules: dict[str, str]) -> list[tuple[str, str]]:
    by_key = {key.lower(): value for key, value in rules.items()}
    seen: set[tuple[str, str]] = set()
    conflicts: list[tuple[str, str]] = []
    for key, value in rules.items():
        if key.lower() == value.lower():  # нормализация регистра, не конфликт
            continue
        back = by_key.get(value.lower())
        if back is None or back.lower() != key.lower():
            continue
        pair = tuple(sorted((key.lower(), value.lower())))
        if pair in seen:
            continue
        seen.add(pair)
        conflicts.append((key, value))
    return conflicts


def find_dead_chains(rules: dict[str, str]) -> list[tuple[str, str, str]]:
    by_key = {key.lower(): (key, value) for key, value in rules.items()}
    chains: list[tuple[str, str, str]] = []
    for key, value in rules.items():
        if key.lower() == value.lower():
            continue
        nxt = by_key.get(value.lower())
        if nxt is None or nxt[1].lower() == key.lower():  # обратные пары — отдельная проверка
            continue
        if nxt[1].lower() == value.lower():  # вторая ступень лишь правит регистр
            continue
        chains.append((key, value, nxt[1]))
    return chains


def find_term_breakers(rules: dict[str, str]) -> list[tuple[str, str]]:
    canonical = {
        value.lower() for value in rules.values() if LATIN_TERM.fullmatch(value)
    }
    return [
        (key, value)
        for key, value in rules.items()
        if key.lower() in canonical and CYRILLIC.search(value)
    ]


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC)
    rules = _load(src)
    print(f"правил: {len(rules)}  ({src})\n")

    reverse = find_reverse_pairs(rules)
    chains = find_dead_chains(rules)
    breakers = find_term_breakers(rules)

    if reverse:
        print(f"ОБРАТНЫЕ ПАРЫ ({len(reverse)}) — оставить только одно направление:")
        for key, value in reverse:
            print(f'  "{key}" → "{value}"   и обратно "{value}" → "{key}"')
        print()

    if breakers:
        print(f"ПОРЧА ТЕРМИНОВ ({len(breakers)}) — правило гасит латиницу, которую даёт биасинг:")
        for key, value in breakers:
            print(f'  "{key}" → "{value}"')
        print()

    if chains:
        print(f"МЁРТВЫЕ ЦЕПОЧКИ ({len(chains)}) — вторая ступень не сработает, замена однопроходная:")
        for key, value, further in chains[:20]:
            print(f'  "{key}" → "{value}", но "{value}" → "{further}" уже не применится')
        if len(chains) > 20:
            print(f"  … ещё {len(chains) - 20}")
        print()

    conflicts = len(reverse) + len(breakers)
    if conflicts == 0 and not chains:
        print("конфликтов не найдено")
        return 0

    print(f"конфликтов: {conflicts}, цепочек под пересмотр: {len(chains)}")
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
