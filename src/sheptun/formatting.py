import re
from functools import partial

_EXTENSIONS = frozenset({
    "env", "php", "py", "js", "ts", "jsx", "tsx", "yaml", "yml",
    "json", "md", "html", "css", "txt", "sh", "sql", "go", "rs",
    "rb", "vue", "svelte", "xml", "toml", "cfg", "ini", "log",
    "gitignore", "dockerignore", "editorconfig", "lock", "map",
})

_SNAKE_EXTENSIONS = frozenset({"py", "rb", "sh", "sql"})
_CAMEL_EXTENSIONS = frozenset({"js", "ts", "jsx", "tsx", "vue", "svelte"})
_PASCAL_EXTENSIONS = frozenset({"php"})

_ALWAYS_SYMBOLS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bнижнее\s+подч[её]ркивание\b", re.IGNORECASE), "_"),
    (re.compile(r"\bточка\s+с\s+запятой\b", re.IGNORECASE), ";"),
    (re.compile(r"\bточка\s+запятая\b", re.IGNORECASE), ";"),
    (re.compile(r"\bтире\s+тире\b", re.IGNORECASE), "--"),
    (re.compile(r"\bтройное\s+равно\b", re.IGNORECASE), "==="),
    (re.compile(r"\bдвойное\s+равно\b", re.IGNORECASE), "=="),
    (re.compile(r"\bжирная\s+стрелка\b", re.IGNORECASE), "=>"),
    (re.compile(r"\bтолстая\s+стрелка\b", re.IGNORECASE), "=>"),
    (re.compile(r"\bобратный\s+сл[эе]ш\b", re.IGNORECASE), "\\"),
    (re.compile(r"\bбэксл[эе]ш\b", re.IGNORECASE), "\\"),
    (re.compile(r"\bвертикальная\s+черта\b", re.IGNORECASE), "|"),
    (re.compile(r"\bоткрыть\s+скобку\b", re.IGNORECASE), "("),
    (re.compile(r"\bзакрыть\s+скобку\b", re.IGNORECASE), ")"),
    (re.compile(r"\bоткрыть\s+круглую\b", re.IGNORECASE), "("),
    (re.compile(r"\bзакрыть\s+круглую\b", re.IGNORECASE), ")"),
    (re.compile(r"\bоткрыть\s+квадратную\b", re.IGNORECASE), "["),
    (re.compile(r"\bзакрыть\s+квадратную\b", re.IGNORECASE), "]"),
    (re.compile(r"\bоткрыть\s+фигурную\b", re.IGNORECASE), "{"),
    (re.compile(r"\bзакрыть\s+фигурную\b", re.IGNORECASE), "}"),
    (re.compile(r"\bодинарная\s+кавычка\b", re.IGNORECASE), "'"),
    (re.compile(r"\bдвойная\s+кавычка\b", re.IGNORECASE), '"'),
    (re.compile(r"\bобратная\s+кавычка\b", re.IGNORECASE), "`"),
    (re.compile(r"\bвосклицательный\s+знак\b", re.IGNORECASE), "!"),
    (re.compile(r"\bвопросительный\s+знак\b", re.IGNORECASE), "?"),
    (re.compile(r"\bандерскор\b", re.IGNORECASE), "_"),
    (re.compile(r"\bзапятая\b", re.IGNORECASE), ","),
    (re.compile(r"\bдвоеточие\b", re.IGNORECASE), ":"),
    (re.compile(r"\bсобачка\b", re.IGNORECASE), "@"),
    (re.compile(r"\bхэш\b", re.IGNORECASE), "#"),
    (re.compile(r"\bамперсанд\b", re.IGNORECASE), "&"),
    (re.compile(r"\bпайп\b", re.IGNORECASE), "|"),
    (re.compile(r"\bбэктик\b", re.IGNORECASE), "`"),
    (re.compile(r"\bмноготочие\b", re.IGNORECASE), "..."),
    (re.compile(r"\bтильда\b", re.IGNORECASE), "~"),
]

_TOCHKA_EXT = re.compile(
    r"\bточка\s+("
    + "|".join(re.escape(e) for e in sorted(_EXTENSIONS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

_SINGLE_TIRE_LATIN = re.compile(
    r"(?<=[a-zA-Z])\s+тире\s+(?=[a-zA-Z])", re.IGNORECASE
)

_SLASH_PREFIXES_LOWER = ("слэш ", "слеш ", "слышь ")

_CASING_SEP = r"[,.\s]*\s+"

_CASING_COMMANDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^к[аэе]мел[ьъ]?" + _CASING_SEP + r"кейс" + _CASING_SEP, re.IGNORECASE), "camel"),
    (re.compile(r"^к[аэе]мел[ьъ]?кейс" + _CASING_SEP, re.IGNORECASE), "camel"),
    (re.compile(r"^camel\s*case" + _CASING_SEP, re.IGNORECASE), "camel"),
    (re.compile(r"^снейк" + _CASING_SEP + r"кейс" + _CASING_SEP, re.IGNORECASE), "snake"),
    (re.compile(r"^снейккейс" + _CASING_SEP, re.IGNORECASE), "snake"),
    (re.compile(r"^snake\s*case" + _CASING_SEP, re.IGNORECASE), "snake"),
    (re.compile(r"^паскал[ьъ]?" + _CASING_SEP + r"кейс" + _CASING_SEP, re.IGNORECASE), "pascal"),
    (re.compile(r"^паскал[ьъ]?кейс" + _CASING_SEP, re.IGNORECASE), "pascal"),
    (re.compile(r"^pascal\s*case" + _CASING_SEP, re.IGNORECASE), "pascal"),
    (re.compile(r"^кебаб" + _CASING_SEP + r"кейс" + _CASING_SEP, re.IGNORECASE), "kebab"),
    (re.compile(r"^кебабкейс" + _CASING_SEP, re.IGNORECASE), "kebab"),
    (re.compile(r"^kebab\s*case" + _CASING_SEP, re.IGNORECASE), "kebab"),
]

_DOT_EXT_SPACE = re.compile(
    r"(\S)\s+\.("
    + "|".join(re.escape(e) for e in sorted(_EXTENSIONS, key=len, reverse=True))
    + r")\b"
)
_UNDERSCORE_SEGMENT = re.compile(r"\w+(?:_\w+)+")


def _to_camel_case(words: list[str]) -> str:
    if not words:
        return ""
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


def _to_pascal_case(words: list[str]) -> str:
    return "".join(w.capitalize() for w in words)


def _to_snake_case(words: list[str]) -> str:
    return "_".join(w.lower() for w in words)


def _to_kebab_case(words: list[str]) -> str:
    return "-".join(w.lower() for w in words)


_CASING_FUNCS = {
    "camel": _to_camel_case,
    "snake": _to_snake_case,
    "pascal": _to_pascal_case,
    "kebab": _to_kebab_case,
}


class TechnicalFormatter:

    def format(self, text: str) -> str:
        result, had_casing = self._apply_casing_command(text)
        if had_casing:
            return result

        for pattern, symbol in _ALWAYS_SYMBOLS:
            text = pattern.sub(partial(lambda r, _m: r, symbol), text)

        text = _TOCHKA_EXT.sub(r".\1", text)
        text = _DOT_EXT_SPACE.sub(r"\1.\2", text)
        text = self._convert_slash(text)
        text = _SINGLE_TIRE_LATIN.sub("-", text)
        text = self._collapse_spaces(text)
        text = self._lowercase_around_underscores(text)
        text = self._auto_casing_by_extension(text)

        return text

    def _apply_casing_command(self, text: str) -> tuple[str, bool]:
        for pattern, style in _CASING_COMMANDS:
            m = pattern.match(text)
            if m:
                words = text[m.end():].split()
                if words:
                    return _CASING_FUNCS[style](words), True
                return "", True
        return text, False

    @staticmethod
    def _convert_slash(text: str) -> str:
        text_lower = text.lstrip().lower()
        for prefix in _SLASH_PREFIXES_LOWER:
            if text_lower.startswith(prefix):
                return text
        return re.sub(r"\bсл[эе]ш\b", "/", text, flags=re.IGNORECASE)

    @staticmethod
    def _collapse_spaces(text: str) -> str:
        text = re.sub(r"\s*_\s*", "_", text)
        text = re.sub(r"\s*/\s*", "/", text)
        text = re.sub(r"--\s+(?=\S)", "--", text)
        return text

    @staticmethod
    def _lowercase_around_underscores(text: str) -> str:
        if "_" not in text:
            return text
        return _UNDERSCORE_SEGMENT.sub(lambda m: m.group(0).lower(), text)

    def _auto_casing_by_extension(self, text: str) -> str:
        m = re.search(r"\.(\w+)$", text)
        if not m:
            return text
        ext = m.group(1).lower()

        if ext not in (_SNAKE_EXTENSIONS | _CAMEL_EXTENSIONS | _PASCAL_EXTENSIONS):
            return text

        before_ext = text[: m.start()]
        last_slash = before_ext.rfind("/")
        if last_slash >= 0:
            prefix = before_ext[: last_slash + 1]
            filename = before_ext[last_slash + 1 :]
        else:
            words = before_ext.split()
            if not words:
                return text
            filename_words: list[str] = []
            for w in reversed(words):
                if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", w):
                    filename_words.insert(0, w)
                else:
                    break
            if len(filename_words) <= 1:
                return text
            prefix = before_ext[: before_ext.rfind(filename_words[0])]
            filename = " ".join(filename_words)

        parts = re.split(r"[_\s]+", filename)
        parts = [p for p in parts if p]
        if len(parts) <= 1:
            return text

        if ext in _SNAKE_EXTENSIONS:
            new_name = _to_snake_case(parts)
        elif ext in _CAMEL_EXTENSIONS:
            new_name = _to_camel_case(parts)
        else:
            new_name = _to_pascal_case(parts)

        return prefix + new_name + "." + m.group(1)
