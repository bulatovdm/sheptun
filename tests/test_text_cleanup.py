import pytest

from sheptun.text_cleanup import TextCleaner


@pytest.fixture
def cleaner() -> TextCleaner:
    return TextCleaner()


class TestPunctuationCollapse:
    """Дубль пунктуации на стыке (главный кейс с .env)."""

    def test_double_dot_before_env(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("..env") == ".env"

    def test_dot_then_dotenv(self, cleaner: TextCleaner) -> None:
        # "точка" → "." плюс правило env→".env" дают ".env" с лишней точкой
        assert cleaner.clean("настройки в . .env файле") == "настройки в .env файле"

    def test_double_comma(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("раз,, два") == "раз, два"

    def test_double_semicolon(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("a;; b") == "a; b"

    def test_double_question(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("правда??") == "правда?"

    def test_preserves_ellipsis(self, cleaner: TextCleaner) -> None:
        # многоточие — легитимный повтор, не трогаем
        assert cleaner.clean("подожди...") == "подожди..."

    def test_preserves_double_dash(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("git commit --amend") == "git commit --amend"

    def test_preserves_triple_equals(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("a === b") == "a === b"


class TestSpokenSymbolPlusSymbol:
    """Слово-команда пунктуации перед реальным символом дублирует его."""

    def test_word_dot_before_dot(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("файл точка .env") == "файл .env"

    def test_word_comma_before_comma(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("раз запятая , два") == "раз, два"


class TestWordRepeat:
    """Повтор целых слов от ASR/замен."""

    def test_repeated_word(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("коммит коммит") == "коммит"

    def test_repeated_word_case_insensitive(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("Docker docker") == "Docker"

    def test_no_false_repeat_different_words(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("да да ну хорошо") == "да ну хорошо"

    def test_keeps_meaningful_repeat_of_short(self, cleaner: TextCleaner) -> None:
        # "чуть-чуть" через дефис не разбивается на повтор
        assert cleaner.clean("чуть-чуть") == "чуть-чуть"


class TestWhitespace:
    def test_double_space(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("раз  два") == "раз два"

    def test_triple_space(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("раз   два") == "раз два"

    def test_space_before_punctuation(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("привет , как дела ?") == "привет, как дела?"

    def test_trim_edges(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("  привет  ") == "привет"


class TestProtectedTokens:
    """Не ломать URL / email / код при схлопывании."""

    def test_url_not_broken(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("https://example.com/path") == "https://example.com/path"

    def test_email_not_broken(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("user@example.com") == "user@example.com"

    def test_arrow_operator_preserved(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("obj->method") == "obj->method"


class TestExtensibility:
    """Модуль должен позволять добавлять правила."""

    def test_empty_stays_empty(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("") == ""

    def test_clean_text_unchanged(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("обычный текст без проблем") == "обычный текст без проблем"

    def test_rules_are_listed(self, cleaner: TextCleaner) -> None:
        # правила доступны как список для расширения
        assert len(cleaner.rules) > 0
        assert all(hasattr(r, "name") for r in cleaner.rules)
