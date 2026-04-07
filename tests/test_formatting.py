# pyright: reportPrivateUsage=false
from sheptun.formatting import TechnicalFormatter

_formatter = TechnicalFormatter()
format_technical = _formatter.format


class TestAlwaysConvertSymbols:
    def test_underscore(self) -> None:
        assert format_technical("document нижнее подчёркивание file") == "document_file"

    def test_underscore_no_yo(self) -> None:
        assert format_technical("document нижнее подчеркивание file") == "document_file"

    def test_underscore_anderskor(self) -> None:
        assert format_technical("document андерскор file") == "document_file"

    def test_underscore_lowercase(self) -> None:
        assert format_technical("Document нижнее подчёркивание File") == "document_file"

    def test_double_dash(self) -> None:
        assert format_technical("тире тире verbose") == "--verbose"

    def test_comma(self) -> None:
        assert "," in format_technical("a запятая b")

    def test_colon(self) -> None:
        assert ":" in format_technical("key двоеточие value")

    def test_semicolon(self) -> None:
        assert ";" in format_technical("a точка с запятой b")

    def test_semicolon_alt(self) -> None:
        assert ";" in format_technical("a точка запятая b")

    def test_at_sign(self) -> None:
        assert "@" in format_technical("user собачка example")

    def test_hash(self) -> None:
        assert "#" in format_technical("хэш comment")

    def test_pipe(self) -> None:
        assert "|" in format_technical("grep пайп sort")

    def test_double_equals(self) -> None:
        assert "==" in format_technical("x двойное равно y")

    def test_triple_equals(self) -> None:
        assert "===" in format_technical("x тройное равно y")

    def test_fat_arrow(self) -> None:
        assert "=>" in format_technical("x жирная стрелка y")

    def test_backslash(self) -> None:
        assert "\\" in format_technical("обратный слэш n")

    def test_parens(self) -> None:
        result = format_technical("func открыть скобку закрыть скобку")
        assert "(" in result
        assert ")" in result

    def test_brackets(self) -> None:
        result = format_technical("arr открыть квадратную 0 закрыть квадратную")
        assert "[" in result
        assert "]" in result

    def test_braces(self) -> None:
        result = format_technical("открыть фигурную закрыть фигурную")
        assert "{" in result
        assert "}" in result

    def test_double_quote(self) -> None:
        assert '"' in format_technical("двойная кавычка hello двойная кавычка")

    def test_backtick(self) -> None:
        assert "`" in format_technical("бэктик hello бэктик")

    def test_exclamation(self) -> None:
        assert "!" in format_technical("восклицательный знак")

    def test_question(self) -> None:
        assert "?" in format_technical("вопросительный знак")

    def test_spread(self) -> None:
        assert "..." in format_technical("многоточие args")

    def test_tilde(self) -> None:
        assert "~" in format_technical("тильда .config")

    def test_normal_speech_not_converted(self) -> None:
        assert format_technical("у него собака дома") == "у него собака дома"
        assert format_technical("это равно пяти") == "это равно пяти"
        assert format_technical("два плюс два") == "два плюс два"
        assert format_technical("a минус b") == "a минус b"
        assert format_technical("стрелка вправо") == "стрелка вправо"


class TestContextSymbols:
    def test_tochka_with_extension(self) -> None:
        assert format_technical("main точка py") == "main.py"

    def test_tochka_env(self) -> None:
        assert format_technical("файл точка env") == "файл.env"

    def test_tochka_without_extension(self) -> None:
        assert format_technical("точка зрения") == "точка зрения"

    def test_tochka_php(self) -> None:
        assert "." in format_technical("controller точка php")
        assert "php" in format_technical("controller точка php")

    def test_slash_between_words(self) -> None:
        assert format_technical("src слэш main") == "src/main"

    def test_slash_at_start_preserved(self) -> None:
        assert format_technical("слэш хелп") == "слэш хелп"

    def test_slash_slesh_variant(self) -> None:
        assert format_technical("src слеш main") == "src/main"

    def test_single_tire_between_latin(self) -> None:
        assert format_technical("feature тире branch") == "feature-branch"

    def test_single_tire_between_cyrillic_preserved(self) -> None:
        assert format_technical("поставь тире здесь") == "поставь тире здесь"

    def test_cyrillic_not_captured_by_auto_casing(self) -> None:
        result = format_technical("создай user service точка py")
        assert "создай" in result
        assert result.endswith(".py")


class TestCasingCommands:
    def test_camel_case(self) -> None:
        assert format_technical("кэмел кейс get user name") == "getUserName"

    def test_camel_case_alt(self) -> None:
        assert format_technical("камел кейс get user name") == "getUserName"

    def test_snake_case(self) -> None:
        assert format_technical("снейк кейс get user name") == "get_user_name"

    def test_pascal_case(self) -> None:
        assert format_technical("паскаль кейс user service") == "UserService"

    def test_pascal_case_alt(self) -> None:
        assert format_technical("паскал кейс user service") == "UserService"

    def test_kebab_case(self) -> None:
        assert format_technical("кебаб кейс my component") == "my-component"

    def test_camel_case_merged(self) -> None:
        assert format_technical("Камелькейс, get user name") == "getUserName"

    def test_snake_case_english(self) -> None:
        assert format_technical("SnakeCase get user name") == "get_user_name"

    def test_camel_case_english(self) -> None:
        assert format_technical("CamelCase get user name") == "getUserName"

    def test_pascal_case_english(self) -> None:
        assert format_technical("PascalCase user service") == "UserService"


class TestAutoCasingByExtension:
    def test_py_snake_case(self) -> None:
        assert format_technical("user service точка py") == "user_service.py"

    def test_php_pascal_case(self) -> None:
        assert format_technical("user service точка php") == "UserService.php"

    def test_js_camel_case(self) -> None:
        assert format_technical("user service точка js") == "userService.js"

    def test_ts_camel_case(self) -> None:
        assert format_technical("user service точка ts") == "userService.ts"

    def test_no_casing_for_unknown_ext(self) -> None:
        result = format_technical("readme точка md")
        assert result == "readme.md"

    def test_single_word_no_casing(self) -> None:
        assert format_technical("main точка py") == "main.py"


class TestCombinations:
    def test_full_path(self) -> None:
        text = "src слэш controllers слэш main точка py"
        assert format_technical(text) == "src/controllers/main.py"

    def test_underscore_with_extension(self) -> None:
        text = "User нижнее подчёркивание Model точка py"
        assert format_technical(text) == "user_model.py"

    def test_complex_path(self) -> None:
        text = "src слэш models слэш User нижнее подчёркивание Model точка py"
        assert format_technical(text) == "src/models/user_model.py"

    def test_cli_flag(self) -> None:
        assert format_technical("git push тире тире force") == "git push --force"


class TestNoOp:
    def test_normal_text(self) -> None:
        text = "Привет, как дела у тебя сегодня"
        assert format_technical(text) == text

    def test_empty_string(self) -> None:
        assert format_technical("") == ""

    def test_english_text(self) -> None:
        text = "Hello world"
        assert format_technical(text) == text
