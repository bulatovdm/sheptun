# pyright: reportPrivateUsage=false
import os
from unittest.mock import patch

from sheptun.settings import _get_bool, _get_float, _get_optional_str, _get_str


class TestGetBool:
    def test_returns_default_when_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _get_bool("NONEXISTENT_KEY", False) is False
            assert _get_bool("NONEXISTENT_KEY", True) is True

    def test_returns_true_for_true_values(self) -> None:
        for value in ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]:
            with patch.dict(os.environ, {"TEST_KEY": value}):
                assert _get_bool("TEST_KEY") is True

    def test_returns_false_for_false_values(self) -> None:
        for value in ["false", "False", "FALSE", "0", "no", "No", "anything"]:
            with patch.dict(os.environ, {"TEST_KEY": value}):
                assert _get_bool("TEST_KEY") is False


class TestGetFloat:
    def test_returns_default_when_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _get_float("NONEXISTENT_KEY", 1.5) == 1.5

    def test_returns_float_value(self) -> None:
        with patch.dict(os.environ, {"TEST_KEY": "2.5"}):
            assert _get_float("TEST_KEY", 0.0) == 2.5

    def test_returns_integer_as_float(self) -> None:
        with patch.dict(os.environ, {"TEST_KEY": "5"}):
            assert _get_float("TEST_KEY", 0.0) == 5.0


class TestGetStr:
    def test_returns_default_when_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _get_str("NONEXISTENT_KEY", "default") == "default"

    def test_returns_string_value(self) -> None:
        with patch.dict(os.environ, {"TEST_KEY": "value"}):
            assert _get_str("TEST_KEY", "default") == "value"


class TestGetOptionalStr:
    def test_returns_none_when_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _get_optional_str("NONEXISTENT_KEY") is None

    def test_returns_none_for_empty_string(self) -> None:
        with patch.dict(os.environ, {"TEST_KEY": ""}):
            assert _get_optional_str("TEST_KEY") is None

    def test_returns_string_value(self) -> None:
        with patch.dict(os.environ, {"TEST_KEY": "value"}):
            assert _get_optional_str("TEST_KEY") == "value"
