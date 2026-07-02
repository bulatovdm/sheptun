from __future__ import annotations

from benchmarks.metrics import (
    damage,
    exact_match_rate,
    latin_tokens,
    term_tokens,
)

TERMS = frozenset({"докер", "гит", "коммит"})


class TestLatinTokens:
    def test_extracts_lowercased_latin(self) -> None:
        assert latin_tokens("подними Docker и GIT") == {"docker", "git"}

    def test_ignores_single_letters_and_cyrillic(self) -> None:
        assert latin_tokens("это a b тест докер") == set()


class TestTermTokens:
    def test_matches_whole_words_case_insensitive(self) -> None:
        assert term_tokens("сделай Коммит в гит", TERMS) == {"коммит", "гит"}

    def test_no_substring_match(self) -> None:
        # "гитара" must NOT count as the term "гит"
        assert term_tokens("возьми гитару", TERMS) == set()


class TestDamage:
    def test_lost_latin_when_english_dropped(self) -> None:
        lost_latin, lost_terms = damage("подними Docker", "подними уокер", TERMS)
        assert lost_latin == {"docker"}
        assert lost_terms == frozenset()

    def test_lost_term_when_translit_mangled(self) -> None:
        _lost_latin, lost_terms = damage("запушь в докер", "запушь в уокер", TERMS)
        assert lost_terms == {"докер"}

    def test_no_damage_when_terms_preserved(self) -> None:
        lost_latin, lost_terms = damage("сделай коммит", "Сделай коммит.", TERMS)
        assert not lost_latin
        assert not lost_terms


class TestExactMatch:
    def test_case_and_space_insensitive(self) -> None:
        rate = exact_match_rate([("Привет  Мир", "привет мир"), ("докер", "доктор")])
        assert rate == 0.5

    def test_none_when_empty(self) -> None:
        assert exact_match_rate([]) is None
