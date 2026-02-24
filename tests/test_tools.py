"""Tests for the single agent tools."""

import pytest

from single_agent.tools import (
    calculator,
    count_words,
    dispatch,
    get_current_datetime,
    reverse_text,
)


class TestCalculator:
    def test_addition(self):
        assert calculator("2 + 3") == "5"

    def test_multiplication(self):
        assert calculator("3 * 4") == "12"

    def test_operator_precedence(self):
        assert calculator("2 + 3 * 4") == "14"

    def test_power(self):
        assert calculator("2 ** 10") == "1024"

    def test_division(self):
        result = calculator("10 / 4")
        assert result == "2.5"

    def test_floor_division(self):
        assert calculator("10 // 3") == "3"

    def test_modulo(self):
        assert calculator("10 % 3") == "1"

    def test_sqrt_function(self):
        assert calculator("sqrt(144)") == "12"

    def test_unary_negation(self):
        assert calculator("-5 + 10") == "5"

    def test_invalid_expression(self):
        result = calculator("import os")
        assert result.startswith("Error")

    def test_unknown_name(self):
        result = calculator("malicious()")
        assert result.startswith("Error")


class TestGetCurrentDatetime:
    def test_returns_string(self):
        result = get_current_datetime()
        assert isinstance(result, str)

    def test_format(self):
        import re
        result = get_current_datetime()
        pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
        assert re.match(pattern, result), f"Unexpected format: {result}"


class TestCountWords:
    def test_simple(self):
        assert count_words("hello world") == "2"

    def test_single_word(self):
        assert count_words("hello") == "1"

    def test_empty_string(self):
        assert count_words("") == "0"

    def test_extra_spaces(self):
        assert count_words("  hello   world  ") == "2"


class TestReverseText:
    def test_simple(self):
        assert reverse_text("hello") == "olleh"

    def test_empty(self):
        assert reverse_text("") == ""

    def test_palindrome(self):
        assert reverse_text("racecar") == "racecar"


class TestDispatch:
    def test_known_tool(self):
        assert dispatch("calculator", {"expression": "1 + 1"}) == "2"

    def test_unknown_tool(self):
        result = dispatch("does_not_exist", {})
        assert "unknown tool" in result

    def test_bad_args(self):
        result = dispatch("count_words", {"wrong_param": "hello"})
        assert result.startswith("Error")
