"""Tests for the calculator tool (wiki §8.1 template)."""

from __future__ import annotations

import pytest

from core.tool import ITool
from tools.calculator import CalculatorTool, safe_eval


class TestSafeEvalValid:
    def test_add_mul_precedence(self):
        assert safe_eval("2+3*4") == 14

    def test_parentheses(self):
        assert safe_eval("(2+3)*4") == 20

    def test_nested_parentheses(self):
        assert safe_eval("((1+2)*(3+4))") == 21

    def test_unary_operators(self):
        assert safe_eval("-3 + +2") == -1

    def test_power(self):
        assert safe_eval("2**10") == 1024

    def test_modulo(self):
        assert safe_eval("7 % 3") == 1

    def test_floor_div(self):
        assert safe_eval("7 // 2") == 3

    def test_true_div_returns_float(self):
        assert safe_eval("1/4") == 0.25

    def test_float_literals(self):
        assert safe_eval("1.5 + 2.5") == 4.0

    def test_negative_exponent(self):
        assert safe_eval("2 ** -2") == 0.25


class TestCalculatorToolRun:
    async def test_run_returns_str(self):
        tool = CalculatorTool()
        result = await tool.run(expression="2+2")
        assert isinstance(result, str)
        assert result == "4"

    async def test_run_float_result(self):
        tool = CalculatorTool()
        result = await tool.run(expression="1/4")
        assert result == "0.25"


class TestZeroDivision:
    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError, match="Division by zero"):
            safe_eval("1/0")

    def test_floor_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_eval("1//0")

    def test_modulo_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_eval("1%0")


class TestInjectionRejected:
    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo hi')",
            "().__class__.__bases__",
            "abs(-1)",
            "x + 1",
            "[1,2]",
            "'a' * 3",
            "True + 1",
            "lambda: 1",
            "1 if 1 else 2",
            "1 < 2",
        ],
    )
    def test_rejected_without_side_effects(self, expression):
        with pytest.raises(ValueError):
            safe_eval(expression)


class TestAntiDos:
    def test_huge_exponent_tower_rejected_quickly(self):
        with pytest.raises(ValueError):
            safe_eval("9**9**9")

    def test_large_exponent_rejected_quickly(self):
        with pytest.raises(ValueError):
            safe_eval("2**100000")

    def test_huge_base_power_rejected_quickly(self):
        with pytest.raises(ValueError, match="Result too large"):
            safe_eval("((10**1000)**1000)**1000")

    def test_moderate_power_still_allowed(self):
        assert safe_eval("2**1000") == 2**1000

    def test_expression_too_long_rejected(self):
        with pytest.raises(ValueError):
            safe_eval("1+" * 150 + "1")


class TestInvalidInput:
    def test_invalid_syntax(self):
        with pytest.raises(ValueError):
            safe_eval("2 +")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            safe_eval("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError):
            safe_eval("   ")


class TestSchemaAndContract:
    def test_is_itool(self):
        assert isinstance(CalculatorTool(), ITool)

    def test_name_is_snake_case(self):
        assert CalculatorTool.name == "calculator"
        assert CalculatorTool.name.islower()

    def test_schema_shape(self):
        params = CalculatorTool.parameters
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert params["required"] == ["expression"]
        assert "expression" in params["properties"]

    def test_from_config_returns_instance(self):
        tool = CalculatorTool.from_config({}, None)
        assert isinstance(tool, CalculatorTool)
