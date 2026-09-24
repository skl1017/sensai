"""Calculator tool: the template every new tool copies (wiki §8.1).

This is deliberately the smallest possible `ITool`: no I/O, no external
dependency, a tiny schema, one method. It exists to validate the `ReAct`
loop end-to-end (model calls a tool, gets an observation, answers) before
any tool with real side effects is wired in — copy this file's shape
(`name`, `description`, `parameters`, `run`, `from_config`) as the starting
point for the next tool.

Safety design
-------------
Arithmetic expressions come from the model, which is untrusted input, so
`safe_eval` never calls `eval()` or `compile()` on them. Instead it parses
the expression into an AST (`ast.parse(..., mode="eval")`) and walks it
itself, allow-listing exactly the node types a calculator needs: numeric
constants, the binary operators (`+ - * / // % **`), and unary `+`/`-`.
Anything else — `Name` (variables/builtins), `Call` (function calls,
`__import__` included), `Attribute`/`Subscript` (attribute or item access,
the classic `().__class__.__bases__` sandbox-escape gadget), `Compare`,
`BoolOp`, `Lambda`, `Tuple`, string/bytes constants, `bool`/`None`/complex
constants — is rejected with `ValueError` before it can be evaluated. `bool`
is a subclass of `int` in Python, so it is rejected explicitly (`True`
would otherwise silently evaluate as `1`).

Two anti-DoS guards keep the walk itself cheap: `MAX_EXPRESSION_CHARS`
rejects absurdly long input before parsing, and `MAX_EXPONENT` rejects a
huge `**` exponent before it is computed (`9**9**9` would otherwise try to
build a number with billions of digits), and `MAX_RESULT_BITS` rejects an
integer power whose result would be huge even with a small exponent
(`((10**1000)**1000)**1000`).
"""

from __future__ import annotations

import ast
import operator

from core.tool import ITool

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 1000
MAX_RESULT_BITS = 10_000  # caps int powers: `(10**1000)**1000` is rejected before computing

_BIN_OPS: dict[type[ast.operator], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expression: str) -> int | float:
    """Evaluate an arithmetic expression without ever calling `eval()`.

    Only numeric constants, `+ - * / // % **` and unary `+`/`-` are
    allowed; any other AST node raises `ValueError`. Division/modulo by
    zero raise `ZeroDivisionError`.
    """
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(
            f"Expression too long ({len(expression)} chars, max {MAX_EXPRESSION_CHARS})"
        )
    if not expression.strip():
        raise ValueError("Empty expression")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expression!r}") from exc

    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"Unsupported constant: {value!r}")
        return value

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported expression element: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"Exponent too large (max {MAX_EXPONENT})")
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(left, int)
            and isinstance(right, int)
            and left.bit_length() * right > MAX_RESULT_BITS
        ):
            raise ValueError(f"Result too large (max {MAX_RESULT_BITS} bits)")
        if isinstance(node.op, ast.Div | ast.FloorDiv | ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero")
        try:
            return op(left, right)
        except OverflowError as exc:
            raise ValueError(f"Result too large to represent: {_expression_repr(node)}") from exc

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported expression element: {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _expression_repr(node: ast.AST) -> str:
    """Best-effort text for an error message; never re-parses or evals."""
    try:
        return ast.unparse(node)
    except Exception:
        return "<expression>"


class CalculatorTool(ITool):
    """Evaluate a small arithmetic expression. The template tool (wiki §8.1)."""

    name = "calculator"
    description = (
        "Evaluate an arithmetic expression (+ - * / // % ** and parentheses). "
        "Use it for any calculation instead of computing in your head."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression to evaluate, e.g. '(2 + 3) * 4'.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    async def run(self, expression: str) -> str:
        # Pure CPU, bounded by MAX_EXPRESSION_CHARS/MAX_EXPONENT: no I/O to
        # push to a thread, unlike tools that touch the filesystem or network.
        return str(safe_eval(expression))

    @classmethod
    def from_config(cls, params: dict, deps: object) -> CalculatorTool:
        # `deps` is untyped here (not `Deps` from app.py): tools depend only
        # on core/, never on app.py, so the concrete type can't be imported.
        return cls()
