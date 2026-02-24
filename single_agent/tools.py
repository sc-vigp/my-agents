"""
Built-in tools available to the single agent.

Each tool is defined as:
  - A Python function that performs the actual work.
  - An OpenAI tool-schema dict (TOOL_SCHEMAS) that tells the model what the
    tool does and what arguments it expects.
"""

import ast
import datetime
import math
import operator
from typing import Any

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES: dict[str, Any] = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a safe mathematical AST node."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        func = _safe_eval(node.func)
        if not callable(func):
            raise ValueError(f"Not callable: {func!r}")
        args = [_safe_eval(a) for a in node.args]
        return func(*args)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result as a string."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        # Return as int when result is a whole number for cleaner output
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def get_current_datetime() -> str:
    """Return the current date and time in ISO-8601 format."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def count_words(text: str) -> str:
    """Count the number of words in the provided text."""
    words = text.split()
    return str(len(words))


def reverse_text(text: str) -> str:
    """Reverse the characters in the provided text."""
    return text[::-1]


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression. "
                "Supports +, -, *, /, **, %, // and standard math functions "
                "such as sqrt(), sin(), cos(), log(), etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4' or 'sqrt(144)'.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Return the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_words",
            "description": "Count the number of words in a piece of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text whose words should be counted.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reverse_text",
            "description": "Reverse the characters in a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to reverse.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Dispatcher: map tool name → function
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "calculator": calculator,
    "get_current_datetime": get_current_datetime,
    "count_words": count_words,
    "reverse_text": reverse_text,
}


def dispatch(tool_name: str, tool_args: dict) -> str:
    """Call the named tool with the supplied arguments and return a string result."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return f"Error: unknown tool '{tool_name}'"
    try:
        return fn(**tool_args)
    except Exception as exc:
        return f"Error calling '{tool_name}': {exc}"
