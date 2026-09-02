"""A tiny local MCP server (stdio) to prove the agent's MCP plumbing end to end.

Tools: calculate, convert_units, roll_dice. Run by the agent via mcp.json; or by hand:
    .venv\\Scripts\\python.exe agent\\mcp_servers\\demo_server.py
"""
from __future__ import annotations

import ast
import operator as op
import random

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
        ast.Mod: op.mod, ast.USub: op.neg, ast.FloorDiv: op.floordiv}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / ** % // are allowed")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. '17 * 23 + 4 / 8'. Use for any math."""
    try:
        v = _eval(ast.parse(expression, mode="eval").body)
        return f"{expression} = {v:g}" if isinstance(v, float) else f"{expression} = {v}"
    except Exception as e:  # noqa: BLE001
        return f"cannot evaluate: {e}"


_UNITS = {("km", "mi"): 0.621371, ("mi", "km"): 1.609344, ("kg", "lb"): 2.204623, ("lb", "kg"): 0.453592,
          ("c", "f"): None, ("f", "c"): None, ("m", "ft"): 3.28084, ("ft", "m"): 0.3048, ("l", "gal"): 0.264172,
          ("gal", "l"): 3.785412}


@mcp.tool()
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between km/mi, kg/lb, m/ft, l/gal, c/f."""
    key = (from_unit.lower(), to_unit.lower())
    if key == ("c", "f"):
        return f"{value} C = {value * 9 / 5 + 32:.1f} F"
    if key == ("f", "c"):
        return f"{value} F = {(value - 32) * 5 / 9:.1f} C"
    if key in _UNITS:
        return f"{value} {from_unit} = {value * _UNITS[key]:.3f} {to_unit}"
    return f"don't know how to convert {from_unit} to {to_unit}"


@mcp.tool()
def roll_dice(sides: int = 6, count: int = 1) -> str:
    """Roll dice, e.g. roll_dice(20, 2) rolls two twenty-sided dice."""
    rolls = [random.randint(1, max(2, sides)) for _ in range(max(1, min(count, 20)))]
    return f"rolled {rolls} (total {sum(rolls)})"


if __name__ == "__main__":
    mcp.run(transport="stdio")
