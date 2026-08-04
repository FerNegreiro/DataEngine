from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAG_PATH = PROJECT_ROOT / "airflow" / "dags" / "dataengine_full_pipeline.py"


def dag_tree() -> ast.Module:
    return ast.parse(DAG_PATH.read_text(encoding="utf-8"), filename=str(DAG_PATH))


def callable_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def dag_call(tree: ast.Module) -> ast.Call:
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call) and callable_name(
                    item.context_expr
                ) == "DAG":
                    return item.context_expr
    raise AssertionError("DAG(...) não encontrado")


def task_calls(tree: ast.Module) -> dict[str, ast.Call]:
    tasks: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and isinstance(node.value, ast.Call)
            and callable_name(node.value) == "BashOperator"
        ):
            tasks[target.id] = node.value
    return tasks


def keyword_node(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"Keyword {name} ausente")


def keyword_value(call: ast.Call, name: str) -> object:
    return ast.literal_eval(keyword_node(call, name))


def _dependency_names(node: ast.expr) -> list[str]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
        return _dependency_names(node.left) + _dependency_names(node.right)
    if isinstance(node, ast.Name):
        return [node.id]
    return []


def dependency_chain(tree: ast.Module) -> list[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.BinOp):
            names = _dependency_names(node.value)
            if names:
                return names
    raise AssertionError("Cadeia de dependências não encontrada")

