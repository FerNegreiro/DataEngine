from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from typing import Any

import pytest

from tests.airflow import DAG_PATH


class _FakeDAG:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> _FakeDAG:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _FakeBashOperator:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __rshift__(self, other: _FakeBashOperator) -> _FakeBashOperator:
        return other


def test_dag_file_imports_with_airflow_api_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = {
        name: ModuleType(name)
        for name in (
            "airflow",
            "airflow.providers",
            "airflow.providers.standard",
            "airflow.providers.standard.operators",
            "airflow.providers.standard.operators.bash",
        )
    }
    modules["airflow"].DAG = _FakeDAG
    modules["airflow.providers.standard.operators.bash"].BashOperator = (
        _FakeBashOperator
    )
    for name, module in modules.items():
        if name != "airflow.providers.standard.operators.bash":
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("dataengine_dag_test", DAG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.dag.kwargs["dag_id"] == "dataengine_full_pipeline"

