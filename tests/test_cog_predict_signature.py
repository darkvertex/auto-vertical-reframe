"""Cog reads predict() choices from source, so they must be list literals."""

from __future__ import annotations

import ast
from pathlib import Path

from verthor.reframe_job import PRESETS, SALIENCY_MODELS

PREDICT = Path(__file__).resolve().parents[1] / "predict.py"


def _static_choices() -> dict[str, list[str]]:
    tree = ast.parse(PREDICT.read_text(encoding="utf-8"))
    predict = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "predict"
    )
    offset = len(predict.args.args) - len(predict.args.defaults)
    found: dict[str, list[str]] = {}
    for arg, default in zip(
        predict.args.args[offset:], predict.args.defaults, strict=True
    ):
        if not isinstance(default, ast.Call):
            continue
        for keyword in default.keywords:
            if keyword.arg != "choices":
                continue
            value = keyword.value
            assert isinstance(value, ast.List), arg.arg
            assert all(isinstance(elt, ast.Constant) for elt in value.elts)
            found[arg.arg] = [elt.value for elt in value.elts]
    return found


def test_cog_choices_are_static_literals() -> None:
    choices = _static_choices()
    assert choices["preset"] == list(PRESETS)
    assert choices["saliency_model"] == list(SALIENCY_MODELS)
