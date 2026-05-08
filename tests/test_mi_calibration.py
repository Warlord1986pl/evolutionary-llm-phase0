from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.mi_calibration import (
    evaluate_mi_function,
    mi_cosine,
    mi_entropy_decomp,
    mi_jsd,
    mi_npmi,
)


def test_mi_cosine_identical_texts_return_one() -> None:
    result = mi_cosine("alpha beta gamma", "alpha beta gamma")
    assert abs(result - 1.0) < 1e-12


def test_mi_cosine_disjoint_vocabulary_returns_zero() -> None:
    result = mi_cosine("cat dog", "table lamp")
    assert abs(result - 0.0) < 1e-12


def test_mi_entropy_decomp_identical_texts_return_one() -> None:
    result = mi_entropy_decomp("alpha beta gamma", "alpha beta gamma")
    assert abs(result - 1.0) < 1e-12


def test_mi_entropy_decomp_empty_text_returns_zero() -> None:
    result = mi_entropy_decomp("alpha beta", "")
    assert abs(result - 0.0) < 1e-12


def test_mi_jsd_identical_texts_return_one() -> None:
    result = mi_jsd("alpha beta gamma", "alpha beta gamma")
    assert abs(result - 1.0) < 1e-12


def test_mi_jsd_disjoint_vocabulary_returns_zero() -> None:
    result = mi_jsd("cat dog", "table lamp")
    assert abs(result - 0.0) < 1e-12


def test_mi_npmi_no_shared_tokens_returns_zero() -> None:
    result = mi_npmi("cat dog", "table lamp")
    assert abs(result - 0.0) < 1e-12


def test_mi_npmi_result_is_in_unit_interval() -> None:
    result = mi_npmi("alpha beta alpha gamma", "alpha delta alpha")
    assert 0.0 <= result <= 1.0


def test_evaluate_mi_function_detects_correct_direction() -> None:
    seed = "good signal"
    results = [
        {"type": "food", "model_output": "good signal good signal"},
        {"type": "food", "model_output": "good signal"},
        {"type": "toxin", "model_output": "noise junk"},
        {"type": "toxin", "model_output": "random tokens"},
        {"type": "noise", "model_output": "random noise"},
    ]

    summary = evaluate_mi_function(mi_func=mi_cosine, results=results, seed_text=seed)
    assert summary["direction"] == "correct"


def test_evaluate_mi_function_detects_reversed_direction() -> None:
    seed = "good signal"
    results = [
        {"type": "food", "model_output": "junk only"},
        {"type": "food", "model_output": "unrelated tokens"},
        {"type": "toxin", "model_output": "good signal good signal"},
        {"type": "toxin", "model_output": "good signal"},
        {"type": "noise", "model_output": "random noise"},
    ]

    summary = evaluate_mi_function(mi_func=mi_cosine, results=results, seed_text=seed)
    assert summary["direction"] == "reversed"
