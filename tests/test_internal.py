import numpy as np
import pandas as pd

from llmreproscore import internal_repeatability, internal_reproducibility, token_entropy_from_logprobs


def test_entropy_deterministic_near_zero():
    h = token_entropy_from_logprobs(np.array([0.0, -100.0]), top_k=2)
    assert h < 1e-6


def test_internal_repeatability_bounds():
    rows = []
    for run in [1, 2]:
        for pos in [0, 1]:
            rows.append([1, "A", run, pos, "x", 0.0])
            rows.append([1, "A", run, pos, "y", -10.0])
    df = pd.DataFrame(rows, columns=["case_id", "condition_id", "run_id", "token_position", "candidate_token", "candidate_logprob"])
    out = internal_repeatability(df, top_k=2)
    score = out["internal_repeatability"].iloc[0]
    assert 0 <= score <= 1
    assert score > 0.9


def test_internal_reproducibility_runs():
    rows = []
    for condition in ["A", "B"]:
        for run in [1, 2]:
            for pos in [0, 1]:
                rows.append([1, condition, run, pos, "x", 0.0])
                rows.append([1, condition, run, pos, "y", -10.0])
    df = pd.DataFrame(rows, columns=["case_id", "condition_id", "run_id", "token_position", "candidate_token", "candidate_logprob"])
    out = internal_reproducibility(df, top_k=2)
    assert out["internal_reproducibility"].iloc[0] > 0.99
