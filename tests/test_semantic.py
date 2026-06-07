import numpy as np
import pandas as pd

from llmreproscore.semantic import semantic_repeatability, semantic_reproducibility


def toy_embed(texts):
    mapping = {
        "a": np.array([1.0, 0.0]),
        "a again": np.array([1.0, 0.0]),
        "b": np.array([0.0, 1.0]),
    }
    return np.vstack([mapping[t] for t in texts])


def test_semantic_repeatability():
    df = pd.DataFrame({
        "case_id": [1, 1],
        "condition_id": ["A", "A"],
        "run_id": [1, 2],
        "output_text": ["a", "a again"],
    })
    out = semantic_repeatability(df, toy_embed)
    assert out["semantic_repeatability"].iloc[0] == 1.0


def test_semantic_reproducibility():
    df = pd.DataFrame({
        "case_id": [1, 1],
        "condition_id": ["A", "B"],
        "run_id": [1, 1],
        "output_text": ["a", "b"],
    })
    out = semantic_reproducibility(df, toy_embed)
    assert abs(out["semantic_reproducibility"].iloc[0] - 0.5) < 1e-8
