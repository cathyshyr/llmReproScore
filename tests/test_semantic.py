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


def test_semantic_reproducibility_preserves_run_level_variation():

    def embed(texts):
        mapping = {
            "x": np.array([1.0, 0.0]),
            "y": np.array([0.0, 1.0]),
        }

        return np.vstack(
            [
                mapping[t]
                for t in texts
            ]
        )

    df = pd.DataFrame(
        {
            "case_id": [
                1, 1, 1, 1,
            ],
            "condition_id": [
                "A", "A", "B", "B",
            ],
            "run_id": [
                1, 2, 1, 2,
            ],
            "output_text": [
                "x", "y", "x", "y",
            ],
        }
    )

    out = semantic_reproducibility(
        df,
        embed,
    )

    # Cross-run cosine values are:
    # 1, 0, 0, 1
    #
    # Raw mean = 0.5
    # Rescaled score = 0.75
    #
    # The former centroid-cosine definition would give 1.0.

    assert np.isclose(
        out[
            "mean_pairwise_cosine"
        ].iloc[0],
        0.5,
    )

    assert np.isclose(
        out[
            "semantic_reproducibility"
        ].iloc[0],
        0.75,
    )

    assert (
        out[
            "n_condition_pairs"
        ].iloc[0]
        == 1
    )
