import pandas as pd

from llmreproscore import noninferiority_test, one_sample_test, paired_comparison


def test_one_sample_test():
    df = pd.DataFrame({"score": [0.95, 0.96, 0.97, 0.98]})
    out = one_sample_test(df, score_col="score", threshold=0.90, n_boot=100, random_state=1)
    assert out["decision"] is True


def test_paired_comparison():
    df = pd.DataFrame({
        "case_id": [1, 1, 2, 2, 3, 3],
        "condition_id": ["A", "B", "A", "B", "A", "B"],
        "score": [0.96, 0.90, 0.95, 0.91, 0.97, 0.92],
    })
    out = paired_comparison(df, "case_id", "condition_id", "score", "A", "B", n_boot=100, random_state=1)
    assert out["estimate_difference"] > 0


def test_noninferiority():
    df = pd.DataFrame({
        "case_id": [1, 1, 2, 2, 3, 3],
        "condition_id": ["A", "B", "A", "B", "A", "B"],
        "score": [0.95, 0.96, 0.94, 0.95, 0.96, 0.97],
    })
    out = noninferiority_test(df, "case_id", "condition_id", "score", "A", "B", margin=0.03, n_boot=100, random_state=1)
    assert out["noninferior"] is True
