import numpy as np
import pandas as pd

from llmreproscore import (
    internal_repeatability,
    internal_reproducibility,
    token_entropy_from_logprobs,
)

COLUMNS = [
    "case_id",
    "condition_id",
    "run_id",
    "token_position",
    "candidate_token",
    "candidate_logprob",
]


def add_distribution(
    rows,
    case_id,
    condition_id,
    run_id,
    position,
    logprobs,
):
    for idx, lp in enumerate(logprobs):
        rows.append(
            [
                case_id,
                condition_id,
                run_id,
                position,
                f"token_{idx}",
                lp,
            ]
        )


def test_entropy_deterministic_near_zero():
    h = token_entropy_from_logprobs(
        np.array([0.0, -100.0]),
        top_k=2,
    )
    assert h < 1e-6


def test_entropy_uniform_equals_log2_k():
    h = token_entropy_from_logprobs(
        np.array([0.0, 0.0]),
        top_k=2,
    )
    assert np.isclose(h, 1.0)


def test_internal_repeatability_identical_high_certainty_is_one():
    rows = []

    for run in [1, 2, 3, 4]:
        for pos in [0, 1]:
            add_distribution(
                rows,
                case_id=1,
                condition_id="A",
                run_id=run,
                position=pos,
                logprobs=[0.0, -10.0],
            )

    df = pd.DataFrame(rows, columns=COLUMNS)

    out = internal_repeatability(df, top_k=2)

    assert np.isclose(
        out["internal_repeatability"].iloc[0],
        1.0,
    )


def test_internal_repeatability_identical_low_certainty_is_one():
    rows = []

    for run in [1, 2, 3, 4]:
        for pos in [0, 1]:
            add_distribution(
                rows,
                case_id=1,
                condition_id="A",
                run_id=run,
                position=pos,
                logprobs=[0.0, 0.0],
            )

    df = pd.DataFrame(rows, columns=COLUMNS)

    out = internal_repeatability(df, top_k=2)

    assert np.isclose(
        out["internal_repeatability"].iloc[0],
        1.0,
    )


def test_internal_repeatability_decreases_when_certainty_varies():
    rows = []

    for run in [1, 2]:
        for pos in [0, 1]:
            add_distribution(
                rows,
                case_id=1,
                condition_id="A",
                run_id=run,
                position=pos,
                logprobs=[0.0, -10.0],
            )

    for run in [3, 4]:
        for pos in [0, 1]:
            add_distribution(
                rows,
                case_id=1,
                condition_id="A",
                run_id=run,
                position=pos,
                logprobs=[0.0, 0.0],
            )

    df = pd.DataFrame(rows, columns=COLUMNS)

    out = internal_repeatability(df, top_k=2)

    score = out["internal_repeatability"].iloc[0]

    assert 0 <= score <= 1
    assert score < 0.1


def test_internal_repeatability_requires_multiple_runs():
    rows = []

    for pos in [0, 1]:
        add_distribution(
            rows,
            case_id=1,
            condition_id="A",
            run_id=1,
            position=pos,
            logprobs=[0.0, -10.0],
        )

    df = pd.DataFrame(rows, columns=COLUMNS)

    out = internal_repeatability(df, top_k=2)

    assert np.isnan(
        out["internal_repeatability"].iloc[0]
    )


def test_internal_reproducibility_identical_condition_certainty_is_one():
    rows = []

    for condition in ["A", "B", "C"]:
        for run in [1, 2, 3]:
            for pos in [0, 1]:
                add_distribution(
                    rows,
                    case_id=1,
                    condition_id=condition,
                    run_id=run,
                    position=pos,
                    logprobs=[0.0, -10.0],
                )

    df = pd.DataFrame(rows, columns=COLUMNS)

    out = internal_reproducibility(df, top_k=2)

    assert np.isclose(
        out["internal_reproducibility"].iloc[0],
        1.0,
    )


def test_internal_reproducibility_decreases_when_conditions_differ():
    rows = []

    for run in [1, 2, 3]:
        for pos in [0, 1]:
            add_distribution(
                rows,
                case_id=1,
                condition_id="A",
                run_id=run,
                position=pos,
                logprobs=[0.0, -10.0],
            )

    for run in [1, 2, 3]:
        for pos in [0, 1]:
            add_distribution(
                rows,
                case_id=1,
                condition_id="B",
                run_id=run,
                position=pos,
                logprobs=[0.0, 0.0],
            )

    df = pd.DataFrame(rows, columns=COLUMNS)

    out = internal_reproducibility(df, top_k=2)

    score = out["internal_reproducibility"].iloc[0]

    assert 0 <= score <= 1
    assert score < 0.1
