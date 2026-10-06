from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_internal_input(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_top_k(top_k: int) -> None:
    if top_k < 2:
        raise ValueError("top_k must be >= 2 for normalized entropy/certainty metrics.")


def token_entropy_from_logprobs(
    logprobs: np.ndarray,
    top_k: int | None = None,
) -> float:
    """
    Compute Shannon entropy from candidate token log probabilities.
    """
    logprobs = np.asarray(logprobs, dtype=float)

    if logprobs.size == 0:
        return np.nan

    if top_k is not None and logprobs.size > top_k:
        idx = np.argsort(logprobs)[-top_k:]
        logprobs = logprobs[idx]

    probs = np.exp(logprobs - np.max(logprobs))
    probs = probs / probs.sum()

    return float(-np.sum(probs * np.log2(probs)))


def _run_level_entropies(
    df: pd.DataFrame,
    case_col: str,
    condition_col: str,
    run_col: str,
    position_col: str,
    logprob_col: str,
    top_k: int,
) -> pd.DataFrame:
    """
    Return run-level entropy H_r and normalized token-generation
    certainty C_r for each case/condition/run.
    """
    _validate_top_k(top_k)

    _validate_internal_input(
        df,
        [
            case_col,
            condition_col,
            run_col,
            position_col,
            logprob_col,
        ],
    )

    token_records = []

    for keys, g in df.groupby(
        [case_col, condition_col, run_col, position_col],
        sort=False,
    ):
        h_i = token_entropy_from_logprobs(
            g[logprob_col].to_numpy(),
            top_k=top_k,
        )

        token_records.append(
            {
                case_col: keys[0],
                condition_col: keys[1],
                run_col: keys[2],
                position_col: keys[3],
                "token_entropy": h_i,
                "n_candidates": len(g),
            }
        )

    token_df = pd.DataFrame.from_records(token_records)

    run_df = (
        token_df.groupby(
            [case_col, condition_col, run_col],
            sort=False,
        )
        .agg(
            H_r=("token_entropy", "mean"),
            n_positions=(position_col, "nunique"),
            mean_n_candidates=("n_candidates", "mean"),
        )
        .reset_index()
    )

    max_entropy = np.log2(top_k)

    run_df["C_r"] = 1.0 - run_df["H_r"] / max_entropy
    run_df["C_r"] = run_df["C_r"].clip(0.0, 1.0)

    return run_df


def internal_repeatability(
    logprob_df: pd.DataFrame,
    top_k: int,
    case_col: str = "case_id",
    condition_col: str = "condition_id",
    run_col: str = "run_id",
    position_col: str = "token_position",
    logprob_col: str = "candidate_logprob",
) -> pd.DataFrame:
    """
    Compute Internal Repeatability per case and condition.

    Internal Repeatability quantifies the consistency of normalized
    run-level token-generation certainty C_r across repeated runs under
    identical conditions.
    """
    run_df = _run_level_entropies(
        logprob_df,
        case_col,
        condition_col,
        run_col,
        position_col,
        logprob_col,
        top_k,
    )

    records = []

    for (case_id, condition_id), g in run_df.groupby(
        [case_col, condition_col],
        sort=False,
    ):
        valid = g.dropna(subset=["C_r"])
        n_runs = valid[run_col].nunique()

        if n_runs == 0:
            H_bar = np.nan
            C_bar = np.nan
            S_C_rpt = np.nan
            score = np.nan
        else:
            H_bar = float(valid["H_r"].mean())
            C_bar = float(valid["C_r"].mean())

            if n_runs < 2:
                S_C_rpt = np.nan
                score = np.nan
            else:
                S_C_rpt = float(valid["C_r"].std(ddof=0))
                score = float(
                    np.clip(
                        1.0 - 2.0 * S_C_rpt,
                        0.0,
                        1.0,
                    )
                )

        records.append(
            {
                case_col: case_id,
                condition_col: condition_id,
                "H_bar": H_bar,
                "C_bar": C_bar,
                "S_C_rpt": S_C_rpt,
                "n_runs": n_runs,
                "mean_n_positions": (
                    float(valid["n_positions"].mean())
                    if len(valid) > 0
                    else np.nan
                ),
                "top_k": top_k,
                "internal_repeatability": score,
            }
        )

    return pd.DataFrame.from_records(records)


def internal_reproducibility(
    logprob_df: pd.DataFrame,
    top_k: int,
    case_col: str = "case_id",
    condition_col: str = "condition_id",
    run_col: str = "run_id",
    position_col: str = "token_position",
    logprob_col: str = "candidate_logprob",
) -> pd.DataFrame:
    """
    Compute Internal Reproducibility per case across conditions.

    Internal Reproducibility quantifies the consistency of mean
    token-generation certainty across pre-specified experimental conditions.
    """
    rpt = internal_repeatability(
        logprob_df,
        top_k,
        case_col,
        condition_col,
        run_col,
        position_col,
        logprob_col,
    )

    records = []

    for case_id, g in rpt.groupby(case_col, sort=False):
        values = g[
            [condition_col, "C_bar"]
        ].dropna(subset=["C_bar"])

        n_conditions = len(values)

        if n_conditions < 2:
            C_bar_across_conditions = (
                float(values["C_bar"].mean())
                if n_conditions == 1
                else np.nan
            )
            S_C_rpd = np.nan
            score = np.nan
        else:
            condition_certainties = values["C_bar"].to_numpy(dtype=float)

            C_bar_across_conditions = float(
                np.mean(condition_certainties)
            )

            S_C_rpd = float(
                np.std(condition_certainties, ddof=0)
            )

            score = float(
                np.clip(
                    1.0 - 2.0 * S_C_rpd,
                    0.0,
                    1.0,
                )
            )

        records.append(
            {
                case_col: case_id,
                "n_conditions": n_conditions,
                "condition_ids": ",".join(
                    map(str, values[condition_col].tolist())
                ),
                "top_k": top_k,
                "C_bar": C_bar_across_conditions,
                "S_C_rpd": S_C_rpd,
                "internal_reproducibility": score,
            }
        )

    return pd.DataFrame.from_records(records)
