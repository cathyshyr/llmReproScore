from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _validate_internal_input(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def token_entropy_from_logprobs(logprobs: np.ndarray, top_k: int | None = None) -> float:
    """Compute Shannon entropy from candidate token log probabilities.

    Log probabilities are exponentiated, normalized over the supplied candidates,
    and entropy is computed using log base 2.
    """
    logprobs = np.asarray(logprobs, dtype=float)
    if logprobs.size == 0:
        return np.nan
    if top_k is not None and logprobs.size > top_k:
        # Keep the top-k highest log probabilities if more candidates were supplied.
        idx = np.argsort(logprobs)[-top_k:]
        logprobs = logprobs[idx]
    # Numerically stable softmax over returned top-k candidates.
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
    """Return one entropy value H_r per case/condition/run."""
    _validate_internal_input(df, [case_col, condition_col, run_col, position_col, logprob_col])
    token_records = []
    for keys, g in df.groupby([case_col, condition_col, run_col, position_col], sort=False):
        h_i = token_entropy_from_logprobs(g[logprob_col].to_numpy(), top_k=top_k)
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
        token_df.groupby([case_col, condition_col, run_col], sort=False)
        .agg(
            H_r=("token_entropy", "mean"),
            n_positions=(position_col, "nunique"),
            mean_n_candidates=("n_candidates", "mean"),
        )
        .reset_index()
    )
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
    """Compute internal repeatability per case and condition.

    Internal repeatability is 1 - H_bar / log2(k), where H_bar is the mean
    run-level entropy across repeated runs.
    """
    run_df = _run_level_entropies(
        logprob_df, case_col, condition_col, run_col, position_col, logprob_col, top_k
    )
    max_entropy = np.log2(top_k)
    out = (
        run_df.groupby([case_col, condition_col], sort=False)
        .agg(H_bar=("H_r", "mean"), n_runs=(run_col, "nunique"), mean_n_positions=("n_positions", "mean"))
        .reset_index()
    )
    out["top_k"] = top_k
    out["internal_repeatability"] = 1.0 - out["H_bar"] / max_entropy
    return out


def internal_reproducibility(
    logprob_df: pd.DataFrame,
    top_k: int,
    case_col: str = "case_id",
    condition_col: str = "condition_id",
    run_col: str = "run_id",
    position_col: str = "token_position",
    logprob_col: str = "candidate_logprob",
) -> pd.DataFrame:
    """Compute internal reproducibility per case across conditions.

    Internal reproducibility is 1 - D / log2(k), where D is the average pairwise
    absolute difference in condition-level mean entropy.
    """
    rpt = internal_repeatability(
        logprob_df, top_k, case_col, condition_col, run_col, position_col, logprob_col
    )
    max_entropy = np.log2(top_k)
    records = []
    for case_id, g in rpt.groupby(case_col, sort=False):
        values = g[[condition_col, "H_bar"]].dropna()
        if len(values) < 2:
            diff_bar = np.nan
            score = np.nan
        else:
            diffs = [abs(a - b) for a, b in combinations(values["H_bar"].to_numpy(), 2)]
            diff_bar = float(np.mean(diffs))
            score = 1.0 - diff_bar / max_entropy
        records.append(
            {
                case_col: case_id,
                "n_conditions": len(values),
                "condition_ids": ",".join(map(str, values[condition_col].tolist())),
                "top_k": top_k,
                "H_rpd": diff_bar,
                "internal_reproducibility": score,
            }
        )
    return pd.DataFrame.from_records(records)
