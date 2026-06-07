from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_scores(df: pd.DataFrame, score_col: str, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Descriptive summary of a score column."""
    group_cols = group_cols or []
    if score_col not in df.columns:
        raise ValueError(f"Missing score column: {score_col}")
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    records = []
    for keys, g in grouped:
        x = g[score_col].dropna().to_numpy(dtype=float)
        rec = {}
        if group_cols:
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec.update(dict(zip(group_cols, keys)))
        rec.update(
            {
                "n": int(x.size),
                "mean": float(np.mean(x)) if x.size else np.nan,
                "sd": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
                "median": float(np.median(x)) if x.size else np.nan,
                "q1": float(np.quantile(x, 0.25)) if x.size else np.nan,
                "q3": float(np.quantile(x, 0.75)) if x.size else np.nan,
                "min": float(np.min(x)) if x.size else np.nan,
                "max": float(np.max(x)) if x.size else np.nan,
            }
        )
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _bootstrap_stat(x: np.ndarray, stat_fn, n_boot: int, random_state: int | None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([])
    return np.array([stat_fn(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)])


def one_sample_test(
    scores: pd.DataFrame | np.ndarray | list[float],
    score_col: str | None = None,
    threshold: float = 0.0,
    alternative: str = "greater",
    statistic: str = "mean",
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int | None = 123,
) -> dict:
    """One-sample bootstrap comparison to a pre-specified threshold tau.

    For alternative='greater', the decision criterion is lower CI > threshold.
    """
    if isinstance(scores, pd.DataFrame):
        if score_col is None:
            raise ValueError("score_col is required when scores is a DataFrame.")
        x = scores[score_col].dropna().to_numpy(dtype=float)
    else:
        x = np.asarray(scores, dtype=float)
        x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("No non-missing scores supplied.")
    stat_fn = np.mean if statistic == "mean" else np.median
    estimate = float(stat_fn(x))
    boot = _bootstrap_stat(x, stat_fn, n_boot, random_state)
    alpha = 1 - ci
    lower, upper = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    if alternative == "greater":
        p_value = float((np.sum(boot <= threshold) + 1) / (len(boot) + 1))
        decision = bool(lower > threshold)
    elif alternative == "less":
        p_value = float((np.sum(boot >= threshold) + 1) / (len(boot) + 1))
        decision = bool(upper < threshold)
    else:
        p_value = float((np.sum(np.abs(boot - estimate) >= abs(estimate - threshold)) + 1) / (len(boot) + 1))
        decision = bool(lower > threshold or upper < threshold)
    return {
        "estimate": estimate,
        "lower_ci": float(lower),
        "upper_ci": float(upper),
        "threshold": threshold,
        "alternative": alternative,
        "p_value_bootstrap": p_value,
        "decision": decision,
        "n": int(x.size),
    }


def paired_comparison(
    df: pd.DataFrame,
    case_col: str,
    condition_col: str,
    score_col: str,
    condition_a: str,
    condition_b: str,
    statistic: str = "mean",
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int | None = 123,
) -> dict:
    """Paired bootstrap comparison between two conditions.

    Returns results for Delta = score(A) - score(B).
    """
    required = [case_col, condition_col, score_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    wide = df[df[condition_col].isin([condition_a, condition_b])].pivot_table(
        index=case_col, columns=condition_col, values=score_col, aggfunc="mean"
    )
    if condition_a not in wide.columns or condition_b not in wide.columns:
        raise ValueError("Both conditions must be present in the data.")
    d = (wide[condition_a] - wide[condition_b]).dropna().to_numpy(dtype=float)
    stat_fn = np.mean if statistic == "mean" else np.median
    estimate = float(stat_fn(d))
    boot = _bootstrap_stat(d, stat_fn, n_boot, random_state)
    alpha = 1 - ci
    lower, upper = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p_value = float((2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "estimate_difference": estimate,
        "lower_ci": float(lower),
        "upper_ci": float(upper),
        "p_value_bootstrap": min(p_value, 1.0),
        "n_pairs": int(d.size),
    }


def noninferiority_test(
    df: pd.DataFrame,
    case_col: str,
    condition_col: str,
    score_col: str,
    candidate: str,
    reference: str,
    margin: float,
    statistic: str = "mean",
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int | None = 123,
) -> dict:
    """Paired noninferiority test for candidate vs reference.

    Noninferiority is demonstrated if lower CI for candidate-reference > -margin.
    """
    res = paired_comparison(
        df,
        case_col=case_col,
        condition_col=condition_col,
        score_col=score_col,
        condition_a=candidate,
        condition_b=reference,
        statistic=statistic,
        n_boot=n_boot,
        ci=ci,
        random_state=random_state,
    )
    res["candidate"] = candidate
    res["reference"] = reference
    res["margin"] = margin
    res["noninferior"] = bool(res["lower_ci"] > -margin)
    return res
