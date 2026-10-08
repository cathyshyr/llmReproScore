from __future__ import annotations

import re
from typing import Iterable

import numpy as np

_DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
    "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "this",
    "these", "those", "or", "if", "then", "there", "their", "his", "her", "they", "them", "we",
    "you", "your", "i", "my", "our", "ours", "but", "not", "no", "yes", "can", "could", "would",
    "should", "may", "might", "most", "likely", "patient", "diagnosis", "response", "answer",
}


def filter_stopwords(text: str, stopwords: Iterable[str] | None = None) -> str:
    """Return a simple stop-word-filtered version of text.

    This is intentionally lightweight and dependency-free. It keeps alphabetic tokens only.
    Users who need domain-specific tokenization should pre-process externally and pass the
    desired text column to the scoring functions.
    """
    if text is None:
        return ""
    sw = set(stopwords or _DEFAULT_STOPWORDS)
    tokens = re.findall(r"[A-Za-z]+", str(text))
    return " ".join(t for t in tokens if t.lower() not in sw)


def average_pairwise_cosine_from_matrix(x: np.ndarray) -> float:
    """Average pairwise cosine similarity among rows of x."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n_items, n_features).")
    n = x.shape[0]
    if n < 2:
        return np.nan
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    z = x / norms
    sim = z @ z.T
    upper = sim[np.triu_indices(n, k=1)]
    return float(np.mean(upper))



def average_cross_cosine_from_matrices(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Average cosine similarity across all rows of x and y.

    If x contains R_p run embeddings and y contains R_q run
    embeddings, this is exactly the mean of all R_p * R_q
    cross-condition cosine similarities.

    The full similarity matrix is not materialized.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            "x and y must both be 2D arrays."
        )

    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "x and y must have the same embedding dimension."
        )

    if x.shape[0] == 0 or y.shape[0] == 0:
        return np.nan

    x_norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    y_norms = np.linalg.norm(
        y,
        axis=1,
        keepdims=True,
    )

    x_norms[x_norms == 0] = 1.0
    y_norms[y_norms == 0] = 1.0

    x_unit = x / x_norms
    y_unit = y / y_norms

    mean_x = x_unit.mean(axis=0)
    mean_y = y_unit.mean(axis=0)

    return float(
        np.dot(
            mean_x,
            mean_y,
        )
    )


def rescale_cosine_to_unit_interval(cosine_value: float) -> float:
    """Rescale cosine similarity from [-1, 1] to [0, 1]."""
    if np.isnan(cosine_value):
        return np.nan
    return float((cosine_value + 1.0) / 2.0)
