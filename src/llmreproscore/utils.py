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


def rescale_cosine_to_unit_interval(cosine_value: float) -> float:
    """Rescale cosine similarity from [-1, 1] to [0, 1]."""
    if np.isnan(cosine_value):
        return np.nan
    return float((cosine_value + 1.0) / 2.0)
