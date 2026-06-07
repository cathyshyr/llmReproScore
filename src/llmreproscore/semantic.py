from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .utils import (
    average_pairwise_cosine_from_matrix,
    filter_stopwords,
    rescale_cosine_to_unit_interval,
)

EmbeddingFn = Callable[[list[str]], np.ndarray]


@dataclass
class SemanticScorer:
    """Compute semantic repeatability and reproducibility from repeated text outputs.

    Parameters
    ----------
    embedding_model:
        SentenceTransformer model name/path. Defaults to all-MiniLM-L6-v2.
    embedding_fn:
        Optional custom embedding function. It must accept list[str] and return a 2D numpy array.
        If supplied, `embedding_model` is ignored.
    filter_stop_words:
        If True, compute embeddings on stop-word-filtered text. Defaults to False in the package.
        Users can also pre-filter text externally and pass that column as `text_col`.
    stopwords:
        Optional stop-word set for the built-in lightweight filter.
    """

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_fn: EmbeddingFn | None = None
    filter_stop_words: bool = False
    stopwords: Iterable[str] | None = None

    def __post_init__(self) -> None:
        self._model = None
        if self.embedding_fn is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.filter_stop_words:
            texts = [filter_stopwords(t, self.stopwords) for t in texts]
        if self.embedding_fn is not None:
            arr = self.embedding_fn(texts)
            return np.asarray(arr)
        return np.asarray(self._model.encode(texts, convert_to_numpy=True))

    def repeatability(
        self,
        df: pd.DataFrame,
        case_col: str = "case_id",
        condition_col: str = "condition_id",
        run_col: str = "run_id",
        text_col: str = "output_text",
    ) -> pd.DataFrame:
        return semantic_repeatability(
            df,
            embedding_fn=self.embed,
            case_col=case_col,
            condition_col=condition_col,
            run_col=run_col,
            text_col=text_col,
        )

    def reproducibility(
        self,
        df: pd.DataFrame,
        case_col: str = "case_id",
        condition_col: str = "condition_id",
        run_col: str = "run_id",
        text_col: str = "output_text",
    ) -> pd.DataFrame:
        return semantic_reproducibility(
            df,
            embedding_fn=self.embed,
            case_col=case_col,
            condition_col=condition_col,
            run_col=run_col,
            text_col=text_col,
        )


def _validate_semantic_input(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def semantic_repeatability(
    df: pd.DataFrame,
    embedding_fn: EmbeddingFn,
    case_col: str = "case_id",
    condition_col: str = "condition_id",
    run_col: str = "run_id",
    text_col: str = "output_text",
) -> pd.DataFrame:
    """Compute semantic repeatability per case and condition.

    The score is the average pairwise cosine similarity across repeated runs,
    rescaled from [-1, 1] to [0, 1].
    """
    _validate_semantic_input(df, [case_col, condition_col, run_col, text_col])
    records = []
    for (case_id, condition_id), g in df.groupby([case_col, condition_col], sort=False):
        g = g.sort_values(run_col)
        texts = g[text_col].fillna("").astype(str).tolist()
        if len(texts) < 2:
            s_bar = np.nan
            score = np.nan
        else:
            emb = embedding_fn(texts)
            s_bar = average_pairwise_cosine_from_matrix(emb)
            score = rescale_cosine_to_unit_interval(s_bar)
        records.append(
            {
                case_col: case_id,
                condition_col: condition_id,
                "n_runs": len(texts),
                "mean_pairwise_cosine": s_bar,
                "semantic_repeatability": score,
            }
        )
    return pd.DataFrame.from_records(records)


def semantic_reproducibility(
    df: pd.DataFrame,
    embedding_fn: EmbeddingFn,
    case_col: str = "case_id",
    condition_col: str = "condition_id",
    run_col: str = "run_id",
    text_col: str = "output_text",
) -> pd.DataFrame:
    """Compute semantic reproducibility per case across conditions.

    For each case, output embeddings are averaged within each condition across runs.
    Reproducibility is the average pairwise cosine similarity between condition-level
    mean embeddings, rescaled from [-1, 1] to [0, 1].
    """
    _validate_semantic_input(df, [case_col, condition_col, run_col, text_col])
    records = []
    for case_id, case_df in df.groupby(case_col, sort=False):
        condition_embeddings = []
        condition_ids = []
        for condition_id, g in case_df.groupby(condition_col, sort=False):
            g = g.sort_values(run_col)
            texts = g[text_col].fillna("").astype(str).tolist()
            if len(texts) == 0:
                continue
            emb = embedding_fn(texts)
            condition_embeddings.append(np.mean(emb, axis=0))
            condition_ids.append(condition_id)
        if len(condition_embeddings) < 2:
            s_bar = np.nan
            score = np.nan
        else:
            mat = np.vstack(condition_embeddings)
            s_bar = average_pairwise_cosine_from_matrix(mat)
            score = rescale_cosine_to_unit_interval(s_bar)
        records.append(
            {
                case_col: case_id,
                "n_conditions": len(condition_embeddings),
                "condition_ids": ",".join(map(str, condition_ids)),
                "mean_pairwise_cosine": s_bar,
                "semantic_reproducibility": score,
            }
        )
    return pd.DataFrame.from_records(records)
