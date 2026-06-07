"""llmReproScore: Repeatability and reproducibility scores for LLM outputs."""

from .semantic import (
    SemanticScorer,
    semantic_repeatability,
    semantic_reproducibility,
)
from .internal import (
    internal_repeatability,
    internal_reproducibility,
    token_entropy_from_logprobs,
)
from .inference import (
    summarize_scores,
    one_sample_test,
    paired_comparison,
    noninferiority_test,
)

__all__ = [
    "SemanticScorer",
    "semantic_repeatability",
    "semantic_reproducibility",
    "internal_repeatability",
    "internal_reproducibility",
    "token_entropy_from_logprobs",
    "summarize_scores",
    "one_sample_test",
    "paired_comparison",
    "noninferiority_test",
]
