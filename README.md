# llmReproScore

`llmReproScore` computes repeatability and reproducibility scores for stochastic large language model (LLM) outputs.

The package supports two classes of metrics:

| Metric | Requires output text | Requires embeddings | Requires token-level top-k log probabilities |
|---|---:|---:|---:|
| Semantic Repeatability | Yes | Yes | No |
| Semantic Reproducibility | Yes | Yes | No |
| Internal Repeatability | No | No | Yes |
| Internal Reproducibility | No | No | Yes |

Semantic metrics can be computed from repeated text outputs. Internal metrics require token-level top-k log probabilities for each generated token. Not all LLM APIs expose these values. If log probabilities are unavailable, users can still compute semantic repeatability and semantic reproducibility, but not internal repeatability or internal reproducibility.

## Installation

```bash
pip install -e .
```

For OpenAI-compatible adapters:

```bash
pip install -e '.[adapters]'
```

## Input schema

### Semantic metrics

Provide a dataframe with one row per generated output:

```text
case_id | condition_id | run_id | output_text
```

`condition_id` can represent an LLM, prompt, decoding configuration, user, site, or any other pre-specified experimental condition.

### Internal metrics

Provide a long-format dataframe with one row per candidate token in the top-k distribution:

```text
case_id | condition_id | run_id | token_position | candidate_token | candidate_logprob
```

Optional columns include:

```text
generated_token | candidate_rank
```

## Semantic repeatability example

```python
import pandas as pd
from llmreproscore import SemanticScorer

outputs = pd.DataFrame({
    "case_id": [1, 1, 1],
    "condition_id": ["prompt_A", "prompt_A", "prompt_A"],
    "run_id": [1, 2, 3],
    "output_text": [
        "The most likely diagnosis is meningitis.",
        "This presentation is consistent with meningitis.",
        "Meningitis is the most likely diagnosis.",
    ],
})

scorer = SemanticScorer()  # default: sentence-transformers/all-MiniLM-L6-v2
repeatability = scorer.repeatability(outputs)
print(repeatability)
```

Users can supply any SentenceTransformer model name/path:

```python
scorer = SemanticScorer(embedding_model="sentence-transformers/all-mpnet-base-v2")
```

or a custom embedding function:

```python
scorer = SemanticScorer(embedding_fn=my_embedding_function)
```

## Internal repeatability example

```python
from llmreproscore import internal_repeatability

scores = internal_repeatability(
    logprob_df,
    top_k=30,
    case_col="case_id",
    condition_col="condition_id",
    run_col="run_id",
    position_col="token_position",
    logprob_col="candidate_logprob",
)
```

For each output position, the package normalizes the probabilities of the returned top-k candidate tokens and computes Shannon entropy:

```math
H_{r,i}
=
-\sum_{v \in K_{r,i}}
\tilde{\pi}_{r,i}(v)
\log_2\left(\tilde{\pi}_{r,i}(v)\right).
```

Token-level entropy is averaged across output positions to obtain run-level entropy:

```math
H_r
=
\frac{1}{L_r}
\sum_{i=1}^{L_r}
H_{r,i}.
```

Run-level entropy is converted to normalized run-level token-generation certainty:

```math
C_r
=
1-\frac{H_r}{\log_2 k}.
```

Internal Repeatability quantifies the consistency of token-generation certainty across repeated runs:

```math
S_C^{\mathrm{Rpt}}
=
\sqrt{
\frac{1}{R}
\sum_{r=1}^{R}
(C_r-\bar{C})^2
}.
```

The Internal Repeatability Score is:

```math
\widetilde{C}^{\mathrm{Rpt}}
=
1-2S_C^{\mathrm{Rpt}}.
```

The score ranges from 0 to 1. Larger values indicate greater consistency in token-generation certainty across repeated runs under identical conditions. A score of 1 indicates identical token-generation certainty across all runs, regardless of the overall level of certainty.

## Internal reproducibility example

```python
from llmreproscore import internal_reproducibility

scores = internal_reproducibility(
    logprob_df,
    top_k=30,
    case_col="case_id",
    condition_col="condition_id",
    run_col="run_id",
    position_col="token_position",
    logprob_col="candidate_logprob",
)
```

For each experimental condition \(p\), normalized run-level token-generation certainty is averaged across repeated runs:

```math
\bar{C}^{(p)}
=
\frac{1}{R}
\sum_{r=1}^{R}
C_r^{(p)}.
```

Internal Reproducibility quantifies the consistency of mean token-generation certainty across pre-specified experimental conditions:

```math
S_C^{\mathrm{Rpd}}
=
\sqrt{
\frac{1}{P}
\sum_{p=1}^{P}
\left(\bar{C}^{(p)}-\bar{C}\right)^2
}.
```

The Internal Reproducibility Score is:

```math
\widetilde{C}^{\mathrm{Rpd}}
=
1-2S_C^{\mathrm{Rpd}}.
```

The score ranges from 0 to 1. Larger values indicate greater consistency in token-generation certainty across different, pre-specified experimental conditions. A score of 1 indicates identical mean token-generation certainty across all conditions, regardless of the overall level of certainty.

## Statistical inference examples

### Descriptive summary

```python
from llmreproscore import summarize_scores

summarize_scores(repeatability, score_col="semantic_repeatability", group_cols=["condition_id"])
```

### One-sample comparison to a threshold

```python
from llmreproscore import one_sample_test

one_sample_test(
    repeatability,
    score_col="semantic_repeatability",
    threshold=0.95,
    alternative="greater",
)
```

### Paired comparison between two conditions

```python
from llmreproscore import paired_comparison

paired_comparison(
    df_scores,
    case_col="case_id",
    condition_col="condition_id",
    score_col="semantic_repeatability",
    condition_a="LLM_A",
    condition_b="LLM_B",
)
```

### Noninferiority comparison

```python
from llmreproscore import noninferiority_test

noninferiority_test(
    df_scores,
    case_col="case_id",
    condition_col="condition_id",
    score_col="semantic_repeatability",
    candidate="new_model",
    reference="reference_model",
    margin=0.02,
)
```

Noninferiority is demonstrated when the lower 95% confidence bound for the paired difference is greater than `-margin`.

## OpenAI-compatible adapters

Adapters are convenience utilities. They do not guarantee that every model or endpoint exposes log probabilities.

```python
from llmreproscore.adapters import extract_logprobs_openai_chat

logprob_df = extract_logprobs_openai_chat(
    response,
    case_id=1,
    condition_id="prompt_A",
    run_id=1,
)
```

This works only when the response contains:

```python
response.choices[0].logprobs.content
```

Some models, API versions, or hosted endpoints do not expose token log probabilities, or expose them only for certain request types. In those cases, internal metrics cannot be computed from that endpoint.

For SGLang/Qwen models that support disabling thinking through chat template kwargs:

```python
from llmreproscore.adapters import sglang_extra_body_disable_thinking

response = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.5,
    max_tokens=256,
    logprobs=True,
    top_logprobs=30,
    extra_body=sglang_extra_body_disable_thinking(),
)
```

## What the package guarantees

`llmReproScore` guarantees that, given data in the required schema, it computes the metric definitions implemented in the package. It does not guarantee that a third-party LLM API will return token-level log probabilities. Users should verify logprob availability before planning analyses that require internal repeatability or internal reproducibility.
