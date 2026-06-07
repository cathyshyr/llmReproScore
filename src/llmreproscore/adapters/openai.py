from __future__ import annotations

from typing import Any

import pandas as pd


def extract_chat_completion_text(response: Any) -> str:
    """Extract visible assistant text from an OpenAI-compatible chat completion response."""
    choice = response.choices[0] if hasattr(response, "choices") else response["choices"][0]
    msg = choice.message if hasattr(choice, "message") else choice["message"]
    content = msg.content if hasattr(msg, "content") else msg.get("content")
    if content is None:
        # Some reasoning endpoints may place text elsewhere; we intentionally do not use hidden reasoning.
        return ""
    return str(content)


def _get_attr_or_key(obj: Any, name: str, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def extract_logprobs_openai_chat(
    response: Any,
    case_id: str | int,
    condition_id: str | int,
    run_id: str | int,
) -> pd.DataFrame:
    """Convert OpenAI-compatible chat-completion logprobs to long format.

    This adapter works only when the response includes `choices[0].logprobs.content`,
    as returned by OpenAI-compatible endpoints that support token log probabilities
    such as some OpenAI, SGLang, and vLLM deployments.
    """
    choice = response.choices[0] if hasattr(response, "choices") else response["choices"][0]
    logprobs_obj = _get_attr_or_key(choice, "logprobs")
    if logprobs_obj is None:
        raise ValueError("Response does not contain choice.logprobs. Internal metrics cannot be computed.")
    content = _get_attr_or_key(logprobs_obj, "content")
    if content is None:
        raise ValueError("Response does not contain choice.logprobs.content. Internal metrics cannot be computed.")

    records = []
    for pos, token_logprob in enumerate(content):
        generated_token = _get_attr_or_key(token_logprob, "token")
        top_logprobs = _get_attr_or_key(token_logprob, "top_logprobs", []) or []
        if len(top_logprobs) == 0:
            # Fall back to the generated token logprob if top_logprobs were not returned.
            lp = _get_attr_or_key(token_logprob, "logprob")
            top_logprobs = [{"token": generated_token, "logprob": lp}]
        for rank, cand in enumerate(top_logprobs, start=1):
            records.append(
                {
                    "case_id": case_id,
                    "condition_id": condition_id,
                    "run_id": run_id,
                    "token_position": pos,
                    "generated_token": generated_token,
                    "candidate_rank": rank,
                    "candidate_token": _get_attr_or_key(cand, "token"),
                    "candidate_logprob": _get_attr_or_key(cand, "logprob"),
                }
            )
    return pd.DataFrame.from_records(records)
