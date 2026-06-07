from __future__ import annotations

from typing import Any

from .openai import extract_chat_completion_text, extract_logprobs_openai_chat


def sglang_extra_body_disable_thinking() -> dict[str, Any]:
    """Return SGLang/Qwen chat-template kwargs to disable thinking when supported."""
    return {"chat_template_kwargs": {"enable_thinking": False}}


__all__ = [
    "extract_chat_completion_text",
    "extract_logprobs_openai_chat",
    "sglang_extra_body_disable_thinking",
]
