"""Versioned RubricEval prompt templates."""

from __future__ import annotations

from forgeai.eval.constants import JUDGE_PROMPT_VERSION


def render_judge_prompt(
    *,
    query: str,
    response_text: str,
    retrieved_context: str,
) -> str:
    """Build structured JSON-only rubric scoring prompt."""

    return (
        f"JUDGE_PROMPT_VERSION={JUDGE_PROMPT_VERSION}\n"
        "You are a strict evaluation judge. Score the candidate response.\n\n"
        "Rubric (0-10 per dimension):\n"
        "- relevance: does the response address the query?\n"
        "- groundedness: is response grounded in retrieved context?\n"
        "- completeness: does it fully answer the query?\n"
        "- conciseness: concise without dropping key facts.\n\n"
        "Return ONLY valid JSON with keys:\n"
        '{"relevance": <0-10>, "groundedness": <0-10>, '
        '"completeness": <0-10>, "conciseness": <0-10>, '
        '"justifications": {"relevance":"...", "groundedness":"...", '
        '"completeness":"...", "conciseness":"..."}}\n\n'
        f"Query:\n{query}\n\n"
        f"Candidate response:\n{response_text}\n\n"
        f"Retrieved context:\n{retrieved_context}\n"
    )


__all__ = ["render_judge_prompt"]
