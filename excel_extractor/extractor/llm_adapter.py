"""Extractor-owned adapter over the shared LLM client."""
from __future__ import annotations

from typing import Type

from pydantic import BaseModel

from shared import llm as _llm


HOOK_MODULE = "extractor"


def structured_call(tier: str, system: str, user: str, schema: Type[BaseModel],
                    retries: int = 1, temperature: float = 0.0,
                    hook_stage: str = "", allow_budget_overrun: bool = False) -> BaseModel:
    """Call the shared structured surface as the extractor module."""
    return _llm.structured_call(
        tier, system, user, schema,
        retries=retries,
        temperature=temperature,
        hook_module=HOOK_MODULE,
        hook_stage=hook_stage,
        allow_budget_overrun=allow_budget_overrun,
    )


def text_call(tier: str, system: str, user: str, temperature: float = 0.0,
              hook_stage: str = "text_call") -> str:
    """Call the shared text surface as the extractor module."""
    return _llm.text_call(
        tier, system, user,
        temperature=temperature,
        hook_module=HOOK_MODULE,
        hook_stage=hook_stage,
    )


__all__ = ["HOOK_MODULE", "structured_call", "text_call"]
