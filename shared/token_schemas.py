"""Pydantic schemas for shared LLM token usage and cost reports."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TierTokenUsage(BaseModel):
    """Token usage for one model tier over the whole run — provider-reported or estimated."""

    model: str = Field(description="model id backing this tier")
    models_seen: list[str] = Field(
        default_factory=list,
        description="every model id this tier ran. More than one means the provider fallback "
                    "latched part-way through, so the tier's tokens span two price lists")
    encoding: str = Field(
        description="tiktoken encoding this model falls back to when usage is unavailable")
    source: str = Field(
        default="local-estimate",
        description="'gateway', 'local-estimate', or 'mixed'")
    calls: int = Field(description="number of LLM invocations on this tier, retries included")
    tokens_in: int = Field(description="new, uncached input tokens")
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    prompt_tokens: int = 0
    tokens_out: int = Field(description="billed output tokens, extended thinking included")
    thinking_tokens: int = Field(default=0, description="a subset of tokens_out")
    tokens_total: int = Field(description="prompt_tokens + tokens_out")


class TierCost(BaseModel):
    """What one tier's tokens cost, from the configured rates."""

    model: str = ""
    rates_basis: str = "tier"
    rates_key: str = ""
    cache_write_billed: bool = True
    input: float = 0.0
    cache_write: float = 0.0
    cache_read: float = 0.0
    output: float = 0.0
    total: float = 0.0
    uncached_equivalent: float = 0.0
    cache_savings: float = 0.0


class RunCost(BaseModel):
    """The run's estimated cost, priced from the configured model/tier rates."""

    priced: bool = False
    currency: str = "USD"
    rates_source: str = ""
    as_of: str = ""
    rates_file: str = ""
    cache_ttl: str = ""
    unpriced_tiers: list[str] = Field(default_factory=list)
    model_mismatch: list[dict] = Field(default_factory=list)
    mixed_model_tiers: list[dict] = Field(default_factory=list)
    by_tier: dict[str, TierCost] = Field(default_factory=dict)
    total_usd: float | None = None
    uncached_equivalent_usd: float | None = None
    cache_savings_usd: float | None = None
    note: str = ""


class TokenUsageReport(BaseModel):
    """Token totals and estimated cost for one accounting window."""

    estimator: str = "tiktoken"
    note: str = ""
    by_tier: dict[str, TierTokenUsage] = Field(default_factory=dict)
    total_tokens_in: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_prompt_tokens: int = 0
    total_tokens_out: int = 0
    total_thinking_tokens: int = 0
    total_tokens: int = 0
    cost: RunCost = Field(default_factory=RunCost)


__all__ = ["RunCost", "TierCost", "TierTokenUsage", "TokenUsageReport"]
