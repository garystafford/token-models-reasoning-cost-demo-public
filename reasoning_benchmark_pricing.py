"""Standard on-demand pricing used by the reasoning benchmark."""

from __future__ import annotations


OPENAI_PRICING_AS_OF = "2026-08-01"
OPENAI_PRICING_SOURCE = "https://aws.amazon.com/bedrock/pricing/"
OPENAI_STANDARD_PRICING_PER_1M = {
    "openai.gpt-5.6-sol": {
        "input": 5.50,
        "cache_write": 6.88,
        "output": 33.00,
    },
    "openai.gpt-5.6-terra": {
        "input": 2.20,
        "cache_write": 2.75,
        "output": 13.20,
    },
    "openai.gpt-5.6-luna": {
        "input": 0.22,
        "cache_write": 0.275,
        "output": 1.32,
    },
    "openai.gpt-5.5": {"input": 5.50, "output": 33.00},
}

ANTHROPIC_PRICING_AS_OF = "2026-08-01"
ANTHROPIC_PRICING_SOURCE = "https://aws.amazon.com/bedrock/pricing/"
# Sonnet uses the announced $3/$15 standard rates that take effect after its
# $2/$10 launch promotion ends on August 31, 2026. Prompt caching is disabled.
ANTHROPIC_STANDARD_PRICING_PER_1M = {
    "anthropic.claude-fable-5": {"input": 10.00, "output": 50.00},
    "anthropic.claude-opus-5": {"input": 5.00, "output": 25.00},
    "anthropic.claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "anthropic.claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


def estimate_anthropic_standard_cost(model: str, usage: dict) -> float | None:
    """Estimate standard on-demand Claude cost with caching disabled."""
    rates = ANTHROPIC_STANDARD_PRICING_PER_1M.get(model)
    if rates is None:
        return None

    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    cost = (input_tokens / 1_000_000) * rates["input"] + (
        output_tokens / 1_000_000
    ) * rates["output"]
    return round(cost, 6)


def estimate_openai_standard_cost(model: str, usage: dict) -> float | None:
    """Estimate standard on-demand cost without cache-read discounts."""
    rates = OPENAI_STANDARD_PRICING_PER_1M.get(model)
    if rates is None:
        return None

    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    raw_usage = usage.get("raw_usage") or {}
    input_details = raw_usage.get("input_tokens_details") or {}
    cache_write_tokens = min(
        input_tokens,
        max(0, input_details.get("cache_write_tokens") or 0),
    )
    base_input_tokens = input_tokens - cache_write_tokens
    cache_write_rate = rates.get("cache_write", rates["input"])
    cost = (
        (base_input_tokens / 1_000_000) * rates["input"]
        + (cache_write_tokens / 1_000_000) * cache_write_rate
        + (output_tokens / 1_000_000) * rates["output"]
    )
    return round(cost, 6)


def openai_pricing_metadata(model: str) -> dict[str, object] | None:
    """Describe the exact pricing basis used for a model estimate."""
    rates = OPENAI_STANDARD_PRICING_PER_1M.get(model)
    if rates is None:
        return None

    return {
        "source": OPENAI_PRICING_SOURCE,
        "as_of": OPENAI_PRICING_AS_OF,
        "currency": "USD",
        "unit": "per_1m_tokens",
        "input": rates["input"],
        "cache_write": rates.get("cache_write"),
        "output": rates["output"],
        "cache_discount_applied": False,
    }
