import unittest

from reasoning_benchmark_pricing import (
    estimate_anthropic_standard_cost,
    estimate_openai_standard_cost,
)


class PricingTests(unittest.TestCase):
    def test_anthropic_cost_uses_input_and_output_rates(self) -> None:
        cost = estimate_anthropic_standard_cost(
            "anthropic.claude-haiku-4-5",
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )
        self.assertEqual(cost, 6.0)

    def test_openai_cost_applies_cache_write_rate(self) -> None:
        cost = estimate_openai_standard_cost(
            "openai.gpt-5.6-sol",
            {
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "raw_usage": {
                    "input_tokens_details": {"cache_write_tokens": 1_000_000}
                },
            },
        )
        self.assertEqual(cost, 39.88)

    def test_openai_model_rates_are_distinct(self) -> None:
        terra_cost = estimate_openai_standard_cost(
            "openai.gpt-5.6-terra",
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )
        luna_cost = estimate_openai_standard_cost(
            "openai.gpt-5.6-luna",
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )
        self.assertEqual(terra_cost, 15.40)
        self.assertEqual(luna_cost, 1.54)

    def test_missing_usage_does_not_look_like_zero_cost(self) -> None:
        self.assertIsNone(
            estimate_anthropic_standard_cost(
                "anthropic.claude-haiku-4-5",
                {"input_tokens": None, "output_tokens": None},
            )
        )
        self.assertIsNone(
            estimate_openai_standard_cost(
                "openai.gpt-5.6-sol",
                {"input_tokens": 100, "output_tokens": None},
            )
        )

    def test_unknown_model_has_no_estimate(self) -> None:
        self.assertIsNone(
            estimate_openai_standard_cost(
                "unknown-model",
                {"input_tokens": 100, "output_tokens": 100},
            )
        )


if __name__ == "__main__":
    unittest.main()
