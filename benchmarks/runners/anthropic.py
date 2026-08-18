#!/usr/bin/env python3
"""
Cycles through Anthropic Claude Opus 5, Sonnet 5, Fable 5, and Haiku 4.5 on
Amazon Bedrock, running a shared six-scenario prompt suite at each model's
supported reasoning level.

Opus 5, Sonnet 5, and Fable 5 use adaptive thinking and an effort level. Haiku
4.5 uses extended thinking, so its low, medium, and high benchmark entries map to
explicit thinking budgets. The script uses the Anthropic Messages API exposed
by the bedrock-mantle endpoint, rather than the OpenAI-compatible Responses
API.

Assumes you're already logged in (for example, `aws sso login`) and uses the
default AWS credential chain, with no profile/session wiring needed.

Requirements:
    pip install aws-bedrock-token-generator requests
"""

import argparse
import time

import requests
from aws_bedrock_token_generator import provide_token
from botocore.exceptions import BotoCoreError

from reasoning_benchmark_evaluation import (
    classify_response_outcome,
    count_outcomes,
    evaluate_answer,
    evaluate_recoverable_answer,
    evaluation_label,
    outcome_label,
)
from reasoning_benchmark_pricing import estimate_anthropic_standard_cost
from benchmarks.runners.common import (
    SuiteConfig,
    create_run_metadata,
    load_suite,
    write_results_checkpoint,
)

# --- Provider configuration -------------------------------------------

AWS_REGION = "us-east-1"
MAX_TOKENS = 65_536
SCENARIO_WIDTH = 18
OUTCOME_WIDTH = 14

MODEL_CONFIGS = {
    "anthropic.claude-fable-5": {
        "reasoning_mode": "adaptive thinking",
        "efforts": ("low", "medium", "high", "xhigh", "max"),
    },
    "anthropic.claude-opus-5": {
        "reasoning_mode": "adaptive thinking",
        "efforts": ("low", "medium", "high", "xhigh", "max"),
    },
    "anthropic.claude-sonnet-5": {
        "reasoning_mode": "adaptive thinking",
        "efforts": ("low", "medium", "high", "xhigh", "max"),
    },
    "anthropic.claude-haiku-4-5": {
        "reasoning_mode": "extended thinking",
        # Haiku 4.5 does not support adaptive-thinking effort. These labels
        # make its explicit thinking budgets comparable to the other benchmarks.
        "thinking_budgets": {
            "low": 1_024,
            "medium": 4_096,
            "high": 16_384,
        },
    },
}

MAX_TOKENS_BY_MODEL = {
    "anthropic.claude-haiku-4-5": 32_768,
}

BASE_URL = f"https://bedrock-mantle.{AWS_REGION}.api.aws/anthropic/v1/messages"


# --- Helpers -------------------------------------------------------------


AUTH_MAX_RETRIES = 2
AUTH_BASE_BACKOFF_SECONDS = 2
AUTH_MAX_BACKOFF_SECONDS = 60


def get_bearer_token(region: str) -> tuple[str, list[dict]]:
    """Generate a Bedrock bearer token using the default AWS credential chain.

    This picks up whatever profile/credentials are active in your shell
    (e.g. via `export AWS_PROFILE=your-profile` + `aws sso login`), so no
    boto3 Session object needs to be threaded through explicitly.
    """
    retry_events = []
    for attempt in range(AUTH_MAX_RETRIES + 1):
        try:
            return provide_token(region=region), retry_events
        except BotoCoreError as exc:
            if attempt >= AUTH_MAX_RETRIES:
                exc.credential_retry_events = retry_events
                exc.credential_request_attempts = attempt + 1
                raise

            wait = min(
                AUTH_BASE_BACKOFF_SECONDS * (2**attempt),
                AUTH_MAX_BACKOFF_SECONDS,
            )
            retry_events.append(
                {
                    "attempt": attempt + 1,
                    "error_type": type(exc).__name__,
                    "backoff_s": wait,
                }
            )
            print(
                f"  Retryable AWS credential error ({type(exc).__name__}), "
                f"attempt {attempt + 1}/{AUTH_MAX_RETRIES}, retrying in {wait}s..."
            )
            time.sleep(wait)


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}
MAX_RETRIES = 2
MAX_TIMEOUT_RETRIES = 1
BASE_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 300
REQUEST_DEADLINE_SECONDS = 600


def get_reasoning_config(model: str, effort: str) -> dict:
    """Build the native Claude thinking configuration for a benchmark entry."""
    config = MODEL_CONFIGS[model]

    if config["reasoning_mode"] == "adaptive thinking":
        return {
            "output_config": {"effort": effort},
        }

    return {
        "thinking": {
            "type": "enabled",
            "budget_tokens": config["thinking_budgets"][effort],
        }
    }


def describe_reasoning(model: str, effort: str) -> str:
    """Return a concise label for terminal output and result records."""
    config = MODEL_CONFIGS[model]
    if config["reasoning_mode"] == "adaptive thinking":
        return f"adaptive/{effort}"
    return f"extended/{effort} ({config['thinking_budgets'][effort]:,} tokens)"


def max_tokens_for_model(model: str) -> int:
    """Return a model-specific output ceiling or the Anthropic default."""
    return MAX_TOKENS_BY_MODEL.get(model, MAX_TOKENS)


def get_retry_delay(response: requests.Response | None, attempt: int) -> float:
    """Use Bedrock's retry hint when present, otherwise exponential backoff."""
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after is not None:
        try:
            return min(float(retry_after), MAX_BACKOFF_SECONDS)
        except ValueError:
            pass

    return min(BASE_BACKOFF_SECONDS * (2**attempt), MAX_BACKOFF_SECONDS)


def call_model(
    bearer_token: str,
    model: str,
    effort: str,
    prompt: str,
    system_prompt: str,
) -> tuple[dict, list[dict]]:
    """Call one model within a bounded retry and wall-clock budget."""
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens_for_model(model),
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        **get_reasoning_config(model, effort),
    }

    retry_events = []
    deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
    for attempt in range(MAX_RETRIES + 1):
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            error = requests.exceptions.Timeout(
                f"Request deadline of {REQUEST_DEADLINE_SECONDS}s exceeded."
            )
            error.retry_events = retry_events
            error.request_attempts = attempt
            raise error

        try:
            resp = requests.post(
                BASE_URL,
                headers=headers,
                json=payload,
                timeout=min(REQUEST_TIMEOUT_SECONDS, remaining_seconds),
            )
        except requests.exceptions.Timeout as error:
            retry_limit = MAX_TIMEOUT_RETRIES
            response = None
            retry_error = error
        except requests.exceptions.ConnectionError as error:
            retry_limit = MAX_RETRIES
            response = None
            retry_error = error
        else:
            if resp.status_code not in RETRYABLE_STATUS_CODES:
                # Success, or a non-retryable client error (400, etc.) — don't
                # retry a bad request, it'll just fail the same way again.
                try:
                    resp.raise_for_status()
                except requests.exceptions.HTTPError as exc:
                    exc.retry_events = retry_events
                    exc.request_attempts = attempt + 1
                    raise
                return resp.json(), retry_events
            retry_error = requests.exceptions.HTTPError(
                f"{resp.status_code} Server Error (retryable) for url: {resp.url}",
                response=resp,
            )
            retry_limit = MAX_RETRIES
            response = resp

        if attempt >= retry_limit:
            retry_error.retry_events = retry_events
            retry_error.request_attempts = attempt + 1
            raise retry_error

        wait = get_retry_delay(response, attempt)
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= wait:
            error = requests.exceptions.Timeout(
                f"Request deadline of {REQUEST_DEADLINE_SECONDS}s exceeded."
            )
            error.retry_events = retry_events
            error.request_attempts = attempt + 1
            raise error

        retry_events.append(
            {
                "attempt": attempt + 1,
                "status_code": response.status_code if response is not None else None,
                "error_type": (
                    type(retry_error).__name__ if response is None else None
                ),
                "backoff_s": wait,
            }
        )
        print(
            f"  Retryable {type(retry_error).__name__}, attempt {attempt + 1}/{retry_limit}, "
            f"retrying in {wait}s..."
        )
        time.sleep(wait)


def extract_text(response_json: dict) -> str:
    """Pull the final answer text out of an Anthropic Messages API payload."""
    chunks = []
    for part in response_json.get("content", []):
        if part.get("type") == "text":
            chunks.append(part.get("text", ""))
    return "\n".join(chunks).strip()


def extract_response_metadata(response_json: dict) -> dict:
    """Capture provider response state without treating it as answer text."""
    stop_reason = response_json.get("stop_reason")
    stop_details = response_json.get("stop_details")
    provider_refusal = stop_reason == "refusal" or (
        isinstance(stop_details, dict) and stop_details.get("type") == "refusal"
    )
    return {
        "response_id": response_json.get("id"),
        "response_model": response_json.get("model"),
        "response_stop_reason": stop_reason,
        "response_stop_sequence": response_json.get("stop_sequence"),
        "response_stop_details": stop_details,
        "provider_refusal": provider_refusal,
        "provider_truncated": stop_reason == "max_tokens",
    }


def extract_usage(response_json: dict) -> dict:
    """Pull input, reasoning, and output counts from a response.

    Typical shape:
        "usage": {
            "input_tokens": ...,
            "output_tokens": ...,
        }
    """
    usage = response_json.get("usage", {})
    output_details = usage.get("output_tokens_details", {}) or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    return {
        "input_tokens": input_tokens,
        # Thinking is billed as output; retain the separate count when the
        # Messages API includes it in either supported usage shape.
        "reasoning_tokens": (
            usage.get("thinking_tokens")
            if usage.get("thinking_tokens") is not None
            else output_details.get("thinking_tokens")
        ),
        "output_tokens": output_tokens,
        "total_tokens": (
            sum(value or 0 for value in (input_tokens, output_tokens))
            if input_tokens is not None or output_tokens is not None
            else None
        ),
        "raw_usage": usage,  # keep the full block too, in case shape differs
    }


def calculate_cost(model: str, usage: dict) -> float | None:
    """Estimate USD cost for one call at standard on-demand token rates.

    Claude bills extended/adaptive-thinking tokens as output tokens.
    """
    return estimate_anthropic_standard_cost(model, usage)


# --- Main benchmark --------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        help=(
            "Run only one model. Partial runs use a repair filename and are "
            "excluded from automatic full-benchmark chart discovery."
        ),
    )
    return parser.parse_args(argv)


def main(suite_config: SuiteConfig, argv: list[str] | None = None) -> None:
    """Run the Anthropic benchmark for one explicitly configured suite."""
    args = parse_args(argv)
    prompts, expected_answers = load_suite(suite_config)
    selected_configs = (
        {args.model: MODEL_CONFIGS[args.model]} if args.model else MODEL_CONFIGS
    )
    verified_answers = suite_config.verify_answer_key()
    print(f"Ground truth verified: {len(verified_answers)}/{len(prompts)} scenarios")
    results = []
    results_basename = (
        suite_config.repair_results_basename
        if args.model and suite_config.repair_results_basename
        else suite_config.results_basename
    )
    run_metadata, results_path = create_run_metadata(
        suite_config,
        results_basename=results_basename,
        provider_metadata={
            "max_tokens": MAX_TOKENS,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "request_deadline_seconds": REQUEST_DEADLINE_SECONDS,
            "max_retries": MAX_RETRIES,
            "max_timeout_retries": MAX_TIMEOUT_RETRIES,
        },
        selected_model=args.model,
    )
    suite_config.results_dir.mkdir(parents=True, exist_ok=True)
    total_calls = len(prompts) * sum(
        len(model_config.get("efforts", model_config.get("thinking_budgets", {})))
        for model_config in selected_configs.values()
    )
    call_number = 0

    for scenario, prompt in prompts:
        for model, model_config in selected_configs.items():
            efforts = model_config.get(
                "efforts", model_config.get("thinking_budgets", {}).keys()
            )
            for effort in efforts:
                request_max_tokens = max_tokens_for_model(model)
                call_number += 1
                print(f"\n{'=' * 70}")
                print(
                    f"[{call_number}/{total_calls}] SCENARIO: {scenario}  |  MODEL: {model}  |  "
                    f"REASONING: {describe_reasoning(model, effort)}"
                )
                print(f"MAX TOKENS: {request_max_tokens:,}")
                print(f"BENCHMARK VARIANT: {suite_config.variant}")
                print(f"RUN ID: {run_metadata['run_id']}")
                print("=" * 70)

                start = time.time()
                credential_retry_events = []
                try:
                    bearer_token, credential_retry_events = get_bearer_token(AWS_REGION)
                    response_json, retry_events = call_model(
                        bearer_token, model, effort, prompt, suite_config.system_prompt
                    )
                    elapsed = time.time() - start

                    answer = extract_text(response_json)
                    response_metadata = extract_response_metadata(response_json)
                    usage = extract_usage(response_json)
                    cost = calculate_cost(model, usage)
                    evaluation = evaluate_answer(answer, expected_answers[scenario])
                    recoverable_evaluation = evaluate_recoverable_answer(
                        answer, expected_answers[scenario]
                    )
                    outcome = classify_response_outcome(
                        evaluation,
                        recoverable_evaluation,
                        provider_refusal=response_metadata["provider_refusal"],
                        provider_truncated=response_metadata["provider_truncated"],
                    )

                    print(f"Time: {elapsed:.2f}s")
                    print(
                        f"Request attempts: {len(retry_events) + 1}  "
                        f"retries: {len(retry_events)}"
                    )
                    print(
                        f"Tokens — input: {usage['input_tokens']}  "
                        f"reasoning: {usage['reasoning_tokens']}  "
                        f"output: {usage['output_tokens']}  "
                        f"total: {usage['total_tokens']}"
                    )
                    print(
                        f"Estimated cost: ${cost:.6f}"
                        if cost is not None
                        else "Estimated cost: N/A (no pricing for this model)"
                    )
                    print(f"Outcome: {outcome_label(outcome)}")
                    print(
                        f"Strict JSON result: {evaluation_label(evaluation)}"
                        + (f" ({evaluation['detail']})" if evaluation["detail"] else "")
                    )
                    if not evaluation["correct"]:
                        print(
                            f"Recoverable JSON result: {evaluation_label(recoverable_evaluation)}"
                            + (
                                f" ({recoverable_evaluation['detail']})"
                                if recoverable_evaluation["detail"]
                                else ""
                            )
                        )
                    print(f"Answer:\n{answer}")

                    results.append(
                        {
                            **run_metadata,
                            "max_tokens": request_max_tokens,
                            "scenario": scenario,
                            "model": model,
                            "effort": effort,
                            "reasoning": describe_reasoning(model, effort),
                            "elapsed_s": round(elapsed, 2),
                            "credential_request_attempts": (
                                len(credential_retry_events) + 1
                            ),
                            "credential_retry_count": len(credential_retry_events),
                            "credential_retry_events": credential_retry_events,
                            "request_attempts": len(retry_events) + 1,
                            "retry_count": len(retry_events),
                            "retry_events": retry_events,
                            **response_metadata,
                            "usage": usage,
                            "cost_usd": cost,
                            "evaluation": evaluation,
                            "recoverable_evaluation": recoverable_evaluation,
                            "outcome": outcome,
                            "answer": answer,
                            **({"raw_response": response_json} if not answer else {}),
                        }
                    )

                except requests.exceptions.HTTPError as e:
                    elapsed = time.time() - start
                    retry_events = getattr(e, "retry_events", [])
                    credential_retry_events = getattr(
                        e, "credential_retry_events", credential_retry_events
                    )
                    request_attempts = getattr(
                        e, "request_attempts", len(retry_events) + 1
                    )
                    print(f"HTTP error: {e}")
                    if e.response is not None:
                        print(f"Response body: {e.response.text}")
                    else:
                        print("Response body: N/A (no response object)")
                    results.append(
                        {
                            **run_metadata,
                            "max_tokens": request_max_tokens,
                            "scenario": scenario,
                            "model": model,
                            "effort": effort,
                            "reasoning": describe_reasoning(model, effort),
                            "elapsed_s": round(elapsed, 2),
                            "credential_request_attempts": getattr(
                                e,
                                "credential_request_attempts",
                                len(credential_retry_events) + 1,
                            ),
                            "credential_retry_count": len(credential_retry_events),
                            "credential_retry_events": credential_retry_events,
                            "request_attempts": request_attempts,
                            "retry_count": len(retry_events),
                            "retry_events": retry_events,
                            "outcome": "endpoint_error",
                            "error": str(e),
                            "error_status_code": (
                                e.response.status_code
                                if e.response is not None
                                else None
                            ),
                            "error_response_body": (
                                e.response.text if e.response is not None else None
                            ),
                            "evaluation": {
                                "status": "error",
                                "correct": False,
                                "detail": str(e),
                            },
                        }
                    )
                except Exception as e:
                    elapsed = time.time() - start
                    retry_events = getattr(e, "retry_events", [])
                    credential_retry_events = getattr(
                        e, "credential_retry_events", credential_retry_events
                    )
                    request_attempts = getattr(
                        e, "request_attempts", len(retry_events) + 1
                    )
                    print(f"Error: {e}")
                    results.append(
                        {
                            **run_metadata,
                            "max_tokens": request_max_tokens,
                            "scenario": scenario,
                            "model": model,
                            "effort": effort,
                            "reasoning": describe_reasoning(model, effort),
                            "elapsed_s": round(elapsed, 2),
                            "credential_request_attempts": getattr(
                                e,
                                "credential_request_attempts",
                                len(credential_retry_events) + 1,
                            ),
                            "credential_retry_count": len(credential_retry_events),
                            "credential_retry_events": credential_retry_events,
                            "request_attempts": request_attempts,
                            "retry_count": len(retry_events),
                            "retry_events": retry_events,
                            "outcome": "endpoint_error",
                            "error": str(e),
                            "evaluation": {
                                "status": "error",
                                "correct": False,
                                "detail": str(e),
                            },
                        }
                    )

                write_results_checkpoint(results_path, results)

    # --- Summary ---
    print(f"\n\n{'#' * 70}")
    print("SUMMARY")
    print("#" * 70)
    time_width = max(
        len("TIME"),
        *(len(f"{r['elapsed_s']:.2f}s") for r in results if "elapsed_s" in r),
    )
    header = (
        f"{'SCENARIO':{SCENARIO_WIDTH}s} | {'MODEL':30s} | {'LEVEL':7s} | {'TIME':>{time_width}s} | {'IN':>6s} | "
        f"{'OUT':>6s} | {'TOTAL':>6s} | {'COST':>9s} | {'OUTCOME':{OUTCOME_WIDTH}s}"
    )
    print(header)
    print("-" * len(header))
    total_cost = 0.0
    for r in results:
        if "error" in r:
            print(
                f"{r['scenario']:{SCENARIO_WIDTH}s} | {r['model']:30s} | {r['effort']:7s} | "
                f"{'ENDPOINT_ERROR':{OUTCOME_WIDTH}s}: {r['error']}"
            )
        else:
            u = r["usage"]
            cost = r.get("cost_usd")
            if cost is not None:
                total_cost += cost
            cost_str = f"${cost:.6f}" if cost is not None else "N/A"
            elapsed_str = f"{r['elapsed_s']:.2f}s"
            result_str = outcome_label(r["outcome"])
            print(
                f"{r['scenario']:{SCENARIO_WIDTH}s} | {r['model']:30s} | {r['effort']:7s} | "
                f"{elapsed_str:>{time_width}s} | {str(u['input_tokens']):>6s} "
                f"| {str(u['output_tokens']):>6s} | {str(u['total_tokens']):>6s} | {cost_str:>9s} | {result_str:{OUTCOME_WIDTH}s}"
            )
    print("-" * len(header))
    print(f"{'TOTAL ESTIMATED COST':>{len(header) - 12}s} | ${total_cost:.6f}")
    outcome_counts = count_outcomes(results)
    semantic_correct = outcome_counts["strict"] + outcome_counts["format_only"]
    print(f"RAW JSON CORRECT: {outcome_counts['strict']}/{len(results)}")
    print(f"SEMANTICALLY CORRECT: {semantic_correct}/{len(results)}")
    print(f"  CORRECT, BARE JSON: {outcome_counts['strict']}")
    print(f"  CORRECT, WRONG FORMAT: {outcome_counts['format_only']}")
    print(f"WRONG ANSWERS: {outcome_counts['semantic_error']}")
    print(f"POLICY REFUSALS: {outcome_counts['policy_refusal']}")
    print(f"TRUNCATED RESPONSES: {outcome_counts['truncated']}")
    print(f"MALFORMED RESPONSES: {outcome_counts['malformed']}")
    total_retries = sum(r.get("retry_count", 0) for r in results)
    retried_calls = sum(bool(r.get("retry_count")) for r in results)
    total_credential_retries = sum(r.get("credential_retry_count", 0) for r in results)
    credential_retried_calls = sum(
        bool(r.get("credential_retry_count")) for r in results
    )
    print(
        f"REQUEST RETRIES: {total_retries} across {retried_calls}/{len(results)} calls"
    )
    print(
        f"CREDENTIAL RETRIES: {total_credential_retries} across "
        f"{credential_retried_calls}/{len(results)} calls"
    )
    print(f"ENDPOINT ERRORS: {outcome_counts['endpoint_error']}/{len(results)}")
    print("(Cost uses standard Bedrock on-demand rates with prompt caching disabled.)")

    write_results_checkpoint(results_path, results)
    print(f"\nFull results written to {results_path}")
