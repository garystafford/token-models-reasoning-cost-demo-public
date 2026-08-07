#!/usr/bin/env python3
"""
Cycles through GPT-5.6 Sol, Terra, Luna, and GPT-5.5 on Amazon Bedrock across
their supported reasoning-effort levels, running a shared six-scenario prompt
suite at each level with prompt caching disabled.

Note: These models are served via the OpenAI-compatible Responses API on the
bedrock-mantle endpoint, not the native bedrock-runtime InvokeModel API.
Assumes you're already logged in (e.g. `aws sso login`) — uses the default
AWS credential chain, no profile/session wiring needed.

Requirements:
    pip install aws-bedrock-token-generator requests
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from aws_bedrock_token_generator import provide_token
from botocore.exceptions import BotoCoreError

from reasoning_benchmark_evaluation import (
    ANSWER_KEY_PATH,
    classify_response_outcome,
    count_outcomes,
    evaluate_answer,
    evaluate_recoverable_answer,
    evaluation_label,
    load_expected_answers,
    outcome_label,
)
from reasoning_benchmark_pricing import (
    estimate_openai_standard_cost,
    openai_pricing_metadata,
)
from operations_verify_answer_key import verify_answer_key

# --- Configuration -----------------------------------------------------

AWS_REGION = "us-east-1"
SCENARIO_WIDTH = 18
OUTCOME_WIDTH = 14
BENCHMARK_VARIANT = "json_contract_v5"
RESULTS_DIR = Path(__file__).with_name("results")
RESULTS_BASENAME = "operations_bedrock_reasoning_benchmark_openai"

# Keep this contract identical in the Anthropic benchmark. The user prompt still
# defines each task's JSON shape; this stable instruction defines the response
# boundary for every scenario and effort level.
SYSTEM_PROMPT = """This is a controlled benchmark of benign, self-contained
operations reasoning. The user tasks may involve workflow mathematics,
scheduling, policy application, incident fact extraction, or small code
corrections. They do not ask you to interact with systems, retrieve data, or
take action outside the supplied text.

Answer the requested task within your normal safety policy.

Return exactly one valid JSON object that matches the schema requested by the user.
Output the object directly.

Do not include Markdown, code fences, prose, analysis, explanations, or any text
before or after the JSON object."""

MODEL_EFFORTS = {
    "openai.gpt-5.6-sol": ("low", "medium", "high", "xhigh", "max"),
    "openai.gpt-5.6-terra": ("low", "medium", "high", "xhigh", "max"),
    "openai.gpt-5.6-luna": ("low", "medium", "high", "xhigh", "max"),
    # GPT-5.5 supports none through xhigh, but the benchmark intentionally starts
    # at low so every entry performs some reasoning.
    "openai.gpt-5.5": ("low", "medium", "high", "xhigh"),
}

PROMPT_SUITE_PATH = Path(__file__).with_name(
    "operations_reasoning_benchmark_prompts.json"
)


def load_prompts() -> tuple[tuple[str, str], ...]:
    """Load and validate the shared benchmark prompt suite."""
    try:
        suite = json.loads(PROMPT_SUITE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load prompt suite: {PROMPT_SUITE_PATH}") from exc

    if suite.get("version") != 2:
        raise ValueError("Prompt suite must declare version 2.")

    scenarios = suite.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Prompt suite must contain at least one scenario.")

    prompts = []
    scenario_ids = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("Each prompt-suite scenario must be an object.")

        scenario_id = scenario.get("id")
        difficulty = scenario.get("difficulty")
        prompt = scenario.get("prompt")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or not isinstance(difficulty, str)
            or not difficulty
            or not isinstance(prompt, str)
            or not prompt
        ):
            raise ValueError(
                "Each scenario requires non-empty id, difficulty, and prompt fields."
            )
        if scenario_id in scenario_ids:
            raise ValueError(f"Duplicate prompt-suite scenario ID: {scenario_id}")

        scenario_ids.add(scenario_id)
        prompts.append((scenario_id, prompt))

    return tuple(prompts)


PROMPTS = load_prompts()
EXPECTED_ANSWERS = load_expected_answers(scenario_id for scenario_id, _ in PROMPTS)

BASE_URL = f"https://bedrock-mantle.{AWS_REGION}.api.aws/openai/v1/responses"


# --- Helpers -------------------------------------------------------------


AUTH_MAX_RETRIES = 6
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


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s, 16s


def call_model(
    bearer_token: str, model: str, effort: str, prompt: str
) -> tuple[dict, list[dict]]:
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": prompt,
        "instructions": SYSTEM_PROMPT,
        "reasoning": {"effort": effort},
    }

    retry_events = []
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt >= MAX_RETRIES:
                exc.retry_events = retry_events
                exc.request_attempts = attempt + 1
                raise

            wait = BASE_BACKOFF_SECONDS * (2**attempt)
            retry_events.append(
                {
                    "attempt": attempt + 1,
                    "status_code": None,
                    "error_type": type(exc).__name__,
                    "backoff_s": wait,
                }
            )
            print(
                f"  Retryable {type(exc).__name__}, attempt {attempt + 1}/{MAX_RETRIES}, "
                f"retrying in {wait}s..."
            )
            time.sleep(wait)
            continue

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

        # Retryable error — back off and try again, unless out of attempts.
        if attempt < MAX_RETRIES:
            wait = BASE_BACKOFF_SECONDS * (2**attempt)
            retry_events.append(
                {
                    "attempt": attempt + 1,
                    "status_code": resp.status_code,
                    "error_type": None,
                    "backoff_s": wait,
                }
            )
            print(
                f"  Retryable error ({resp.status_code}), attempt {attempt + 1}/{MAX_RETRIES}, "
                f"retrying in {wait}s..."
            )
            time.sleep(wait)

    # Exhausted retries — surface the last error to the caller.
    error = requests.exceptions.HTTPError(
        f"{resp.status_code} Server Error (retryable) for url: {resp.url}",
        response=resp,
    )
    error.retry_events = retry_events
    error.request_attempts = MAX_RETRIES + 1
    raise error


def extract_text(response_json: dict) -> str:
    """Pull the final answer text out of a Responses API payload."""
    chunks = []
    for item in response_json.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text"):
                    chunks.append(part.get("text", ""))
    return "\n".join(chunks).strip()


def extract_response_metadata(response_json: dict) -> dict:
    """Capture Responses API state and refusal details separately from text."""
    refusals = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "refusal":
                refusals.append(part.get("refusal") or part.get("text") or "")

    response_error = response_json.get("error")
    error_type = (
        response_error.get("type") if isinstance(response_error, dict) else None
    )
    provider_refusal = (
        bool(refusals)
        or response_json.get("status") == "refused"
        or (isinstance(error_type, str) and "refusal" in error_type)
    )
    incomplete_details = response_json.get("incomplete_details")
    incomplete_reason = (
        incomplete_details.get("reason")
        if isinstance(incomplete_details, dict)
        else None
    )
    provider_truncated = response_json.get(
        "status"
    ) == "incomplete" and incomplete_reason in {"max_output_tokens", "max_tokens"}
    return {
        "response_id": response_json.get("id"),
        "response_model": response_json.get("model"),
        "response_status": response_json.get("status"),
        "response_incomplete_details": incomplete_details,
        "response_error": response_error,
        "response_refusals": refusals,
        "provider_refusal": provider_refusal,
        "provider_truncated": provider_truncated,
    }


def extract_usage(response_json: dict) -> dict:
    """Pull input, reasoning, and output counts from a cold-cache response.

    Typical shape:
        "usage": {
            "input_tokens": ...,
            "output_tokens": ...,
            "output_tokens_details": {"reasoning_tokens": ...},
            "total_tokens": ...
        }
    """
    usage = response_json.get("usage", {})
    input_details = usage.get("input_tokens_details", {}) or {}
    output_details = usage.get("output_tokens_details", {}) or {}
    cache_write_tokens = input_details.get("cache_write_tokens") or 0
    cached_tokens = input_details.get("cached_tokens") or 0
    if cache_write_tokens or cached_tokens:
        raise RuntimeError(
            "Prompt caching was disabled, but the OpenAI response reported "
            f"{cache_write_tokens} cache-write and {cached_tokens} cache-read tokens."
        )

    return {
        "input_tokens": usage.get("input_tokens"),
        "reasoning_tokens": output_details.get("reasoning_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "raw_usage": usage,  # keep the full block too, in case shape differs
    }


def calculate_cost(model: str, usage: dict) -> float | None:
    """Estimate USD cost for one call at standard on-demand token rates.

    output_tokens as returned by the Responses API already includes
    reasoning tokens (they're billed at the output rate), so no separate
    line item is needed for reasoning.
    """
    return estimate_openai_standard_cost(model, usage)


def create_run_metadata() -> tuple[dict[str, object], Path]:
    """Create a UTC run identity and its non-overwriting result path."""
    started_at = datetime.now(timezone.utc)
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{RESULTS_BASENAME}_{BENCHMARK_VARIANT}_{timestamp}"
    metadata = {
        "run_id": run_id,
        "run_started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
        "benchmark_variant": BENCHMARK_VARIANT,
        "prompt_suite_sha256": hashlib.sha256(
            PROMPT_SUITE_PATH.read_bytes()
        ).hexdigest(),
        "answer_key_sha256": hashlib.sha256(ANSWER_KEY_PATH.read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    return metadata, RESULTS_DIR / f"{run_id}.json"


def write_results_checkpoint(results_path: Path, results: list[dict]) -> None:
    """Atomically preserve completed calls without waiting for benchmark completion."""
    temporary_path = results_path.with_suffix(f"{results_path.suffix}.tmp")
    temporary_path.write_text(
        f"{json.dumps(results, indent=2)}\n",
        encoding="utf-8",
    )
    temporary_path.replace(results_path)


# --- Main benchmark --------------------------------------------------------


def main():
    verified_answers = verify_answer_key()
    print(f"Ground truth verified: {len(verified_answers)}/{len(PROMPTS)} scenarios")
    results = []
    run_metadata, results_path = create_run_metadata()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_calls = len(PROMPTS) * sum(len(efforts) for efforts in MODEL_EFFORTS.values())
    call_number = 0

    for scenario, prompt in PROMPTS:
        for model, efforts in MODEL_EFFORTS.items():
            for effort in efforts:
                call_number += 1
                print(f"\n{'=' * 70}")
                print(
                    f"[{call_number}/{total_calls}] SCENARIO: {scenario}  |  MODEL: {model}  |  "
                    f"REASONING EFFORT: {effort}"
                )
                print(f"BENCHMARK VARIANT: {BENCHMARK_VARIANT}")
                print(f"RUN ID: {run_metadata['run_id']}")
                print("=" * 70)

                start = time.time()
                credential_retry_events = []
                try:
                    bearer_token, credential_retry_events = get_bearer_token(AWS_REGION)
                    response_json, retry_events = call_model(
                        bearer_token, model, effort, prompt
                    )
                    elapsed = time.time() - start

                    answer = extract_text(response_json)
                    response_metadata = extract_response_metadata(response_json)
                    usage = extract_usage(response_json)
                    cost = calculate_cost(model, usage)
                    evaluation = evaluate_answer(answer, EXPECTED_ANSWERS[scenario])
                    recoverable_evaluation = evaluate_recoverable_answer(
                        answer, EXPECTED_ANSWERS[scenario]
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
                            "scenario": scenario,
                            "model": model,
                            "effort": effort,
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
                            "pricing": openai_pricing_metadata(model),
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
                            "scenario": scenario,
                            "model": model,
                            "effort": effort,
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
                            "pricing": openai_pricing_metadata(model),
                            "outcome": "endpoint_error",
                            "error": str(e),
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
                            "scenario": scenario,
                            "model": model,
                            "effort": effort,
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
                            "pricing": openai_pricing_metadata(model),
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
        f"{'SCENARIO':{SCENARIO_WIDTH}s} | {'MODEL':30s} | {'EFFORT':7s} | {'TIME':>{time_width}s} | {'IN':>6s} | "
        f"{'REASON':>7s} | {'OUT':>6s} | {'TOTAL':>6s} | {'COST':>9s} | {'OUTCOME':{OUTCOME_WIDTH}s}"
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
                f"| {str(u['reasoning_tokens']):>7s} "
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


if __name__ == "__main__":
    main()
