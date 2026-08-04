#!/usr/bin/env python3
"""Independently derive and verify every benchmark answer-key entry."""

from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ANSWER_KEY_PATH = Path(__file__).with_name("reasoning_benchmark_expected_answers.json")


def solve_pipeline_simple() -> dict[str, Any]:
    extract_minutes = 28
    transform_minutes = 34
    schema_validation_minutes = 18
    load_minutes = 22
    post_load_verification_minutes = 7
    total_minutes = (
        extract_minutes
        + max(transform_minutes, schema_validation_minutes)
        + load_minutes
        + post_load_verification_minutes
    )
    return {
        "total_minutes": total_minutes,
        "alert": total_minutes > 90,
    }


def solve_pipeline_moderate() -> dict[str, Any]:
    items = {
        "A": {"minutes": 25, "risk_reduction": 7, "dependency": None, "backup": False},
        "B": {"minutes": 20, "risk_reduction": 7, "dependency": "A", "backup": True},
        "C": {"minutes": 35, "risk_reduction": 10, "dependency": None, "backup": False},
        "D": {"minutes": 15, "risk_reduction": 6, "dependency": "C", "backup": True},
        "E": {"minutes": 30, "risk_reduction": 7, "dependency": None, "backup": False},
    }
    best: tuple[int, int, tuple[str, ...]] | None = None

    for size in range(len(items) + 1):
        for selected in itertools.combinations(items, size):
            selected_set = set(selected)
            if any(
                item["dependency"] is not None
                and item["dependency"] not in selected_set
                for item in (items[name] for name in selected)
            ):
                continue
            if not any(items[name]["backup"] for name in selected):
                continue

            total_minutes = sum(items[name]["minutes"] for name in selected)
            if total_minutes > 90:
                continue
            total_risk_reduction = sum(
                items[name]["risk_reduction"] for name in selected
            )
            candidate = (total_risk_reduction, -total_minutes, selected)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        raise AssertionError("No valid moderate pipeline selection.")

    total_risk_reduction, negative_minutes, selected_items = best
    return {
        "selected_items": list(selected_items),
        "total_minutes": -negative_minutes,
        "total_risk_reduction": total_risk_reduction,
    }


def solve_pipeline_complex() -> dict[str, Any]:
    candidates: list[tuple[int, int, int, int, int]] = []

    for extract_workers in range(1, 11):
        for transform_workers in range(1, 11):
            validate_workers = 12 - extract_workers - transform_workers
            if validate_workers < 1:
                continue

            extract_minutes = 240 // min(extract_workers, 6)
            transform_minutes = max(60, 96 - 6 * max(transform_workers - 4, 0))
            validate_minutes = max(42, 72 - 5 * max(validate_workers - 3, 0))
            load_minutes = (
                25
                + (10 if extract_workers > 4 else 0)
                + (5 if transform_workers > 5 else 0)
            )
            pipeline_minutes = (
                extract_minutes + transform_minutes + validate_minutes + load_minutes
            )
            if pipeline_minutes > 240:
                continue

            total_cost_dollars = 2 * (
                extract_workers * extract_minutes
                + transform_workers * transform_minutes
                + validate_workers * validate_minutes
            )
            candidates.append(
                (
                    total_cost_dollars,
                    pipeline_minutes,
                    extract_workers,
                    transform_workers,
                    validate_workers,
                )
            )

    if not candidates:
        raise AssertionError("No valid complex pipeline allocation.")

    cost, pipeline_minutes, extract_workers, transform_workers, validate_workers = min(
        candidates
    )
    return {
        "extract_workers": extract_workers,
        "transform_workers": transform_workers,
        "validate_workers": validate_workers,
        "pipeline_minutes": pipeline_minutes,
        "total_cost_dollars": cost,
    }


def solve_policy() -> dict[str, Any]:
    account_active = True
    diagnostic_code = "H2"
    incident_age_days = 18
    fraud_hold = False
    replacement_value_dollars = 680
    recent_replacement_claims = 2
    return {
        "expedited_replacement": (
            account_active
            and diagnostic_code in {"H2", "H4"}
            and incident_age_days <= 30
            and not fraud_hold
        ),
        "manager_review": (
            replacement_value_dollars > 750 or recent_replacement_claims >= 2
        ),
    }


def solve_extraction() -> dict[str, Any]:
    incident_start = datetime.fromisoformat("2026-01-01T09:14:00+00:00")
    rollback_complete = datetime.fromisoformat("2026-01-01T09:28:00+00:00")
    resolution_window_end = rollback_complete.replace(minute=43)
    deployments = (
        ("DEP-768", "Payment API", datetime.fromisoformat("2026-01-01T08:56:00+00:00")),
        ("DEP-771", "Payment API", datetime.fromisoformat("2026-01-01T09:08:00+00:00")),
        ("DEP-772", "Catalog API", datetime.fromisoformat("2026-01-01T09:12:00+00:00")),
    )
    affected_service = "Payment API"
    candidates = [
        (completed_at, deployment_id)
        for deployment_id, service, completed_at in deployments
        if service == affected_service
        and 0 <= (incident_start - completed_at).total_seconds() <= 15 * 60
    ]
    _, root_cause_deployment = max(candidates)

    post_rollback_samples = (
        (datetime.fromisoformat("2026-01-01T09:31:00+00:00"), 0.8),
        (datetime.fromisoformat("2026-01-01T09:36:00+00:00"), 1.1),
        (datetime.fromisoformat("2026-01-01T09:41:00+00:00"), 0.4),
        (datetime.fromisoformat("2026-01-01T09:47:00+00:00"), 0.2),
    )
    resolution_window_rates = [
        rate
        for sampled_at, rate in post_rollback_samples
        if rollback_complete <= sampled_at <= resolution_window_end
    ]
    max_resolution_window_rate = max(resolution_window_rates)

    return {
        "incident_id": "INC-4821",
        "root_cause_deployment": root_cause_deployment,
        "customer_impact_minutes": int(
            (rollback_complete - incident_start).total_seconds() / 60
        ),
        "max_error_rate_in_resolution_window_pct": max_resolution_window_rate,
        "resolved": max_resolution_window_rate < 1.0,
    }


def evaluate_corrected_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive observable task results from the corrected cost rules."""
    eligible_task_ids = []
    rounded_billable_minutes = {}
    final_task_cost_dollars = {}
    total = 0

    for task in tasks:
        if not task["active"] or task["cancelled"]:
            continue

        task_id = task["id"]
        billable = max(task["minutes"] - task["free_minutes"], 0)
        rounded = ((billable + 14) // 15) * 15
        task_cost = rounded * task["workers"] * task["rate"]
        if task["priority"] == "emergency":
            task_cost = task_cost * 3 // 2

        eligible_task_ids.append(task_id)
        rounded_billable_minutes[task_id] = rounded
        final_task_cost_dollars[task_id] = task_cost
        total += task_cost

    return {
        "eligible_task_ids": sorted(eligible_task_ids),
        "rounded_billable_minutes": rounded_billable_minutes,
        "final_task_cost_dollars": final_task_cost_dollars,
        "correct_total_dollars": total,
    }


def solve_debugging() -> dict[str, Any]:
    tasks = [
        {
            "id": "A",
            "active": True,
            "cancelled": False,
            "minutes": 38,
            "free_minutes": 5,
            "workers": 2,
            "rate": 3,
            "priority": "normal",
        },
        {
            "id": "B",
            "active": False,
            "cancelled": False,
            "minutes": 90,
            "free_minutes": 0,
            "workers": 4,
            "rate": 5,
            "priority": "normal",
        },
        {
            "id": "C",
            "active": True,
            "cancelled": False,
            "minutes": 31,
            "free_minutes": 1,
            "workers": 2,
            "rate": 4,
            "priority": "emergency",
        },
        {
            "id": "D",
            "active": True,
            "cancelled": True,
            "minutes": 60,
            "free_minutes": 0,
            "workers": 3,
            "rate": 2,
            "priority": "normal",
        },
        {
            "id": "E",
            "active": True,
            "cancelled": False,
            "minutes": 5,
            "free_minutes": 5,
            "workers": 8,
            "rate": 10,
            "priority": "normal",
        },
    ]
    return evaluate_corrected_tasks(tasks)


def derive_answers() -> dict[str, dict[str, Any]]:
    """Return answers independently derived from the benchmark rules."""
    return {
        "pipeline_simple": solve_pipeline_simple(),
        "pipeline_moderate": solve_pipeline_moderate(),
        "pipeline_complex": solve_pipeline_complex(),
        "policy": solve_policy(),
        "extraction": solve_extraction(),
        "debugging": solve_debugging(),
    }


def verify_answer_key() -> dict[str, dict[str, Any]]:
    """Verify the saved key and return the independently derived answers."""
    answer_key = json.loads(ANSWER_KEY_PATH.read_text(encoding="utf-8"))
    expected_answers = answer_key["answers"]
    derived_answers = derive_answers()

    if derived_answers != expected_answers:
        raise SystemExit(
            "Answer key does not match the deterministic reference solver:\n"
            f"derived: {json.dumps(derived_answers, indent=2)}\n"
            f"answer key: {json.dumps(expected_answers, indent=2)}"
        )

    return derived_answers


def main() -> None:
    derived_answers = verify_answer_key()
    print(
        f"Verified {len(derived_answers)} answer-key entries with deterministic reference solvers."
    )


if __name__ == "__main__":
    main()
