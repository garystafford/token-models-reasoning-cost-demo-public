#!/usr/bin/env python3
"""Independently derive and verify the Resilience answer key."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

ANSWER_KEY_PATH = Path(__file__).with_name("expected_answers.json")


def solve_access_policy() -> dict[str, Any]:
    role_permissions = {
        "curator": {"read:atlas", "read:birch", "read:delta", "export:reports"},
        "auditor": {"read:birch", "read:cedar", "query:audit"},
        "responder": {"read:cedar", "read:incident", "query:audit"},
    }
    permissions = set().union(
        *(role_permissions[role] for role in ("curator", "auditor"))
    )
    permissions.add("read:incident")
    permissions.difference_update({"read:birch", "export:reports"})
    return {
        "effective_read_resources": sorted(
            permission.removeprefix("read:")
            for permission in permissions
            if permission.startswith("read:")
        ),
        "can_export_reports": "export:reports" in permissions,
        "can_query_audit": "query:audit" in permissions,
    }


def solve_integrity_checksum() -> dict[str, Any]:
    records = {
        "ARC-57243-7": (57243, 7),
        "ARC-18426-3": (18426, 3),
        "ARC-93017-6": (93017, 6),
    }

    def check_digit(payload: int) -> int:
        digits = [int(character) for character in f"{payload:05d}"]
        return (
            sum(weight * digit for weight, digit in zip((7, 3, 9, 5, 1), digits)) % 10
        )

    valid = sorted(
        record_id
        for record_id, (payload, declared_check_digit) in records.items()
        if check_digit(payload) == declared_check_digit
    )
    return {
        "valid_record_ids": valid,
        "valid_payload_sum": sum(records[record_id][0] for record_id in valid),
    }


def solve_audit_log_replay() -> dict[str, Any]:
    balances = {"red": 10, "blue": 7}
    capacity = 14
    events = (
        ("E-10", "09:00", "allocate", "red", 4),
        ("E-12", "09:05", "release", "blue", 2),
        ("E-11", "09:05", "move", "red", "blue", 5),
        ("E-13", "09:10", "allocate", "blue", 15),
        ("E-14", "09:15", "release", "red", 1),
        ("E-15", "09:20", "move", "blue", "red", 8),
    )
    rejected = []
    for event in sorted(events, key=lambda item: (item[1], item[0])):
        event_id, _, action, *details = event
        if action == "allocate":
            pool, amount = details
            if balances[pool] < amount:
                rejected.append(event_id)
            else:
                balances[pool] -= amount
        elif action == "release":
            pool, amount = details
            if balances[pool] + amount > capacity:
                rejected.append(event_id)
            else:
                balances[pool] += amount
        else:
            source, destination, amount = details
            if balances[source] < amount or balances[destination] + amount > capacity:
                rejected.append(event_id)
            else:
                balances[source] -= amount
                balances[destination] += amount
    return {
        "final_balances": dict(sorted(balances.items())),
        "rejected_event_ids": rejected,
    }


def solve_version_resolution() -> dict[str, Any]:
    engines = {
        "E4": {"api": 4, "features": {"core", "trace"}, "risk": 6},
        "E5": {"api": 5, "features": {"core", "streaming"}, "risk": 8},
        "E6": {"api": 6, "features": {"core", "streaming", "trace"}, "risk": 11},
    }
    parsers = {
        "P4": {"apis": {4}, "formats": {"CSV"}, "risk": 3},
        "P5": {"apis": {5, 6}, "formats": {"CSV", "JSON"}, "risk": 5},
        "P6": {"apis": {6}, "formats": {"CSV", "JSON", "XML"}, "risk": 7},
    }
    consoles = {
        "C1": {"formats": {"CSV"}, "features": set(), "risk": 2},
        "C2": {"formats": {"JSON"}, "features": {"streaming"}, "risk": 4},
        "C3": {"formats": {"XML"}, "features": {"trace"}, "risk": 5},
    }
    candidates = []
    for engine_id, parser_id, console_id in itertools.product(
        engines, parsers, consoles
    ):
        engine, parser, console = (
            engines[engine_id],
            parsers[parser_id],
            consoles[console_id],
        )
        if engine["api"] not in parser["apis"]:
            continue
        if not console["formats"] <= parser["formats"]:
            continue
        if not console["features"] <= engine["features"]:
            continue
        if not {"CSV", "JSON"} <= parser["formats"]:
            continue
        if not {"streaming", "trace"} <= engine["features"]:
            continue
        risk = engine["risk"] + parser["risk"] + console["risk"]
        candidates.append((risk, engine_id, parser_id, console_id))
    risk, engine_id, parser_id, console_id = min(candidates)
    return {
        "engine_release": engine_id,
        "parser_release": parser_id,
        "console_release": console_id,
        "total_risk_score": risk,
    }


def solve_replica_placement() -> dict[str, Any]:
    sites = {
        "A": ("North", 7, 2, 5, False),
        "B": ("North", 9, 1, 8, True),
        "C": ("South", 8, 2, 6, False),
        "D": ("South", 10, 1, 9, True),
        "E": ("East", 7, 1, 7, True),
        "F": ("East", 11, 3, 6, False),
        "G": ("West", 8, 2, 5, False),
        "H": ("West", 9, 1, 8, True),
    }
    candidates = []
    for selection in itertools.combinations(sites, 4):
        values = [sites[site_id] for site_id in selection]
        if {value[0] for value in values} != {"North", "South", "East", "West"}:
            continue
        if not any(value[4] for value in values):
            continue
        total_cost = sum(value[3] for value in values)
        if total_cost > 26:
            continue
        score = sum(value[1] for value in values) - 2 * sum(
            value[2] for value in values
        )
        candidates.append((-score, total_cost, selection))
    negative_score, total_cost, selection = min(candidates)
    return {
        "site_ids": list(selection),
        "resilience_score": -negative_score,
        "total_cost_units": total_cost,
    }


def solve_recovery_schedule() -> dict[str, Any]:
    tasks = {
        "Identity": ("Platform", 40, (), 0),
        "Ledger": ("Platform", 60, ("Identity",), 0),
        "Search": ("Applications", 50, ("Identity",), 4),
        "Portal": ("Applications", 60, ("Search",), 11),
        "Media": ("Applications", 60, ("Identity",), 10),
        "Analytics": ("Applications", 40, ("Search",), 7),
        "Audit": ("Platform", 40, ("Ledger", "Analytics"), 8),
    }
    mandatory = {"Identity", "Ledger"}
    task_ids = tuple(tasks)
    feasible = []
    for count in range(len(mandatory), len(tasks) + 1):
        for selected_tuple in itertools.combinations(task_ids, count):
            selected = set(selected_tuple)
            if not mandatory <= selected:
                continue
            if any(not set(tasks[task_id][2]) <= selected for task_id in selected):
                continue
            crew_tasks = {
                crew: [task_id for task_id in selected if tasks[task_id][0] == crew]
                for crew in ("Platform", "Applications")
            }
            for platform_order in itertools.permutations(crew_tasks["Platform"]):
                for application_order in itertools.permutations(
                    crew_tasks["Applications"]
                ):
                    orders = {
                        "Platform": platform_order,
                        "Applications": application_order,
                    }
                    starts: dict[str, int] = {}
                    finishes: dict[str, int] = {}
                    cursors = {"Platform": 0, "Applications": 0}
                    pending = {task_id for task_id in selected}
                    while pending:
                        ready = []
                        for crew, order in orders.items():
                            position = sum(task_id in finishes for task_id in order)
                            if position >= len(order):
                                continue
                            task_id = order[position]
                            dependencies = tasks[task_id][2]
                            if all(
                                dependency in finishes for dependency in dependencies
                            ):
                                ready.append((crew, task_id))
                        if not ready:
                            break
                        for crew, task_id in ready:
                            start = max(
                                cursors[crew],
                                *(
                                    finishes[dependency]
                                    for dependency in tasks[task_id][2]
                                ),
                                0,
                            )
                            starts[task_id] = start
                            finishes[task_id] = start + tasks[task_id][1]
                            cursors[crew] = finishes[task_id]
                            pending.remove(task_id)
                    if pending or max(finishes.values()) > 180:
                        continue
                    priority = sum(tasks[task_id][3] for task_id in selected)
                    feasible.append(
                        (-priority, max(finishes.values()), tuple(sorted(selected)))
                    )
    negative_priority, completion, selected = min(feasible)
    return {
        "recovered_task_ids": list(selected),
        "total_priority_points": -negative_priority,
        "completion_time_utc": f"{8 + completion // 60:02d}:{completion % 60:02d}",
    }


def derive_answers() -> dict[str, dict[str, Any]]:
    """Return answers independently derived from the suite rules."""
    return {
        "access_policy": solve_access_policy(),
        "integrity_checksum": solve_integrity_checksum(),
        "audit_log_replay": solve_audit_log_replay(),
        "version_resolution": solve_version_resolution(),
        "replica_placement": solve_replica_placement(),
        "recovery_schedule": solve_recovery_schedule(),
    }


def verify_answer_key() -> dict[str, dict[str, Any]]:
    """Verify the saved key and return independently derived answers."""
    answer_key = json.loads(ANSWER_KEY_PATH.read_text(encoding="utf-8"))
    derived_answers = derive_answers()
    if derived_answers != answer_key["answers"]:
        raise SystemExit(
            "Resilience answer key does not match the deterministic solver:\n"
            f"derived: {json.dumps(derived_answers, indent=2)}\n"
            f"answer key: {json.dumps(answer_key['answers'], indent=2)}"
        )
    return derived_answers


def main() -> None:
    derived_answers = verify_answer_key()
    print(
        f"Verified {len(derived_answers)} Resilience answer-key entries with deterministic reference solvers."
    )


if __name__ == "__main__":
    main()
