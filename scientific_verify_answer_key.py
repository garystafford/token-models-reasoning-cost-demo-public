#!/usr/bin/env python3
"""Independently derive and verify the scientific benchmark answer key."""

from __future__ import annotations

import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


ANSWER_KEY_PATH = Path(__file__).with_name(
    "scientific_reasoning_benchmark_expected_answers.json"
)


def solve_chemistry_mixture() -> dict[str, Any]:
    first_volume_ml = 200
    first_concentration = Fraction(12)
    second_volume_ml = 300
    second_concentration = Fraction(4)
    blank_bias = Fraction(1, 5)
    mixed = (
        first_volume_ml * first_concentration + second_volume_ml * second_concentration
    ) / (first_volume_ml + second_volume_ml)
    corrected = mixed - blank_bias
    return {
        "corrected_concentration_mg_per_l": float(corrected),
        "contamination_flag": corrected > 7,
    }


def solve_genetics_cross() -> dict[str, Any]:
    allele_combinations = ("AA", "Aa", "aA", "aa")
    recessive_count = sum(genotype == "aa" for genotype in allele_combinations)
    probability = Fraction(recessive_count, len(allele_combinations))
    expected_offspring = probability * 128
    return {
        "recessive_probability": float(probability),
        "expected_recessive_offspring": int(expected_offspring),
        "follow_up_required": expected_offspring >= 30,
    }


def solve_ecology_stratified() -> dict[str, Any]:
    strata = (
        (Fraction(40, 100), Fraction(85, 10)),
        (Fraction(35, 100), Fraction(11)),
        (Fraction(25, 100), Fraction(6)),
    )
    weighted_density = sum(weight * density for weight, density in strata)
    estimated_population = weighted_density * 240
    return {
        "weighted_density_per_ha": float(weighted_density),
        "estimated_population": int(estimated_population),
        "management_review": (
            estimated_population < 2_000
            or any(density < Fraction(65, 10) for _, density in strata)
        ),
    }


def minutes(time_text: str) -> int:
    """Convert an HH:MM time to minutes after midnight."""
    hours, minute = (int(part) for part in time_text.split(":"))
    return hours * 60 + minute


def solve_astronomy_transit() -> dict[str, Any]:
    predicted_midpoint = minutes("03:00")
    observations = (
        ("OBS-41", "K-17", "02:34", "03:14", 50_000, 49_100),
        ("OBS-44", "K-17", "02:46", "03:20", 48_000, 47_040),
        ("OBS-47", "M-22", "02:40", "03:18", 60_000, 58_200),
    )
    candidates = []
    for observation_id, target, start, end, baseline, minimum in observations:
        start_minute = minutes(start)
        end_minute = minutes(end)
        midpoint = Fraction(start_minute + end_minute, 2)
        depth = Fraction(baseline - minimum, baseline) * 100
        if (
            target == "K-17"
            and abs(midpoint - predicted_midpoint) <= 12
            and depth >= Fraction(3, 2)
        ):
            candidates.append(
                (
                    abs(midpoint - predicted_midpoint),
                    -depth,
                    observation_id,
                    start_minute,
                    end_minute,
                    baseline,
                    depth,
                )
            )

    _, _, selected, start, end, baseline, depth = min(candidates)
    follow_up_samples = (
        (minutes("03:25"), 47_900),
        (minutes("03:34"), 47_700),
        (minutes("03:42"), 47_950),
    )
    in_window = [
        flux for sampled_at, flux in follow_up_samples if end <= sampled_at <= end + 20
    ]
    recovered = all(
        Fraction(abs(flux - baseline), baseline) * 100 <= Fraction(1, 2)
        for flux in in_window
    )
    return {
        "selected_observation": selected,
        "duration_minutes": end - start,
        "flux_depth_pct": float(depth),
        "recovered": recovered,
    }


def solve_lab_allocation() -> dict[str, Any]:
    candidates: list[tuple[Fraction, int, int, int]] = []
    for spectroscopy in range(2, 9):
        for microscopy in range(2, 9):
            sequencing = 12 - spectroscopy - microscopy
            if sequencing < 3:
                continue
            variance = (
                Fraction(120, spectroscopy)
                + Fraction(90, microscopy)
                + Fraction(60, sequencing)
            )
            candidates.append((variance, -sequencing, -microscopy, spectroscopy))

    variance, negative_sequencing, negative_microscopy, spectroscopy = min(candidates)
    return {
        "spectroscopy_blocks": spectroscopy,
        "microscopy_blocks": -negative_microscopy,
        "sequencing_blocks": -negative_sequencing,
        "variance_proxy": float(variance),
    }


def solve_vessel_schedule() -> dict[str, Any]:
    sites = {
        "A": {"duration": 50, "window": (0, 180), "priority": 7},
        "B": {"duration": 60, "window": (90, 300), "priority": 9},
        "C": {"duration": 70, "window": (150, 390), "priority": 12},
        "D": {"duration": 45, "window": (0, 240), "priority": 6},
    }
    travel = {
        frozenset(("BASE", "A")): 30,
        frozenset(("BASE", "B")): 45,
        frozenset(("BASE", "C")): 60,
        frozenset(("BASE", "D")): 50,
        frozenset(("A", "B")): 25,
        frozenset(("A", "C")): 40,
        frozenset(("A", "D")): 35,
        frozenset(("B", "C")): 30,
        frozenset(("B", "D")): 20,
        frozenset(("C", "D")): 25,
    }
    candidates: list[tuple[int, int, tuple[str, ...]]] = []
    for count in range(1, len(sites) + 1):
        for route in itertools.permutations(sites, count):
            if "C" not in route:
                continue
            clock = 0
            location = "BASE"
            valid = True
            for site in route:
                clock += travel[frozenset((location, site))]
                window_start, window_end = sites[site]["window"]
                clock = max(clock, window_start)
                clock += sites[site]["duration"]
                if clock > window_end:
                    valid = False
                    break
                location = site
            if not valid:
                continue
            clock += travel[frozenset((location, "BASE"))]
            if clock > 360:
                continue
            priority = sum(sites[site]["priority"] for site in route)
            candidates.append((-priority, clock, route))

    negative_priority, return_minutes, route = min(candidates)
    absolute_return = minutes("06:00") + return_minutes
    return {
        "site_order": list(route),
        "total_priority": -negative_priority,
        "return_time_utc": f"{absolute_return // 60:02d}:{absolute_return % 60:02d}",
    }


def derive_answers() -> dict[str, dict[str, Any]]:
    """Return answers independently derived from the scientific rules."""
    return {
        "chemistry_mixture": solve_chemistry_mixture(),
        "genetics_cross": solve_genetics_cross(),
        "ecology_stratified": solve_ecology_stratified(),
        "astronomy_transit": solve_astronomy_transit(),
        "lab_allocation": solve_lab_allocation(),
        "vessel_schedule": solve_vessel_schedule(),
    }


def verify_answer_key() -> dict[str, dict[str, Any]]:
    """Verify the saved key and return independently derived answers."""
    answer_key = json.loads(ANSWER_KEY_PATH.read_text(encoding="utf-8"))
    expected_answers = answer_key["answers"]
    derived_answers = derive_answers()
    if derived_answers != expected_answers:
        raise SystemExit(
            "Scientific answer key does not match the deterministic solver:\n"
            f"derived: {json.dumps(derived_answers, indent=2)}\n"
            f"answer key: {json.dumps(expected_answers, indent=2)}"
        )
    return derived_answers


def main() -> None:
    derived_answers = verify_answer_key()
    print(
        f"Verified {len(derived_answers)} scientific answer-key entries "
        "with deterministic reference solvers."
    )


if __name__ == "__main__":
    main()
