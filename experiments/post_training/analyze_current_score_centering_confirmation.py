# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predeclared paired-seed analysis for the current Snowball confirmation.

Inputs are audited endpoint summaries and terminal reserved-allocation costs.
Evaluation repeats share a training seed and are averaged before inference.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t

MAIN_ARMS = ("older_tis32", "older_sc32", "fresh_tis32", "older_incumbent")
CONTRASTS = (
    ("older_sc32", "older_tis32", "centering at equal scheduling and capture"),
    ("older_sc32", "fresh_tis32", "centering with older rollouts versus fresh TIS"),
    ("older_sc32", "older_incumbent", "centering versus current merged incumbent"),
)


def paired_interval(values: list[float], alpha: float = 0.05) -> dict:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("paired uncertainty requires at least two finite training-seed differences")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = float(t.ppf(1 - alpha / 2, len(values) - 1)) * sd / math.sqrt(len(values))
    return {"n_seeds": len(values), "mean": mean, "sample_sd": sd, "low": mean - half, "high": mean + half}


def ratio_interval(log_ratios: list[float]) -> dict:
    """Exponentiate paired log-ratio means and bounds; sample SD stays in log units."""
    return {
        key: math.exp(value) if key in {"mean", "low", "high"} else value
        for key, value in paired_interval(log_ratios).items()
    }


def primary_curve(endpoint: dict, protocol: dict) -> dict[int, float]:
    if endpoint["primary_membership_sha256"] != protocol["primary_membership_sha256"]:
        raise ValueError("confirmation endpoint changed frozen primary membership")
    rows = [row for row in endpoint["evaluations"] if row["scope"] == "primary"]
    expected = {(step, name) for step in protocol["evaluation_steps"] for name in (None, "greedy_repeat")}
    if {(row["step"], row["evaluation_name"]) for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("confirmation endpoint is missing or repeats a declared evaluation")
    curve = {}
    for step in protocol["evaluation_steps"]:
        members = [row for row in rows if row["step"] == step]
        if any(row["members"] != protocol["primary_members"] for row in members):
            raise ValueError("confirmation endpoint has a different primary denominator")
        values = [row["completed_correct"] for row in members]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= protocol["primary_members"]
            for value in values
        ):
            raise ValueError("completed-answer endpoint is not a valid count")
        curve[step] = statistics.mean(values)
    return curve


def analyze_confirmation(runs: list[dict], protocol: dict) -> dict:
    """Require the full frozen design before producing confirmation conclusions."""
    if protocol["status"] != "frozen":
        raise ValueError("confirmation requires a frozen protocol")
    seeds = protocol["seeds"]
    expected = {(arm, seed) for arm in MAIN_ARMS for seed in seeds}
    main = [run for run in runs if run["arm"] in MAIN_ARMS]
    if {(run["arm"], run["seed"]) for run in main} != expected or len(main) != len(expected):
        raise ValueError("confirmation is missing or repeats a frozen arm/seed")
    companion_seeds = protocol.get("capture_cost_companion_seeds", [])
    companions = [run for run in runs if run["arm"] == "older_tis0"]
    if {run["seed"] for run in companions} != set(companion_seeds) or len(companions) != len(companion_seeds):
        raise ValueError("capture-cost companion is missing or repeats a declared seed")
    declared = {(row["arm"], row["seed"]): row for row in protocol["configuration_manifest"]}
    indexed = {}
    for run in main + companions:
        declaration = declared[(run["arm"], run["seed"])]
        if run["run_id"] != declaration["run_id"] or run["configuration_sha256"] != declaration["sha256"]:
            raise ValueError("confirmation run differs from its frozen input configuration")
        if run["source_commit"] != protocol["source_commit"]:
            raise ValueError("confirmation pooled different source semantics")
        if not run["scientific_work_complete"]:
            raise ValueError("confirmation includes incomplete scientific work")
        cost = run["reserved_h100_task_hours"]
        elapsed = run["allocated_wall_seconds"]
        if any(not math.isfinite(x) or x <= 0 for x in (cost, elapsed)):
            raise ValueError("confirmation requires finite positive terminal costs and elapsed times")
        endpoint = run["endpoint"]
        if endpoint["run_id"] != run["run_id"]:
            raise ValueError("confirmation endpoint belongs to a different run")
        indexed[(run["arm"], run["seed"])] = {**run, "curve": primary_curve(endpoint, protocol)}
    final_step = protocol["evaluation_steps"][-1]
    contrasts = []
    for target, control, meaning in CONTRASTS:
        differences, adjusted, log_elapsed, log_cost = [], [], [], []
        for seed in seeds:
            a, b = indexed[(target, seed)], indexed[(control, seed)]
            delta = a["curve"][final_step] - b["curve"][final_step]
            differences.append(delta)
            adjusted.append(delta - (a["curve"][0] - b["curve"][0]))
            log_elapsed.append(math.log(a["allocated_wall_seconds"] / b["allocated_wall_seconds"]))
            log_cost.append(math.log(a["reserved_h100_task_hours"] / b["reserved_h100_task_hours"]))
        contrasts.append(
            {
                "target": target,
                "control": control,
                "meaning": meaning,
                "paired_final_completed_answer_differences": differences,
                "individual_95_percent": paired_interval(differences),
                "three_contrast_family_95_percent": paired_interval(differences, alpha=0.05 / 3),
                "baseline_adjusted_sensitivity_95_percent": paired_interval(adjusted),
                "allocated_elapsed_ratio_individual_95_percent": ratio_interval(log_elapsed),
                "reserved_compute_ratio_individual_95_percent": ratio_interval(log_cost),
            }
        )
    capture_cost = None
    if companion_seeds:
        elapsed, compute, quality = [], [], []
        for seed in companion_seeds:
            captured, plain = indexed[("older_tis32", seed)], indexed[("older_tis0", seed)]
            elapsed.append(math.log(captured["allocated_wall_seconds"] / plain["allocated_wall_seconds"]))
            compute.append(math.log(captured["reserved_h100_task_hours"] / plain["reserved_h100_task_hours"]))
            quality.append(captured["curve"][final_step] - plain["curve"][final_step])
        capture_cost = {
            "scope": (
                "Secondary paired capture/rescoring cost comparison; async generation interactions remain measured."
            ),
            "seeds": companion_seeds,
            "captured_over_plain_elapsed_ratio_95_percent": ratio_interval(elapsed),
            "captured_over_plain_compute_ratio_95_percent": ratio_interval(compute),
            "final_quality_sensitivity_95_percent": paired_interval(quality),
            "reserved_h100_task_hours": sum(run["reserved_h100_task_hours"] for run in companions),
        }
    arms = []
    for arm in MAIN_ARMS:
        selected = [indexed[(arm, seed)] for seed in seeds]
        arms.append(
            {
                "arm": arm,
                "mean_completed_correct_answers_by_step": {
                    step: statistics.mean(run["curve"][step] for run in selected)
                    for step in protocol["evaluation_steps"]
                },
                "mean_reserved_h100_task_hours": statistics.mean(run["reserved_h100_task_hours"] for run in selected),
                "mean_allocated_wall_seconds": statistics.mean(run["allocated_wall_seconds"] for run in selected),
            }
        )
    return {
        "scope": "Frozen primary final endpoint; training seed is the uncertainty unit.",
        "secondary": "Curves, baseline adjustment, time and compute.",
        "quality_loss_margin": None,
        "equivalence_claim": False,
        "primary_members": protocol["primary_members"],
        "seeds": seeds,
        "final_training_step": final_step,
        "contrasts": contrasts,
        "arms": arms,
        "capture_cost_companion": capture_cost,
        "total_main_reserved_h100_task_hours": sum(run["reserved_h100_task_hours"] for run in main),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_confirmation(json.loads(args.runs.read_text()), json.loads(args.protocol.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["contrasts"], indent=2))


if __name__ == "__main__":
    main()
