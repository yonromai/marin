# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from experiments.post_training.analyze_current_score_centering_confirmation import MAIN_ARMS, analyze_confirmation


def experiment():
    protocol = {
        "status": "frozen",
        "seeds": [1, 2, 3],
        "evaluation_steps": [0, 40],
        "primary_members": 756,
        "primary_membership_sha256": "same-members",
        "source_commit": "same-source",
    }
    runs = []
    for arm in MAIN_ARMS:
        for seed in protocol["seeds"]:
            rows = []
            for step in protocol["evaluation_steps"]:
                for name in [None, "greedy_repeat"]:
                    # The repeat adds two answers, averaged within seed. SC final
                    # differences are 10,20,30; initial differences are all five.
                    extra = (10 * seed if step else 5) if arm == "older_sc32" else 0
                    rows.append(
                        {
                            "scope": "primary",
                            "step": step,
                            "evaluation_name": name,
                            "members": 756,
                            "completed_correct": 100 + extra + (2 if name else 0),
                        }
                    )
            run_id = f"{arm}-{seed}"
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "source_commit": "same-source",
                    "scientific_work_complete": True,
                    "run_id": run_id,
                    "reserved_h100_task_hours": 10 if arm == "older_sc32" else 20,
                    "allocated_wall_seconds": 100 if arm == "older_sc32" else 200,
                    "endpoint": {"run_id": run_id, "primary_membership_sha256": "same-members", "evaluations": rows},
                }
            )
    protocol["capture_cost_companion_seeds"] = [1, 2, 3]
    for seed in protocol["seeds"]:
        run = copy.deepcopy(next(row for row in runs if row["arm"] == "older_tis32" and row["seed"] == seed))
        run.update(
            arm="older_tis0", run_id=f"older_tis0-{seed}", reserved_h100_task_hours=10, allocated_wall_seconds=100
        )
        run["endpoint"]["run_id"] = run["run_id"]
        runs.append(run)
    protocol["configuration_manifest"] = []
    for run in runs:
        run["configuration_sha256"] = run["run_id"] + "-hash"
        protocol["configuration_manifest"].append(
            {"arm": run["arm"], "seed": run["seed"], "run_id": run["run_id"], "sha256": run["configuration_sha256"]}
        )
    return runs, protocol


def test_repeats_are_averaged_before_paired_seed_uncertainty():
    runs, protocol = experiment()
    result = analyze_confirmation(runs, protocol)
    contrast = result["contrasts"][0]
    ci = contrast["individual_95_percent"]
    assert contrast["paired_final_completed_answer_differences"] == [10, 20, 30]
    assert ci["n_seeds"] == 3
    assert ci["mean"] == 20
    assert ci["sample_sd"] == 10
    # Hand Student t: df2 critical4.30265273 * 10/sqrt3.
    assert ci["low"] == pytest.approx(-4.841377, abs=1e-5)
    assert ci["high"] == pytest.approx(44.841377, abs=1e-5)
    assert contrast["baseline_adjusted_sensitivity_95_percent"]["mean"] == 15
    assert contrast["allocated_elapsed_ratio_individual_95_percent"]["mean"] == 0.5
    assert contrast["three_contrast_family_95_percent"]["high"] > ci["high"]
    assert result["capture_cost_companion"]["captured_over_plain_compute_ratio_95_percent"]["mean"] == 2
    assert result["capture_cost_companion"]["final_quality_sensitivity_95_percent"]["mean"] == 0


@pytest.mark.parametrize(
    "changed", ["missing", "duplicate", "source", "membership", "repeat", "unfinished", "run_identity", "config"]
)
def test_refuses_incomplete_or_changed_confirmation(changed):
    runs, protocol = copy.deepcopy(experiment())
    if changed == "missing":
        runs.pop()
    elif changed == "duplicate":
        runs.append(runs[0])
    elif changed == "source":
        runs[0]["source_commit"] = "other"
    elif changed == "membership":
        runs[0]["endpoint"]["primary_membership_sha256"] = "other"
    elif changed == "repeat":
        runs[0]["endpoint"]["evaluations"].pop()
    elif changed == "unfinished":
        runs[0]["scientific_work_complete"] = False
    elif changed == "config":
        runs[0]["configuration_sha256"] = "other"
    elif changed == "run_identity":
        runs[0]["endpoint"]["run_id"] = "other"
    with pytest.raises(ValueError):
        analyze_confirmation(runs, protocol)
