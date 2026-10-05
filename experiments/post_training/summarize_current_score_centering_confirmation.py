# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize predeclared exposure, timing and reliability sensitivities.

Requires the entire frozen design and original primary analysis code. These
descriptive summaries do not change the primary endpoint or its uncertainty.
Core time excludes step-end callbacks; cycle time includes their evaluations,
checkpoint work and bookkeeping. Allocated time also includes startup and teardown.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from experiments.post_training.analyze_current_score_centering_confirmation import (
    CONTRASTS,
    analyze_confirmation,
)


def exposure_overlap(target: Counter, control: Counter) -> dict:
    """Compare actual consumed prompt multiplicities, counting responses."""
    union = sum((target | control).values())
    shared = sum((target & control).values())
    return {
        "target_distinct_prompts": len(target),
        "control_distinct_prompts": len(control),
        "shared_distinct_prompts": len(target.keys() & control.keys()),
        "shared_response_assignments": shared,
        "response_assignment_multiset_jaccard": shared / union if union else None,
    }


def summarize_run(run: dict, base: Path, final_step: int) -> tuple[dict, list[dict]]:
    directory = (base / run["endpoint_summary"]).parent
    training = json.loads((directory / "training_records.json").read_text())
    ages = json.loads((directory / "ages.json").read_text())["aggregate"]
    raw = (base / run["source_metrics"]["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != run["source_metrics"]["sha256"]:
        raise ValueError("source metrics differ from their terminal audit")
    train = [row for row in json.loads(gzip.decompress(raw)) if row["kind"] == "train"]
    if sorted(row["step"] for row in train) != list(range(1, final_step + 1)):
        raise ValueError("secondary summaries require each completed training step exactly once")
    if training["run_id"] != run["run_id"] or not training["complete_consumed_training_evidence"]:
        raise ValueError("secondary training evidence belongs to a different or incomplete run")
    tokens = training["consumed_loss_tokens"]
    if tokens != ages["loss_tokens"] or tokens != sum(row["async/performance/consumed_loss_tokens"] for row in train):
        raise ValueError("consumed token counts disagree across independent audits and source metrics")
    sequences = sum(row["consumed/sequences"] for row in train)
    if sequences != training["consumed_records"]:
        raise ValueError("consumed response counts disagree with exact training records")
    known = sum(row["consumed/known_stop_count"] for row in train)
    length_stops = sum(row["consumed/length_stop_count"] for row in train)
    core = sum(row["async/performance/core_seconds"] for row in train)
    cycle = sum(row["async/performance/cycle_seconds"] for row in train)
    if core <= 0 or cycle < core:
        raise ValueError("core and cycle timing are inconsistent")
    retained = training["retained_records"]
    summary = {
        "arm": run["arm"],
        "seed": run["seed"],
        "run_id": run["run_id"],
        "loss_tokens": tokens,
        "consumed_responses": sequences,
        "consumed_response_tokens_mean": training["consumed_response_tokens"] / sequences,
        "consumed_stop_reason_coverage": known / sequences,
        "consumed_length_stop_fraction": length_stops / sequences if known == sequences else None,
        "retained_responses": retained,
        "retained_normal_completion_fraction": training["retained_normal_completions"] / retained,
        "finished_unconsumed_responses": training["completed_unconsumed_records"],
        "finished_unconsumed_response_tokens": training["completed_unconsumed_response_tokens"],
        "rejected_groups": sum(row["async/rejected_count"] for row in train),
        "rejected_stale_groups": sum(row["async/rejected_count/stale"] for row in train),
        "actual_consumed_optimizer_age": ages,
        "core_seconds_excluding_step_end_callbacks": core,
        "cycle_seconds_including_step_end_callbacks": cycle,
        "step_end_callback_seconds": cycle - core,
        "allocated_seconds_outside_step_cycles": run["allocated_wall_seconds"] - cycle,
        "summed_step_phase_seconds": {
            phase: sum(row[f"timing/step_wall/{phase}"] for row in train)
            for phase in ("evaluation", "checkpoint_work", "group_admission", "policy_training", "weight_sync")
        },
        "loss_tokens_per_core_second": tokens / core,
        "loss_tokens_per_cycle_second": tokens / cycle,
        "allocated_wall_seconds": run["allocated_wall_seconds"],
        "reserved_h100_task_hours": run["reserved_h100_task_hours"],
        "minimum_raw_grad_norm": min(row["policy/raw_grad_norm"] for row in train),
        "maximum_raw_grad_norm": max(row["policy/raw_grad_norm"] for row in train),
        "source_metrics_sha256": run["source_metrics"]["sha256"],
        "consumed_evidence": str(directory.relative_to(base) / "training_records.json"),
        "diagnostics_by_step": [
            {
                "step": row["step"],
                **{
                    key: value
                    for key, value in row.items()
                    if key.startswith(("policy/score_centering/", "policy/correction/", "policy/mismatch/"))
                    or key in ("policy/ppo_clip_ratio", "timing/sync_weights", "timing/fwd_logprobs_values_reward")
                },
            }
            for row in train
        ],
    }
    return summary, training["steps"]


def summarize_confirmation(runs: list[dict], protocol: dict, base: Path, costs: dict) -> dict:
    analyze_confirmation(runs, protocol)
    if len(runs) != len(protocol["configuration_manifest"]):
        raise ValueError("secondary summaries received undeclared extra runs")
    cost_by_job = {row["job_id"]: row for row in costs["attempts"]}
    if len(cost_by_job) != len(costs["attempts"]):
        raise ValueError("cost ledger repeats an allocation")
    for run in runs:
        if run["job_id"] not in cost_by_job or (
            cost_by_job[run["job_id"]]["reserved_h100_task_hours"] != run["reserved_h100_task_hours"]
        ):
            raise ValueError("cost ledger lacks or changes a successful confirmation allocation")
    summaries, steps = [], {}
    for run in runs:
        summary, consumed = summarize_run(run, base, protocol["completed_training_steps"])
        summaries.append(summary)
        steps[(run["arm"], run["seed"])] = consumed
    comparisons = []
    for target, control, _ in CONTRASTS:
        for seed in protocol["seeds"]:
            for step in protocol["evaluation_steps"][1:]:
                assignments = []
                for arm in (target, control):
                    counts = Counter()
                    for row in steps[(arm, seed)]:
                        if row["training_step"] <= step:
                            counts.update(row["uid_response_counts"])
                    assignments.append(counts)
                comparisons.append(
                    {"target": target, "control": control, "seed": seed, "step": step, **exposure_overlap(*assignments)}
                )

    failures = []
    for attempt in costs["attempts"]:
        if not attempt["result"].startswith("score_centering_current_confirmation_failures/"):
            continue
        failure = json.loads((base / "results" / attempt["result"]).read_text())
        if failure["scientific_work_complete"] or failure["source_commit"] != protocol["source_commit"]:
            raise ValueError("excluded confirmation failure has inconsistent scientific or source identity")
        if (
            failure["job_id"] != attempt["job_id"]
            or failure["reserved_h100_task_hours"] != attempt["reserved_h100_task_hours"]
        ):
            raise ValueError("failed confirmation allocation differs from the cost ledger")
        failures.append({key: failure[key] for key in ("arm", "seed", "run_id", "reserved_h100_task_hours")})
    operational_costs = []
    completions = [
        {
            **{key: run[key] for key in ("arm", "seed", "run_id", "reserved_h100_task_hours")},
            "operational_completion": run["operational_completion"],
        }
        for run in runs
        if "operational_completion" in run
    ]
    for arm in sorted({run["arm"] for run in runs}):
        selected = [run for run in runs if run["arm"] == arm]
        failed = [row for row in failures if row["arm"] == arm]
        success_hours = sum(run["reserved_h100_task_hours"] for run in selected)
        failure_hours = sum(row["reserved_h100_task_hours"] for row in failed)
        operational_costs.append(
            {
                "arm": arm,
                "successful_runs": len(selected),
                "scientifically_completed_runs_with_worker_failures": sum(
                    "operational_completion" in run for run in selected
                ),
                "excluded_failed_attempts": len(failed),
                "successful_reserved_h100_task_hours": success_hours,
                "failed_reserved_h100_task_hours": failure_hours,
                "total_reserved_h100_task_hours": success_hours + failure_hours,
                "failed_cost_fraction": failure_hours / (success_hours + failure_hours),
                "mean_successful_allocated_wall_seconds": statistics.mean(
                    run["allocated_wall_seconds"] for run in selected
                ),
            }
        )
    return {
        "scope": "Predeclared descriptive secondary measures; primary inference is unchanged.",
        "timing_scope": (
            "Core excludes step-end callbacks; cycle includes their evaluation, checkpoint and bookkeeping work. "
            "Allocation additionally includes startup, initial evaluations and teardown."
        ),
        "observation_limit": (
            "Finished unused responses are observed exactly. Tokens in unfinished or cancelled generation are "
            "not fully observed; their allocation remains counted. Retained completion includes unused responses. "
            "Per-step diagnostics retain their original reduction and quantiles, without pooling quantiles."
        ),
        "prompt_exposure_basis": (
            "UIDs index rows in the same ordered, filtered training dataset. Compare consumed response assignments "
            "per dataset row; duplicate text in distinct rows is not merged. "
            "Tokenizer and filtering settings are frozen."
        ),
        "diagnostic_age_scope": (
            "Source mismatch bins use trainer global step minus the admitted group's policy step. "
            "Exact per-token optimizer ages come from consumed version spans and applied-update ledgers; "
            "a response spanning publications can contain several ages within one source diagnostic bin."
        ),
        "runs": summaries,
        "cumulative_prompt_exposure_comparisons": comparisons,
        "excluded_confirmation_failures": failures,
        "completed_confirmation_operational_exceptions": completions,
        "confirmation_operational_costs_by_arm": operational_costs,
        "all_new_completed_reserved_h100_task_hours": costs["current_completed_reserved_h100_task_hours"],
        "accounted_completed_attempts": costs["accounted_completed_attempts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.protocol_sha256:
        raise ValueError("secondary protocol differs from immutable publication")
    protocol = json.loads(raw)
    base = args.protocol.resolve().parent.parent
    if hashlib.sha256((base / protocol["analysis_code"]).read_bytes()).hexdigest() != protocol["analysis_sha256"]:
        raise ValueError("secondary summaries require the original frozen primary analysis code")
    result = summarize_confirmation(
        json.loads(args.runs.read_text()), protocol, base, json.loads(args.costs.read_text())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2) + "\n").encode()
    args.output.write_bytes(gzip.compress(payload, mtime=0) if args.output.suffix == ".gz" else payload)
    print(f"Wrote full confirmation secondary summaries to {args.output}")


if __name__ == "__main__":
    main()
