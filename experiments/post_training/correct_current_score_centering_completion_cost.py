# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exclude unallocated controller wait after an audited manual job completion.

The original collector and scientific audits remain frozen. Iris job complete
can close pending tasks after their original pods have disappeared. Preserve
that controller result, then use the original attempt allocation timestamps.
The immutable operational policy names the one audited run and input bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def correct_cost(run: dict, attempts: list[dict], policy: dict) -> dict:
    """Keep scientific results and report the completed original allocation."""
    if (
        run["run_id"] != policy["run_id"]
        or run["configuration_sha256"] != policy["configuration_sha256"]
        or run["source_commit"] != policy["source_commit"]
        or run["driver"]["sha256"] != policy["driver_sha256"]
        or run["resolved_launch"]["sha256"] != policy["resolved_sha256"]
        or run["checkpoint"]["bytes"] != policy["checkpoint_bytes"]
        or not run["scientific_work_complete"]
        or not run["complete_consumed_training_evidence"]
        or run["completed_training_steps"] != 40
        or run["optimizer_updates_applied"] != 40
    ):
        raise ValueError("operational completion differs from the audited frozen run")
    tasks = {row["task_id"]: row for row in run["tasks"]}
    if len(tasks) != 5 or len(attempts) != 5 or {row["task_id"] for row in attempts} != set(tasks):
        raise ValueError("original attempt ledger must cover all five tasks exactly once")
    starts, ends = [], []
    cleanup_ms = policy["scientific_cleanup_completed_at_ms"]
    failures = []
    for attempt in attempts:
        task = tasks[attempt["task_id"]]
        if (
            attempt["attempt_id"] != 0
            or attempt["current_attempt_id"] != 0
            or task["current_attempt_id"] != 0
            or task["state"] != 4
            or task["job_state"] != 4
            or task["priority_band"] != 2
            or task["job_priority"] != 2
        ):
            raise ValueError("completion has a retry, priority change or nonterminal task")
        start, end = attempt["created_at_ms"], attempt["finished_at_ms"]
        if start != task["started_at_ms"] or end is None or not start < cleanup_ms < end:
            raise ValueError("original allocation does not contain completed scientific work")
        if end > task["finished_at_ms"]:
            raise ValueError("original allocation ended after controller completion")
        if attempt["state"] == 4:
            if attempt["exit_code"] != 0:
                raise ValueError("successful original attempt did not exit zero")
        elif attempt["state"] == 7 and attempt["error"] == "PodDeleted: pod was deleted while the attempt was active":
            failures.append(attempt)
        elif attempt["state"] == 11 and "bounced for atomic re-scheduling" in attempt["error"]:
            failures.append(attempt)
        else:
            raise ValueError("operational exception includes an unaudited failure")
        starts.append(start)
        ends.append(end)
    if sum(row["state"] == 7 for row in failures) != 1 or sum(row["state"] == 11 for row in failures) != 2:
        raise ValueError("operational exception differs from the published PodDeleted ledger")
    if attempts[0]["task_id"] != run["job_id"] + "/0" or attempts[0]["state"] != 4:
        raise ValueError("source driver did not independently exit zero")
    return {
        **run,
        "allocated_wall_seconds": (max(ends) - min(starts)) / 1000,
        "reserved_h100_task_hours": sum(8 * (end - start) / 3_600_000 for start, end in zip(starts, ends, strict=True)),
        "operational_completion": {
            "kind": "Iris job complete after verified scientific completion and exact pod release",
            "policy_sha256": policy["policy_sha256"],
            "original_attempts_sha256": policy["original_attempts_sha256"],
            "cost_basis": "Original attempt creation and terminal timestamps; controller wait has no GPU allocation.",
            "controller_allocated_wall_seconds_before_correction": run["allocated_wall_seconds"],
            "controller_reserved_h100_task_hours_before_correction": run["reserved_h100_task_hours"],
            "original_attempts": attempts,
            "limitation": "PodDeleted was observed after cleanup; the deletion initiator is unknown.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.policy.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.policy_sha256:
        raise ValueError("operational completion policy differs from its immutable publication")
    policy = json.loads(raw)
    if hashlib.sha256(args.attempts.read_bytes()).hexdigest() != policy["original_attempts_sha256"]:
        raise ValueError("original attempt ledger differs from its immutable publication")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != policy["cost_correction_code_sha256"]:
        raise ValueError("cost correction code changed after policy publication")
    policy["policy_sha256"] = args.policy_sha256
    result = correct_cost(json.loads(args.run.read_text()), json.loads(args.attempts.read_text()), policy)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run_id": result["run_id"], "reserved_h100_task_hours": result["reserved_h100_task_hours"]}))


if __name__ == "__main__":
    main()
