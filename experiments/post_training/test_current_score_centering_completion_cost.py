# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression for pending controller wait after completed GPU work."""

from copy import deepcopy

import pytest

from experiments.post_training.correct_current_score_centering_completion_cost import correct_cost


def test_controller_wait_does_not_change_original_allocation_or_scientific_results():
    policy = {
        "run_id": "completed-run",
        "configuration_sha256": "configuration",
        "source_commit": "source",
        "driver_sha256": "driver",
        "resolved_sha256": "resolved",
        "checkpoint_bytes": 939_000_000_000,
        "scientific_cleanup_completed_at_ms": 1000,
        "policy_sha256": "policy",
        "original_attempts_sha256": "attempts",
    }
    run = {
        **{key: policy[key] for key in ("run_id", "configuration_sha256", "source_commit")},
        "job_id": "/owner/completed-run",
        "scientific_work_complete": True,
        "complete_consumed_training_evidence": True,
        "completed_training_steps": 40,
        "optimizer_updates_applied": 40,
        "driver": {"sha256": "driver"},
        "resolved_launch": {"sha256": "resolved"},
        "checkpoint": {"bytes": 939_000_000_000},
        "endpoint": {"evidence": "unchanged held-out responses"},
        "tasks": [],
    }
    attempts = []
    for rank in range(5):
        task_id = run["job_id"] + f"/{rank}"
        state = 4 if rank in (0, 2) else 7 if rank == 1 else 11
        attempts.append(
            {
                "task_id": task_id,
                "attempt_id": 0,
                "current_attempt_id": 0,
                "created_at_ms": 0,
                "finished_at_ms": (rank + 3) * 1000,
                "state": state,
                "exit_code": 0 if state == 4 else None,
                "error": (
                    "PodDeleted: pod was deleted while the attempt was active"
                    if state == 7
                    else "Coscheduled sibling bounced for atomic re-scheduling"
                ),
            }
        )
        run["tasks"].append(
            {
                "task_id": task_id,
                "current_attempt_id": 0,
                "state": 4,
                "job_state": 4,
                "priority_band": 2,
                "job_priority": 2,
                "started_at_ms": 0,
                "finished_at_ms": 10_000,
            }
        )
    for controller_wait_ms in (600_000, 6_000_000):
        observed = deepcopy(run)
        observed.update(allocated_wall_seconds=controller_wait_ms / 1000, reserved_h100_task_hours=999)
        for task in observed["tasks"]:
            task["finished_at_ms"] += controller_wait_ms
        result = correct_cost(observed, attempts, policy)
        # Five original allocations last 3, 4, 5, 6 and 7 seconds on eight GPUs.
        assert result["reserved_h100_task_hours"] == pytest.approx(200 / 3600)
        assert result["allocated_wall_seconds"] == 7
        assert result["endpoint"] == run["endpoint"]
        assert result["tasks"] == observed["tasks"]
        assert observed["reserved_h100_task_hours"] == 999
