# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Account for every owned allocation, including failed startup and teardown."""

import argparse
import gzip
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def account(protocol, report_dir):
    rows = []
    for allocation in protocol["owned_allocations"]:
        assert allocation.get("controller_terminal_snapshot"), "Wait for terminal owned allocations"
        assert allocation.get("provider_pods_absent") is True
        start = allocation["controller_start_snapshot"]
        terminal = allocation["controller_terminal_snapshot"]
        assert terminal and all(row["max_retries_failure"] == row["max_retries_preemption"] == "0" for row in terminal)
        assert all(row["task_finished_at_ms"] for row in terminal)
        submitted = int(start["submitted_at_ms"]) / 1000
        finished = max(int(row["finished_at_ms"]) for row in terminal) / 1000
        controller_started = int(start["started_at_ms"]) / 1000
        absence = epoch(allocation["provider_absence_observed_at_utc"])
        gpus = allocation.get("gpus_admitted", allocation.get("gpus", 0))
        row = {
            "job_id": allocation["job_id"],
            "cluster": allocation["cluster"],
            "state": allocation["state"],
            "admitted_h100s": gpus,
            "submitted_to_terminal_seconds": finished - submitted,
            "controller_runtime_seconds": finished - controller_started,
            "controller_runtime_h100_hours": gpus * (finished - controller_started) / 3600,
            "provider_upper_bound_h100_hours": gpus * (absence - submitted) / 3600,
            "terminal_to_verified_absence_seconds": absence - finished,
            "automatic_retries": 0,
        }
        initial_sync = allocation.get("initial_weight_sync_completed_at_utc")
        admitted = [
            condition["lastTransitionTime"]
            for condition in allocation.get("admission", [])
            if condition["type"] == "Admitted" and condition["status"] == "True"
        ]
        allocation_started = epoch(min(admitted)) if admitted else controller_started
        assert finished >= allocation_started
        row["allocated_h100_hours"] = gpus * (finished - allocation_started) / 3600
        row["allocation_start_basis"] = (
            ("Kueue admission" if admitted else "controller start proxy; admission time unavailable")
            if gpus
            else "verified never admitted"
        )
        if admitted:
            row["submission_to_admission_seconds"] = allocation_started - submitted
        if initial_sync and admitted:
            setup_seconds = epoch(initial_sync) - epoch(min(admitted))
            row["admission_to_initial_sync_seconds"] = setup_seconds
            row["admission_to_initial_sync_h100_hours"] = gpus * setup_seconds / 3600
        name = allocation["job_id"].split("/")[-1] + ".json"
        path = report_dir / name
        if not path.exists():
            path = report_dir / (name + ".gz")
        if not path.exists():
            path = report_dir / (name + ".zip")
        if path.exists():
            stored = path.read_bytes()
            if path.suffix == ".gz":
                stored = gzip.decompress(stored)
            elif path.suffix == ".zip":
                with zipfile.ZipFile(io.BytesIO(stored)) as archive:
                    assert len(archive.namelist()) == 1
                    stored = archive.read(archive.namelist()[0])
            data = json.loads(stored)
            windows = data["windows"] + data.get("native_periodic_windows", [])
            native_seconds = sum(window["total_seconds"] for window in windows)
            row.update(
                complete_report=data["complete"],
                applied_updates=data["applied_updates"],
                native_cycles_including_profiled_warmup_seconds=native_seconds,
                residual_setup_qualification_io_idle_teardown_seconds=finished - controller_started - native_seconds,
                initial_weight_sync_timings=data["setup_timings"],
            )
        rows.append(row)
    return {
        "allocations": rows,
        "totals": {
            key: sum(row[key] for row in rows)
            for key in ("allocated_h100_hours", "controller_runtime_h100_hours", "provider_upper_bound_h100_hours")
        },
        "limits": (
            "Controller RUNNING includes admission wait; the withdrawn East gang was verified never admitted "
            "and uses zero GPUs. Allocated hours start at Kueue admission when recorded; earlier Qwen runs use "
            "controller start as a proxy because their exact admission timestamps were not retained. "
            "The separate controller proxy can include unallocated queue time. Provider upper bounds run from "
            "submission through verified pod absence, "
            "so they include queue, startup, qualification, report/profile work, idle and teardown. "
            "The residual is accounted but cannot be separated into those categories from retained timers. "
            "Native totals include profiled warmup; they are not steady-state estimates. No dollar-cost claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = account(json.loads(args.protocol.read_text()), args.report_dir)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["totals"]))


if __name__ == "__main__":
    main()
