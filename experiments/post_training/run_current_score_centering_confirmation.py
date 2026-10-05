# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the frozen confirmation in spare-capacity waves using the Iris CLI.

Run with the pinned SkyRL package from the protocol. The default is read-only;
--launch submits at most one next input per check. --watch performs repeated
checks, while the authoring agent audits outputs and handles failures separately.
Authentication uses the existing marin-env/Iris integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path

import yaml
from rigging.filesystem.factory import filesystem
from rigging.filesystem.s3_compat import configure_coreweave_s3

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "experiments/post_training/results/score_centering_current_confirmation_protocol.json"
CLUSTER = "lib/iris/config/cw-rno2a.yaml"
BOARD = Path("/home/romain/.local/state/resource-board/board.sqlite3")


def cli_json(arguments: list[str]) -> dict:
    result = subprocess.run(
        [
            "marin-env",
            "uv",
            "run",
            "--frozen",
            "--package",
            "marin-core",
            "--prerelease=allow",
            "iris",
            "--cluster",
            CLUSTER,
            *arguments,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=90,
        check=True,
    )
    return json.loads(result.stdout[result.stdout.index("{") :])


def check_first_steps(fs, cache_path: Path, first_runs: list[dict]) -> bool:
    """Return False while first-step evidence is missing; raise ValueError if it is invalid."""
    if not cache_path.exists():
        return False
    cache = json.loads(cache_path.read_text())
    for run in first_runs:
        name = run.get("driver_cache_key", run["run_id"].removeprefix("score-centering-current-"))
        if name not in cache:
            return False
        raw = fs.cat(cache[name]).decode(errors="replace")
        if "WANDB_MIRROR kind=train step=1 metrics=" not in raw:
            return False
        line = next(line for line in raw.splitlines() if "WANDB_MIRROR kind=train step=1 metrics=" in line)
        metrics, _ = json.JSONDecoder().raw_decode(line.split("metrics=", 1)[1])
        norm = metrics.get("policy/raw_grad_norm", 0)
        if not math.isfinite(norm) or norm <= 0 or metrics.get("policy/policy_update_steps", 0) != 1:
            raise ValueError(f"first-step GPU qualification failed for {run['run_id']}")
    return True


def announce(board_id: int, origin: str, job_id: str, note: str) -> None:
    with sqlite3.connect(BOARD, timeout=5) as connection:
        row = connection.execute(
            "SELECT job_refs FROM intentions WHERE id=? AND origin_session=? AND phase <> 'closed'",
            (board_id, origin),
        ).fetchone()
        if row is None:
            raise ValueError("campaign resource intention is missing or belongs to another session")
        refs = row[0].splitlines()
        if job_id not in refs:
            refs.append(job_id)
        connection.execute(
            "UPDATE intentions SET job_refs=?,phase='running',note=?,updated_at=unixepoch() "
            "WHERE id=? AND origin_session=? AND phase <> 'closed'",
            ("\n".join(refs), note, board_id, origin),
        )


def run(args) -> None:
    protocol_path = args.protocol.resolve()
    protocol_bytes = protocol_path.read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != args.protocol_sha256:
        raise ValueError("confirmation protocol changed after its immutable publication")
    protocol = json.loads(protocol_bytes)
    if protocol["status"] != "frozen" or not 1 <= args.max_active <= 6:
        raise ValueError("requires the frozen protocol and at most six active jobs")
    declarations = {(row["arm"], row["seed"]): row for row in protocol["configuration_manifest"]}
    order = [declarations[(row["arm"], row["seed"])] for row in protocol["launch_order"]]
    for row in order:
        if not re.fullmatch(r"[a-z0-9-]+", row["run_id"]):
            raise ValueError("invalid frozen run identity")
        configuration_bytes = (protocol_path.parent.parent / row["path"]).read_bytes()
        if hashlib.sha256(configuration_bytes).hexdigest() != row["sha256"]:
            raise ValueError(f"frozen input bytes changed: {row['path']}")
        configuration = yaml.safe_load(configuration_bytes)
        if configuration["run"]["id"] != row["run_id"] or configuration["iris"]["job_name"] != row["run_id"]:
            raise ValueError("protocol registry differs from the frozen input run/job identity")
    binding = json.loads(args.binding.read_text())
    native_id = binding["native_id"]
    if os.environ.get("CODEX_THREAD_ID", native_id) != native_id:
        raise ValueError("resource-board binding differs from the current native session")
    origin = f"{binding['origin']}/{binding['harness']}/{native_id}"
    fs = filesystem("s3")
    previous_outside = None
    first_steps_passed = False
    while True:
        quoted = ",".join("'/romain/" + row["run_id"] + "'" for row in order)
        sql = (
            "SELECT t.job_id,t.task_id,t.state,t.current_attempt_id,t.priority_band,j.state AS job_state,"
            "c.priority_band AS job_priority FROM tasks t JOIN jobs j ON j.job_id=t.job_id "
            "JOIN job_config c ON c.job_id=j.job_id WHERE t.job_id IN (" + quoted + ") ORDER BY t.task_id"
        )
        result = cli_json(["rpc", "controller", "execute-raw-query", "--sql", sql])
        columns = [column["name"] for column in result["columns"]]
        tasks = [dict(zip(columns, json.loads(row), strict=True)) for row in result["rows"]]
        if any(
            row["state"] in (5, 6, 7) or row["job_state"] in (5, 6) or row["current_attempt_id"] not in (0, None)
            for row in tasks
        ):
            raise ValueError("campaign failure or retry requires author audit before additional submissions")
        if any(row["priority_band"] != 2 or row["job_priority"] != 2 for row in tasks):
            raise ValueError("campaign effective priority differs from interactive")
        known = {row["job_id"] for row in tasks}
        succeeded = {row["job_id"] for row in tasks if row["job_state"] == 4}
        active = known - succeeded
        if len(succeeded) == len(order):
            print(
                json.dumps({"phase": "all jobs terminal; research output audits remain", "jobs": len(order)}), flush=True
            )
            return
        backend = cli_json(["rpc", "controller", "list-backends"])["backends"][0]
        availability = backend["availability"]
        observed_ms = int(availability["observation_epoch_ms"])
        if time.time() * 1000 - observed_ms > 60_000:
            raise ValueError("capacity observation is stale")
        free = int(availability["amounts"].get("h100", 0))
        total = int(availability["total_amounts"]["h100"])
        empty_nodes = sum(
            node["gpu_count"] == 8 and node["ready"] and node["schedulable"] and node["running_pods"] == 0
            for node in backend["detail"]["kubernetes"]["nodes"]
        )
        own_held = 8 * sum(row["state"] == 3 for row in tasks)
        outside = total - free - own_held
        rising = previous_outside is not None and outside >= previous_outside + 8
        previous_outside = outside
        if not first_steps_passed:
            first_steps_passed = check_first_steps(fs, args.driver_cache, order[:4])
        candidate = next((row for row in order if "/romain/" + row["run_id"] not in known), None)
        eligible = (
            candidate is not None
            and len(active) < args.max_active
            and free >= 40 + args.spare_after
            and empty_nodes >= 5
            and first_steps_passed
            and not rising
        )
        print(
            json.dumps(
                {
                    "active": len(active),
                    "succeeded": len(succeeded),
                    "h100_free": free,
                    "empty_whole_nodes": empty_nodes,
                    "other_demand_rose": rising,
                    "first_steps_passed": first_steps_passed,
                    "next": candidate["run_id"] if candidate else None,
                    "eligible": eligible,
                    "launch_enabled": args.launch,
                }
            ),
            flush=True,
        )
        if args.launch and eligible:
            from cloud.iris.launch import main as launch_main  # noqa: PLC0415  # optional pinned SkyRL package

            assert candidate is not None
            job_id = "/romain/" + candidate["run_id"]
            note = (
                f"Frozen Snowball confirmation: {len(succeeded)} complete, {len(active)} active; "
                f"next {candidate['arm']} seed {candidate['seed']}. Pool {free}/{total} H100 free, "
                f"{empty_nodes} empty whole nodes; interactive, 40 H100. At most {args.max_active} jobs; "
                f"retain {args.spare_after} spare H100, pause new launches when other demand rises. "
                "51 fixed jobs; no score-based selection. Research audits remain."
            )
            announce(args.board_id, origin, job_id, note)
            code = launch_main(["iris", "launch", "--config", str(protocol_path.parent.parent / candidate["path"])])
            if code:
                raise ValueError(f"submission failed for {job_id}; reconcile live state before retrying")
            print(json.dumps({"submitted": job_id, "configuration_sha256": candidate["sha256"]}), flush=True)
        if not args.watch:
            return
        time.sleep(args.check_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--board-id", type=int, required=True)
    parser.add_argument("--driver-cache", type=Path, default=Path("/tmp/score-centering-current-driver-cache.json"))
    parser.add_argument("--max-active", type=int, default=6)
    parser.add_argument("--spare-after", type=int, default=80)
    parser.add_argument("--check-seconds", type=int, default=60)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.check_seconds < 60 or args.spare_after < 80:
        raise ValueError("require at least 60 seconds between checks and 80 spare H100 after a launch")
    configure_coreweave_s3()
    run(args)


if __name__ == "__main__":
    main()
