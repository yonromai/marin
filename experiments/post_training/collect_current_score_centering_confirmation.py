# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect terminal confirmation evidence without choosing runs from their scores.

The collector requires complete immutable training and evaluation evidence,
source/configuration pins, finite optimizer ledgers, and the final checkpoint.
It leaves running or failed jobs for the authoring agent to audit separately.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import fsspec
import yaml
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.analyze_current_policy_versions import summarize
from experiments.post_training.analyze_current_score_centering_evaluations import (
    audit_evaluations,
    canonical_hash,
    load_membership,
    read_archive,
)
from experiments.post_training.analyze_current_score_centering_training_records import analyze as audit_training
from experiments.post_training.run_current_score_centering_confirmation import cli_json

ANSI = re.compile(r"\x1b\[[0-9;]*m")
METRIC = re.compile(r"WANDB_MIRROR kind=(\w+) step=(\d+) metrics=(.*)")


def parse_metrics(raw: str) -> list[dict]:
    rows = []
    for line in raw.splitlines():
        clean = ANSI.sub("", line)
        match = METRIC.search(clean)
        if not match:
            continue
        values, _ = json.JSONDecoder().raw_decode(match[3])
        stamp = re.search(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+", clean)
        if stamp is None:
            raise ValueError("source metrics have no observed UTC timestamp")
        values.update(kind=match[1], step=int(match[2]), timestamp_utc=stamp[0] + "+00:00")
        rows.append(values)
    return rows


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_resolved_launch(raw: bytes, *, run_id: str, seed: int, source_commit: str) -> dict:
    """Read the native TrainingDriver document and verify its launch provenance."""
    configuration = yaml.safe_load(raw)["config"]
    if configuration["runtime"]["launcher_commit"] != source_commit or configuration["run"]["id"] != run_id:
        raise ValueError("published resolved launch names different source or run identity")
    if configuration["run"]["seed"] != seed:
        raise ValueError("published resolved launch has a different training seed")
    return configuration


def collect(row: dict, tasks: list[dict], protocol: dict, base: Path, fs, cache: dict) -> dict:
    run_id = row["run_id"]
    short = run_id.removeprefix("score-centering-current-")
    job_id = "/romain/" + run_id
    if len(tasks) != 5 or any(t["state"] != 4 or t["job_state"] != 4 for t in tasks):
        raise ValueError("confirmation collection requires five successful terminal GPU tasks")
    if any(t["current_attempt_id"] != 0 or t["priority_band"] != 2 or t["job_priority"] != 2 for t in tasks):
        raise ValueError("confirmation retry or effective priority requires author audit")
    if hashlib.sha256((base / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("frozen configuration bytes changed")
    configuration = yaml.safe_load((base / row["path"]).read_bytes())
    if configuration["run"]["id"] != run_id or configuration["iris"]["job_name"] != run_id:
        raise ValueError("registry differs from the frozen input run/job identity")
    durable = configuration["artifacts"]["attempts_root"].rsplit("/", 1)[0]
    resolved_uri = configuration["artifacts"]["resolved_config_uri"]
    resolved_raw = fs.cat_file(resolved_uri)
    read_resolved_launch(resolved_raw, run_id=run_id, seed=row["seed"], source_commit=protocol["source_commit"])
    uri = cache.get(row.get("driver_cache_key", short))
    if uri is None:
        source = configuration["ray"]["log_dir"]
        candidates = fs.glob(source + "/rank0-*/session_latest/worker-*.err", detail=True)
        for filename, info in sorted(candidates.items(), key=lambda pair: pair[1]["size"], reverse=True)[:16]:
            if info["size"] > 100 and b":task_name:skyrl_entrypoint" in fs.cat_file(filename, start=0, end=256):
                uri = filename
                break
    if uri is None:
        raise ValueError("successful job has no identified source driver yet")
    raw = fs.cat_file(uri)
    metrics = parse_metrics(raw.decode(errors="replace"))
    train = [r for r in metrics if r["kind"] == "train"]
    expected = list(range(1, protocol["completed_training_steps"] + 1))
    if sorted(r["step"] for r in train) != expected:
        raise ValueError("uploaded source metrics are missing completed training steps")
    if any(not math.isfinite(r["policy/raw_grad_norm"]) for r in train):
        raise ValueError("nonfinite raw gradients require author audit")
    directory = base / "results/score_centering_current_confirmation_runs" / run_id
    records, inputs = read_archive(durable + "/attempts/trajectories")
    endpoint = audit_evaluations(
        records,
        load_membership(base / "results/score_centering_current_snowball_heldout_manifest.json"),
        protocol["evaluation_steps"],
    )
    members = endpoint.pop("member_results")
    endpoint.update(
        archive=durable + "/attempts/trajectories",
        inputs=inputs,
        member_result_count=len(members),
        member_results_sha256=canonical_hash(members),
    )
    save_json(directory / "endpoint.json", endpoint)
    versions = configuration["skyrl"]["trainer"]["token_policy_version_archive"]
    vfs, vroot = fsspec.core.url_to_fs(versions)
    ledger = [json.loads(vfs.cat_file(path)) for path in sorted(vfs.glob(vroot + "/step-*.json"))]
    if sorted(r["training_step"] for r in ledger) != expected:
        raise ValueError("optimizer-version ledger is incomplete")
    by_step = {r["step"]: r for r in train}
    if any(r["optimizer_updates_applied"] != by_step[r["training_step"]]["policy/policy_update_steps"] for r in ledger):
        raise ValueError("applied-update archive disagrees with source training metrics")
    save_json(directory / "ages.json", summarize(ledger))
    training = audit_training(durable + "/attempts/trajectories", versions, run_id, expected)
    if any(
        r["responses"] != 512
        or r["distinct_prompt_uids"] != 128
        or any(count != 4 for count in r["uid_response_counts"].values())
        for r in training["steps"]
    ):
        raise ValueError("consumed prompt exposure differs from frozen batch geometry")
    save_json(directory / "training_records.json", training)
    checkpoint_root = configuration["artifacts"]["checkpoint_root"]
    if int(fs.cat_file(checkpoint_root + "/latest_ckpt_global_step.txt")) != expected[-1]:
        raise ValueError("checkpoint marker does not name the declared final step")
    checkpoint_files = fs.find(checkpoint_root + f"/global_step_{expected[-1]}", detail=True)
    checkpoint_bytes = sum(info["size"] for info in checkpoint_files.values())
    if checkpoint_bytes < 900_000_000_000:
        raise ValueError("final Snowball trainer/optimizer checkpoint is incomplete")
    raw_sha = hashlib.sha256(raw).hexdigest()
    preserved = durable + "/confirmation-evidence/driver-" + raw_sha + ".err"
    if fs.exists(preserved):
        if fs.cat_file(preserved) != raw:
            raise ValueError("immutable source driver identity has conflicting bytes")
    else:
        fs.pipe(preserved, raw)
    resolved_sha = hashlib.sha256(resolved_raw).hexdigest()
    preserved_resolved = durable + "/confirmation-evidence/resolved-" + resolved_sha + ".json"
    if fs.exists(preserved_resolved):
        if fs.cat_file(preserved_resolved) != resolved_raw:
            raise ValueError("immutable resolved launch identity has conflicting bytes")
    else:
        fs.pipe(preserved_resolved, resolved_raw)
    directory.mkdir(parents=True, exist_ok=True)
    compressed = gzip.compress(json.dumps(metrics, sort_keys=True).encode(), mtime=0)
    (directory / "source_metrics.json.gz").write_bytes(compressed)
    starts = [t["started_at_ms"] for t in tasks]
    ends = [t["finished_at_ms"] for t in tasks]
    if any(value is None for value in starts + ends):
        raise ValueError("successful tasks have incomplete allocation timestamps")
    result = {
        "arm": row["arm"],
        "seed": row["seed"],
        "run_id": run_id,
        "job_id": job_id,
        "source_commit": protocol["source_commit"],
        "configuration_sha256": row["sha256"],
        "state": "succeeded",
        "scientific_work_complete": True,
        "completed_training_steps": len(train),
        "complete_consumed_training_evidence": True,
        "optimizer_updates_applied": sum(r["optimizer_updates_applied"] for r in ledger),
        "allocated_wall_seconds": (max(ends) - min(starts)) / 1000,
        "reserved_h100_task_hours": sum(8 * (end - start) / 3_600_000 for start, end in zip(starts, ends, strict=True)),
        "tasks": tasks,
        "endpoint_summary": str((directory / "endpoint.json").relative_to(base)),
        "endpoint": endpoint,
        "driver": {"source": fs.unstrip_protocol(uri), "preserved_uri": preserved, "sha256": raw_sha, "bytes": len(raw)},
        "resolved_launch": {"source": resolved_uri, "preserved_uri": preserved_resolved, "sha256": resolved_sha},
        "source_metrics": {
            "path": str((directory / "source_metrics.json.gz").relative_to(base)),
            "sha256": hashlib.sha256(compressed).hexdigest(),
        },
        "checkpoint": {
            "root": checkpoint_root,
            "step": expected[-1],
            "objects": len(checkpoint_files),
            "bytes": checkpoint_bytes,
        },
        "evaluation_elapsed_seconds": {
            str(r["step"]): (datetime.fromisoformat(r["timestamp_utc"]).timestamp() * 1000 - min(starts)) / 1000
            for r in metrics
            if r["kind"] == "eval"
        },
        "training_core_seconds": sum(r["async/performance/core_seconds"] for r in train),
        "training_cycle_seconds": sum(r["async/performance/cycle_seconds"] for r in train),
    }
    save_json(directory / "run.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver-cache", type=Path, default=Path("/tmp/score-centering-current-driver-cache.json"))
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.protocol_sha256:
        raise ValueError("protocol bytes differ from the immutable publication")
    protocol = json.loads(raw)
    if protocol["status"] != "frozen":
        raise ValueError("collection requires the frozen protocol")
    base = args.protocol.resolve().parent.parent
    for file_key, sha_key in (
        ("endpoint_audit", "endpoint_audit_sha256"),
        ("training_record_audit", "training_record_audit_sha256"),
        ("policy_version_audit", "policy_version_audit_sha256"),
        ("terminal_collector", "terminal_collector_sha256"),
    ):
        if hashlib.sha256((base / protocol[file_key]).read_bytes()).hexdigest() != protocol[sha_key]:
            raise ValueError("audit code changed after confirmation freeze")
    quoted = ",".join("'/romain/" + row["run_id"] + "'" for row in protocol["configuration_manifest"])
    sql = (
        "SELECT t.job_id,t.task_id,t.state,t.current_attempt_id,t.started_at_ms,t.finished_at_ms,t.priority_band,"
        "j.state AS job_state,c.priority_band AS job_priority FROM tasks t JOIN jobs j ON j.job_id=t.job_id "
        "JOIN job_config c ON c.job_id=j.job_id WHERE t.job_id IN (" + quoted + ") ORDER BY t.task_id"
    )
    value = cli_json(["rpc", "controller", "execute-raw-query", "--sql", sql])
    columns = [c["name"] for c in value["columns"]]
    tasks = [dict(zip(columns, json.loads(row), strict=True)) for row in value["rows"]]
    cache = json.loads(args.driver_cache.read_text()) if args.driver_cache.exists() else {}
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    runs = []
    for row in protocol["configuration_manifest"]:
        selected = [t for t in tasks if t["job_id"] == "/romain/" + row["run_id"]]
        if len(selected) == 5 and all(t["state"] == 4 and t["job_state"] == 4 for t in selected):
            runs.append(collect(row, selected, protocol, base, fs, cache))
            print(
                json.dumps(
                    {"collected": row["run_id"], "reserved_h100_task_hours": runs[-1]["reserved_h100_task_hours"]}
                ),
                flush=True,
            )
    save_json(args.output, runs)
    print(json.dumps({"terminal_audited_runs": len(runs), "declared_runs": len(protocol["configuration_manifest"])}))


if __name__ == "__main__":
    main()
