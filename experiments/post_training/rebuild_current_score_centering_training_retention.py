# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare missing training archives from the run's original stored rollout groups.

Run with the campaign's pinned MarinSkyRL package. Existing native records are
byte controls for its serializer. This command writes local artifacts only;
publication requires a separate immutable policy naming every input and output.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import zipfile
from collections import Counter, defaultdict
from importlib.metadata import distribution
from pathlib import Path

import yaml
from skyrl_train.tokenizer import create_tokenizer
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from skyrl_train.trajectory_runners.trajectory_retention import (
    TrajectorySink,
    _archive_payload,
    _SelectedRecord,
    _Selection,
    _serialized_record,
    build_trajectory_records,
    canonical_json_bytes,
)
from skyrl_train.trajectory_runners.trajectory_retention_config import parse_trajectory_retention_config

from experiments.post_training.analyze_current_score_centering_training_records import TrainingRecordAudit, response_key


def checked_bytes(path: Path, sha: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"source bytes changed: {path}")
    return raw


def rebuild(case: Path, configuration: Path, configuration_sha: str, source_commit: str, output: Path) -> dict:
    """Reproduce all native records before preparing independently backed missing records."""
    installed = json.loads(distribution("marinskyrl").read_text("direct_url.json"))
    if installed.get("vcs_info", {}).get("commit_id") != source_commit:
        raise ValueError("reconstruction requires the frozen runtime source")
    config = yaml.safe_load(checked_bytes(configuration, configuration_sha))
    audit_source = json.loads((case / "audit.json").read_text())
    groups = json.loads((case / "original_group_index.json").read_text())
    original = {}
    audit = TrainingRecordAudit(audit_source["run_id"])
    for item in audit_source["training_archives"]:
        path = case / (item["sha256"] + ".zip")
        checked_bytes(path, item["sha256"])
        with zipfile.ZipFile(path) as bundle:
            for entry in json.loads(bundle.read("manifest.json"))["records"]:
                payload = bundle.read(entry["entry"])
                if len(payload) != entry["bytes"]:
                    raise ValueError("original archive manifest disagrees with its record")
                record = json.loads(gzip.decompress(payload))
                audit.retain(record)
                original[record["record_id"]] = canonical_json_bytes(record)
    reference = json.loads(next(iter(original.values())))
    provenance = reference["provenance"]
    retention = parse_trajectory_retention_config(
        {
            **config["skyrl"]["generator"]["trajectory_retention"],
            "run_id": audit_source["run_id"],
            **{
                key: provenance[key]
                for key in ("model_path", "model_source_identity", "resume_path", "inference_backend")
            },
        }
    )
    tokenizer = create_tokenizer(
        config["inputs"]["model"]["tokenizer_uri"],
        disable_fast_tokenizer=False,
        revision=config["inputs"]["model"]["tokenizer_revision"],
    )
    reproduced = set()
    recovered = {}
    source_groups = []
    for item in groups:
        group = pickle.loads(checked_bytes(case / "original_groups" / (item["sha256"] + ".pkl"), item["sha256"]))
        request, _ = prepare_trajectory_request(
            [group.prompt],
            config["skyrl"]["generator"]["n_samples_per_prompt"],
            provenance["sampling"],
            config["skyrl"]["environment"]["env_class"],
            "train",
            group.policy_step,
        )
        # Retention metrics are appended after the native record is serialized.
        batch = {
            **group.trajectory_batch,
            "rollout_metrics": {
                key: value
                for key, value in (group.trajectory_batch.get("rollout_metrics") or {}).items()
                if not key.startswith("generate/trajectory_retention/")
            },
        }
        records = build_trajectory_records(request, batch, retention, tokenizer, runner_name=provenance["runner"])
        recovered_ids = []
        for record in records:
            serialized = _serialized_record(record, retention.redact_fields)
            raw = canonical_json_bytes(serialized)
            if record.record_id in original:
                if raw != original[record.record_id]:
                    raise ValueError("native record was not reproduced byte for byte")
                reproduced.add(record.record_id)
            else:
                if record.record_id in recovered:
                    raise ValueError("original groups repeat a recovered record identity")
                recovered[record.record_id] = (record, raw)
                recovered_ids.append(record.record_id)
                audit.retain(serialized)
        if recovered_ids:
            source_groups.append({**item, "recovered_record_ids": recovered_ids})
    if reproduced != original.keys():
        raise ValueError("some native records could not be reproduced from original groups")
    ledger = []
    for item in audit_source["version_inputs"]:
        filename = Path(item["uri"]).name
        ledger.append(json.loads(checked_bytes(case / ("ledger-" + filename), item["sha256"])))
    training = audit.consume(ledger, list(range(1, 41)))
    if len(recovered) != audit_source["missing_consumed_responses"]:
        raise ValueError("recovered record count differs from the independently detected gap")
    consumed = Counter(
        response_key(str(row["uid"]), row["response_token_ids"], row["loss_mask"])
        for step in ledger
        for row in step["rows"]
    )
    recovered_keys = Counter(
        response_key(str(record.trajectory.instance_id), record.response.token_ids, record.response.loss_mask)
        for record, _ in recovered.values()
    )
    if recovered_keys - consumed:
        raise ValueError("recovery includes responses outside the consumption ledger")
    by_step = defaultdict(list)
    for record, raw in recovered.values():
        by_step[record.global_step].append(
            _SelectedRecord(record, gzip.compress(raw, mtime=0), _Selection(("fraction",)))
        )
    archives = []
    for _step, selected in sorted(by_step.items()):
        payload = _archive_payload(sorted(selected, key=lambda item: item.record.record_id))
        relative = TrajectorySink._archive_path(selected[0].record, payload)
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        archives.append(
            {"relative_path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        )
    report = {
        "run_id": audit_source["run_id"],
        "source_commit": source_commit,
        "configuration_sha256": configuration_sha,
        "native_records_reproduced_byte_for_byte": len(reproduced),
        "recovered_records": len(recovered),
        "original_source_groups": source_groups,
        "archives": archives,
        "training_audit_after_recovery": training,
        "scope": "Original immutable rollout groups; frozen serializer; all native records used as byte controls.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = rebuild(args.case, args.configuration, args.configuration_sha256, args.source_commit, args.output)
    print(
        json.dumps(
            {key: report[key] for key in ("run_id", "native_records_reproduced_byte_for_byte", "recovered_records")}
        )
    )


if __name__ == "__main__":
    main()
