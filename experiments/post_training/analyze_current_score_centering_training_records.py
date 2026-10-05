# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit full training retention against exact consumed responses and masks.

Retention request steps do not establish optimizer age. The separate consumed
version analysis uses its applied-update ledger for that purpose.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.analyze_current_score_centering_evaluations import ACCEPTED_STOPS


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def response_key(uid, tokens: list, mask: list) -> str:
    if len(tokens) != len(mask) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens
    ):
        raise ValueError("training response token IDs or mask alignment are invalid")
    if any(value not in (0, 1, False, True) for value in mask):
        raise ValueError("training response mask is not binary")
    return digest([str(uid), tokens, [bool(value) for value in mask]])


class TrainingRecordAudit:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.available: Counter[str] = Counter()
        self.record_ids: set[str] = set()
        self.prompt_hashes: dict[str, str] = {}
        self.retained_tokens = 0
        self.retained_loss_tokens = 0
        self.retained_normal_completions = 0

    def retain(self, record: dict) -> None:
        if record["schema_version"] != 6 or record["phase"] != "train" or record["run_id"] != self.run_id:
            raise ValueError("training retention schema, phase, or run identity changed")
        identity = record["record_id"]
        if identity in self.record_ids:
            raise ValueError("duplicate immutable training record identity")
        self.record_ids.add(identity)
        uid = str(record["trajectory"]["instance_id"])
        prompt_hash = digest(record["prompt"]["token_ids"])
        if self.prompt_hashes.setdefault(uid, prompt_hash) != prompt_hash:
            raise ValueError("same training UID names different exact prompt tokens")
        response = record["response"]
        tokens, mask = response["token_ids"], response["loss_mask"]
        self.available[response_key(uid, tokens, mask)] += 1
        self.retained_tokens += len(tokens)
        self.retained_loss_tokens += sum(bool(value) for value in mask)
        self.retained_normal_completions += response["stop_reason"] in ACCEPTED_STOPS and not any(
            record["disposition"].values()
        )

    def consume(self, records: list[dict], expected_steps: list[int]) -> dict:
        if sorted(record["training_step"] for record in records) != expected_steps:
            raise ValueError("consumed training ledger is missing steps or contains duplicate steps")
        steps = []
        consumed: Counter[str] = Counter()
        consumed_tokens = consumed_loss_tokens = 0
        for record in sorted(records, key=lambda value: value["training_step"]):
            uids: Counter[str] = Counter()
            tokens_in_step = loss_in_step = 0
            for row in record["rows"]:
                uid = str(row["uid"])
                tokens, mask = row["response_token_ids"], row["loss_mask"]
                consumed[response_key(uid, tokens, mask)] += 1
                uids[uid] += 1
                tokens_in_step += len(tokens)
                loss_in_step += sum(bool(value) for value in mask)
            consumed_tokens += tokens_in_step
            consumed_loss_tokens += loss_in_step
            steps.append(
                {
                    "training_step": record["training_step"],
                    "responses": len(record["rows"]),
                    "distinct_prompt_uids": len(uids),
                    "uid_response_counts": dict(sorted(uids.items())),
                    "response_tokens": tokens_in_step,
                    "loss_tokens": loss_in_step,
                }
            )
        missing = consumed - self.available
        if missing:
            raise ValueError(f"{sum(missing.values())} consumed responses lack exact retained UID/token/mask evidence")
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "complete_consumed_training_evidence": True,
            "retained_records": len(self.record_ids),
            "retained_response_tokens": self.retained_tokens,
            "retained_loss_tokens": self.retained_loss_tokens,
            "retained_normal_completions": self.retained_normal_completions,
            "consumed_records": sum(consumed.values()),
            "consumed_response_tokens": consumed_tokens,
            "consumed_loss_tokens": consumed_loss_tokens,
            "completed_unconsumed_records": sum((self.available - consumed).values()),
            "completed_unconsumed_response_tokens": self.retained_tokens - consumed_tokens,
            "prompt_uid_token_manifest_sha256": digest(self.prompt_hashes),
            "steps": steps,
            "scope": (
                "Exact multiset of UID/response token IDs/loss mask; identical repeated responses matched by count. "
                "Prompt tokens are stable within each UID. Retention request steps do not establish optimizer age. "
                "Finished retained responses outside consumption are reported; tokens in unfinished or cancelled "
                "generation are not fully observed. Reserved task costs include that work."
            ),
        }


def analyze(archive: str, versions: str, run_id: str, expected_steps: list[int]) -> dict:
    fs, root = url_to_fs(archive)
    audit = TrainingRecordAudit(run_id)
    inputs = []
    for filename in sorted(fs.glob(prefix_join(root, "schema_v6/archives/phase=train/step=*/*.zip"))):
        data = fs.cat_file(filename)
        sha = hashlib.sha256(data).hexdigest()
        if Path(filename).stem != sha:
            raise ValueError("training archive has the wrong content hash")
        inputs.append({"uri": fs.unstrip_protocol(filename), "sha256": sha, "bytes": len(data)})
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            for entry in json.loads(bundle.read("manifest.json"))["records"]:
                payload = bundle.read(entry["entry"])
                record = json.loads(gzip.decompress(payload))
                if len(payload) != entry["bytes"] or record["record_id"] != entry["record_id"]:
                    raise ValueError("training record disagrees with its immutable archive manifest")
                audit.retain(record)
    vfs, vroot = url_to_fs(versions)
    ledger, ledger_inputs = [], []
    for filename in sorted(vfs.glob(prefix_join(vroot, "step-*.json"))):
        data = vfs.cat_file(filename)
        ledger.append(json.loads(data))
        ledger_inputs.append(
            {"uri": vfs.unstrip_protocol(filename), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        )
    result = audit.consume(ledger, expected_steps)
    result.update(archive=archive, versions=versions, archive_inputs=inputs, version_inputs=ledger_inputs)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--versions", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configure_coreweave_s3()
    result = analyze(args.archive, args.versions, args.run_id, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("steps", "archive_inputs", "version_inputs")}
        )
    )


if __name__ == "__main__":
    main()
