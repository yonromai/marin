# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit current completed-answer endpoints against frozen exact prompt tokens.

Read schema-six retained evaluations after the job and its required publisher
are terminal. Named repeats are separate passes, not separate training seeds.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

ACCEPTED_STOPS = frozenset({"complete", "end_turn", "eos", "stop"})
EVALUATION_NAMES = (None, "greedy_repeat")


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def load_membership(path: Path) -> dict:
    membership = json.loads(path.read_text())
    data = (path.parent / membership["members_file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != membership["members_file_sha256"]:
        raise ValueError("compressed frozen membership has the wrong byte hash")
    membership["members"] = json.loads(gzip.decompress(data))
    return membership


def audit_evaluations(records: list[dict], membership: dict, expected_steps: list[int], answer_cap: int = 4096) -> dict:
    members = membership["members"]
    if canonical_hash(members) != membership["membership_sha256"]:
        raise ValueError("frozen membership hash does not match its members")
    primary = set(membership["primary_member_ordinals"])
    selected = [row for row in members if row["ordinal"] in primary]
    if (
        len(selected) != membership["primary_member_count"]
        or canonical_hash(selected) != membership["primary_membership_sha256"]
    ):
        raise ValueError("frozen primary membership does not match its hash or count")
    lookup = {}
    for member in members:
        if canonical_hash(member["content"]) != member["member_sha256"]:
            raise ValueError("frozen member content hash is inconsistent")
        key = tuple(member["rendered_prompt_token_ids"])
        if key in lookup:
            raise ValueError("frozen prompt token sequences are ambiguous")
        lookup[key] = member
    groups = defaultdict(dict)
    ids = set()
    run_ids = set()
    for record in records:
        if record["schema_version"] != 6 or record["phase"] != "eval":
            raise ValueError("expected schema-six evaluation records")
        record_id = record["record_id"]
        if record_id in ids:
            raise ValueError("retained evaluation repeats a record identity")
        ids.add(record_id)
        run_ids.add(record["run_id"])
        step, name = record["global_step"], record["evaluation_name"]
        if step not in expected_steps or name not in EVALUATION_NAMES:
            raise ValueError("evaluation step or named profile differs from the frozen contract")
        if record["provenance"]["model_version_step"] != step:
            raise ValueError("evaluation does not name the completed weights at its step")
        if record["provenance"]["sampling"]["temperature"] != 0.0:
            raise ValueError("evaluation is not greedy")
        if record["trajectory"]["repetition_id"] != 0:
            raise ValueError("each pass requires one response per member")
        member = lookup.get(tuple(record["prompt"]["token_ids"]))
        if member is None:
            raise ValueError("evaluation prompt tokens differ from frozen membership")
        ordinal, content = member["ordinal"], member["content"]
        trajectory = record["trajectory"]
        extras = trajectory["environment_extras"]
        if trajectory["environment_class"] != content["env_class"] or extras["reward_spec"] != content["reward_spec"]:
            raise ValueError("evaluation environment or native grading rule changed")
        if extras["data_source"] != content["extra_info"]["data_source"]:
            raise ValueError("evaluation data source changed")
        response = record["response"]
        if response["generation_limit"] != answer_cap or len(response["token_ids"]) > answer_cap:
            raise ValueError("evaluation answer limit differs from the frozen contract")
        if len(response["token_ids"]) != len(response["loss_mask"]):
            raise ValueError("evaluation token evidence has an inconsistent mask")
        outcome = float(record["reward"]["outcome"])
        if not math.isfinite(outcome):
            raise ValueError("evaluation has a nonfinite native verifier outcome")
        error = record["disposition"]
        completed = response["stop_reason"] in ACCEPTED_STOPS and not any(error.values())
        if ordinal in groups[(step, name)]:
            raise ValueError("evaluation pass repeats a held-out member")
        groups[(step, name)][ordinal] = {
            "ordinal": ordinal,
            "member_sha256": member["member_sha256"],
            "source": extras["data_source"],
            "completed": completed,
            "completed_correct": completed and outcome > 0,
            "native_outcome": outcome,
            "response_tokens": len(response["token_ids"]),
            "stop_reason": response["stop_reason"],
            "record_id": record_id,
        }
    if len(run_ids) != 1:
        raise ValueError("endpoint requires exactly one run identity")
    expected_groups = {(step, name) for step in expected_steps for name in EVALUATION_NAMES}
    if set(groups) != expected_groups:
        raise ValueError("evaluation is missing a step or a repeat")
    evaluations = []
    for step in expected_steps:
        for name in EVALUATION_NAMES:
            by_member = groups[(step, name)]
            if set(by_member) != {member["ordinal"] for member in members}:
                raise ValueError("evaluation pass has missing or extra held-out members")
            rows = [by_member[index] for index in sorted(by_member)]
            scopes = {"primary": [row for row in rows if row["ordinal"] in primary]}
            for source in sorted({row["source"] for row in rows}):
                scopes[source] = [row for row in rows if row["source"] == source]
            for scope, values in scopes.items():
                correct = sum(row["completed_correct"] for row in values)
                evaluations.append(
                    {
                        "step": step,
                        "evaluation_name": name,
                        "scope": scope,
                        "members": len(values),
                        "completed": sum(row["completed"] for row in values),
                        "completed_correct": correct,
                        "completed_correct_fraction": correct / len(values),
                        "response_tokens_mean": statistics.mean(row["response_tokens"] for row in values),
                        "response_tokens_max": max(row["response_tokens"] for row in values),
                    }
                )
    return {
        "run_id": next(iter(run_ids)),
        "membership_sha256": membership["membership_sha256"],
        "primary_membership_sha256": membership["primary_membership_sha256"],
        "evaluations": evaluations,
        "member_results": [
            {"step": step, "evaluation_name": name, **row}
            for (step, name), rows in groups.items()
            for row in rows.values()
        ],
    }


def read_archive(root: str) -> tuple[list[dict], list[dict]]:
    fs, path = fsspec.core.url_to_fs(root)
    files = sorted(fs.glob(f"{path.rstrip('/')}/schema_v6/archives/phase=eval/step=*/*.zip"))
    if not files:
        raise ValueError("evaluation archive has no schema-six evaluation records")
    records, inputs = [], []
    for filename in files:
        data = fs.cat_file(filename)
        digest = hashlib.sha256(data).hexdigest()
        if Path(filename).stem != digest:
            raise ValueError("content-addressed evaluation archive has the wrong byte hash")
        inputs.append({"uri": fs.unstrip_protocol(filename), "sha256": digest})
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
            for entry in manifest["records"]:
                payload = bundle.read(entry["entry"])
                if len(payload) != entry["bytes"]:
                    raise ValueError("retained record byte count disagrees with its manifest")
                record = json.loads(gzip.decompress(payload))
                if record["record_id"] != entry["record_id"]:
                    raise ValueError("retained record identity disagrees with its manifest")
                records.append(record)
    return records, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Keep full exact-token evidence in immutable source archives and hash member results in the summary",
    )
    args = parser.parse_args()
    configure_coreweave_s3()
    records, inputs = read_archive(args.archive)
    result = audit_evaluations(records, load_membership(args.membership), args.steps)
    result.update({"archive": args.archive, "inputs": inputs})
    if args.compact:
        member_results = result.pop("member_results")
        result.update(member_result_count=len(member_results), member_results_sha256=canonical_hash(member_results))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([row for row in result["evaluations"] if row["scope"] == "primary"]))


if __name__ == "__main__":
    main()
