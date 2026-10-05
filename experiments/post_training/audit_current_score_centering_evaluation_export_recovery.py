# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit original evaluation exports when native baseline records were not published.

This writes derived quality evidence. It never creates native trajectory records
or response token IDs. The fixed case has 768 missing first-pass baseline records;
all other records and all post-training evaluation passes retain native tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from importlib.metadata import distribution
from pathlib import Path

import fsspec
import yaml
from rigging.filesystem.s3_compat import configure_coreweave_s3
from skyrl_train.trajectory_runners.trajectory_reward_shaping import NormalizedReward
from transformers import AutoTokenizer

from experiments.post_training.analyze_current_score_centering_evaluations import (
    ACCEPTED_STOPS,
    audit_evaluations,
    canonical_hash,
    load_membership,
    read_archive,
)


def audit_exports(records: list[dict], membership: dict, config: dict, proof: dict, fs) -> dict:
    """Verify same-run exports against every native control and the frozen endpoint."""
    run_id = config["run"]["id"]
    if proof["run_id"] != run_id or proof["missing_native_records_backed_by_original_exports"] != 768:
        raise ValueError("export recovery differs from the frozen operational case")
    source = json.loads(distribution("marinskyrl").read_text("direct_url.json"))
    if source.get("vcs_info", {}).get("commit_id") != "5f53efd1300ac41fcff0811db56842dacb4c735d":
        raise ValueError("export recovery requires the original runtime source")
    generator = config["skyrl"]["generator"]
    shaping = generator.get("reward_shaping") or {}
    if shaping.get("enabled", False) or generator["eval_sampling_params"]["temperature"] != 0.0:
        raise ValueError("original exports require unshaped greedy evaluation")
    tokenizer = AutoTokenizer.from_pretrained(
        config["inputs"]["model"]["tokenizer_uri"],
        revision=config["inputs"]["model"]["tokenizer_revision"],
        trust_remote_code=True,
    )
    lookup = {tokenizer.decode(member["rendered_prompt_token_ids"]): member for member in membership["members"]}
    if len(lookup) != 1199:
        raise ValueError("decoded frozen prompts are ambiguous")
    native = {}
    for record in records:
        member = lookup[tokenizer.decode(record["prompt"]["token_ids"])]
        response, provenance = record["response"], record["provenance"]
        if (
            record["schema_version"] != 6
            or record["phase"] != "eval"
            or record["run_id"] != run_id
            or record["prompt"]["token_ids"] != member["rendered_prompt_token_ids"]
            or provenance["model_version_step"] != record["global_step"]
            or provenance["sampling"]["temperature"] != 0.0
            or record["trajectory"]["repetition_id"] != 0
            or response["generation_limit"] != 4096
            or len(response["token_ids"]) != len(response["loss_mask"])
            or len(response["token_ids"]) > 4096
        ):
            raise ValueError("available native evaluation evidence changed")
        key = record["global_step"], record["evaluation_name"], member["ordinal"]
        if key in native:
            raise ValueError("native evaluation repeats a member")
        native[key] = record
    groups = defaultdict(dict)
    missing = set()
    expected_missing = {(0, None, ordinal) for ordinal in range(256, 1024)}
    export_root = config["artifacts"]["export_root"]
    for item in proof["original_export_inputs"]:
        step, name = item["step"], item["evaluation_name"]
        prefix = export_root + ("/" + name if name else "") + f"/dumped_evals/global_step_{step}_evals/"
        if step not in (0, 10, 20, 30, 40) or name not in (None, "greedy_repeat"):
            raise ValueError("export names a different evaluation")
        if not item["uri"].startswith(prefix) or not item["uri"].endswith(".jsonl"):
            raise ValueError("export is outside the original evaluation directory")
        raw = fs.cat_file(item["uri"])
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("original export bytes changed")
        # Split bytes on actual JSONL line endings; decoded answers can contain
        # Unicode line separators inside a valid JSON string.
        for line_number, line in enumerate(raw.splitlines()):
            row = json.loads(line)
            member = lookup[row["input_prompt"]]
            ordinal, content = member["ordinal"], member["content"]
            if (
                row["env_class"] != content["env_class"]
                or row["env_extras"]["reward_spec"] != content["reward_spec"]
                or row["data_source"] != content["extra_info"]["data_source"]
                or not isinstance(row["score"], list)
                or len(row["score"]) > 4096
            ):
                raise ValueError("export membership, grading rule or token limit changed")
            outcome = NormalizedReward.from_output(row["score"]).outcome
            if not math.isfinite(outcome):
                raise ValueError("export has a nonfinite outcome")
            flags = {key: row[key] for key in ("exception_type", "error_treatment", "server_error")}
            key = step, name, ordinal
            record = native.get(key)
            if record is not None:
                trajectory = record["trajectory"]
                if (
                    record["disposition"] != flags
                    or record["response"]["stop_reason"] != row["stop_reason"]
                    or tokenizer.decode(record["response"]["token_ids"]) != row["output_response"]
                    or len(record["response"]["token_ids"]) != len(row["score"])
                    or float(record["reward"]["outcome"]) != outcome
                    or trajectory["environment_class"] != row["env_class"]
                    or trajectory["environment_extras"]["reward_spec"] != row["env_extras"]["reward_spec"]
                    or trajectory["environment_extras"]["data_source"] != row["data_source"]
                ):
                    raise ValueError("original export disagrees with native quality evidence")
                record_id = record["record_id"]
            else:
                if key not in expected_missing:
                    raise ValueError("missing native record is outside the fixed baseline case")
                missing.add(key)
                record_id = f"original-eval-export:{item['sha256']}:line:{line_number}"
            completed = row["stop_reason"] in ACCEPTED_STOPS and not any(flags.values())
            if ordinal in groups[(step, name)]:
                raise ValueError("export pass repeats a member")
            groups[(step, name)][ordinal] = {
                "ordinal": ordinal,
                "member_sha256": member["member_sha256"],
                "source": row["data_source"],
                "completed": completed,
                "completed_correct": completed and outcome > 0,
                "native_outcome": outcome,
                "response_tokens": len(row["score"]),
                "stop_reason": row["stop_reason"],
                "record_id": record_id,
            }
    if len(native) != 11222 or missing != expected_missing or len(groups) != 10:
        raise ValueError("export recovery lacks the fixed native controls or missing members")
    primary = set(membership["primary_member_ordinals"])
    evaluations = []
    member_results = []
    for step in (0, 10, 20, 30, 40):
        for name in (None, "greedy_repeat"):
            by_member = groups[(step, name)]
            if set(by_member) != {member["ordinal"] for member in membership["members"]}:
                raise ValueError("export pass has missing or extra frozen members")
            rows = [by_member[ordinal] for ordinal in sorted(by_member)]
            scopes = {"primary": [row for row in rows if row["ordinal"] in primary]}
            for scope in sorted({row["source"] for row in rows}):
                scopes[scope] = [row for row in rows if row["source"] == scope]
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
            member_results.extend({"step": step, "evaluation_name": name, **row} for row in rows)
    original = audit_evaluations([row for row in records if row["global_step"] > 0], membership, [10, 20, 30, 40])
    if original["evaluations"] != [row for row in evaluations if row["step"] > 0]:
        raise ValueError("derived aggregation differs from the unchanged frozen auditor")
    return {
        "run_id": run_id,
        "membership_sha256": membership["membership_sha256"],
        "primary_membership_sha256": membership["primary_membership_sha256"],
        "evaluations": evaluations,
        "member_results": member_results,
        "recovery": {
            "native_records_verified": len(native),
            "baseline_quality_records_from_original_exports": len(missing),
            "missing_raw_response_token_id_records": len(missing),
            "response_lengths_source": (
                "Original token reward array lengths; equal to token counts in every native control."
            ),
            "complete_native_evaluation_transport_evidence": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.configuration.read_bytes())
    configure_coreweave_s3()
    records, inputs = read_archive(config["artifacts"]["attempts_root"] + "/trajectories")
    result = audit_exports(
        records, load_membership(args.membership), config, json.loads(args.proof.read_bytes()), fsspec.filesystem("s3")
    )
    members = result.pop("member_results")
    result.update(inputs=inputs, member_result_count=len(members), member_results_sha256=canonical_hash(members))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["recovery"]))


if __name__ == "__main__":
    main()
