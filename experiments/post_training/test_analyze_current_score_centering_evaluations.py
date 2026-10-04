# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from experiments.post_training.analyze_current_score_centering_evaluations import audit_evaluations, canonical_hash


def evidence():
    members = []
    records = []
    for ordinal, source in enumerate(["val-gsm8k", "val-math500", "val-secondary"]):
        content = {
            "env_class": "gsm8k" if ordinal == 0 else "aime",
            "reward_spec": {"ground_truth": str(ordinal), "method": "rule"},
            "extra_info": {"data_source": source},
        }
        members.append(
            {
                "ordinal": ordinal,
                "member_sha256": canonical_hash(content),
                "content": content,
                "rendered_prompt_token_ids": [ordinal + 10],
            }
        )
        for name in [None, "greedy_repeat"]:
            records.append(
                {
                    "record_id": f"{ordinal}:{name}",
                    "schema_version": 6,
                    "phase": "eval",
                    "run_id": "same-run",
                    "global_step": 0,
                    "evaluation_name": name,
                    "prompt": {"token_ids": [ordinal + 10]},
                    "trajectory": {
                        "environment_class": content["env_class"],
                        "environment_extras": {"reward_spec": content["reward_spec"], "data_source": source},
                        "repetition_id": 0,
                    },
                    "provenance": {"model_version_step": 0, "sampling": {"temperature": 0.0}},
                    "response": {
                        "generation_limit": 4096,
                        "token_ids": [20, 21],
                        "loss_mask": [1, 1],
                        "stop_reason": "length" if ordinal == 0 and name is None else "stop",
                    },
                    "reward": {"outcome": 5 if ordinal == 2 else -1 if ordinal == 1 and name is None else 1},
                    "disposition": {"exception_type": None, "error_treatment": None, "server_error": None},
                }
            )
    membership = {
        "members": members,
        "membership_sha256": canonical_hash(members),
        "primary_member_ordinals": [0, 1],
        "primary_member_count": 2,
        "primary_membership_sha256": canonical_hash(members[:2]),
    }
    return records, membership


def test_completed_correct_excludes_truncation_and_secondary_reward_scale():
    records, membership = evidence()
    result = audit_evaluations(records, membership, [0])
    primary = [row for row in result["evaluations"] if row["scope"] == "primary"]
    assert [row["completed_correct"] for row in primary] == [0, 2]
    assert [row["members"] for row in primary] == [2, 2]
    assert [row["completed"] for row in primary] == [1, 2]


@pytest.mark.parametrize("correction", ["tokens", "weights", "cap", "missing_repeat", "duplicate", "grading"])
def test_endpoint_refuses_changed_contract_or_incomplete_evidence(correction):
    records, membership = evidence()
    records = copy.deepcopy(records)
    if correction == "tokens":
        records[0]["prompt"]["token_ids"] = [99]
    elif correction == "weights":
        records[0]["provenance"]["model_version_step"] = 1
    elif correction == "cap":
        records[0]["response"]["generation_limit"] = 1536
    elif correction == "missing_repeat":
        records = [record for record in records if record["evaluation_name"] is None]
    elif correction == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif correction == "grading":
        records[0]["trajectory"]["environment_extras"]["reward_spec"]["ground_truth"] = "changed"
    with pytest.raises(ValueError):
        audit_evaluations(records, membership, [0])
