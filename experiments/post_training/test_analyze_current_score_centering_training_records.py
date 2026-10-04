# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from experiments.post_training.analyze_current_score_centering_training_records import TrainingRecordAudit


def record(identity, uid="row1", tokens=None):
    return {
        "schema_version": 6,
        "phase": "train",
        "run_id": "run",
        "record_id": identity,
        "trajectory": {"instance_id": uid},
        "prompt": {"token_ids": [11, 12]},
        "response": {"token_ids": tokens or [5, 6], "loss_mask": [1, 0], "stop_reason": "eos"},
        "disposition": {"exception_type": None, "error_treatment": None, "server_error": None},
    }


def consumed(count=1):
    return [
        {
            "training_step": 1,
            "rows": [{"uid": "row1", "response_token_ids": [5, 6], "loss_mask": [True, False]}] * count,
        }
    ]


def test_exact_multiset_requires_every_repeated_consumed_response():
    audit = TrainingRecordAudit("run")
    audit.retain(record("a"))
    with pytest.raises(ValueError, match="1 consumed responses"):
        audit.consume(consumed(2), [1])
    audit.retain(record("b"))
    audit.retain(record("unused", uid="row2", tokens=[7, 8]))
    result = audit.consume(consumed(2), [1])
    assert result["consumed_records"] == 2
    assert result["consumed_loss_tokens"] == 2
    assert result["completed_unconsumed_records"] == 1
    assert result["completed_unconsumed_response_tokens"] == 2


@pytest.mark.parametrize("field,value", [("uid", "row2"), ("response_token_ids", [6, 5]), ("loss_mask", [0, 1])])
def test_uid_tokens_and_mask_each_matter(field, value):
    audit = TrainingRecordAudit("run")
    audit.retain(record("a"))
    rows = consumed()
    rows[0]["rows"][0][field] = value
    with pytest.raises(ValueError, match="lack exact retained"):
        audit.consume(rows, [1])


def test_duplicate_identity_changed_prompt_and_incomplete_steps_rejected():
    audit = TrainingRecordAudit("run")
    original = record("a")
    audit.retain(original)
    with pytest.raises(ValueError, match="duplicate immutable"):
        audit.retain(original)
    changed = deepcopy(original)
    changed.update(record_id="b", prompt={"token_ids": [99]})
    with pytest.raises(ValueError, match="different exact prompt"):
        audit.retain(changed)
    with pytest.raises(ValueError, match="missing steps"):
        audit.consume(consumed(), [1, 2])
