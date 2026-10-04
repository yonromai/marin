from collections import Counter

import pytest

from experiments.post_training.analyze_current_policy_versions import age_summary, summarize


def test_optimizer_ages_count_skipped_updates_as_zero_and_preserve_mixed_weights():
    records = [
        {
            "schema_version": 1,
            "training_step": step,
            "consuming_policy_version": step - 1,
            "optimizer_updates_applied": applied,
            "loss_token_publication_gap_counts": {},
            "rows": [],
        }
        for step, applied in ((1, 2), (2, 0), (3, 3), (4, 1))
    ]
    records[-1].update(
        loss_token_publication_gap_counts={"3": 1, "2": 2, "0": 1},
        rows=[
            {
                "uid": "resumed-response",
                "response_token_ids": [10, 11, 12, 13, 14],
                "loss_mask": [1, 1, 1, 0, 1],
                "policy_version_spans": [
                    {"start": 0, "end": 1, "version": 0},
                    {"start": 1, "end": 3, "version": 1},
                    {"start": 3, "end": 4, "version": -1},
                    {"start": 4, "end": 5, "version": 3},
                ],
            }
        ],
    )
    result = summarize(records)
    # At consuming step3, version0 is five optimizer updates old, version1 is
    # three old (step2 was skipped), and version3 is fresh. Masked assembly does not count.
    assert result["aggregate"]["optimizer_age_counts"] == {0: 1, 3: 2, 5: 1}
    assert result["aggregate"]["optimizer_age_mean"] == 2.75
    assert result["aggregate"]["mixed_policy_response_fraction"] == 1.0
    assert age_summary(Counter({0: 1, 3: 2, 5: 1}))["optimizer_age_p50"] == 3
    with pytest.raises(ValueError, match="missing earlier"):
        summarize(records[1:])
