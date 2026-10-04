# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.collect_current_score_centering_confirmation import parse_metrics


def test_source_ansi_log_keeps_observed_timestamp_and_finite_update_payload():
    raw = (
        "ordinary setup line\n\x1b[32m2026-10-04 23:15:12.123\x1b[0m | INFO | "
        'WANDB_MIRROR kind=train step=2 metrics={"policy/raw_grad_norm":0.17,"policy/policy_update_steps":1}'
        "\x1b[0m\n"
    )
    rows = parse_metrics(raw)
    assert rows == [
        {
            "policy/raw_grad_norm": 0.17,
            "policy/policy_update_steps": 1,
            "kind": "train",
            "step": 2,
            "timestamp_utc": "2026-10-04 23:15:12.123+00:00",
        }
    ]


def test_metrics_without_source_time_cannot_supply_elapsed_evidence():
    with pytest.raises(ValueError, match="observed UTC timestamp"):
        parse_metrics('WANDB_MIRROR kind=train step=2 metrics={"policy/raw_grad_norm":0.17}')
