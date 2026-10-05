# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json

import pytest

from experiments.post_training.run_current_score_centering_confirmation import run


def test_misdeclared_identity_is_rejected_before_controller_or_submission(tmp_path, monkeypatch):
    config = tmp_path / "configs/input.yaml"
    config.parent.mkdir()
    config.write_text("run:\n  id: actual-id\niris:\n  job_name: actual-id\n")
    protocol = tmp_path / "results/protocol.json"
    protocol.parent.mkdir()
    protocol.write_text(
        json.dumps(
            {
                "status": "frozen",
                "configuration_manifest": [
                    {
                        "arm": "older_tis32",
                        "seed": 1,
                        "run_id": "declared-id",
                        "path": "configs/input.yaml",
                        "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    }
                ],
                "launch_order": [{"arm": "older_tis32", "seed": 1}],
            }
        )
    )

    def unexpected_network(*args, **kwargs):
        raise AssertionError("identity rejection must precede any controller call")

    monkeypatch.setattr(
        "experiments.post_training.run_current_score_centering_confirmation.cli_json", unexpected_network
    )
    args = argparse.Namespace(
        protocol=protocol, protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(), max_active=6
    )
    with pytest.raises(ValueError, match="registry differs"):
        run(args)
