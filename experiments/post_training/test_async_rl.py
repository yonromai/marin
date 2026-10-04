# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from experiments.post_training import async_rl
from experiments.post_training.curriculum_rl.launch import SNOWBALL_POLICY


def test_launcher_builds_complete_smoke_run(monkeypatch) -> None:
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")

    run = async_rl.build_run(SNOWBALL_POLICY, async_rl.SMOKE_PRESET, version="2026.09.18")

    assert run.rl.name == "users/alice/checkpoints/async-rl/snowball-smoke"
    assert any(dep.name == async_rl.POOL_ARTIFACT_NAME for dep in run.rl.deps)
    assert any(dep.name == SNOWBALL_POLICY.adopted_model.name for dep in run.rl.deps)
    assert run.evaluation.deps == (run.rl,)
    assert run.evaluation.name == "evals/alice-async-rl-snowball-smoke/gsm8k-smoke"


@pytest.mark.integration
def test_launcher_composes_unified_rollout_buffer_config(tmp_path, monkeypatch) -> None:
    from pathlib import Path
    from cloud.iris.launch_config import load_launch_config
    from marin.execution.lazy import StepContext

    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")

    run = async_rl.build_run(
        SNOWBALL_POLICY,
        async_rl.SMOKE_PRESET,
        version="2026.09.18",
        settings=("trainer.rollout_buffer.max_in_flight=null",),
    )
    launch = run.rl.build_config(StepContext.for_fingerprint(run.rl.runtime_args, run.rl.deps))
    path: Path = tmp_path / "launch.yaml"
    path.write_text(launch.launch_config_yaml)
    config = load_launch_config(path)
    assert config.skyrl.trainer.rollout_buffer.max_in_flight is None
    assert config.skyrl.trainer.rollout_buffer.batch_policy == "full_batch"
    assert config.runtime.entrypoint == "skyrl_train.entrypoints.main_base"
    assert config.skyrl.generator.weight_sync_pause.mode == "keep"
