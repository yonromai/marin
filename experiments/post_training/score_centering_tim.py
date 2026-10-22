# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite SC runtime and TIM cost checks; measurement worktree only."""

from dataclasses import replace

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)

from experiments.models import qwen3_0_6b
from experiments.post_training.cat_count_canary.data import TRAIN_FILENAME, VALIDATION_FILENAME, cat_count_data_step
from experiments.post_training.cat_count_canary.launcher import role_plan, training_config

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def startup(version: str, *, probes: bool = False) -> ArtifactStep:
    data = cat_count_data_step(
        user_owned_name("documents/score-centering-tim-cat-count"),
        version,
        train_ns=(4, 6, 8, 12),
        train_rows=64,
        seed=17,
    )
    plan = replace(role_plan(batch_size=4, group_size=4, micro_train_batch_size=1), colocate_all=True)
    config = training_config(
        preset="dry",
        lane="sync",
        model="qwen3-0.6b",
        batch_size=4,
        group_size=4,
        micro_train_batch_size=1,
        train_ns=(4, 6, 8, 12),
        seed=17,
        checkpoint=False,
        export=False,
        settings=(
            f"trainer.max_steps={4 if probes else 2}",
            "trainer.update_epochs_per_batch=1",
            "trainer.logger=console",
        ),
    )
    _materialize_role_plan_config(config, plan)
    config["context_budget"] = {"request_window_tokens": 512, "max_new_tokens_per_turn": 128, "max_turns": 1}
    trainer = config["trainer"]
    trainer.update(
        callbacks=[],
        eval_before_train=False,
        eval_interval=-1,
        ckpt_interval=-1,
        hf_save_interval=-1,
        flash_attn=True,
        offload_optimizer_during_rollouts=False,
    )
    trainer["algorithm"].update(
        advantage_estimator="grpo" if probes else "uniform",
        score_centering_topk=0 if probes else 128,
        score_centering_enabled=not probes,
        score_centering_diagnostic_tokens=0 if probes else 32,
    )
    trainer["algorithm"].setdefault("score_centering_probe", {})["interval"] = 2 if probes else 10
    config["generator"].update(
        gpu_memory_utilization=0.35, max_num_seqs=16, max_num_batched_tokens=512, enable_http_endpoint=False
    )
    config["generator"]["sampling_params"].update(
        logprobs=0 if probes else 128, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0
    )
    config["trajectory_runner"] = {"rollout_workers": {"num_workers": 2, "cpus_per_worker": 2}}
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"checkpoints/score-centering-tim/{'probe-startup' if probes else 'startup'}"),
            version=version,
            config_yaml=yaml.safe_dump(config),
            model=ArtifactHfModel(step=qwen3_0_6b, relative_path="", tokenizer_uri=MODEL, tokenizer_revision=REVISION),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=2, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=17,
        ),
        IrisSkyRLExecution(
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            target_cluster="cw-rno2a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            cpu=32,
            memory="128GB",
            disk="200GB",
            priority="interactive",
            max_retries=0,
            coordinator_timeout_hours=2,
            job_timeout_seconds=2700,
        ),
        export_hf=False,
    )


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(["startup", "probe-startup"]), default="startup")
@rl_build_options
def main(stage: str) -> ArtifactStep:
    return startup(resolve_version("score-centering-tim", None), probes=stage == "probe-startup")


if __name__ == "__main__":
    main()
