# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect final reference evidence and analyze paired training-seed endpoints."""

import argparse
import gzip
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from botocore.config import Config

mpl.use("Agg")
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
AUTHOR_SOURCE = "7c56e9ee2972aa57f446cf564de1a1658d14b321"
BUCKET = "marin-us-east-02a"
PREFIX = "marin/users/romain/score-centering-overnight-01a123c0/reference/"


def collect(client, run_id: str, cache: Path) -> Path:
    prefix = PREFIX + run_id + "/"
    root = cache / run_id
    objects = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    selected = []
    for item in objects:
        name = item["Key"].removeprefix(prefix)
        retain = name.endswith(".json") or (name.startswith("samples/") and "-eval-" in name)
        if retain:
            selected.append((item, root / name))

    def download(item_and_path):
        item, destination = item_and_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.stat().st_size != item["Size"]:
            client.download_file(BUCKET, item["Key"], str(destination))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download, selected))
    root.mkdir(parents=True, exist_ok=True)
    (root / "object_inventory.json").write_text(
        json.dumps([{k: value[k] for k in ("Key", "Size", "ETag")} for value in objects], indent=2) + "\n"
    )
    return root


def audit(root: Path, noise: float, seed: int) -> dict:
    metadata = json.loads((root / "metadata.json").read_text())
    terminal = json.loads((root / "terminal.json").read_text())
    assert metadata["source"] == AUTHOR_SOURCE and not metadata["source_dirty"]
    assert metadata["model_revision"] == MODEL_REVISION
    assert terminal["status"] == "succeeded" and terminal["completed_training_batches"] == 300
    metrics = [json.loads((root / "metrics" / f"{step:04d}.json").read_text()) for step in range(300)]
    assert [m["step"] for m in metrics] == list(range(300))
    assert all(math.isfinite(m[k]) for m in metrics for k in ("loss", "grad_norm", "update_norm", "weight_norm"))
    cfg_path = root / "score-centering-reproduction" / root.name / "config.json"
    cfg = json.loads(cfg_path.read_text())
    assert cfg["run"]["seed"] == seed
    assert cfg["model"]["source"] == "Qwen/Qwen3-0.6B"
    assert (cfg["rl"]["num_prompts"], cfg["rl"]["group_size"], cfg["rl"]["seq_len"]) == (64, 8, 512)
    assert cfg["rl"]["minibatches"] == 1 and cfg["rl"]["reward_mode"] == "group_centered"
    assert cfg["rl"]["is"]["high"] == 2 and cfg["rl"]["is"]["outside"] == "clamp"
    assert cfg["sampler"]["weight_noise_scale"] == noise and cfg["sampler"]["staleness"] == 0
    assert cfg["sampler"]["vocab_logprobs"] == 128
    assert cfg["opt"]["optimizer"] == "sgd" and cfg["opt"]["lr"] == 0.01 and cfg["opt"]["lr_warmup"] == 0
    assert cfg["stop"]["steps"] == 300 and cfg["stop"]["collapse"] == 0
    assert cfg["eval"]["num_prompts"] == 64 and cfg["eval"]["every_steps"] == 20
    assert cfg["env"]["chat_template_kwargs"]["enable_thinking"] is False
    inventory = json.loads((root / "object_inventory.json").read_text())
    assert any(
        value["Key"].endswith("final_model/model.safetensors") and value["Size"] > 1000000000 for value in inventory
    )
    assert sum("-train-" in value["Key"] and value["Key"].endswith(".npz") for value in inventory) == 300
    points, membership = [], None
    for path in sorted((root / "samples").glob("*-eval-*.json.gz")):
        with gzip.open(path, "rt") as stream:
            record = json.load(stream)
        rows = record["records"]
        assert len(rows) == 512
        current_membership = hashlib.sha256(
            json.dumps([{k: r[k] for k in ("prompt", "info", "example_id")} for r in rows], sort_keys=True).encode()
        ).hexdigest()
        if membership is None:
            membership = current_membership
        assert membership == current_membership
        tokens = np.load(path.with_suffix("").with_suffix(".npz"))
        assert tokens["tokens"].shape == (512, 512)
        truncated = missing_answer = 0
        for index, row in enumerate(rows):
            prompt = int(tokens["prompt_lengths"][index])
            response = tokens["tokens"][index, prompt:]
            truncated += not any(int(token) in (151643, 151645) for token in response)
            missing_answer += "<answer>" not in row["text"] or "</answer>" not in row["text"]
            assert row["reward"] in (0, 1)
        rate = float(np.mean([row["reward"] for row in rows]))
        step = record["completed_updates"]
        np.testing.assert_allclose(rate, metrics[step]["eval_reward"], atol=0, rtol=0)
        points.append(
            {
                "completed_updates": step,
                "quality": rate,
                "responses": 512,
                "truncated": int(truncated),
                "missing_answer_tags": int(missing_answer),
                "infrastructure_errors": 0,
            }
        )
    assert [p["completed_updates"] for p in points] == [*range(0, 300, 20), 299]
    return {
        "run_id": root.name,
        "seed": seed,
        "noise": noise,
        "score_centering": cfg["rl"]["sc"],
        "endpoint": points[-1]["quality"],
        "initial_quality": points[0]["quality"],
        "eval_points": points,
        "membership_sha256": membership,
        "consumed_loss_tokens": round(sum(m["mean_completion_len"] * 512 for m in metrics)),
        "environment_lock_sha256": metadata["environment_lock_sha256"],
        "artifacts": terminal["artifact_uri"],
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["CW_S3_ENDPOINT"],
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"}),
    )
    runs = []
    for noise, seeds in ((0.05, (0, 1, 2)), (0.0, (0,))):
        label = "noise005" if noise else "clean"
        for seed in seeds:
            for arm in ("tis", "sc"):
                run_id = f"sc-ref-main-{label}-{arm}-s{seed}-a0-01a123c0"
                root = collect(client, run_id, args.cache)
                runs.append(audit(root, noise, seed))
    assert len({run["environment_lock_sha256"] for run in runs}) == 1
    pairs = []
    for seed in (0, 1, 2):
        a, b = [run for run in runs if run["seed"] == seed and run["noise"] == 0.05]
        assert a["membership_sha256"] == b["membership_sha256"]
        pairs.append(
            {"seed": seed, "tis": a["endpoint"], "sc": b["endpoint"], "difference": b["endpoint"] - a["endpoint"]}
        )
    differences = np.array([p["difference"] for p in pairs])
    # For df=2, F(t)=1/2+t/(2*sqrt(t*t+2)); invert at0.975 for three paired seeds.
    quantile = math.sqrt(2 * 0.95**2 / (1 - 0.95**2))
    half_width = float(quantile * differences.std(ddof=1) / np.sqrt(3))
    result = {
        "replication_unit": "training_seed",
        "endpoint": "clean held-out evaluation at299completed updates, before update300",
        "pairs": pairs,
        "mean_difference": float(differences.mean()),
        "paired_t_95_interval": [float(differences.mean() - half_width), float(differences.mean() + half_width)],
        "runs": [{k: v for k, v in run.items() if k != "metrics"} for run in runs],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reference_results.json").write_text(json.dumps(result, indent=2) + "\n")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for run in runs:
        color = "#007b83" if run["score_centering"] else "#c36c44"
        label = f"{'TIS+SC' if run['score_centering'] else 'TIS'} seed{run['seed']}"
        panel = axes[0, 0] if run["noise"] else axes[0, 1]
        panel.plot(
            [p["completed_updates"] for p in run["eval_points"]],
            [100 * p["quality"] for p in run["eval_points"]],
            color=color,
            alpha=0.7,
            label=label,
        )
        if run["noise"]:
            axes[1, 0].plot(
                [m["step"] + 1 for m in run["metrics"]],
                [m["grad_norm"] for m in run["metrics"]],
                color=color,
                alpha=0.65,
            )
            axes[1, 1].plot(
                [m["step"] + 1 for m in run["metrics"]],
                [1 - m["head_mass"] for m in run["metrics"]],
                color=color,
                alpha=0.65,
            )
    for panel, title in zip(
        axes.flat,
        ["Noise0.05: all three seeds", "No-noise sanity pair", "Gradient norm", "Omitted behavior mass"],
        strict=True,
    ):
        panel.set_title(title)
        panel.set_xlabel("Completed optimizer updates")
        panel.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Clean held-out correct (%)")
    axes[0, 1].set_ylabel("Clean held-out correct (%)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output / "reference_quality_stability.png", dpi=170)
    figure.savefig(args.output / "reference_quality_stability.svg")
    plt.close(figure)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
