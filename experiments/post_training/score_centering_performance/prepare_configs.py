# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the retained readiness identities to finite performance qualifications."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml


def relocated(value, old_prefix, prefix):
    if isinstance(value, dict):
        return {key: relocated(item, old_prefix, prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [relocated(item, old_prefix, prefix) for item in value]
    return value.replace(old_prefix, prefix) if isinstance(value, str) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-runtime", required=True)
    parser.add_argument("--baseline-runtime", required=True)
    parser.add_argument("--candidate-product", required=True)
    parser.add_argument("--qwen-attempt", choices=("a0", "a1", "a2"), default="a0")
    parser.add_argument("--qwen-cluster", choices=("cw-us-east-02a", "cw-rno2a"), default="cw-rno2a")
    parser.add_argument("--workload", choices=("qwen", "snowball"))
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument("--snowball-cluster", choices=("cw-us-east-02a", "cw-rno2a"), default="cw-us-east-02a")
    parser.add_argument("--snowball-attempt", choices=("a0", "a1", "a2", "a3"), default="a0")
    parser.add_argument("--snowball-selection", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for cluster in ("cw-us-east-02a", "cw-rno2a"):
        source = yaml.safe_load(
            (args.reference_dir.parent / "score_centering_overnight" / f"{cluster}-client.yaml").read_text()
        )
        client = {
            key: source[key]
            for key in ("name", "dashboard_url", "platform", "storage", "controller", "kubernetes_provider")
        }
        client["defaults"] = {key: source["defaults"][key] for key in ("worker", "task_env")}
        (root / f"{cluster}-client.yaml").write_text(
            "# Client connection facts only; this file is not a deployment configuration.\n"
            + yaml.safe_dump(client, sort_keys=False)
        )
    snowball_placement = "-rno" if args.snowball_cluster == "cw-rno2a" else ""
    qwen_placement = "-east" if args.qwen_cluster == "cw-us-east-02a" else ""
    phases = (
        ("qwen", "candidate", f"qwen-{args.qwen_attempt}{qwen_placement}", args.candidate_runtime, args.qwen_cluster, 2),
        (
            "snowball",
            "baseline",
            f"snowball-baseline{snowball_placement}-{args.snowball_attempt}",
            args.baseline_runtime,
            args.snowball_cluster,
            6,
        ),
        (
            "snowball",
            "candidate",
            f"snowball-candidate{snowball_placement}-{args.snowball_attempt}",
            args.candidate_runtime,
            args.snowball_cluster,
            6,
        ),
    )
    for workload, variant, suffix, runtime, cluster, repetitions in phases:
        if args.workload is not None and args.workload != workload:
            continue
        if args.variant is not None and args.variant != variant:
            continue
        source = yaml.safe_load((args.reference_dir / f"{workload}-prepare.yaml").read_text())
        old_prefix = source["ray"]["log_dir"].removesuffix("/ray-logs")
        run_id = f"sc-performance-{suffix}-01a146a3"
        prefix = f"s3://marin-us-east-02a/marin/users/romain/score-centering-performance-01a146a3/{run_id}"

        config = relocated(source, old_prefix, prefix)
        config["run"]["id"] = run_id
        config["runtime"]["launcher_commit"] = runtime
        config["iris"].update(cluster=cluster, cluster_config=str(root / f"{cluster}-client.yaml"), job_name=run_id)
        trainer = config["skyrl"]["trainer"]
        trainer["max_steps"] = 128
        trainer.pop("score_centering_readiness")
        trainer["score_centering_performance"] = {
            "workload": workload,
            "variant": variant,
            "output_uri": f"{prefix}/performance.json",
            "candidate": (
                args.candidate_product if variant == "candidate" else "325f819f94c3ed14a550ef5bc502fb6ff2d5b5d8"
            ),
            "repetitions": repetitions,
        }
        if workload == "snowball" and args.snowball_selection is not None:
            selection = json.loads(args.snowball_selection.read_text())
            trainer["score_centering_performance"].update(
                source_rows=selection["campaign_source_rows"],
                source_data_sha256=selection["source_sha256"],
                window_timeout_seconds=600,
            )
        if variant == "candidate":
            trainer["algorithm"]["score_centering_topk"] = 0
            trainer["algorithm"]["score_centering_probe"] = {"interval": 2, "topk": 128, "tokens_per_response": 32}
            config["skyrl"]["generator"]["sampling_params"]["logprobs"] = 0
        config["iris"]["max_retries"] = 0
        if workload == "qwen" and args.qwen_attempt != "a0":
            config["iris"]["priority"] = "batch"
        config["iris"]["timeout"] = 3600 if workload == "qwen" else 21600
        config["run"]["submission"] = "prepare"
        (root / f"{suffix}-prepare.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        detached = deepcopy(config)
        detached["run"]["submission"] = "detach"
        (root / f"{suffix}-detach.yaml").write_text(yaml.safe_dump(detached, sort_keys=False))


if __name__ == "__main__":
    main()
