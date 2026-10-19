# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind retained workload controls to finite integration qualifications."""

import argparse
from pathlib import Path

import yaml

from experiments.post_training.score_centering_performance.prepare_configs import relocated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("qwen", "snowball"), required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--cluster", choices=("cw-us-east-02a", "cw-rno2a"), required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--attempt", default="a0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    retained = root.parent / "score_centering_performance"
    short = args.workload == "qwen"
    for cluster in ("cw-us-east-02a", "cw-rno2a"):
        (root / f"{cluster}-client.yaml").write_bytes((retained / f"{cluster}-client.yaml").read_bytes())
    source = "qwen-a2-east-prepare.yaml" if short else "snowball-candidate-a3-prepare.yaml"
    cfg = yaml.safe_load((retained / source).read_text())
    placement = "rno" if args.cluster == "cw-rno2a" else "east"
    suffix = f"{args.workload}-{args.variant}-{args.attempt}-{placement}"
    run_id = f"sc-integration-{suffix}-01a146a3"
    prefix = f"s3://marin-us-east-02a/marin/users/romain/score-centering-integration-01a146a3/{run_id}"
    cfg = relocated(cfg, cfg["ray"]["log_dir"].removesuffix("/ray-logs"), prefix)
    cfg["run"]["id"] = run_id
    cfg["runtime"]["launcher_commit"] = args.runtime
    cfg["iris"].update(
        cluster=args.cluster,
        cluster_config=str(root / f"{args.cluster}-client.yaml"),
        job_name=run_id,
        priority="interactive" if short else "batch",
        max_retries=0,
        timeout=3600 if short else 10800,
    )
    trainer = cfg["skyrl"]["trainer"]
    trainer["max_steps"] = 128
    trainer["score_centering_performance"].update(
        candidate=args.product,
        variant=args.variant,
        repetitions=2 if short else 6,
        async_steps=4 if short else 12,
        async_warmup_steps=1 if short else 2,
        async_phase_timeout_seconds=600 if short else 2400,
    )
    if short:
        cfg["skyrl"]["context_budget"]["request_window_tokens"] = 8192
    for submission in ("prepare", "detach"):
        cfg["run"]["submission"] = submission
        path = root / f"{suffix}-{submission}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
