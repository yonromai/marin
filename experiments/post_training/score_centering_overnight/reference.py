# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run pinned authors' code with token evidence and durable artifact publication."""

import argparse
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
from botocore.config import Config
from jax.sharding import PartitionSpec as P


class ReferenceRecorder:
    def __init__(self, root: Path, artifact_uri: str, capture_rows: int):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        target = urlparse(artifact_uri)
        if target.scheme != "s3" or not target.netloc:
            raise ValueError("reference artifacts require an S3 run-owned URI")
        self.bucket = target.netloc
        self.prefix = target.path.strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL", os.environ.get("CW_S3_ENDPOINT")),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", os.environ.get("CW_KEY_ID")),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("CW_KEY_SECRET")),
            config=Config(s3={"addressing_style": "virtual"}),
        )
        self.capture_rows = capture_rows
        self.train_samples = 0
        self.sample_index = 0
        self.pending = None
        self.uploaded = {}

    def write_json(self, name: str, value) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def publish(self) -> None:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            name = str(path.relative_to(self.root))
            signature = (path.stat().st_mtime_ns, path.stat().st_size)
            if self.uploaded.get(name) == signature:
                continue
            self.client.upload_file(str(path), self.bucket, f"{self.prefix}/{name}")
            self.uploaded[name] = signature

    def wrap_generation(self, generate):
        def recorded(*args, **kwargs):
            started = time.monotonic()
            result = generate(*args, **kwargs)
            tokens, chosen_logprobs, _, head_logprobs, head_ids = result
            prompt_lengths = np.asarray(args[3])
            train = kwargs.get("logprobs_topk", 0) > 0
            phase = "train" if train else "eval"
            step = self.train_samples
            stem = f"samples/{self.sample_index:06d}-{phase}-update{step:04d}"
            (self.root / "samples").mkdir(exist_ok=True)
            payload = {
                "tokens": np.asarray(tokens),
                "prompt_lengths": prompt_lengths,
                "behavior_chosen_logprobs": np.asarray(chosen_logprobs),
            }
            if train:
                payload["evidence_row_indices"] = np.arange(min(len(tokens), self.capture_rows))
                payload["behavior_topk_ids"] = np.asarray(
                    head_ids.at[: self.capture_rows].get(out_sharding=P(None, None, None))
                )
                payload["behavior_topk_logprobs"] = np.asarray(
                    head_logprobs.at[: self.capture_rows].get(out_sharding=P(None, None, None))
                )
                response_scores = []
                for row, prompt in enumerate(prompt_lengths[: self.capture_rows]):
                    generated = payload["tokens"][row, prompt:]
                    stop_ids = {args[1].tokenizer.eos_token_id, args[1].tokenizer.pad_token_id}
                    length = next((i + 1 for i, token in enumerate(generated) if token in stop_ids), len(generated))
                    response_scores.append(payload["behavior_topk_logprobs"][row, prompt - 1 : prompt + length - 1])
                values = np.concatenate(response_scores)
                if not np.isfinite(values).all() or np.std(values) < 1e-6:
                    raise ValueError("reference behavior capture is nonfinite or uniform")
                self.train_samples += 1
            np.savez_compressed(self.root / f"{stem}.npz", **payload)
            self.pending = {
                "stem": stem,
                "phase": phase,
                "completed_updates": step,
                "generation_seconds": time.monotonic() - started,
            }
            self.sample_index += 1
            return result

        return recorded

    def wrap_score(self, score):
        def recorded(rubric, rows, texts, group_size):
            rewards = score(rubric, rows, texts, group_size)
            assert self.pending is not None
            records = [
                {
                    "prompt": row["prompt"],
                    "info": row.get("info", {}),
                    "answer": row.get("answer"),
                    "text": text,
                    "reward": float(reward),
                    "example_id": row["example_id"],
                }
                for row, text, reward in zip(rows, texts, rewards, strict=True)
            ]
            with gzip.open(self.root / f"{self.pending['stem']}.json.gz", "wt") as stream:
                json.dump({**self.pending, "records": records}, stream, allow_nan=False)
            self.pending = None
            return rewards

        return recorded

    def wrap_metrics(self, log_metrics):
        def recorded(step, metrics, run_dir=None):
            log_metrics(step, metrics, run_dir)
            clean = {name: np.asarray(value).item() for name, value in metrics.items()}
            self.write_json(f"metrics/{step:04d}.json", clean)
            self.publish()

        return recorded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--capture-rows", type=int, default=2)
    parser.add_argument("--run-id", required=True)
    args, overrides = parser.parse_known_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.85")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    # Recorders wrap I/O only; model, sampler, verifier, advantages, and loss stay in released code.
    sampling = importlib.import_module("models.sampling")
    models = importlib.import_module("models")
    model = importlib.import_module("models.model")
    rollout = importlib.import_module("tasks.rollout")
    utilities = importlib.import_module("utils")
    download = model.snapshot_download

    def pinned_download(*positional, **keywords):
        return download(*positional, **{**keywords, "revision": args.model_revision})

    model.snapshot_download = pinned_download
    config_download = models.hf_hub_download

    def pinned_config(*positional, **keywords):
        return config_download(*positional, **{**keywords, "revision": args.model_revision})

    models.hf_hub_download = pinned_config
    output = Path("/tmp/score-centering-reference-results") / args.run_id
    recorder = ReferenceRecorder(output, args.artifact_uri, args.capture_rows)
    sampling.generate = recorder.wrap_generation(sampling.generate)
    rollout.score = recorder.wrap_score(rollout.score)
    utilities.log_metrics = recorder.wrap_metrics(utilities.log_metrics)
    versions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    metadata = {
        "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "source_dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip(),
        "model_revision": args.model_revision,
        "environment_lock_sha256": hashlib.sha256(args.lock.read_bytes()).hexdigest(),
        "python": sys.version,
        "versions": versions,
        "overrides": overrides,
        "capture_rows": args.capture_rows,
        "behavior_score_alignment": "column i scores token at i+1, with its exact prefix through i",
    }
    recorder.write_json("metadata.json", metadata)
    recorder.publish()
    sys.argv = [str(source / "train_rl.py"), *overrides, f"log.dir={output}", f"log.run_name={args.run_id}"]
    status = "failed"
    started = time.monotonic()
    try:
        runpy.run_path(str(source / "train_rl.py"), run_name="__main__")
        status = "succeeded"
    finally:
        recorder.write_json(
            "terminal.json",
            {
                "status": status,
                "completed_training_batches": recorder.train_samples,
                "elapsed_seconds": time.monotonic() - started,
                "artifact_uri": args.artifact_uri,
            },
        )
        recorder.publish()
        diagnostic = Path(os.environ["IRIS_OUTPUT_DIR"])
        diagnostic.mkdir(parents=True, exist_ok=True)
        for name in ("metadata.json", "terminal.json"):
            shutil.copyfile(output / name, diagnostic / name)


if __name__ == "__main__":
    main()
