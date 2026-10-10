# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze seeded curriculum membership for the paper and native context windows."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=MODEL_REVISION)
    source_files = {split: args.source / f"{split}.parquet" for split in ("train", "validation")}
    frames = {split: pd.read_parquet(path) for split, path in source_files.items()}
    prompt_ids = {}
    for split, frame in frames.items():
        prompt_ids[split] = [
            tokenizer.apply_chat_template(
                list(prompt), tokenize=True, add_generation_prompt=True, enable_thinking=False
            )["input_ids"]
            for prompt in frame["prompt"]
        ]
        assert all(ids and isinstance(ids[0], int) for ids in prompt_ids[split])
    manifest = {
        "source": "curriculum-rl-pool@2026.08.29.1",
        "source_files": {split: hashlib.sha256(path.read_bytes()).hexdigest() for split, path in source_files.items()},
        "tokenizer_revision": MODEL_REVISION,
        "thinking": False,
        "membership": [],
    }
    for seed in (0, 1, 2):
        order = list(range(len(frames["train"])))
        random.Random(seed).shuffle(order)
        for window, prompt_limit in (("paper-window", 511), ("native-window", 4096)):
            target = args.output / f"seed{seed}" / window
            target.mkdir(parents=True, exist_ok=True)
            for split, frame in frames.items():
                source_order = order if split == "train" else list(range(len(frame)))
                keep = [index for index in source_order if len(prompt_ids[split][index]) <= prompt_limit]
                assert window != "native-window" or len(keep) == len(frame)
                derived = frame.iloc[keep].copy()
                derived["campaign_source_row"] = keep
                derived["reference_prompt_ids"] = [prompt_ids[split][index] for index in keep]
                path = target / f"{split}.parquet"
                derived.to_parquet(path, index=False)
                manifest["membership"].append(
                    {
                        "seed": seed,
                        "window": window,
                        "split": split,
                        "source_rows": len(frame),
                        "retained_rows": len(keep),
                        "excluded_rows": len(frame) - len(keep),
                        "source_row_order_sha256": hashlib.sha256(json.dumps(keep).encode()).hexdigest(),
                        "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "maximum_prompt_tokens": max(len(prompt_ids[split][index]) for index in keep),
                    }
                )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
