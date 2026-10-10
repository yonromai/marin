# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the complete native Snowball corpus and its three seeded input orders."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer", revision=TOKENIZER_REVISION)
    manifest = {
        "source": "curriculum-rl-pool@2026.09.18",
        "tokenizer_revision": TOKENIZER_REVISION,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "chat_template_kwargs": {},
        "maximum_prompt_tokens": 4096,
        "membership": [],
    }
    for split in ("train", "validation"):
        source = args.source / f"{split}.parquet"
        frame = pd.read_parquet(source)
        prompts = [
            tokenizer.apply_chat_template(list(prompt), tokenize=True, add_generation_prompt=True)["input_ids"]
            for prompt in frame["prompt"]
        ]
        assert all(ids and isinstance(ids[0], int) for ids in prompts)
        lengths = [len(ids) for ids in prompts]
        # Preserve the native corpus; refuse silent runtime filtering.
        assert max(lengths) <= 4096, f"{split} contains prompts outside the predeclared window"
        for seed in (0, 1, 2):
            order = list(range(len(frame)))
            if split == "train":
                random.Random(seed).shuffle(order)
            derived = frame.iloc[order].copy()
            derived["campaign_source_row"] = order
            derived["reference_prompt_ids"] = [prompts[index] for index in order]
            target = args.output / f"seed{seed}"
            target.mkdir(parents=True, exist_ok=True)
            path = target / f"{split}.parquet"
            derived.to_parquet(path, index=False)
            manifest["membership"].append(
                {
                    "seed": seed,
                    "split": split,
                    "source_rows": len(frame),
                    "retained_rows": len(derived),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_row_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                    "maximum_prompt_tokens": max(lengths),
                    "domains": frame["data_source"].value_counts().to_dict(),
                    "environments": frame["env_class"].value_counts().to_dict(),
                }
            )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
