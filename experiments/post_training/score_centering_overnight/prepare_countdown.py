# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the exact released Countdown membership and prompts for SkyRL."""

import argparse
import hashlib
import importlib.util
import json
import random
from collections.abc import Mapping
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("countdown_reference", args.reference_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module._rows(random.Random(0), 20000)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=args.model_revision)
    for seed in (0, 1, 2):
        dataset = Dataset.from_list(rows).shuffle(seed=seed)
        root = args.output / f"seed{seed}"
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "source": "martin-marek/score-centering@7c56e9ee2972aa57f446cf564de1a1658d14b321",
            "dataset_seed": 0,
            "shuffle_seed": seed,
            "held_out_prompts": 64,
            "train_prompts": 19936,
            "model_revision": args.model_revision,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for name, subset in (("validation", dataset.select(range(64))), ("train", dataset.select(range(64, 20000)))):
            exported = []
            digests = []
            for row in subset:
                tokens = tokenizer.apply_chat_template(
                    row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
                )
                tokens = tokens["input_ids"] if isinstance(tokens, Mapping) else tokens
                if len(tokens) >= 512:
                    raise ValueError("An original prompt consumes the entire 512-token sequence")
                digests.append(hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
                exported.append(
                    {
                        **row,
                        "env_class": "countdown_reference",
                        "data_source": "authors_countdown_7c56e9ee",
                        "reference_prompt_ids": tokens,
                    }
                )
            Dataset.from_list(exported).to_parquet(root / f"{name}.parquet")
            manifest[f"{name}_ordered_membership_sha256"] = hashlib.sha256("\n".join(digests).encode()).hexdigest()
        manifest["file_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.parquet")}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(seed, manifest["validation_ordered_membership_sha256"], manifest["train_ordered_membership_sha256"])


if __name__ == "__main__":
    main()
