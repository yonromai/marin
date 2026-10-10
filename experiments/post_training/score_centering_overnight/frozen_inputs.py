# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create common FP32 inputs without depending on either loss implementation."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(61009)
    logits = rng.normal(size=(16, 4, 9)).astype(np.float32)
    noise = rng.normal(size=logits.shape).astype(np.float32)
    tokens = rng.integers(0, 9, size=(16, 4), dtype=np.int32)
    mask = np.zeros((16, 4), dtype=bool)
    for row in range(16):
        mask[row, 1 : 1 + row % 4] = True
    rewards = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1], np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, logits=logits, noise=noise, tokens=tokens, mask=mask, rewards=rewards)


if __name__ == "__main__":
    main()
