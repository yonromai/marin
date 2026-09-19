# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit the first divergent ladder MoE sum against exact captured terms.

The trace stores each weighted expert output as a BF16 value in a float32
array. Summing these values in float64 avoids accumulation rounding; it does
not measure the error inside an individual expert's matrix multiplications.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

LAYER = "layer_0_"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files if key.startswith(LAYER)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bf16(value: np.ndarray) -> np.ndarray:
    return torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16).float().numpy().astype(np.float64)


def _difference(observed: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    error = observed.astype(np.float64) - reference.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(error))),
        "rms": float(np.sqrt(np.mean(np.square(error)))),
        "changed_elements": int(np.count_nonzero(error)),
        "elements": int(error.size),
    }


def audit(capture_dir: Path) -> dict:
    names = ["ladder-h100-ep1-experts-rank0.npz"]
    names += [f"ladder-h100-ep8-experts-rank{rank}.npz" for rank in range(8)]
    names += [
        "ladder-h100-ep8-fp32-rank0.npz",
        "ladder-h100-ep8-fp32full-rank0.npz",
        "ladder-h100-ep8-fp32full-rank7.npz",
    ]
    traces = {name: _load(capture_dir / name) for name in names}
    ep1 = traces[names[0]]
    ep8 = [traces[name] for name in names[1:9]]
    fp32_collective = traces[names[9]]
    fp32_full = traces[names[10]]
    fp32_full_rank7 = traces[names[11]]

    for trace in ep8:
        for field in ("mlp_input", "selected_experts", "expert_topk_ids", "expert_topk_weights"):
            if not np.array_equal(trace[LAYER + field], ep1[LAYER + field]):
                raise ValueError(f"layer-0 inputs or routes differ: {field}")

    expert_ep1 = ep1[LAYER + "per_expert_output"]
    expert_ep8 = np.stack([trace[LAYER + "per_expert_output"] for trace in ep8])
    local_bf16 = np.stack([trace[LAYER + "local_precombine"] for trace in ep8])
    expert_sum = expert_ep8.astype(np.float64).sum(axis=0)
    local_exact = expert_ep8.astype(np.float64).sum(axis=2)
    total_exact = expert_ep1.astype(np.float64).sum(axis=1)
    local_bf16_sum = local_bf16.astype(np.float64).sum(axis=0)
    original = ep8[0][LAYER + "routed_combined"]
    ep1_sum = ep1[LAYER + "routed_combined"]
    collective_fp32 = fp32_collective[LAYER + "routed_combined"]
    full_fp32 = fp32_full[LAYER + "routed_combined"]

    identities = {
        "same_weighted_experts_across_ep1_ep8": _difference(expert_sum, expert_ep1),
        "local_bf16_equals_rounded_exact": _difference(local_bf16, _bf16(local_exact)),
        "fp32_collective_equals_rounded_local_bf16_sum": _difference(collective_fp32, _bf16(local_bf16_sum)),
        "fp32_local_and_collective_equals_rounded_exact": _difference(full_fp32, _bf16(total_exact)),
        "ep1_equals_rounded_exact": _difference(ep1_sum, _bf16(total_exact)),
        "fp32_rank7_local_equals_exact": _difference(
            fp32_full_rank7[LAYER + "local_precombine"], local_exact[7]
        ),
    }
    if any(value["changed_elements"] for value in identities.values()):
        raise ValueError(f"captured stage identity failed: {identities}")
    for trace in ep8[1:]:
        if not np.array_equal(trace[LAYER + "routed_combined"], original):
            raise ValueError("EP8 ranks disagree on the combined result")

    return {
        "scope": "trained Hero d1536, H100, BF16, first MoE block, six captured rows",
        "reference": "float64 sum of exact captured BF16 weighted expert outputs, then BF16 round where stated",
        "identities": identities,
        "stage_errors": {
            "individual_expert_topology_difference": _difference(expert_sum, expert_ep1),
            "bf16_local_rounding_before_collective": _difference(local_bf16_sum, total_exact),
            "bf16_collective_extra_over_fp32_collective": _difference(original, collective_fp32),
            "ep1_total_vs_exact": _difference(ep1_sum, total_exact),
            "ep8_original_total_vs_exact": _difference(original, total_exact),
            "ep8_fp32_collective_total_vs_exact": _difference(collective_fp32, total_exact),
            "ep8_fp32_local_and_collective_total_vs_exact": _difference(full_fp32, total_exact),
        },
        "input_sha256": {name: _sha256(capture_dir / name) for name in names},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.capture_dir), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.write_text(result)
