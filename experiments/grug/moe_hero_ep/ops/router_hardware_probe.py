# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay one captured Hero router operation on an allocated GPU."""

import argparse
import hashlib
import io
import json
import platform
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

INPUT_ROOT = "s3://marin-us-east-02a/marin/users/romain/hero-numerical-triangulation/" "router-layer15-hw-01a0b566-a1"
INPUT_SHA256 = "b04ac6771903cccc08819e37504c2a53d502dfdb644e81b229c0aa371881779f"


def _bf16_tensor(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.copy()).view(torch.bfloat16).to("cuda")


def _rank(scores: np.ndarray) -> dict:
    # Explicit expert-ID tiebreak, matching the native top-k contract.
    ids = np.lexsort((np.arange(len(scores)), -scores))[:12]
    return {
        "top12_ids": ids.astype(int).tolist(),
        "top12_values": scores[ids].astype(float).tolist(),
        "cutoff_gap": float(scores[ids[7]] - scores[ids[8]]),
        "all_scores": scores.astype(float).tolist(),
    }


def _observe(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> dict:
    cpu_x = x.float().cpu().numpy().astype(np.float64)
    cpu_weight = weight.float().cpu().numpy().astype(np.float64)
    cpu_bias = bias.float().cpu().numpy().astype(np.float64)
    reference = _rank(cpu_weight @ cpu_x + cpu_bias)
    modes = {
        "pinned_vllm_fp32": lambda: F.linear(x.float(), weight.float()).float() + bias.float(),
        "native_like_bf16": lambda: F.linear(x, weight).float() + bias.float(),
    }
    result = {"fp64_reference": reference}
    for name, fn in modes.items():
        repeats = [fn().cpu().numpy().astype(np.float64) for _ in range(3)]
        ranked = _rank(repeats[0])
        ranked["max_abs_vs_fp64"] = float(np.max(np.abs(repeats[0] - reference["all_scores"])))
        ranked["three_repeats_equal"] = all(np.array_equal(repeats[0], row) for row in repeats[1:])
        result[name] = ranked
    return result


def run(output_uri: str) -> None:
    configure_coreweave_s3()
    destination = StoragePath(output_uri)
    if destination.exists():
        raise FileExistsError(output_uri)
    array_bytes = (StoragePath(INPUT_ROOT) / "inputs.npz").read_bytes()
    assert hashlib.sha256(array_bytes).hexdigest() == INPUT_SHA256
    metadata = json.loads((StoragePath(INPUT_ROOT) / "inputs.json").read_text())
    assert metadata["array_sha256"] == INPUT_SHA256
    assert torch.cuda.is_available()
    with np.load(io.BytesIO(array_bytes), allow_pickle=False) as arrays:
        weight = _bf16_tensor(arrays["weight_bits"])
        bias = _bf16_tensor(arrays["bias_bits"])
        xs = {name: _bf16_tensor(arrays[f"input_{name}_bits"]) for name in ("short", "full")}
    result = {
        "input_uri": INPUT_ROOT,
        "input_sha256": INPUT_SHA256,
        "source_export": metadata["source_export"],
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "host_architecture": platform.machine(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "cases": {name: _observe(x, weight, bias) for name, x in xs.items()},
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"ROUTER_HARDWARE_PROBE={output_uri}", flush=True)
    for name, case in result["cases"].items():
        for mode, row in case.items():
            print(name, mode, row["top12_ids"], row["cutoff_gap"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-uri", required=True)
    args = parser.parse_args()
    try:
        run(args.output_uri)
    except Exception:
        traceback.print_exc()
        raise
