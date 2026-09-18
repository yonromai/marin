# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the pinned full-Hero checkpoint for bounded-memory vLLM loading.

This is experimental qualification tooling. It keeps the model's ordinary
Hugging Face names except that each routed-expert bank is split into the common
``experts.<id>.<projection>.weight`` form. That prevents a streaming loader
from staging a complete 384-expert tensor on every vLLM rank.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import draccus
import equinox as eqx
import jax
import numpy as np
from fray.types import GpuConfig, ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.rpc.proto_display import priority_band_value
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from safetensors.numpy import save_file

from experiments.grug.moe_hero_ep import hero_recipe
from experiments.grug.moe_hero_ep.model import grugmoe_inference_state_dict
from experiments.grug.moe_hero_ep.ops.forward_goldens import (
    CONTROLLER_CLUSTER,
    GoldenRequest,
    pinned_request,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.jobs import IrisSamplingJobs
from experiments.grug.moe_hero_ep.ops.vibe_check.sample import (
    COMPUTE_POLICY,
    restore_model_state,
)

logger = logging.getLogger(__name__)

JOB_USER = "hero-vllm"
EXPERT_AXIS_SIZE = 32
EXPORT_GPUS_PER_TASK = 2
EXPORT_TASKS = EXPERT_AXIS_SIZE // EXPORT_GPUS_PER_TASK
DEFAULT_STORE_ROOT = "s3://marin-us-east-02a/marin/users/romain/hero-vllm-b200/hero-535b-step108000-bf16-split-v2"
INDEX_FILENAME = "model.safetensors.index.json"
MANIFEST_FILENAME = "export-manifest.json"
_SPLIT_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def _group_name(name: str) -> str:
    match = re.match(r"^model\.layers\.(\d+)\.", name)
    return f"layer-{int(match.group(1)):03d}" if match is not None else "global"


def _split_experts(name: str, value: np.ndarray) -> dict[str, np.ndarray]:
    match = _SPLIT_EXPERT_RE.fullmatch(name)
    if match is None:
        return {name: value}
    if value.ndim != 3:
        raise ValueError(f"Routed expert bank must be 3D: {name} has shape {value.shape}")
    return {
        f"{match.group('prefix')}.{expert_id}.{match.group('projection')}.weight": value[expert_id]
        for expert_id in range(value.shape[0])
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(32 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _upload(local_path: Path, remote_path: StoragePath) -> tuple[int, str]:
    if remote_path.exists():
        raise FileExistsError(f"Refusing to overwrite {remote_path}")
    size = local_path.stat().st_size
    sha256 = _sha256(local_path)
    with local_path.open("rb") as source, remote_path.open("wb") as target:
        shutil.copyfileobj(source, target, length=32 * 1024 * 1024)
    return size, sha256


def _write_json(root: StoragePath, filename: str, value: dict) -> None:
    target = root / filename
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def export(request: GoldenRequest, store_root: str) -> None:
    DistributedConfig().initialize()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    configure_coreweave_s3()
    if jax.default_backend() != "gpu" or jax.device_count() != request.spec.batch_size:
        raise ValueError("Full-Hero export requires the request's 32 GB200s")
    if request.spec.batch_size != EXPERT_AXIS_SIZE:
        raise ValueError(f"Expected {EXPERT_AXIS_SIZE} export ranks, got {request.spec.batch_size}")

    root = StoragePath(store_root)
    if (root / MANIFEST_FILENAME).exists() or (root / INDEX_FILENAME).exists():
        raise FileExistsError(f"Refusing to overwrite an export at {store_root}")

    mesh = compact_grug_mesh(expert_axis_size=EXPERT_AXIS_SIZE, replica_axis_size=1)
    with jax.set_mesh(mesh):
        logger.info("Restore authoritative checkpoint on mesh %s", dict(mesh.shape))
        restored = restore_model_state(request, mesh)
        if restored.weights_key != "params":
            raise ValueError(f"Expected authoritative params tree, got {restored.weights_key}")
        authoritative_dtypes = sorted(
            {str(leaf.dtype) for leaf in jax.tree.leaves(restored.model) if eqx.is_inexact_array(leaf)}
        )
        if authoritative_dtypes != ["float32"]:
            raise ValueError(f"Expected authoritative FP32 weights, got {authoritative_dtypes}")
        pending_qb_betas = jax.sharding.reshard(restored.pending_qb_betas, P())
        jax.block_until_ready(pending_qb_betas)
        pending_qb_sha256 = None
        if jax.process_index() == 0:
            pending_qb_sha256 = hashlib.sha256(np.asarray(pending_qb_betas).tobytes(order="C")).hexdigest()

        logger.info("Cast effective weights to BF16")
        model = COMPUTE_POLICY.cast_to_compute(restored.model)
        jax.block_until_ready(model)
        del restored, pending_qb_betas
        gc.collect()

        state_dict = grugmoe_inference_state_dict(model)
        groups: dict[str, list[str]] = {}
        for name in state_dict:
            groups.setdefault(_group_name(name), []).append(name)

        weight_map: dict[str, str] = {}
        shard_records: list[dict[str, object]] = []
        total_size = 0
        with TemporaryDirectory(prefix="hero-vllm-export-") as directory:
            local_root = Path(directory)
            for group_name, names in groups.items():
                tensors: dict[str, np.ndarray] = {}
                logger.info("Materialize %s (%d source tensors)", group_name, len(names))
                for name in names:
                    replicated = jax.sharding.reshard(state_dict[name], P())
                    jax.block_until_ready(replicated)
                    if jax.process_index() == 0:
                        host = np.ascontiguousarray(np.asarray(replicated))
                        tensors.update(_split_experts(name, host))
                    del replicated
                    multihost_utils.sync_global_devices(f"export-{group_name}-{name}")

                if jax.process_index() == 0:
                    filename = f"model-{group_name}.safetensors"
                    local_path = local_root / filename
                    save_file(tensors, local_path, metadata={"format": "pt"})
                    size, sha256 = _upload(local_path, root / filename)
                    total_size += size
                    shard_records.append(
                        {
                            "filename": filename,
                            "bytes": size,
                            "sha256": sha256,
                            "tensor_count": len(tensors),
                        }
                    )
                    for name in tensors:
                        weight_map[name] = filename
                    logger.info(
                        "Uploaded %s: %.2f GiB, %d tensors",
                        filename,
                        size / 1024**3,
                        len(tensors),
                    )
                    del tensors
                    local_path.unlink()
                    gc.collect()

        if jax.process_index() == 0:
            model_config = request.spec.model
            decoded = draccus.decode(hero_recipe.GrugModelConfig, model_config)
            hf_config = decoded.to_hf_config(decoded.vocab_size).to_dict()
            _write_json(root, "config.json", hf_config)
            _write_json(
                root,
                INDEX_FILENAME,
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            )
            _write_json(
                root,
                MANIFEST_FILENAME,
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "asset_root": store_root,
                    "checkpoint": request.checkpoint.model_dump(mode="json"),
                    "golden_bundle": (
                        "s3://marin-us-east-02a/marin/reference/hero-forward/hero-535b-step108000-bf16-v1-dcfe4ced165a"
                    ),
                    "source_revision": request.source_revision,
                    "task_id": os.environ.get("IRIS_TASK_ID"),
                    "process_count": jax.process_count(),
                    "global_device_count": jax.device_count(),
                    "mesh": dict(mesh.shape),
                    "authoritative_weight_tree": "params",
                    "authoritative_weight_dtype": "float32",
                    "effective_weight_dtype": "bfloat16",
                    "pending_qb_rule": ("applied exactly once by restore_model_state before BF16 conversion"),
                    "pending_qb_betas_sha256": pending_qb_sha256,
                    "expert_tensor_layout": ("experts.<global expert id>.<gate_proj|up_proj|down_proj>.weight"),
                    "total_safetensors_bytes": total_size,
                    "tensor_count": len(weight_map),
                    "shards": shard_records,
                    "request": request.model_dump(mode="json"),
                },
            )
            logger.info("Export complete: %s", store_root)

    multihost_utils.sync_global_devices("hero-vllm-export-complete")


def submit(store_root: str, attempt: str) -> None:
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    request = pinned_request("required", revision)
    name_digest = hashlib.sha256(
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "store_root": store_root,
                "attempt": attempt,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    name = f"hero-vllm-export-{name_digest}"
    with connect_controller(cluster_name=CONTROLLER_CLUSTER) as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            jobs = IrisSamplingJobs(
                client,
                endpoint,
                Path.cwd(),
                store_root,
                ResourceConfig(
                    cpu=8,
                    ram="64g",
                    disk="64g",
                    device=GpuConfig(variant="GB200", count=EXPORT_GPUS_PER_TASK),
                    replicas=EXPORT_TASKS,
                ),
                EXPORT_GPUS_PER_TASK,
                sampler_module="experiments.grug.moe_hero_ep.ops.export_vllm",
                user=JOB_USER,
                environment_overrides={"XLA_FLAGS": "--xla_gpu_deterministic_ops=true"},
            )
            jobs.submit(request, name, priority_band_value("interactive"))
    print(f"Submitted /{JOB_USER}/{name} for {store_root}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("submit")
        parser.add_argument("--store-root", default=DEFAULT_STORE_ROOT)
        parser.add_argument("--attempt", default="initial")
        args = parser.parse_args()
        submit(args.store_root, args.attempt)
        return
    configure_logging(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--store-root", required=True)
    args = parser.parse_args()
    export(GoldenRequest.model_validate_json(args.request.read_bytes()), args.store_root)


if __name__ == "__main__":
    main()
