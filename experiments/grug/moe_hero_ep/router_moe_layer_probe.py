# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time one Hero-shaped routed MoE layer with natural and fixed routing."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from iris.runtime.jax_init import initialize_jax
from jax import P
from jax.sharding import NamedSharding
from levanter.grug._moe.ep_ragged_all_to_all import RAGGED_REQUIRED_XLA_FLAGS
from levanter.grug.grug_moe import moe_mlp
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.moe_hero_ep.router_causal_probe import scores

CASES = (
    ("natural_current", "current", False),
    ("natural_preferred", "preferred_fp32", False),
    ("natural_hybrid", "preferred_current_vjp", False),
    ("fixed_current", "current", True),
    ("fixed_preferred", "preferred_fp32", True),
)


def _configure_ragged_runtime():
    """Use the relevant Hero EP runtime settings without importing its training launcher."""
    defaults = {
        "LD_PRELOAD": "libjemalloc.so.2",
        "MALLOC_CONF": "background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0,narenas:2",
        "JAX_ENABLE_PGLE": "false",
        "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": "192",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    flags = os.environ.get("XLA_FLAGS", "").split()
    required = (
        "--xla_gpu_experimental_parallel_collective_overlap_limit=1",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_memory_limit_slop_factor=85",
        "--xla_gpu_enable_command_buffer=",
        *RAGGED_REQUIRED_XLA_FLAGS,
    )
    required_names = {flag.partition("=")[0] for flag in required}
    flags = [flag for flag in flags if flag.partition("=")[0] not in required_names]
    os.environ["XLA_FLAGS"] = " ".join([*flags, *required])
    installed = importlib.metadata.version("jax-cuda13-pjrt")
    if not installed.startswith(f"{jax.__version__}+marin."):
        raise RuntimeError(f"Ragged MoE requires the Marin patched PJRT; found {installed}")


def _fixture(uri: str):
    if uri.startswith("s3://"):
        configure_coreweave_s3()
        with fsspec.open(uri, "rb") as handle:
            content = handle.read()
    else:
        content = Path(uri).read_bytes()
    with np.load(io.BytesIO(content), allow_pickle=False) as saved:
        valid = saved["valid"].reshape(-1)
        return (
            hashlib.sha256(content).hexdigest(),
            saved["inputs"][valid].copy(),
            saved["weight"].copy(),
            saved["bias"].copy(),
        )


def _routes(logits, bias):
    _, indices = jax.lax.top_k(logits + bias, 9)
    selected = indices[:, :8].astype(jnp.int32)
    selected_logits = jnp.take_along_axis(logits, selected, axis=-1)
    gates = jax.nn.sigmoid(selected_logits)
    gates = gates * (2.5 / (jnp.sum(gates, axis=-1, keepdims=True) + 1e-9))
    return selected, gates.astype(jnp.bfloat16)


def _make_case(variant: str, fixed: bool, *, bias, token_valid, mesh, capacity_factor, implementation):
    def loss(x, router_weight, w_up_gate, w_down, cotangent, router_cotangent, fixed_indices, fixed_gates):
        logits = scores(x, router_weight, variant)
        if fixed:
            indices, gates = fixed_indices, fixed_gates
        else:
            indices, gates = _routes(logits, bias)
        output, drops = moe_mlp(
            x[:, : w_down.shape[1]],
            indices,
            gates,
            w_up_gate.astype(jnp.bfloat16),
            w_down.astype(jnp.bfloat16),
            token_valid=token_valid,
            activation=jax.nn.silu,
            implementation=implementation,
            mesh=mesh,
            capacity_factor=capacity_factor,
            report_capacity_overflow=True,
        )
        objective = jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))
        if fixed:
            # The router remains in the gradient graph while dispatch receives identical inputs.
            objective = objective + 0.001 * jnp.sum(logits * router_cotangent)
        return objective, (output, drops)

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))


def _time_cases(executables, arguments, *, warmup, repeats):
    for _ in range(warmup):
        for executable in executables.values():
            jax.block_until_ready(executable(*arguments))
    order = list(executables)
    rng = random.Random(31337)
    samples = {name: [] for name in order}
    for trial in range(repeats):
        rng.shuffle(order)
        for position, name in enumerate(order):
            start = time.perf_counter()
            jax.block_until_ready(executables[name](*arguments))
            samples[name].append({"trial": trial, "order": position, "seconds": time.perf_counter() - start})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-backend", action="store_true")
    parser.add_argument("--local-rows", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-prefix")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.local_rows, args.warmup, args.repeats) < 1:
        raise ValueError("local rows, warmup and repeats must be positive")
    use_ragged_backend = not args.smoke or args.gpu_backend
    if use_ragged_backend:
        _configure_ragged_runtime()
    initialize_jax()
    hidden, latent, intermediate, experts = (32, 16, 16, 16) if args.smoke else (6144, 3072, 3072, 384)
    local_rows = 8 if args.smoke else args.local_rows
    implementation = "ragged_all_to_all" if use_ragged_backend else "fixed_all_to_all"
    fixture_hash, saved, router_weight_host, bias_host = _fixture(args.fixture)
    if saved.shape[1] != 6144 or router_weight_host.shape != (6144, 384):
        raise ValueError("Fixture is not the production router shape")
    mesh = compact_grug_mesh(expert_axis_size=jax.device_count(), replica_axis_size=1)
    output_dir = Path(os.environ.get("IRIS_OUTPUT_DIR", "router-moe-layer-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    is_leader = jax.process_index() == 0
    with jax.set_mesh(mesh):
        batch_axes = ("replica_dcn", "data", "expert")
        batch_sharding = NamedSharding(mesh, P(batch_axes, None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        local_count = local_rows * jax.local_device_count()
        local_saved = saved[:, :hidden].astype(ml_dtypes.bfloat16)
        tiled = np.tile(local_saved, (int(np.ceil(local_count / len(local_saved))), 1))[:local_count]
        x = jax.make_array_from_process_local_data(
            batch_sharding, tiled, global_shape=(local_rows * jax.device_count(), hidden)
        )
        router_weight = jax.device_put(router_weight_host[:hidden, :experts], NamedSharding(mesh, P(None, None)))
        bias = jax.device_put(bias_host[:experts], NamedSharding(mesh, P(None)))
        token_valid = jax.jit(
            lambda: jnp.ones((x.shape[0],), dtype=jnp.bool_), out_shardings=NamedSharding(mesh, P(batch_axes))
        )()
        w_up_gate = (
            jax.random.normal(
                jax.random.PRNGKey(11),
                (experts, latent, 2 * intermediate),
                dtype=jnp.float32,
                out_sharding=expert_sharding,
            )
            * 0.005
        )
        w_down = (
            jax.random.normal(
                jax.random.PRNGKey(12),
                (experts, intermediate, latent),
                dtype=jnp.float32,
                out_sharding=expert_sharding,
            )
            * 0.005
        )
        cotangent = jax.lax.stop_gradient(x[:, :latent])
        router_cotangent = jax.lax.stop_gradient(x[:, :experts].astype(jnp.float32))
        baseline_logits = scores(x, router_weight, "current")
        preferred_logits = scores(x, router_weight, "preferred_fp32")
        hybrid_logits = scores(x, router_weight, "preferred_current_vjp")
        fixed_indices, fixed_gates = _routes(baseline_logits, bias)
        preferred_indices, preferred_gates = _routes(preferred_logits, bias)
        route_changes = int(np.asarray(jnp.sum(jnp.any(fixed_indices != preferred_indices, axis=1))))
        gate_exact = float(np.asarray(jnp.mean(fixed_gates == preferred_gates)))
        hybrid_exact = bool(np.asarray(jnp.all(hybrid_logits == preferred_logits)))
        if not hybrid_exact:
            raise ValueError("Hybrid forward scores differ from preferred scores")
        arguments = (x, router_weight, w_up_gate, w_down, cotangent, router_cotangent, fixed_indices, fixed_gates)
        executables = {}
        for name, variant, fixed in CASES:
            compiled = (
                _make_case(
                    variant,
                    fixed,
                    bias=bias,
                    token_valid=token_valid,
                    mesh=mesh,
                    capacity_factor=1.15,
                    implementation=implementation,
                )
                .lower(*arguments)
                .compile()
            )
            executables[name] = compiled
            if is_leader:
                (output_dir / f"{name}-optimized-hlo.txt").write_text(compiled.as_text())
                print(f"compiled {name}", flush=True)
        fixed_current = executables["fixed_current"](*arguments)[0][1][0]
        fixed_preferred = executables["fixed_preferred"](*arguments)[0][1][0]
        fixed_output_max_abs = float(
            np.asarray(jnp.max(jnp.abs(fixed_current.astype(jnp.float32) - fixed_preferred.astype(jnp.float32))))
        )
        if fixed_output_max_abs != 0.0:
            raise ValueError(f"Fixed downstream output differs: {fixed_output_max_abs}")
        samples = _time_cases(executables, arguments, warmup=args.warmup, repeats=args.repeats)
        if args.profile:
            if is_leader:
                with jax.profiler.trace(str(output_dir / "timeline"), create_perfetto_link=False):
                    for name, executable in executables.items():
                        with jax.profiler.TraceAnnotation(name):
                            jax.block_until_ready(executable(*arguments))
            else:
                for executable in executables.values():
                    jax.block_until_ready(executable(*arguments))
    if is_leader:
        report = {
            "command": [sys.executable, *sys.argv],
            "fixture_sha256": fixture_hash,
            "source_commit": args.source_commit,
            "hardware": [str(device) for device in jax.local_devices()],
            "process_count": jax.process_count(),
            "device_count": jax.device_count(),
            "local_rows_per_device": local_rows,
            "hidden": hidden,
            "latent": latent,
            "intermediate": intermediate,
            "experts": experts,
            "implementation": implementation,
            "jax_version": jax.__version__,
            "jaxlib_version": jax.lib.__version__,
            "backend_version": jax.devices()[0].client.platform_version,
            "python_version": platform.python_version(),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "route_order_changed_rows": route_changes,
            "gate_exact_fraction": gate_exact,
            "hybrid_scores_exact": hybrid_exact,
            "fixed_output_max_abs": fixed_output_max_abs,
            "samples": samples,
        }
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if args.output_prefix:
            configure_coreweave_s3()
            for path in output_dir.rglob("*"):
                if path.is_file():
                    target = args.output_prefix.rstrip("/") + "/" + path.relative_to(output_dir).as_posix()
                    with fsspec.open(target, "wb") as handle:
                        handle.write(path.read_bytes())
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
