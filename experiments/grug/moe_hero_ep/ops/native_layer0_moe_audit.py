# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture the first Hero MoE's expert outputs and combine arithmetic.

The 4095/4096-token cases have identical causal prefixes. Both are evaluated
in the same 32-row, 32-GB200 diagnostic batch, with one row on each data rank.
This probe intentionally stops after layer zero; it does not treat a later
route flip or a full-model score as evidence about the first operation.
"""

import argparse
import hashlib
import io
import json
import logging
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.rpc.proto_display import priority_band_value
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.grug._moe.common import (
    _interleave_gate_up,
    _prepare_moe_dispatch,
    _prepare_moe_dispatch_indices_with_assignment_ids,
)
from levanter.grug.attention import AttentionMask, fa4_cute_segment_bounds, token_validity_from_attention_mask
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from transformers import AutoTokenizer

from experiments.grug.moe_hero_ep.model import _batch_reshard, _batch_spec, _embedding_gather
from experiments.grug.moe_hero_ep.ops.forward_goldens import (
    CONTROLLER_CLUSTER,
    DETERMINISTIC_XLA_FLAGS,
    JOB_USER,
    GoldenRequest,
    _validate_authoritative_weights,
    build_inputs,
    pinned_request,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.config import SAMPLING_GPUS_PER_NODE, sampling_resources
from experiments.grug.moe_hero_ep.ops.vibe_check.jobs import IrisSamplingJobs
from experiments.grug.moe_hero_ep.ops.vibe_check.sample import COMPUTE_POLICY, restore_model_state

logger = logging.getLogger(__name__)

STORE_ROOT = "s3://marin-us-east-02a/marin/users/romain/hero-numerical-resolution/native-layer0-moe-audit"
PROBE_POSITIONS = (2046, 2047, 2048)
PAIR_ROWS = (6, 7)


@eqx.filter_jit
def capture_layer0(model, tokens: jax.Array, segment_ids: jax.Array) -> dict[str, jax.Array]:
    """Replay the original first block and expose its expert/combine boundary."""
    # QuACK's CUTLASS dependency is available in the GB200 task image, not the
    # local CPU environment used to submit the task.
    from levanter.grug._moe.sonic_cute import _expert_mlp  # noqa: PLC0415

    batch, sequence = tokens.shape
    config = model.config
    hidden = _embedding_gather(model.token_embed, tokens)
    hidden = model.embed_gated_norm(model.embed_norm(hidden))
    segments = _batch_reshard(segment_ids)
    mask = AttentionMask(is_causal=True, sliding_window=config.sliding_window, segment_ids=(segments, segments))
    bounds, valid = fa4_cute_segment_bounds(
        mask, batch_size=batch, seq_len=sequence, sliding_window=config.sliding_window
    )
    mask = mask.with_fa4_bounds(_batch_reshard(bounds), _batch_reshard(valid))
    layer = model.stacked_blocks.get_layer(0)
    attention_input = layer.attn_gated_norm(layer.rms_attn(hidden))
    attention_output = layer.attn(attention_input, mask, disable_rope=False, is_global=False)
    if layer.sconv_attn is not None:
        attention_output = layer.sconv_attn(attention_output, segments)
    after_attention = hidden + attention_output
    mlp_input = layer.mlp_gated_norm(layer.rms_mlp(after_attention))
    token_valid = token_validity_from_attention_mask(mask, batch_size=batch, sequence_length=sequence)

    mlp = layer.mlp
    flat = mlp_input.reshape(batch * sequence, -1)
    valid_flat = token_valid.reshape(batch * sequence)
    logits = jnp.einsum("td,de->te", flat, reshard(mlp.router, P(None, None))).astype(jnp.float32)
    biased = logits + jax.lax.stop_gradient(mlp.router_bias)
    _, top_indices = jax.lax.top_k(biased, config.num_experts_per_token + 1)
    selected = top_indices[:, :-1]
    combine = jax.nn.sigmoid(jnp.take_along_axis(logits, selected, axis=-1))
    combine = combine * (2.5 / (jnp.sum(combine, axis=-1, keepdims=True) + 1e-9))
    combine = combine.astype(flat.dtype)

    routed_input = flat
    if mlp.w_latent_down is not None and mlp.latent_norm is not None:
        routed_input = jnp.einsum("td,dl->tl", flat, mlp.w_latent_down.astype(flat.dtype), out_sharding=_batch_spec())
        routed_input = mlp.latent_norm(routed_input)

    experts = mlp.expert_mlp
    w_gate_up = reshard(jnp.concatenate((experts.w_gate, experts.w_up), axis=-1), P(None, None, None))
    w_down = reshard(experts.w_down, P(None, None, None))

    # This is the same per-data-rank boundary as the production local MoE.
    # Sorting a globally sharded token axis is invalid in explicit JAX
    # sharding; each rank sorts only its own 4096-token row.
    def local_capture(local_input, local_selected, local_combine, local_valid, local_w13, local_w2):
        local_tokens = local_input.shape[0]
        local_topk = local_selected.shape[1]
        local_hidden = local_input.shape[1]
        x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
            local_input, local_selected, local_combine, local_valid, num_experts=config.num_experts
        )
        _, dispatch_positions, _, _ = _prepare_moe_dispatch_indices_with_assignment_ids(
            local_selected, local_valid, num_experts=config.num_experts
        )
        cumulative = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes).astype(jnp.int32)))
        w13_interleaved = _interleave_gate_up(local_w13, local_w2.shape[1])
        expert_dispatch = _expert_mlp(x_dispatch, w13_interleaved, local_w2, group_sizes, cumulative)
        expert_route = jnp.take(expert_dispatch, dispatch_positions.reshape(-1), axis=0).reshape(
            local_tokens, local_topk, local_hidden
        )
        scatter = jnp.zeros_like(local_input).at[token_dispatch].add(expert_dispatch * w_dispatch[:, None], mode="drop")
        weighted = expert_route * local_combine[:, :, None]
        fixed_sum = jnp.sum(weighted.astype(jnp.float32), axis=1).astype(local_input.dtype)
        fp32_scatter = (
            jnp.zeros_like(local_input, dtype=jnp.float32)
            .at[token_dispatch]
            .add((expert_dispatch * w_dispatch[:, None]).astype(jnp.float32), mode="drop")
        )
        positions = jnp.asarray(PROBE_POSITIONS)
        return (
            jnp.take(expert_route, positions, axis=0),
            jnp.take(scatter, positions, axis=0),
            jnp.take(fixed_sum, positions, axis=0),
            jnp.take(fp32_scatter.astype(local_input.dtype), positions, axis=0),
        )

    mesh = jax.sharding.get_abstract_mesh()
    token_spec = _batch_spec()
    expert_spec = P(token_spec[0], None, None)
    local_results = shard_map(
        local_capture,
        mesh=mesh,
        in_specs=(token_spec, token_spec, token_spec, token_spec, P(None, None, None), P(None, None, None)),
        out_specs=(expert_spec, token_spec, token_spec, token_spec),
        check_rep=False,
    )(
        routed_input,
        selected.astype(jnp.int32),
        combine,
        valid_flat,
        w_gate_up,
        w_down,
    )
    expert_route = local_results[0].reshape(batch, len(PROBE_POSITIONS), config.num_experts_per_token, -1)
    scatter = local_results[1].reshape(batch, len(PROBE_POSITIONS), -1)
    fixed_sum = local_results[2].reshape(batch, len(PROBE_POSITIONS), -1)
    fp32_scatter = local_results[3].reshape(batch, len(PROBE_POSITIONS), -1)
    positions = jnp.asarray(PROBE_POSITIONS)

    def sample(value, *tail):
        shaped = value.reshape(batch, sequence, *tail)
        return jax.sharding.reshard(jnp.take(shaped, positions, axis=1), P())

    # Evaluate the eight actual experts with the production BF16 inputs and
    # weights, but promote both GEMM contractions and SwiGLU to FP64.
    with jax.enable_x64():
        reference_input = sample(routed_input, routed_input.shape[-1])[PAIR_ROWS[0], -1].astype(jnp.float64)
        reference_experts = sample(selected, config.num_experts_per_token)[PAIR_ROWS[0], -1]

        def reference_expert(expert_id):
            gate = reference_input @ w_gate_up[expert_id, :, : w_down.shape[1]].astype(jnp.float64)
            up = reference_input @ w_gate_up[expert_id, :, w_down.shape[1] :].astype(jnp.float64)
            return (jax.nn.silu(gate) * up) @ w_down[expert_id].astype(jnp.float64)

        individual_reference = jax.lax.map(reference_expert, reference_experts)
    if mlp.w_latent_up is not None:
        scatter_up = jnp.einsum(
            "bpl,ld->bpd", scatter, mlp.w_latent_up.astype(scatter.dtype), out_sharding=P(token_spec[0], None, None)
        )
        fixed_up = jnp.einsum(
            "bpl,ld->bpd", fixed_sum, mlp.w_latent_up.astype(fixed_sum.dtype), out_sharding=P(token_spec[0], None, None)
        )
    else:
        scatter_up, fixed_up = scatter, fixed_sum

    return {
        "mlp_input": sample(mlp_input, config.hidden_dim),
        "routed_input": sample(routed_input, routed_input.shape[-1]),
        "selected_experts": sample(selected, config.num_experts_per_token),
        "combine_weights": sample(combine, config.num_experts_per_token),
        "expert_output": jax.sharding.reshard(expert_route, P()),
        "individual_expert_fp64_reference": jax.sharding.reshard(individual_reference, P()),
        "scatter_bf16": jax.sharding.reshard(scatter, P()),
        "fixed_fp32_sum": jax.sharding.reshard(fixed_sum, P()),
        "scatter_fp32": jax.sharding.reshard(fp32_scatter, P()),
        "scatter_after_latent_up": jax.sharding.reshard(scatter_up, P()),
        "fixed_after_latent_up": jax.sharding.reshard(fixed_up, P()),
    }


def _bundle_id(revision: str, attempt: int) -> str:
    return f"hero-layer0-moe-{revision[:12]}-attempt{attempt}"


def produce(request: GoldenRequest, store_root: str, attempt: int) -> None:
    DistributedConfig().initialize()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    configure_coreweave_s3()
    if jax.default_backend() != "gpu" or jax.device_count() != request.spec.batch_size:
        raise ValueError("Native layer-0 audit requires the full 32-GB200 mesh")
    tokenizer = AutoTokenizer.from_pretrained(request.spec.tokenizer, revision=request.spec.tokenizer_revision)
    input_arrays, cases = build_inputs(request, tokenizer)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        restored = restore_model_state(request, mesh)
        _validate_authoritative_weights(restored.model, restored.weights_key)
        model = COMPUTE_POLICY.cast_to_compute(restored.model)
        batch_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert")))

        def batch_array(value: np.ndarray) -> jax.Array:
            return jax.make_array_from_callback(value.shape, batch_sharding, lambda index: value[index])

        captured = capture_layer0(
            model,
            batch_array(input_arrays["tokens"]),
            batch_array(input_arrays["segment_ids"]),
        )
        jax.block_until_ready(captured)
    if jax.process_index() != 0:
        multihost_utils.sync_global_devices("hero-layer0-moe-audit-written")
        return
    arrays = {
        key: np.asarray(value)[list(PAIR_ROWS)]
        for key, value in captured.items()
        if key != "individual_expert_fp64_reference"
    }
    arrays["individual_expert_fp64_reference"] = np.asarray(captured["individual_expert_fp64_reference"])
    arrays["probe_positions"] = np.asarray(PROBE_POSITIONS, dtype=np.int32)
    arrays["row_valid_lengths"] = input_arrays["valid_lengths"][list(PAIR_ROWS)]
    arrays["row_tokens"] = input_arrays["tokens"][list(PAIR_ROWS)]
    arrays["row_validity"] = input_arrays["token_validity"][list(PAIR_ROWS)]
    if not np.array_equal(arrays["row_valid_lengths"], np.array([4095, 4096])):
        raise ValueError("The selected rows are not the native 4095/4096 pair")
    data = io.BytesIO()
    np.savez_compressed(data, **arrays)
    payload = data.getvalue()
    bundle_id = _bundle_id(request.source_revision, attempt)
    root = StoragePath(prefix_join(store_root, bundle_id))
    manifest_target = conditional_object(str(root / "manifest.json"))
    if manifest_target.version() is not None or (root / "arrays.npz").exists():
        raise FileExistsError(str(root))
    manifest = {
        "bundle_id": bundle_id,
        "source_revision": request.source_revision,
        "checkpoint": request.checkpoint.model_dump(mode="json"),
        "created_at": datetime.now(UTC).isoformat(),
        "target_cluster": request.target_cluster,
        "accelerator": "GB200",
        "host_architecture": platform.machine(),
        "xla_flags": DETERMINISTIC_XLA_FLAGS,
        "input_request": request.model_dump(mode="json"),
        "cases": [cases[row] for row in PAIR_ROWS],
        "array_sha256": hashlib.sha256(payload).hexdigest(),
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "contract": (
            "raw selected expert output before BF16 weighting, FP64 evaluation of the selected BF16 "
            "expert inputs and weights at row 6 / position 2048, original BF16 scatter, and fixed FP32 sum"
        ),
    }
    with (root / "arrays.npz").open("wb") as target:
        shutil.copyfileobj(io.BytesIO(payload), target, length=8 * 1024 * 1024)
    manifest_target.write((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(), expected_version=None)
    logger.info("Layer-0 MoE audit uploaded: %s", root)
    multihost_utils.sync_global_devices("hero-layer0-moe-audit-written")


def submit(store_root: str, attempt: int) -> None:
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], check=True, stdout=subprocess.DEVNULL)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    request = pinned_request("shape-audit", revision)
    bundle_id = _bundle_id(revision, attempt)
    configure_coreweave_s3()
    if (StoragePath(prefix_join(store_root, bundle_id)) / "manifest.json").exists():
        raise FileExistsError(bundle_id)
    name = f"hero-layer0-moe-{revision[:10]}-{attempt}"
    with connect_controller(cluster_name=CONTROLLER_CLUSTER) as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            jobs = IrisSamplingJobs(
                client,
                endpoint,
                Path.cwd(),
                store_root,
                sampling_resources(),
                SAMPLING_GPUS_PER_NODE,
                sampler_module="experiments.grug.moe_hero_ep.ops.native_layer0_moe_audit",
                user=JOB_USER,
                environment_overrides={"XLA_FLAGS": DETERMINISTIC_XLA_FLAGS},
            )
            jobs.submit(request, name, priority_band_value("interactive"))
    print(f"Submitted /{JOB_USER}/{name} for {prefix_join(store_root, bundle_id)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--store-root", default=STORE_ROOT)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("submit", nargs="?")
    args = parser.parse_args()
    if args.attempt < 0:
        raise ValueError("attempt must be nonnegative")
    if args.submit == "submit":
        submit(args.store_root, args.attempt)
        return
    if args.submit is not None or args.request is None:
        parser.error("pass submit or --request REQUEST")
    produce(GoldenRequest.model_validate_json(args.request.read_bytes()), args.store_root, args.attempt)


if __name__ == "__main__":
    main()
