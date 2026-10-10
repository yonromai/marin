# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the released JAX loss and gradients on the common frozen inputs."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authors", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.authors))
    losses = importlib.import_module("losses")
    source = np.load(args.input)
    mesh = jax.make_mesh((1, 1), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    logits = jax.device_put(source["logits"], NamedSharding(mesh, P("data", None, None)))
    advantages = losses.compute_advantages(source["rewards"], 8, "group_centered")
    result = {"advantages": advantages}
    values = []
    for mismatch in (0.0, 2.5):
        q_logp = jax.nn.log_softmax(logits + mismatch * source["noise"], -1)
        labels = jnp.roll(jnp.asarray(source["tokens"]), -1, axis=1)
        chosen_q = jnp.take_along_axis(q_logp, labels[..., None], -1)[..., 0]
        for width in (1, 3, 9):
            head_q, ids = jax.lax.top_k(q_logp, width)
            batch = {
                "tokens": jnp.asarray(source["tokens"]),
                "mask": jnp.asarray(source["mask"]),
                "advantages": jax.device_put(advantages, NamedSharding(mesh, P("data"))),
                "sampling_token_logprobs": chosen_q,
                "center_ids": ids,
                "center_logprobs": head_q,
            }
            for enabled in (False, True):
                name = f"m{mismatch}-k{width}-sc{int(enabled)}"

                def objective(hidden):
                    return losses.loss_fn_rl(
                        hidden, lambda x: x, batch,
                        losses.ISCorrection(level="token", high=2.0, outside="clamp"),
                        score_center=enabled,
                    )

                (loss, metrics), grad = jax.value_and_grad(objective, has_aux=True)(logits)
                result[f"{name}-gradient"] = np.asarray(grad)
                result[f"{name}-loss"] = np.asarray(loss)
                values.append({"case": name, "loss": float(loss), **{k: float(v) for k, v in metrics.items()}})
    np.savez(args.output, **result)
    args.output.with_suffix(".json").write_text(json.dumps(values, indent=2) + "\n")


if __name__ == "__main__":
    main()
