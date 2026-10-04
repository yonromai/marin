# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure same-token engine mismatch and stale-weight drift from native archives.

These frozen probes measure counterfactual drift at known optimizer ages. They
do not measure the age distribution of rollouts consumed by an async trainer.
Run with the campaign's frozen Marin environment and pass --archive and --output.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from finestore.reader import ReadView
from finestore.rl import mismatch_probe as schema
from rigging.filesystem.s3_compat import configure_coreweave_s3


def distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        f"{prefix}_signed_mean": float(values.mean()),
        f"{prefix}_abs_mean": float(absolute.mean()),
        f"{prefix}_abs_p50": float(np.quantile(absolute, 0.5)),
        f"{prefix}_abs_p95": float(np.quantile(absolute, 0.95)),
        f"{prefix}_abs_p99": float(np.quantile(absolute, 0.99)),
        f"{prefix}_abs_max": float(absolute.max()),
    }


def summarize(archive: str) -> tuple[list[dict], dict]:
    view = ReadView(archive)
    manifests = [schema.ManifestRow.model_validate(row) for row in view.scan(schema.MANIFEST_TABLE).to_pylist()]
    if len(manifests) != 1 or manifests[0].status != schema.ArchiveStatus.COMPLETE:
        raise ValueError("analysis requires one complete native mismatch archive")
    manifest = manifests[0]
    probes = [schema.ProbeRow.model_validate(row) for row in view.scan(schema.PROBE_TABLE).to_pylist()]
    scorings = [schema.ScoreRow.model_validate(row) for row in view.scan(schema.SCORES_TABLE).to_pylist()]
    scores = {}
    for score in scorings:
        key = (score.sample_id, score.scorer, score.update, score.mode, score.cache_mode)
        if key in scores or score.probe_hash != manifest.probe_hash:
            raise ValueError("duplicate scoring or inconsistent token hash")
        scores[key] = score

    def get(probe: schema.ProbeRow, scorer: str, update: int, mode: str = "", cache: str | None = None):
        row = scores[(probe.sample_id, scorer, update, mode, cache)]
        if len(row.logprobs) != len(probe.vllm_output_ids):
            raise ValueError("scoring vector does not cover the frozen response")
        expected_step = steps[update]
        if row.global_step != expected_step:
            raise ValueError("scoring step disagrees with the manifest")
        result = np.asarray(row.logprobs, dtype=np.float64)
        if not np.isfinite(result).all() or (result > 0).any():
            raise ValueError("scoring contains invalid log probabilities")
        return result

    steps = dict(zip(manifest.scored_updates, manifest.scored_global_steps, strict=True))
    if len(steps) != len(manifest.scored_updates) or steps.get(0) != manifest.starting_global_step:
        raise ValueError("manifest has an invalid starting step or duplicate update")
    if any(update > 0 for update in steps) and manifest.optimizer_steps_per_update < 1:
        raise ValueError("completed updates lack a positive optimizer-step count")
    if not probes or len({probe.sample_id for probe in probes}) != len(probes):
        raise ValueError("probe archive is empty or repeats sample identities")
    for probe in probes:
        if probe.probe_hash != manifest.probe_hash:
            raise ValueError("probe hash disagrees with the manifest")
        if probe.prompt_token_ids != probe.trainer_prompt_ids or probe.vllm_output_ids != probe.trainer_input_ids:
            raise ValueError("engine and trainer did not score identical tokens and prefixes")

    rows = []
    for update, global_step in sorted(steps.items()):
        if global_step < manifest.starting_global_step:
            raise ValueError("scoring precedes the generating weights")
        pieces = []
        for probe in probes:
            active = np.asarray(probe.loss_mask, dtype=bool) & np.asarray(probe.response_mask, dtype=bool)
            if not active.any():
                raise ValueError("frozen response has no eligible loss tokens")
            a = get(probe, "vllm.generate", 0)
            b = get(probe, "trainer", 0, "native")
            c = get(probe, "trainer", update, "native")
            repeat_b = get(probe, "trainer", 0, "repeat")
            repeat_c = get(probe, "trainer", update, "repeat")
            reread_a = get(probe, "vllm.rescore", 0, cache="off")
            arrays = np.stack((b - a, c - b, c - a, repeat_b - b, repeat_c - c, reread_a - a))[:, active]
            pieces.append((probe.prompt_id, arrays))
        scopes = {"all": np.concatenate([array for _, array in pieces], axis=1)}
        for prompt_id in sorted({prompt for prompt, _ in pieces}):
            scopes[f"prompt:{prompt_id}"] = np.concatenate(
                [array for prompt, array in pieces if prompt == prompt_id], axis=1
            )
        for scope, arrays in scopes.items():
            engine, stale, combined, baseline_repeat, current_repeat, reread = arrays
            residual = engine + stale - combined
            if np.max(np.abs(residual)) > 1e-12:
                raise ValueError("signed engine and stale components do not reconstruct combined mismatch")
            row = {
                "update": update,
                "global_step": global_step,
                "optimizer_age": (global_step - manifest.starting_global_step) * manifest.optimizer_steps_per_update,
                "scope": scope,
                "tokens": arrays.shape[1],
                "opposite_sign_fraction": float(np.mean(engine * stale < 0)),
                "canceled_absolute_mean": float(np.mean(np.abs(engine) + np.abs(stale) - np.abs(combined))),
                "reconstruction_abs_max": float(np.max(np.abs(residual))),
                "consumption_tis_cap_1_05_fraction": float(np.mean(combined > np.log(1.05))),
            }
            for values, name in (
                (engine, "engine"),
                (stale, "stale"),
                (combined, "combined"),
                (baseline_repeat, "baseline_repeat"),
                (current_repeat, "current_repeat"),
                (reread, "generation_reread"),
            ):
                row.update(distribution(values, name))
            rows.append(row)
    provenance = {
        "archive": archive,
        "manifest": manifest.model_dump(mode="json"),
        "prompt_count": len({probe.prompt_id for probe in probes}),
        "sample_count": len(probes),
        "analysis_scope": "fixed generated tokens at generating weights, rescored after known optimizer updates",
        "limitations": [
            "Selected validation prompts are calibration evidence, not an estimate over the held-out population.",
            "Optimizer age here describes frozen probes, not the async consumed-token age distribution.",
            "PPO stored old at consumption is C before the next update, not B at generating weights.",
            "TIS clipping fraction uses C/A at consumption; centering tails require separate captured-head evidence.",
        ],
    }
    return rows, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configure_coreweave_s3()
    rows, provenance = summarize(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "components.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"archive": args.archive, "rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
