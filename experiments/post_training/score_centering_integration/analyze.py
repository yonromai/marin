# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check and summarize complete integration qualifications without pooling allocations."""

import argparse
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from experiments.post_training.score_centering_performance.analyze import comparison, describe, devices, replay_scalar

MODES = ("ordinary", "capture", "sc", "armed", "probe")
LOCK_SHA256 = "a9686c1263fae6e77f560c22ab8faa99b694747ac57e09f816ae7ce923c19f99"


def read_report(path):
    stored = path.read_bytes()
    if path.suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(stored)) as archive:
            assert len(archive.namelist()) == 1
            raw = archive.read(archive.namelist()[0])
    else:
        raw = stored
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def observed_range(values):
    values = list(values)
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values), "values": values}


def summarize(path, cadence):
    data, digest = read_report(path)
    assert data["complete"] is True
    snowball = data["workload"] == "snowball"
    repetitions, async_steps, async_warmup = (6, 12, 2) if snowball else (2, 4, 1)
    expected = 3 + repetitions * len(MODES) + 6 + 4 * async_steps
    assert data["applied_updates"] == expected
    assert len(data["windows"]) == repetitions * len(MODES)
    assert len(data["native_periodic_windows"]) == 6
    assert len(data["asynchronous_windows"]) == 4 * async_steps
    assert data["runtime"]["lock_sha256"] == LOCK_SHA256
    serialized = data["windows"]
    shape_keys = ("tokens", "loss_tokens", "sequence_shape")
    for key in shape_keys:
        assert all(row[key] == serialized[0][key] for row in serialized)
    for key in ("prompt_lengths", "row_loss_tokens", "response_lengths"):
        assert all(sorted(row[key]) == sorted(serialized[0][key]) for row in serialized)
    all_windows = serialized + data["native_periodic_windows"] + data["asynchronous_windows"]
    for row in all_windows:
        metrics = row["metrics"]
        assert metrics["policy/policy_update_steps"] == 1 and metrics["policy/skipped_steps"] == 0
        assert metrics["async/staleness_tokens_stamped"] == 1
        assert metrics["async/staleness_tokens_max"] <= (4 if row["asynchronous"] else 0)
        assert math.isfinite(metrics["policy/raw_grad_norm"])
        assert (
            metrics["policy/raw_grad_norm"] > 0 if row["synthetic_advantages"] else metrics["policy/raw_grad_norm"] >= 0
        )
        sampling = row["inference_sampling"]
        assert len(sampling) == 8
        assert all(
            params == {"temperature": row.get("temperature", 1.0), "top_p": 1.0, "top_k": -1, "min_p": 0.0}
            for params in sampling
        )
    result = {
        "report": str(path),
        "sha256": digest,
        "product": data["candidate"],
        "variant": data["variant"],
        "workload": data["workload"],
        "applied_updates": expected,
        "runtime": data["runtime"],
        "source_selection": data["source_selection"],
        "prompt_sha256": data["prompt_sha256"],
        "controlled_shape": {key: serialized[0][key] for key in shape_keys},
        "response_lengths": dict(Counter(serialized[0]["response_lengths"])),
        "setup_timings": data["setup_timings"],
        "serialized": {},
        "asynchronous": {},
    }
    for mode in MODES:
        rows = sorted(
            (row for row in serialized if row["mode"] == mode and not row["warmup"]),
            key=lambda row: row["repetition"],
        )
        assert len(rows) == repetitions - 1
        exclusive = sorted(key for key in rows[0]["metrics"] if key.startswith("timing/step_wall/"))
        summary = {
            "cycle_seconds": describe(row["total_seconds"] for row in rows),
            "response_tokens_per_second": sum(row["tokens"] for row in rows) / sum(row["total_seconds"] for row in rows),
            "exclusive_wall_seconds": {key: describe(row["metrics"][key] for row in rows) for key in exclusive},
            "metric_emission_seconds": describe(row["metric_emission_seconds"] for row in rows),
            "exclusive_wall_residual_seconds": [
                row["core_seconds"] - sum(row["metrics"][key] for key in exclusive) for row in rows
            ],
            "nested_timers_seconds": {
                key: describe(row["timings"].get(key, 0) for row in rows)
                for key in sorted(set().union(*(row["timings"] for row in rows)))
            },
            "optimizer_tensor_bytes": rows[0]["optimizer_input_tensor_bytes"],
            "driver_rss_current_max": max(row["driver_host"]["rss_current"] for row in rows),
            "driver_rss_cumulative_peak": max(row["driver_host"]["rss_cumulative_peak"] for row in rows),
        }
        for role in ("learner", "inference"):
            memory = [device for row in rows for device in devices(row[f"{role}_memory"])]
            assert memory
            summary[f"{role}_memory_bytes"] = {
                key: max(device[key] for device in memory)
                for key in ("peak_allocated", "peak_reserved", "sampled_device_peak_used")
            }
            summary[f"{role}_host_bytes"] = {
                key: max(device["host"][key] for device in memory) for key in ("rss_current", "rss_cumulative_peak")
            }
        if mode == "probe":
            summary["capture_tokens"] = [sum(group["response_tokens"] for group in row["probe_capture"]) for row in rows]
            summary["sampled_tokens"] = [row["metrics"]["score_centering_probe/sampled_tokens"] for row in rows]
            summary["coverage"] = [row["metrics"]["score_centering_probe/coverage"] for row in rows]
            if snowball:
                assert summary["capture_tokens"] == [16384] * (repetitions - 1)
        result["serialized"][mode] = summary
    for mode in ("armed", "probe"):
        assert (
            result["serialized"][mode]["optimizer_tensor_bytes"]
            == result["serialized"]["ordinary"]["optimizer_tensor_bytes"]
        )
    result["serialized_comparisons"] = {
        f"{treatment}_versus_{control}": comparison(
            result["serialized"][treatment]["cycle_seconds"]["values"],
            result["serialized"][control]["cycle_seconds"]["values"],
        )
        for treatment, control in (
            ("capture", "ordinary"),
            ("sc", "ordinary"),
            ("sc", "capture"),
            ("armed", "ordinary"),
            ("probe", "armed"),
            ("probe", "ordinary"),
        )
    }
    ordinary, armed, probe = (
        result["serialized"][name]["cycle_seconds"]["values"] for name in ("ordinary", "armed", "probe")
    )
    modeled = [(a * (cadence - 1) + p) / cadence for a, p in zip(armed, probe, strict=True)]
    result["cadence_model"] = {
        "interval": cadence,
        "overhead_versus_off": comparison(modeled, ordinary),
        "seconds_between_observations": statistics.mean(modeled) * cadence,
        "incremental_probe_seconds": describe(p - a for p, a in zip(probe, armed, strict=True)),
        "scope": "Serialized complete cycles, one long-response group per observation; not an asynchronous ceiling.",
    }
    for phase in data["asynchronous_phases"]:
        rows = [row for row in data["asynchronous_windows"] if row["mode"] == phase["mode"] and not row["warmup"]]
        assert len(rows) == async_steps - async_warmup
        assert all(not row["synthetic_advantages"] for row in rows)
        seconds = phase["measured_wall_seconds"]
        tokens = sum(row["tokens"] for row in rows)
        losses = sum(row["loss_tokens"] for row in rows)
        lengths = np.asarray([length for row in rows for length in row["response_lengths"]])
        observed = [row for row in rows if row["metrics"]["score_centering_probe/observed"]]
        result["asynchronous"][phase["mode"]] = {
            "continuous_wall_seconds": seconds,
            "updates": len(rows),
            "updates_per_hour": len(rows) * 3600 / seconds,
            "response_tokens": tokens,
            "response_tokens_per_second": tokens / seconds,
            "loss_tokens_per_second": losses / seconds,
            "responses_per_second": len(lengths) / seconds,
            "cycle_seconds": observed_range(row["total_seconds"] for row in rows),
            "learner_wait_seconds": observed_range(row["timings"]["wait_for_generation_buffer"] for row in rows),
            "learner_wait_fraction": sum(row["timings"]["wait_for_generation_buffer"] for row in rows) / seconds,
            "token_age_mean": (
                sum(row["metrics"]["async/staleness_tokens_mean"] * row["loss_tokens"] for row in rows) / losses
            ),
            "token_age_max": max(row["metrics"]["async/staleness_tokens_max"] for row in rows),
            "source_versions_stamped_fraction": min(row["metrics"]["async/staleness_tokens_stamped"] for row in rows),
            "responses_spanning_versions_fraction": statistics.mean(
                row["metrics"]["async/staleness_tokens_spanning"] for row in rows
            ),
            "response_length_mean": float(lengths.mean()),
            "response_length_std": float(lengths.std()),
            "response_length_range": [int(lengths.min()), int(lengths.max())],
            "response_length_p50_p90": np.percentile(lengths, [50, 90]).tolist(),
            "raw_reward": observed_range(row["metrics"]["reward/avg_raw_reward"] for row in rows),
            "verifier_score_coverage": min(row["metrics"]["reward/verifier_score_coverage"] for row in rows),
            "observations": len(observed),
            "observation_phases": [row["phase"] for row in observed],
            "observation_coverage": [row["metrics"]["score_centering_probe/coverage"] for row in observed],
            "seconds_between_observation_completions": [
                right["cycle_completed_at_monotonic"] - left["cycle_completed_at_monotonic"]
                for left, right in pairwise(observed)
            ],
        }
    by_phase = {row["phase"]: row for row in all_windows}
    result["independent_scalar_replay"] = [replay_scalar(row, by_phase[row["phase"]]) for row in data["probe_evidence"]]
    qualification = data["qualification"]
    for name in ("reference", "temperature_0_7_reference", "compact_temperature_0_7_reference"):
        assert sum(row["positions"] for row in qualification[name]) > 0
        assert max(row["chosen_max_error"] for row in qualification[name]) <= 1e-5
        assert max(row["head_max_error"] for row in qualification[name]) <= 1e-5
    result["qualification"] = {
        key: {name: value for name, value in row.items() if name != "samples"} if isinstance(row, dict) else row
        for key, row in qualification.items()
    }
    result["asynchronous_controls"] = data["asynchronous_controls"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cadence", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"candidate": summarize(args.candidate, args.cadence)}
    if args.baseline:
        result["baseline"] = summarize(args.baseline, args.cadence)
        left, right = result["candidate"], result["baseline"]
        for key in (
            "prompt_sha256",
            "source_selection",
            "controlled_shape",
            "response_lengths",
            "asynchronous_controls",
        ):
            assert left[key] == right[key], f"Unmatched {key}"
        for key in ("lock_sha256", "versions", "policy_parallelism"):
            assert left["runtime"][key] == right["runtime"][key], f"Unmatched {key}"
        result["between_allocations"] = {
            mode: comparison(
                left["serialized"][mode]["cycle_seconds"]["values"],
                right["serialized"][mode]["cycle_seconds"]["values"],
                paired=False,
            )
            for mode in MODES
        }
    result["limits"] = (
        "Serialized intervals use five measured rotating repetitions per Snowball allocation "
        "and describe these allocations only. "
        "Asynchronous modes use continuous windows, normal task rewards and EOS, two warmup and ten measured updates; "
        "Phase boundaries exclude initial warmup and final context teardown, retain finite prefetch, "
        "and can bias short windows. "
        "One allocation per implementation, ordered modes, changing weights, response variation, "
        "caches and shared hardware limit causal claims. "
        "Only exclusive StepWallTime fields partition the cycle; nested timers and process profiles overlap. "
        "Memory device peaks are sampled every 0.1 seconds; allocator caches persist and host peaks are cumulative. "
        "No learning-gain claim."
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": result["candidate"]["sha256"]}))


if __name__ == "__main__":
    main()
