# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize finite native windows and explicitly selected complete rotations."""

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


def summarize(path, cadence, controls_repetitions=None):
    data, digest = read_report(path)
    snowball = data["workload"] == "snowball"
    scope = data.get("scope", "full")
    source_serialized = data["windows"]
    if controls_repetitions is not None:
        assert snowball and scope == "full" and data["complete"] is False
        scope = "controls"
        repetitions = controls_repetitions
        serialized = [row for row in source_serialized if row["repetition"] < repetitions]
    else:
        assert data["complete"] is True
        repetitions = (3 if scope == "controls" else 6) if snowball else 2
        serialized = source_serialized
    async_steps, async_warmup = (12, 2) if snowball else (4, 1)
    async_probe_steps = data.get("asynchronous_controls", {}).get("probe_steps", async_steps)
    native = data.get("native_periodic_windows", [])
    asynchronous = data.get("asynchronous_windows", [])
    assert scope in ("full", "controls", "async")
    assert len(serialized) == (0 if scope == "async" else repetitions * len(MODES))
    if serialized:
        assert {(row["mode"], row["repetition"]) for row in serialized} == {
            (mode, repetition) for mode in MODES for repetition in range(repetitions)
        }
    assert len(native) == (0 if scope == "controls" else 6)
    assert len(asynchronous) == (0 if scope == "controls" else 3 * async_steps + async_probe_steps)
    assert [row["mode"] for row in data.get("asynchronous_phases", [])] == (
        [] if scope == "controls" else ["ordinary", "capture", "sc", "probes"]
    )
    expected = 3 + len(serialized) + len(native) + len(asynchronous)
    if controls_repetitions is not None:
        assert data["applied_updates"] >= expected
        assert all(row["applied_updates"] <= expected for row in serialized)
    else:
        assert data["applied_updates"] == expected
    assert data["runtime"]["lock_sha256"] == LOCK_SHA256
    shape_keys = ("tokens", "loss_tokens", "sequence_shape")
    if serialized:
        for key in shape_keys:
            assert all(row[key] == serialized[0][key] for row in serialized)
        for key in ("prompt_lengths", "row_loss_tokens", "response_lengths"):
            assert all(sorted(row[key]) == sorted(serialized[0][key]) for row in serialized)
    all_windows = serialized + native + asynchronous
    included_phases = {row["phase"] for row in serialized}
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
        "applied_updates": data["applied_updates"],
        "source_report_complete": data["complete"],
        "analysis_scope": scope,
        "analyzed_applied_updates": expected,
        "excluded_completed_serialized_phases": [
            row["phase"] for row in source_serialized if row["phase"] not in included_phases
        ],
        "runtime": data["runtime"],
        "source_selection": data["source_selection"],
        "prompt_sha256": data["prompt_sha256"],
        "controlled_shape": {key: serialized[0][key] for key in shape_keys} if serialized else None,
        "response_lengths": dict(Counter(serialized[0]["response_lengths"])) if serialized else {},
        "setup_timings": data["setup_timings"],
        "serialized": {},
        "asynchronous": {},
    }
    for mode in MODES if serialized else ():
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
    for mode in ("armed", "probe") if serialized else ():
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
            (
                ("capture", "ordinary"),
                ("sc", "ordinary"),
                ("sc", "capture"),
                ("armed", "ordinary"),
                ("probe", "armed"),
                ("probe", "ordinary"),
            )
            if serialized
            else ()
        )
    }
    if serialized:
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
    for phase in data.get("asynchronous_phases", []):
        phase_rows = [row for row in asynchronous if row["mode"] == phase["mode"]]
        rows = [row for row in phase_rows if not row["warmup"]]
        assert len(rows) == (async_probe_steps if phase["mode"] == "probes" else async_steps) - async_warmup
        assert all(not row["synthetic_advantages"] for row in rows)
        seconds = phase["measured_wall_seconds"]
        tokens = sum(row["tokens"] for row in rows)
        losses = sum(row["loss_tokens"] for row in rows)
        lengths = np.asarray([length for row in rows for length in row["response_lengths"]])
        observed = [row for row in rows if row["metrics"]["score_centering_probe/observed"]]
        all_observed = [row for row in phase_rows if row["metrics"]["score_centering_probe/observed"]]
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
            "observations_including_warmup": len(all_observed),
            "observation_phases_including_warmup": [row["phase"] for row in all_observed],
            "observation_coverage_including_warmup": [
                row["metrics"]["score_centering_probe/coverage"] for row in all_observed
            ],
            "seconds_between_observation_completions_including_warmup": [
                right["cycle_completed_at_monotonic"] - left["cycle_completed_at_monotonic"]
                for left, right in pairwise(all_observed)
            ],
        }
    by_phase = {row["phase"]: row for row in all_windows}
    result["independent_scalar_replay"] = [
        replay_scalar(row, by_phase[row["phase"]]) for row in data.get("probe_evidence", []) if row["phase"] in by_phase
    ]
    qualification = data["qualification"]
    if not snowball:
        cases = qualification["native_chat_compact_parity"]
        assert len(cases) == 4
        assert {(row["temperature"], row["response_length"]) for row in cases} == {
            (1.0, 128),
            (1.0, 4096),
            (0.7, 128),
            (0.7, 4096),
        }
        assert all(
            row["tokens_message_finish_source_chosen_head_exact"]
            and row["display_logprobs_absent"]
            and row["ids_dtype"] == "int32"
            and row["scores_dtype"] == "float32"
            for row in cases
        )
    reference_names = ["reference", "temperature_0_7_reference"]
    if serialized:
        reference_names.append("compact_temperature_0_7_reference")
    for name in reference_names:
        assert sum(row["positions"] for row in qualification[name]) > 0
        assert max(row["chosen_max_error"] for row in qualification[name]) <= 1e-5
        assert max(row["head_max_error"] for row in qualification[name]) <= 1e-5
    result["qualification"] = {
        key: {name: value for name, value in row.items() if name != "samples"} if isinstance(row, dict) else row
        for key, row in qualification.items()
    }
    result["asynchronous_controls"] = data.get("asynchronous_controls")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-controls-repetitions", type=int, choices=(3,))
    parser.add_argument("--cadence", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"candidate": summarize(args.candidate, args.cadence)}
    if args.baseline:
        result["baseline"] = summarize(args.baseline, args.cadence, args.baseline_controls_repetitions)
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
            for mode in left["serialized"]
        }
        result["between_allocations_normalized_to_ordinary"] = {
            mode: comparison(
                [
                    value / ordinary
                    for value, ordinary in zip(
                        left["serialized"][mode]["cycle_seconds"]["values"],
                        left["serialized"]["ordinary"]["cycle_seconds"]["values"],
                        strict=True,
                    )
                ],
                [
                    value / ordinary
                    for value, ordinary in zip(
                        right["serialized"][mode]["cycle_seconds"]["values"],
                        right["serialized"]["ordinary"]["cycle_seconds"]["values"],
                        strict=True,
                    )
                ],
                paired=False,
            )
            for mode in left["serialized"]
        }
        result["asynchronous_between_allocations"] = {
            mode: {
                f"{key}_percent": 100 * (left["asynchronous"][mode][key] / right["asynchronous"][mode][key] - 1)
                for key in ("updates_per_hour", "response_tokens_per_second", "loss_tokens_per_second")
            }
            for mode in left["asynchronous"]
        }
    result["limits"] = (
        "Serialized intervals describe the measured rotating repetitions of these allocations only. "
        "A preregistered incomplete baseline prefix retains two complete measured rotations after warmup; "
        "the raw complete flag and extra completed phases remain explicit. This limits precision. "
        "Asynchronous modes use continuous windows, normal task rewards and EOS with declared warmup/measured counts; "
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
