# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize finite native windows; intervals describe these serialized allocations."""

import argparse
import gzip
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

T975 = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.77644510520,
    5: 2.57058183564,
    6: 2.44691185114,
    7: 2.36462425159,
    8: 2.30600413503,
}
TIMERS = (
    "wait_for_generation_buffer",
    "assemble_generation_group_mini_batch",
    "postprocess_trajectory_batch",
    "convert_to_training_input",
    "fwd_logprobs_values_reward",
    "train_critic_and_policy",
    "policy_train",
    "backload_policy_optimizer_to_gpu",
    "offload_policy_optimizer_to_cpu",
    "sync_weights",
    "score_centering_probe_collation",
    "score_centering_probe_summary",
)


def describe(values):
    values = list(values)
    result = {"mean": statistics.mean(values), "min": min(values), "max": max(values), "values": values}
    if len(values) > 1:
        half = T975[len(values) - 1] * statistics.stdev(values) / math.sqrt(len(values))
        result["descriptive_t95_mean"] = [result["mean"] - half, result["mean"] + half]
    return result


def comparison(treatment, control, *, paired=True):
    left, right = list(treatment), list(control)
    assert left and len(left) == len(right)
    result = {"ratio_of_means_percent": 100 * (statistics.mean(left) / statistics.mean(right) - 1)}
    log_left, log_right = list(map(math.log, left)), list(map(math.log, right))
    center = statistics.mean(log_left) - statistics.mean(log_right)
    result["geometric_percent"] = 100 * math.expm1(center)
    if len(left) > 1:
        if paired:
            differences = [a - b for a, b in zip(log_left, log_right, strict=True)]
            error = statistics.stdev(differences) / math.sqrt(len(differences))
            df = len(differences) - 1
        else:
            a, b = statistics.variance(log_left) / len(left), statistics.variance(log_right) / len(right)
            error = math.sqrt(a + b)
            df = (a + b) ** 2 / (a * a / (len(left) - 1) + b * b / (len(right) - 1)) if a + b else 4
        # Rounding Welch df down makes the table-based interval conservative.
        half = T975[max(1, min(8, math.floor(df + 1e-9)))] * error
        result["descriptive_t95_percent"] = [100 * math.expm1(center - half), 100 * math.expm1(center + half)]
        result["degrees_of_freedom"] = df
    result["pairing"] = "same-repetition within allocation" if paired else "independent allocations; Welch log times"
    return result


def devices(value):
    if isinstance(value, dict):
        if "peak_allocated" in value:
            yield value
        else:
            for nested in value.values():
                yield from devices(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from devices(nested)


def replay_scalar(evidence, window):
    """Independently evaluate the current=old special case with NumPy FP64."""
    current = np.asarray(evidence["current_old_head"], dtype=np.float64)
    behavior = np.asarray(evidence["behavior_head"], dtype=np.float64)
    advantages = np.asarray(evidence["advantages"], dtype=np.float64)
    p, q = np.exp(current), np.exp(behavior)
    p_tail, q_tail = np.maximum(1 - p.sum(-1), 1e-6), np.maximum(1 - q.sum(-1), 1e-6)
    weighted = q * np.exp(np.minimum(current - behavior, math.log(2)))
    tail_scale = q_tail / p_tail * np.minimum(p_tail / q_tail, 2)
    unit = ((weighted - tail_scale[:, None] * p) * current).sum(-1)
    correction = advantages * unit
    expected = {
        "hypothetical_loss_mean": correction.mean(),
        "hypothetical_loss_abs_mean": np.abs(correction).mean(),
        "unit_advantage_loss_mean": unit.mean(),
        "unit_advantage_loss_abs_mean": np.abs(unit).mean(),
    }
    actual = window["metrics"]
    errors = {name: abs(float(value) - actual[f"score_centering_probe/{name}"]) for name, value in expected.items()}
    return {
        "phase": evidence["phase"],
        "positions": len(unit),
        "fp64_values": {k: float(v) for k, v in expected.items()},
        "abs_error_from_recorded_fp32": errors,
    }


def summarize(path, cadence):
    stored = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(stored)
    elif path.suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(stored)) as archive:
            assert len(archive.namelist()) == 1
            raw = archive.read(archive.namelist()[0])
    else:
        raw = stored
    data = json.loads(raw)
    assert data["complete"] is True, "Complete worker report required"
    candidate = data["variant"] == "candidate"
    snowball = data["workload"] == "snowball"
    modes = ("ordinary", "capture", "sc", "armed", "probe") if candidate else ("ordinary", "capture", "sc")
    repetitions = 6 if snowball else 2
    expected_updates = 3 + repetitions * len(modes) + (6 if candidate else 0)
    assert data["applied_updates"] == expected_updates
    assert len(data["windows"]) == repetitions * len(modes)
    windows = data["windows"]
    assert len({row["tokens"] for row in windows}) == len({row["loss_tokens"] for row in windows}) == 1
    assert len({tuple(row["sequence_shape"]) for row in windows}) == 1
    assert all(sorted(row["prompt_lengths"]) == sorted(windows[0]["prompt_lengths"]) for row in windows)
    assert all(sorted(row["row_loss_tokens"]) == sorted(windows[0]["row_loss_tokens"]) for row in windows)
    for row in windows + data.get("native_periodic_windows", []):
        metrics = row["metrics"]
        assert metrics["policy/policy_update_steps"] == 1 and metrics["policy/skipped_steps"] == 0
        assert metrics["async/staleness_tokens_max"] == 0 and metrics["async/staleness_tokens_stamped"] == 1
        norm = metrics["policy/raw_grad_norm"]
        assert math.isfinite(norm) and (norm > 0 if row["synthetic_advantages"] else norm >= 0)
        if candidate:
            assert "inference_sampling" in row, "Final candidate must record effective native sampling"
            sampling = row["inference_sampling"]
            assert len(sampling) == 8
            assert all(
                params == {"temperature": row.get("temperature", 1.0), "top_p": 1.0, "top_k": -1, "min_p": 0.0}
                for params in sampling
            ), f"Native sampling differs from declared controls: {row['phase']}"
    result = {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stored_sha256": hashlib.sha256(stored).hexdigest(),
        "workload": data["workload"],
        "variant": data["variant"],
        "product": data["candidate"],
        "applied_updates": expected_updates,
        "prompt_sha256": data["prompt_sha256"],
        "source_selection": data.get("source_selection"),
        "native_sampling_recorded": all("inference_sampling" in row for row in windows),
        "runtime": data["runtime"],
        "setup_sync_timings": data["setup_timings"],
        "tokens_per_window": windows[0]["tokens"],
        "loss_tokens_per_window": windows[0]["loss_tokens"],
        "sequence_shape": windows[0]["sequence_shape"],
        "prompt_length_counts": dict(Counter(windows[0]["prompt_lengths"])),
        "row_loss_token_counts": dict(Counter(windows[0]["row_loss_tokens"])),
        "response_length_counts": dict(Counter(windows[0]["response_lengths"])),
        "modes": {},
        "comparisons": {},
    }
    frontend = data.get("profiles", {}).get("capture", {}).get("frontend", [])
    result["capture_frontend_profile_selected"] = {
        name: {
            metric: sum(row["selected"].get(name, {}).get(metric, 0) for row in frontend)
            for metric in ("calls", "own_seconds", "cumulative_seconds")
        }
        for name in sorted({name for row in frontend for name in row["selected"]})
    }
    result["profile_limit"] = (
        "Profiled warmup only; cumulative process timings can overlap and are not a cycle-time decomposition."
    )
    for mode in modes:
        rows = sorted(
            (row for row in windows if row["mode"] == mode and not row["warmup"]), key=lambda row: row["repetition"]
        )
        assert len(rows) == repetitions - 1
        summary = {
            "measured_windows": len(rows),
            "total_seconds": describe(row["total_seconds"] for row in rows),
            "timings_seconds": {key: describe(row["timings"].get(key, 0) for row in rows) for key in TIMERS},
            "response_tokens_per_second": sum(row["tokens"] for row in rows) / sum(row["total_seconds"] for row in rows),
            "loss_tokens_per_second": (
                sum(row["loss_tokens"] for row in rows) / sum(row["total_seconds"] for row in rows)
            ),
            "excluding_sync_seconds": describe(row["total_seconds"] - row["timings"]["sync_weights"] for row in rows),
            "optimizer_tensor_bytes": rows[0]["optimizer_input_tensor_bytes"],
            "driver_rss_current_max": max(row["driver_host"]["rss_current"] for row in rows),
            "driver_rss_cumulative_peak": max(row["driver_host"]["rss_cumulative_peak"] for row in rows),
        }
        if "metric_emission_seconds" in rows[0]:
            summary["metric_emission_seconds"] = describe(row["metric_emission_seconds"] for row in rows)
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
            summary["probe_capture_response_tokens"] = [
                sum(row["response_tokens"] for row in window.get("probe_capture", [])) for window in rows
            ]
            summary["probe_sampled_tokens"] = [row["metrics"]["score_centering_probe/sampled_tokens"] for row in rows]
            summary["probe_coverage"] = [row["metrics"]["score_centering_probe/coverage"] for row in rows]
            if snowball:
                assert summary["probe_capture_response_tokens"] == [16384] * (repetitions - 1)
        result["modes"][mode] = summary
    pairs = [("capture", "ordinary"), ("sc", "capture"), ("sc", "ordinary")]
    if candidate:
        pairs.extend((("armed", "ordinary"), ("probe", "armed"), ("probe", "ordinary")))
    for treatment, control in pairs:
        result["comparisons"][f"{treatment}_versus_{control}"] = {
            "cycle": comparison(
                result["modes"][treatment]["total_seconds"]["values"],
                result["modes"][control]["total_seconds"]["values"],
            ),
            "excluding_sync": comparison(
                result["modes"][treatment]["excluding_sync_seconds"]["values"],
                result["modes"][control]["excluding_sync_seconds"]["values"],
            ),
        }
    qualification = data["qualification"]
    if candidate and not snowball:
        parity = qualification["native_chat_id_only_parity"]
        assert parity["same_engine"] and parity["temperatures"] == [1.0, 0.7]
        assert parity["tokens_message_source_steps_chosen_and_head_scores_exact"]
        assert parity["chosen_display_bytes_suppressed"]
    result["qualification"] = {
        key: value
        for key, value in qualification.items()
        if key not in ("prefix_replay", "temperature_0_7_prefix_replay", "stale_prefix_replay")
    }
    for key in ("prefix_replay", "temperature_0_7_prefix_replay", "stale_prefix_replay"):
        result["qualification"][key] = {name: value for name, value in qualification[key].items() if name != "samples"}
    reference_keys = ["reference", "temperature_0_7_reference"]
    if candidate:
        reference_keys.append("compact_temperature_0_7_reference")
    for key in reference_keys:
        rows = qualification[key]
        assert sum(row["positions"] for row in rows) > 0
        assert max(row["chosen_max_error"] for row in rows) <= 1e-5
        assert max(row["head_max_error"] for row in rows) <= 1e-5
    if candidate:
        ordinary = result["modes"]["ordinary"]["total_seconds"]["values"]
        armed, probe = (result["modes"][name]["total_seconds"]["values"] for name in ("armed", "probe"))
        amortized = [(a * (cadence - 1) + p) / cadence for a, p in zip(armed, probe, strict=True)]
        result["default_cadence_model"] = {
            "interval": cadence,
            "average_cycle_overhead": comparison(amortized, ordinary),
            "seconds_between_observations": statistics.mean(armed) * (cadence - 1) + statistics.mean(probe),
            "incremental_probe_seconds": describe(p - a for p, a in zip(probe, armed, strict=True)),
            "assumptions": (
                "One complete probe cycle per interval; remaining cycles armed; startup probes immediately. "
                "A capture group with four responses of 4096 tokens bounds collection in this workload. "
                "Within-run noise, admission rejection and async overlap limit extrapolation."
            ),
        }
        for name in ("armed", "probe"):
            assert (
                result["modes"][name]["optimizer_tensor_bytes"] == result["modes"]["ordinary"]["optimizer_tensor_bytes"]
            )
        natural = data["native_periodic_windows"]
        assert len(natural) == 6 and qualification["repeated_periodic_observations"] >= 3
        result["native_periodic_windows"] = [
            {
                "phase": row["phase"],
                "total_seconds": row["total_seconds"],
                "response_tokens": row["tokens"],
                "loss_tokens": row["loss_tokens"],
                "probe_capture": row.get("probe_capture"),
                "metrics": {
                    key: value
                    for key, value in row["metrics"].items()
                    if key.startswith(
                        (
                            "score_centering_probe/",
                            "policy/log_ratio",
                            "policy/raw_grad_norm",
                            "policy/policy_update",
                            "policy/skipped",
                            "async/staleness_tokens",
                        )
                    )
                },
            }
            for row in natural
        ]
        by_phase = {row["phase"]: row for row in windows + natural}
        result["independent_scalar_replay"] = [
            replay_scalar(row, by_phase[row["phase"]]) for row in data["probe_evidence"]
        ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cadence", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.cadence >= 1
    result = {"candidate": summarize(args.candidate, args.cadence)}
    if args.baseline:
        result["baseline"] = summarize(args.baseline, args.cadence)
        left, right = result["candidate"], result["baseline"]
        for key in (
            "prompt_sha256",
            "source_selection",
            "tokens_per_window",
            "loss_tokens_per_window",
            "sequence_shape",
            "prompt_length_counts",
            "row_loss_token_counts",
        ):
            assert left[key] == right[key], f"Unmatched {key}"
        for key in ("lock_sha256", "versions", "policy_parallelism"):
            assert left["runtime"][key] == right["runtime"][key], f"Unmatched runtime {key}"
        for key, value in left["runtime"]["sampling"].items():
            if key != "logprobs":
                assert value == right["runtime"]["sampling"][key], f"Unmatched sampling {key}"
        result["between_allocations"] = {
            mode: comparison(
                left["modes"][mode]["total_seconds"]["values"],
                right["modes"][mode]["total_seconds"]["values"],
                paired=False,
            )
            for mode in ("ordinary", "capture", "sc")
        }
    result["limits"] = (
        "Serialized native core plus normal metric emission on Snowball; persistent caches and changing weights, "
        "fixed timing advantages and masks. One allocation per implementation. Descriptive Student-t paired log "
        "intervals assume approximately independent normal log differences; Welch comparisons omit unmeasured "
        "allocation effects. No production async throughput or learning-gain claim. Timer fields can overlap; "
        "memory device peaks sample every 0.1 seconds and host peaks are cumulative."
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "modes": {
                    name: {key: value[key] for key in ("response_tokens_per_second", "total_seconds")}
                    for name, value in result["candidate"]["modes"].items()
                },
                "comparisons": result["candidate"]["comparisons"],
                "cadence": result["candidate"].get("default_cadence_model"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
