# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert consumed token versions to ages using the actual optimizer-update ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join


def age_summary(counts: Counter[int]) -> dict[str, Any]:
    total = sum(counts.values())
    result: dict[str, Any] = {"loss_tokens": total, "optimizer_age_counts": dict(sorted(counts.items()))}
    if not total:
        return result
    ordered = sorted(counts.items())
    result.update(
        optimizer_age_min=ordered[0][0],
        optimizer_age_max=ordered[-1][0],
        optimizer_age_mean=sum(age * count for age, count in ordered) / total,
    )
    for label, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        threshold = math.ceil(quantile * total)
        seen = 0
        for age, count in ordered:
            seen += count
            if seen >= threshold:
                result[f"optimizer_age_{label}"] = age
                break
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate every exact span and sum applied updates between generating and consuming weights."""
    ledger: dict[int, int] = {}
    for record in records:
        step, applied = record["training_step"], record["optimizer_updates_applied"]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (step, applied)):
            raise ValueError("optimizer-update ledger requires nonnegative integer steps and applied counts")
        if step in ledger:
            raise ValueError(f"duplicate training step {step} in immutable version archive")
        ledger[step] = applied
    steps = []
    all_counts: Counter[int] = Counter()
    all_mixed = all_active_responses = 0
    for record in sorted(records, key=lambda row: row["training_step"]):
        consuming = record["consuming_policy_version"]
        if record["schema_version"] != 1 or consuming != record["training_step"] - 1:
            raise ValueError("version archive has an unsupported schema or consuming step")
        counts: Counter[int] = Counter()
        publication_counts: Counter[str] = Counter()
        mixed = active_responses = 0
        for row in record["rows"]:
            tokens, mask = row["response_token_ids"], row["loss_mask"]
            if len(tokens) != len(mask):
                raise ValueError("consumed response token IDs and loss mask have different lengths")
            offset = 0
            active_versions = set()
            for span in row["policy_version_spans"]:
                start, end, version = span["start"], span["end"], span["version"]
                if start != offset or not start < end <= len(tokens):
                    raise ValueError("policy-version spans do not exactly partition the response")
                offset = end
                eligible = sum(bool(value) for value in mask[start:end])
                if not eligible:
                    continue
                if isinstance(version, bool) or not isinstance(version, int) or not 0 <= version <= consuming:
                    raise ValueError("consumed token has an unknown or future generating-policy version")
                required = range(version + 1, consuming + 1)
                if any(step not in ledger for step in required):
                    raise ValueError("optimizer-age conversion is missing earlier applied-update ledger rows")
                age = sum(ledger[step] for step in required)
                counts[age] += eligible
                publication_counts[str(consuming - version)] += eligible
                active_versions.add(version)
            if offset != len(tokens):
                raise ValueError("policy-version spans leave unmeasured response tokens")
            active_responses += bool(active_versions)
            mixed += len(active_versions) > 1
        if dict(publication_counts) != record["loss_token_publication_gap_counts"]:
            raise ValueError("archived publication-gap histogram disagrees with exact consumed token spans")
        steps.append(
            {
                "training_step": record["training_step"],
                "consuming_policy_version": consuming,
                "optimizer_updates_applied": ledger[record["training_step"]],
                "active_responses": active_responses,
                "mixed_policy_responses": mixed,
                **age_summary(counts),
            }
        )
        all_counts.update(counts)
        all_mixed += mixed
        all_active_responses += active_responses
    return {
        "scope": "Exact loss tokens after trajectory selection; optimizer ages use applied update counts.",
        "steps": steps,
        "aggregate": {
            **age_summary(all_counts),
            "active_responses": all_active_responses,
            "mixed_policy_responses": all_mixed,
            "mixed_policy_response_fraction": all_mixed / all_active_responses if all_active_responses else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    fs, root = url_to_fs(args.archive)
    files = sorted(fs.glob(prefix_join(root, "step-*.json")))
    if not files:
        raise ValueError("consumed policy-version archive contains no completed training batches")
    records, inputs = [], []
    for path in files:
        data = fs.cat_file(path)
        records.append(json.loads(data))
        inputs.append({"uri": fs.unstrip_protocol(path), "sha256": hashlib.sha256(data).hexdigest()})
    result = summarize(records)
    result["inputs"] = inputs
    result["archive"] = args.archive
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"]))


if __name__ == "__main__":
    main()
