# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw the full frozen confirmation with training seeds as the uncertainty unit.

Run after terminal collection, with the original frozen analysis code present.
Curve intervals are descriptive, pointwise 95% intervals. Final quality contrasts
use the predeclared three-contrast family interval. Time and compute include
startup, evaluation and checkpoint work according to each panel's stated endpoint.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from experiments.post_training.analyze_current_score_centering_confirmation import (
    MAIN_ARMS,
    analyze_confirmation,
    paired_interval,
    primary_curve,
)

ARM_LABELS = {
    "older_tis32": "Older TIS, capture 32",
    "older_sc32": "Older TIS + SC, capture 32",
    "fresh_tis32": "Fresh TIS, capture 32",
    "older_incumbent": "Older incumbent PPO",
}


def task_hours_to_elapsed(tasks: list[dict], elapsed_seconds: float) -> float:
    """Count each eight-GPU task's allocation only up to the evaluation endpoint."""
    endpoint_ms = min(task["started_at_ms"] for task in tasks) + 1000 * elapsed_seconds
    return sum(
        8 * max(0, min(task["finished_at_ms"], endpoint_ms) - task["started_at_ms"]) / 3_600_000 for task in tasks
    )


def plot_confirmation(runs: list[dict], protocol: dict, base: Path, output: Path) -> None:
    # This validates the entire 51-run design before reading quality or drawing figures.
    result = analyze_confirmation(runs, protocol)
    output.mkdir(parents=True, exist_ok=True)
    denominator = protocol["primary_members"]
    steps = protocol["evaluation_steps"]
    points = []
    for run in runs:
        if run["arm"] not in MAIN_ARMS:
            continue
        raw = (base / run["source_metrics"]["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != run["source_metrics"]["sha256"]:
            raise ValueError("source metrics differ from their terminal audit")
        metrics = json.loads(gzip.decompress(raw))
        train = {row["step"]: row for row in metrics if row["kind"] == "train"}
        if set(train) != set(range(1, protocol["completed_training_steps"] + 1)) or sum(
            row["kind"] == "train" for row in metrics
        ) != len(train):
            raise ValueError("learning curve lacks complete consumed-token metrics")
        curve = primary_curve(run["endpoint"], protocol)
        for step in steps:
            elapsed = run["evaluation_elapsed_seconds"][str(step)]
            if not 0 <= elapsed <= run["allocated_wall_seconds"]:
                raise ValueError("evaluation timestamp falls outside allocated run time")
            points.append(
                {
                    "arm": run["arm"],
                    "seed": run["seed"],
                    "step": step,
                    "loss_tokens_millions": (
                        sum(train[index]["async/performance/consumed_loss_tokens"] for index in range(1, step + 1)) / 1e6
                    ),
                    "elapsed_to_second_evaluation_hours": elapsed / 3600,
                    "reserved_h100_task_hours_to_second_evaluation": task_hours_to_elapsed(run["tasks"], elapsed),
                    "completed_correct_percent": 100 * curve[step] / denominator,
                }
            )

    coordinates = (
        ("step", "Applied optimizer updates"),
        ("loss_tokens_millions", "Consumed loss tokens (millions)"),
        ("elapsed_to_second_evaluation_hours", "Allocated hours to evaluation"),
        ("reserved_h100_task_hours_to_second_evaluation", "Reserved H100 task-hours to evaluation"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True, layout="constrained")
    for arm in MAIN_ARMS:
        selected = [[row for row in points if row["arm"] == arm and row["step"] == step] for step in steps]
        intervals = [paired_interval([row["completed_correct_percent"] for row in rows]) for rows in selected]
        for ax, (key, label) in zip(axes, coordinates, strict=True):
            xs = [statistics.mean(row[key] for row in rows) for rows in selected]
            ys = [interval["mean"] for interval in intervals]
            line = ax.plot(xs, ys, marker="o", markersize=4, label=ARM_LABELS[arm])[0]
            ax.fill_between(
                xs,
                [interval["low"] for interval in intervals],
                [interval["high"] for interval in intervals],
                color=line.get_color(),
                alpha=0.12,
            )
            ax.set_xlabel(label)
            ax.grid(alpha=0.25)
    for ax in axes:
        ax.set_xlim(left=0)
    axes[0].set_ylabel(f"Completed correct (% of {denominator} held-out members)")
    fig.suptitle(f"Snowball confirmation: mean of {len(protocol['seeds'])} seeds; pointwise 95% intervals")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    fig.savefig(output / "score_centering_current_confirmation_curves.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    comparisons = result["contrasts"]
    labels = ["SC vs older TIS", "SC vs fresh TIS", "SC vs incumbent"]
    for ax, key, factor, reference, title in (
        (
            axes[0],
            "three_contrast_family_95_percent",
            100 / denominator,
            0,
            "Final quality difference (pp)\nFamily 95% intervals",
        ),
        (
            axes[1],
            "allocated_elapsed_ratio_individual_95_percent",
            1,
            1,
            "Total allocated time ratio\nIndividual 95% intervals",
        ),
        (
            axes[2],
            "reserved_compute_ratio_individual_95_percent",
            1,
            1,
            "Total reserved compute ratio\nIndividual 95% intervals",
        ),
    ):
        for index, comparison in enumerate(comparisons):
            interval = comparison[key]
            mean, low, high = (factor * interval[name] for name in ("mean", "low", "high"))
            ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt="o", capsize=4)
        ax.axvline(reference, color="0.4", linestyle="--", linewidth=1)
        ax.set_yticks(range(3), labels)
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Step 40: paired seeds; total costs include startup, evaluations, checkpoint and teardown")
    fig.savefig(output / "score_centering_current_confirmation_contrasts.svg")
    plt.close(fig)
    (output / "score_centering_current_confirmation_curve_points.json").write_text(json.dumps(points, indent=2) + "\n")
    for filename in output.glob("score_centering_current_confirmation_*.svg"):
        filename.write_text("\n".join(line.rstrip() for line in filename.read_text().splitlines()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.protocol_sha256:
        raise ValueError("plot protocol differs from immutable publication")
    protocol = json.loads(raw)
    base = args.protocol.resolve().parent.parent
    if hashlib.sha256((base / protocol["analysis_code"]).read_bytes()).hexdigest() != protocol["analysis_sha256"]:
        raise ValueError("plots require the original frozen primary analysis code")
    plot_confirmation(json.loads(args.runs.read_text()), protocol, base, args.output)
    print(f"Wrote full confirmation figures to {args.output}")


if __name__ == "__main__":
    main()
