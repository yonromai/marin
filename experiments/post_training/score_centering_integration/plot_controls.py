# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot completed primary controls without pooling failed starts or East data."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
MODES = ("ordinary", "capture", "sc", "armed", "probe")


def main():
    plt.switch_backend("Agg")
    data = json.loads((ROOT / "evidence/snowball-controlled-cost-matched-rno.json").read_text())
    for arm in ("baseline", "candidate"):
        assert data[arm]["source_report_complete"] and data[arm]["analysis_scope"] == "controls"
        assert data[arm]["applied_updates"] == 18
        assert all(len(data[arm]["serialized"][mode]["cycle_seconds"]["values"]) == 2 for mode in MODES)
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "score-centering-controls-rno"})
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})
    colors = {"baseline": "#727d89", "candidate": "#2376bb"}
    labels = {"baseline": "Chat objects", "candidate": "Numeric arrays"}
    x = np.arange(len(MODES))
    for arm, offset in (("baseline", -0.19), ("candidate", 0.19)):
        rows = [data[arm]["serialized"][mode]["cycle_seconds"] for mode in MODES]
        means = np.array([row["mean"] for row in rows])
        bounds = np.array([row["descriptive_t95_mean"] for row in rows])
        axes[0].bar(x + offset, means, 0.35, color=colors[arm], label=labels[arm], alpha=0.88)
        axes[0].errorbar(
            x + offset,
            means,
            yerr=np.array([means - bounds[:, 0], bounds[:, 1] - means]),
            fmt="none",
            ecolor="#28323b",
            capsize=3,
            linewidth=1,
        )
        for index, row in enumerate(rows):
            axes[0].scatter([index + offset] * 2, row["values"], color="#172c3b", s=10, zorder=3)
            axes[0].text(
                index + offset,
                means[index] / 2,
                f"{means[index]:.2f}",
                ha="center",
                va="center",
                rotation=90,
                color="white",
                fontsize=9,
            )
    axes[0].set_xticks(x, ["TIS", "Capture /\nrescore", "SC", "Probe\narmed", "Full\nprobe"])
    axes[0].set_ylabel("Complete serialized cycle (seconds)")
    axes[0].set_title("Whole cycles after warmup")
    axes[0].legend(frameon=False, loc="upper left")

    bars = [("baseline", "ordinary"), ("baseline", "sc"), ("candidate", "ordinary"), ("candidate", "sc")]
    phases = [
        ("group_admission", "Batch admission wait", "#7299b5"),
        ("batch_assembly", "Batch assembly", "#f0ad51"),
        ("training_preparation", "Old-policy scoring / preparation", "#c89162"),
        ("policy_training", "Policy training", "#69a788"),
        ("weight_sync", "Weight publication", "#9b8db5"),
        ("other", "Other + metric emission", "#c8cdd1"),
    ]
    bottom = np.zeros(len(bars))
    for key, label, color in phases:
        heights = []
        for arm, mode in bars:
            row = data[arm]["serialized"][mode]
            exclusive = row["exclusive_wall_seconds"]
            if key == "other":
                known = sum(exclusive[f"timing/step_wall/{phase[0]}"]["mean"] for phase in phases[:-1])
                value = row["cycle_seconds"]["mean"] - known
            else:
                value = exclusive[f"timing/step_wall/{key}"]["mean"]
            assert value >= 0
            heights.append(value)
        axes[1].bar(np.arange(len(bars)), heights, 0.65, bottom=bottom, label=label, color=color)
        bottom += heights
    assert np.allclose(bottom, [data[arm]["serialized"][mode]["cycle_seconds"]["mean"] for arm, mode in bars])
    axes[1].set_xticks(np.arange(len(bars)), ["Objects\nTIS", "Objects\nSC", "Arrays\nTIS", "Arrays\nSC"])
    axes[1].set_title("Exclusive driver walls")
    axes[1].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, fontsize=8)
    for axis in axes:
        axis.set_ylim(0, 110)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.16)
        axis.set_axisbelow(True)
    figure.suptitle("Snowball · 4 x 8 H100 learner + 1 x 8 H100 inference · RNO · pinned vLLM", fontsize=13)
    figure.text(
        0.5,
        0.015,
        "Two measured cycles per mode and arm, after one profiled warmup; one allocation per arm. "
        "Whiskers: descriptive 95% t intervals.\n"
        "Admission wait includes generation and its rollout handoff. Nested timers are not stacked. "
        "Failed starts and East data are excluded from this comparison.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.94))
    figure.savefig(ROOT / "controlled-cycle-costs.svg", metadata={"Date": None})
    figure.savefig(ROOT / "controlled-cycle-costs.png", dpi=150, metadata={"Software": "Matplotlib"})
    plt.close(figure)
    print(ROOT / "controlled-cycle-costs.svg")


if __name__ == "__main__":
    main()
