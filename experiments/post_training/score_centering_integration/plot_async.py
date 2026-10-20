# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Show both throughput denominators alongside response lengths and waiting."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
MODES = ("ordinary", "capture", "sc", "probes")


def main():
    plt.switch_backend("Agg")
    data = json.loads((ROOT / "evidence/snowball-asynchronous-matched-rno.json").read_text())
    for arm in ("baseline", "candidate"):
        assert data[arm]["source_report_complete"] and data[arm]["analysis_scope"] == "async"
        assert data[arm]["applied_updates"] == 67
        assert [data[arm]["asynchronous"][mode]["updates"] for mode in MODES] == [10, 10, 10, 20]
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "score-centering-async-rno"})
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    colors = {"baseline": "#727d89", "candidate": "#2376bb"}
    labels = {"baseline": "Chat objects", "candidate": "Numeric arrays"}
    panels = [
        ("updates_per_hour", 1, "Applied updates / hour", ".1f"),
        ("response_tokens_per_second", 1, "Response tokens / second", ".0f"),
        ("response_length_mean", 1, "Mean response length (tokens)", ".0f"),
        ("learner_wait_fraction", 100, "Learner wait (% of measured window)", ".1f"),
    ]
    x = np.arange(len(MODES))
    for axis, (key, scale, title, spec) in zip(axes.flat, panels, strict=True):
        highest = 0
        for arm, offset in (("baseline", -0.19), ("candidate", 0.19)):
            values = [data[arm]["asynchronous"][mode][key] * scale for mode in MODES]
            axis.bar(x + offset, values, 0.35, color=colors[arm], label=labels[arm], alpha=0.9)
            highest = max(highest, max(values))
            for index, value in enumerate(values):
                axis.annotate(
                    format(value, spec),
                    (index + offset, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        axis.set_title(title, loc="left")
        axis.set_xticks(x, ["TIS", "Capture /\nrescore", "SC", "Probes"])
        axis.set_ylim(0, highest * 1.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.16)
        axis.set_axisbelow(True)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    figure.suptitle("Real-reward Snowball · RNO · 4 x 8 H100 learner + 1 x 8 H100 inference", fontsize=13)
    figure.text(
        0.5,
        0.025,
        "Both arms: 67 applied updates; 10 measured updates per cost mode and 20 for probes, after two warmups.\n"
        "One allocation per arm; ordered modes, changing weights, finite prefetch and stochastic response lengths.\n"
        "Window totals are descriptive. The different response lengths prevent an isolated causal speedup claim.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.92), h_pad=2)
    svg = ROOT / "asynchronous-throughput.svg"
    figure.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    figure.savefig(ROOT / "asynchronous-throughput.png", dpi=150, metadata={"Software": "Matplotlib"})
    plt.close(figure)
    print(ROOT / "asynchronous-throughput.svg")


if __name__ == "__main__":
    main()
