#!/usr/bin/env python3
"""Plot full-replay mean optimizer core time from an archived summary."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.archive / "summary.json").read_text())
    rows = {(row["threads"], row["backend"]): row for row in summary["timing"]}
    details = {(row["threads"], row["backend"]): row["stages_ms_per_call"]
               for row in summary["detailed_stage_statistics"]}
    threads = [1]
    backends = ["native", "native_batch", "ceres_scalar", "ceres_batch"]
    colors = ["#2477b9", "#e58b19", "#259f5c", "#c73e43"]
    plt.rcParams.update({"font.size": 13, "pdf.fonttype": 42, "hatch.linewidth": 0.4})
    fig, ax = plt.subplots(figsize=(11, 7.5))
    x = np.arange(len(backends))
    width = 0.62
    for index, (backend, color) in enumerate(zip(backends, colors)):
        values = np.array([rows[t, backend]["core_mean_ms_per_call"] for t in threads])
        build = np.array([details[t, backend]["build_ms"]["mean"] for t in threads])
        solve = np.array([details[t, backend]["solve_ms"]["mean"] for t in threads])
        # Keep the original total exactly, including reset, destruction and
        # the small timing gaps outside the explicitly timed phases.
        other = values - build - solve
        phases = np.array([build, solve, other])
        if (not np.all(np.isfinite(phases)) or np.any(phases < 0)
                or not np.allclose(phases.sum(axis=0), values)):
            raise ValueError(f"Invalid phase decomposition for {backend}")
        positions = np.array([x[index]])
        bottom = np.zeros(len(threads))
        for heights, hatch in [(build, "//////"), (solve, ""), (other, "\\\\\\\\\\\\")]:
            ax.bar(positions, heights, width, bottom=bottom, facecolor=color,
                   edgecolor="#000000", linewidth=1.0, hatch=hatch)
            bottom += heights
        for xpos, total in zip(positions, values):
            ax.annotate(f"{total:.2f}", (xpos, total), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=11)
    ax.set_xticks(x, ["slict", "slict+splbatch", "ceres", "ceres+splbatch"])
    ax.margins(x=0.1)
    ax.set_ylabel("Mean optimization core time [ms / call]")
    ax.set_title("NTU VIRAL eee_03 replay", pad=15, fontsize=19)
    ax.set_ylim(0, 40)
    ax.set_yticks(np.arange(0, 41, 5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.22)
    fig.legend(handles=[Patch(facecolor="#dddddd", edgecolor="black", hatch=h, label=label)
                        for label, h in [("Build", "//////"), ("Solve", ""),
                                         ("Other (reset / destroy / overhead)", "\\\\\\\\\\\\")]],
               loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False,
               columnspacing=1.0, handlelength=1.8, handletextpad=0.5)
    fig.text(0.5, 0.035, "Bar tops: total ms / call; 5,418 calls per configuration.\n"
             "Solve includes evaluation, assembly, linear solve and update.",
             ha="center", va="center", fontsize=11, color="#555555")
    fig.subplots_adjust(left=0.105, right=0.975, top=0.79, bottom=0.16)
    for extension in ("png", "pdf"):
        output = args.archive / f"optimization_timing.{extension}"
        fig.savefig(output, dpi=220, facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
