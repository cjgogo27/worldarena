from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_figure() -> plt.Figure:
    rollout_times = np.array([1, 3, 5])
    distance_thresholds = np.array([10, 25, 200, 750, 2000])

    distance_accuracy = np.array(
        [
            [0.1410, 0.1650, 0.3420, 0.6830, 0.8210],
            [0.1680, 0.1940, 0.3520, 0.6910, 0.8290],
            [0.1780, 0.2050, 0.3570, 0.6960, 0.8330],
        ]
    )

    prf_scores = np.array(
        [
            [58.64, 60.90, 59.75],
            [59.58, 61.72, 62.63],
            [62.04, 62.04, 62.97],
        ]
    )

    per_sample_seconds = np.array([6.00, 21.00, 47.00])

    body_font_pt = 34
    label_font_pt = 38
    tick_font_pt = 30
    legend_font_pt = 26

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": body_font_pt,
            "axes.labelsize": label_font_pt,
            "xtick.labelsize": tick_font_pt,
            "ytick.labelsize": tick_font_pt,
            "legend.fontsize": legend_font_pt,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 1.0,
        }
    )

    rollout_colors = ["#4C78A8", "#F58518", "#54A24B"]
    metric_colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 15.8), constrained_layout=True)
    ax_left, ax_right = axes

    x_dist = np.arange(len(distance_thresholds))
    width_dist = 0.24
    for idx, rollout in enumerate(rollout_times):
        bars = ax_left.bar(
            x_dist + (idx - 1) * width_dist,
            distance_accuracy[idx],
            width=width_dist,
            color=rollout_colors[idx],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.92,
            label=f"Rollout={rollout}",
        )
        if rollout == 5:
            for bar in bars:
                bar.set_hatch("///")
    ax_left.set_xticks(x_dist)
    ax_left.set_xticklabels([f"{t}\nkm" for t in distance_thresholds])
    ax_left.set_ylim(0, 0.92)
    ax_left.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax_left.set_xlabel("Distance threshold")
    ax_left.set_ylabel("Accuracy")
    ax_left.tick_params(axis="x", labelrotation=0)
    ax_left.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=True,
    )

    metric_labels = ["Precision", "Recall", "F1"]
    x = np.arange(len(rollout_times))
    width = 0.22
    for idx, metric_name in enumerate(metric_labels):
        ax_right.bar(
            x + (idx - 1) * width,
            prf_scores[:, idx],
            width=width,
            color=metric_colors[idx],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.92,
            label=metric_name,
        )

    ax_right.set_xticks(x)
    ax_right.set_xticklabels([str(v) for v in rollout_times])
    ax_right.set_ylim(30.0, 66.0)
    ax_right.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax_right.set_xlabel("Rollout times")
    ax_right.set_ylabel("PRF score")

    ax_time = ax_right.twinx()
    time_line_color = "#8C2D04"
    ax_time.plot(
        x,
        per_sample_seconds,
        color=time_line_color,
        marker="D",
        linestyle="--",
        linewidth=3.2,
        markersize=10.5,
        label="Time / sample",
    )
    for xpos, t in zip(x, per_sample_seconds):
        ax_time.text(
            xpos,
            t + 1.1,
            f"{t:.2f}",
            ha="center",
            va="bottom",
            fontsize=28,
            color=time_line_color,
        )
    ax_time.set_ylim(3.0, 55.0)
    ax_time.set_ylabel("Time / sample")

    handles_left, labels_left = ax_right.get_legend_handles_labels()
    handles_right, labels_right = ax_time.get_legend_handles_labels()
    ax_right.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=True,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rollout sensitivity figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to save generated figure files.",
    )
    parser.add_argument(
        "--stem",
        default="rollout_sensitivity_georc",
        help="Output file stem (without extension).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    png_path = args.output_dir / f"{args.stem}.png"
    pdf_path = args.output_dir / f"{args.stem}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
