# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import run_kan_moo_convergence as exp


MARKERS = {
    "Knowledge-constrained NSGA-II": "D",
    "NSGA-II": "^",
    "NSGA-III": "o",
    "MOEA/D": "s",
}
LABELS = {
    "Knowledge-constrained NSGA-II": "Knowledge-constrained NSGA-II",
    "NSGA-II": "NSGA-II",
    "NSGA-III": "NSGA-III",
    "MOEA/D": "MOEA/D",
}


def output_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def load_formal_moead_config(results_dir: Path) -> None:
    exp.configure_mode("full30", "results")
    exp.maybe_load_selected_moead_config(results_dir)


def extract_front_for_method(ctx: exp.ProblemContext, method: str, seed: int) -> pd.DataFrame:
    if method == "MOEA/D":
        _, _, x_final, f_raw_final, _ = exp.run_moead(ctx, seed)
    else:
        _, _, x_final, f_raw_final, _ = exp.run_nsga(ctx, method, seed)
    f_report = exp.selection_objectives(ctx, x_final, f_raw_final)
    nd = exp.nondominated_indices(f_report)
    f_raw_front = f_raw_final[nd]
    f_report_front = f_report[nd]
    f_norm_front = exp.normalize_objectives(ctx, f_report_front)
    rows = []
    best_idx = int(np.argmin(f_norm_front.mean(axis=1)))
    for local_idx, (raw, report, norm) in enumerate(zip(f_raw_front, f_report_front, f_norm_front)):
        rows.append(
            {
                "method": method,
                "seed": seed,
                "front_index": local_idx,
                "is_selected": method == "Knowledge-constrained NSGA-II" and local_idx == best_idx,
                "raw_f1_iv": float(raw[0]),
                "raw_f2_deg": float(raw[1]),
                "raw_f3_cta": float(raw[2]),
                "constraint_aware_f1_iv": float(report[0]),
                "constraint_aware_f2_deg": float(report[1]),
                "constraint_aware_f3_cta": float(report[2]),
                "norm_f1_iv": float(norm[0]),
                "norm_f2_deg": float(norm[1]),
                "norm_f3_cta": float(norm[2]),
                "normalized_average": float(norm.mean()),
            }
        )
    return pd.DataFrame(rows)


def set_equalish_3d_view(ax: plt.Axes, data: pd.DataFrame) -> None:
    setters = {
        "norm_f1_iv": ax.set_xlim,
        "norm_f2_deg": ax.set_ylim,
        "norm_f3_cta": ax.set_zlim,
    }
    for col, setter in setters.items():
        values = data[col].to_numpy(dtype=float)
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
        span = max(high - low, 1e-6)
        pad = span * 0.08
        setter(low - pad, high + pad)


def plot_fronts(fronts: pd.DataFrame, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 7.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.7,
        }
    )
    fig = plt.figure(figsize=(4.75, 4.25))
    ax = fig.add_subplot(111, projection="3d")

    for method in exp.METHODS:
        sub = fronts[fronts["method"] == method]
        ax.scatter(
            sub["norm_f1_iv"],
            sub["norm_f2_deg"],
            sub["norm_f3_cta"],
            s=14 if method == "Knowledge-constrained NSGA-II" else 11,
            marker=MARKERS[method],
            color=exp.COLORS[method],
            alpha=0.78 if method == "Knowledge-constrained NSGA-II" else 0.62,
            edgecolors="none",
            label=LABELS[method],
            depthshade=False,
        )

    selected = fronts[fronts["is_selected"]]
    if not selected.empty:
        row = selected.iloc[0]
        ax.scatter(
            [row["norm_f1_iv"]],
            [row["norm_f2_deg"]],
            [row["norm_f3_cta"]],
            s=155,
            marker="*",
            color="#FFD43B",
            edgecolors="black",
            linewidths=0.8,
            label="Selected solution",
            depthshade=False,
            zorder=10,
        )

    ax.set_xlabel("F1", color="#D62728", fontweight="bold", labelpad=8)
    ax.set_ylabel("F2", color="#D62728", fontweight="bold", labelpad=8)
    ax.set_zlabel("F3", color="#D62728", fontweight="bold", labelpad=8)
    ax.view_init(elev=25, azim=-48)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor("#CFCFCF")
    set_equalish_3d_view(ax, fronts)
    handles = [
        Line2D([0], [0], marker=MARKERS[method], color="none", markerfacecolor=exp.COLORS[method],
               markeredgecolor="none", markersize=4.0, label=LABELS[method])
        for method in exp.METHODS
    ]
    handles.append(
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#FFD43B",
               markeredgecolor="black", markeredgewidth=0.5, markersize=8.0, label="Selected solution")
    )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.08, 1.01), frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and plot final Pareto fronts in normalized F1/F2/F3 space.")
    parser.add_argument("--seed", type=int, default=42, help="Representative seed used to extract the final fronts.")
    args = parser.parse_args()

    results_dir = output_dir()
    csv_path = results_dir / f"final_pareto_front_points_seed{args.seed}.csv"
    output_base = results_dir / f"final_pareto_front_3d_seed{args.seed}"
    if csv_path.exists():
        fronts = pd.read_csv(csv_path)
        print("Loaded existing Pareto front points from", csv_path)
    else:
        load_formal_moead_config(results_dir)
        exp.set_seed(args.seed)
        ctx = exp.build_context(exp.project_root())
        frames = []
        for method in exp.METHODS:
            print(f"Extracting final Pareto front for {method}, seed={args.seed} ...")
            frames.append(extract_front_for_method(ctx, method, args.seed))
        fronts = pd.concat(frames, ignore_index=True)
        fronts.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_fronts(fronts, output_base)
    print("Final Pareto front plot written to", results_dir)


if __name__ == "__main__":
    main()
