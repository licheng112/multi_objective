# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METHODS = (
    "Knowledge-constrained NSGA-II",
    "NSGA-II",
    "NSGA-III",
    "MOEA/D",
)
OURS = "Knowledge-constrained NSGA-II"
BASELINES = ("NSGA-II", "NSGA-III", "MOEA/D")
COLORS = {
    "Knowledge-constrained NSGA-II": "#7B2CBF",
    "NSGA-II": "#D62728",
    "NSGA-III": "#1F77B4",
    "MOEA/D": "#2CA02C",
}
METRICS = {
    "Final_F1_IV": "min",
    "Final_F2_DEG": "min",
    "Final_F3_CTA": "min",
    "Average_objective": "min",
    "HV": "max",
    "IGD": "min",
    "PAM": "min",
}


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        current = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, current)
        adjusted[idx] = running_max
    return adjusted


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    wins = 0
    losses = 0
    for av in a:
        wins += int(np.sum(av > b))
        losses += int(np.sum(av < b))
    denom = len(a) * len(b)
    return float((wins - losses) / denom) if denom else float("nan")


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) < 2 or np.allclose(diff, 0.0):
        return 1.0
    return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)


def make_statistical_tests(per_seed: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for metric, direction in METRICS.items():
        metric_p_values = []
        metric_rows = []
        for baseline in BASELINES:
            ours = per_seed[per_seed["method"] == OURS].sort_values("seed")[metric].to_numpy(dtype=float)
            other = per_seed[per_seed["method"] == baseline].sort_values("seed")[metric].to_numpy(dtype=float)
            n = min(len(ours), len(other))
            ours = ours[:n]
            other = other[:n]
            p_value = paired_wilcoxon(ours, other)
            better = ours < other if direction == "min" else ours > other
            median_diff = float(np.median(ours - other))
            if direction == "max":
                median_diff = -median_diff
            row = {
                "metric": metric,
                "direction": direction,
                "comparison": f"{OURS} vs {baseline}",
                "runs": int(n),
                "ours_mean": float(np.mean(ours)) if n else float("nan"),
                "baseline_mean": float(np.mean(other)) if n else float("nan"),
                "ours_better_rate": float(np.mean(better)) if n else float("nan"),
                "paired_median_advantage": -median_diff if direction == "min" else median_diff,
                "cliffs_delta_raw": cliffs_delta(ours, other),
                "p_value": p_value,
            }
            metric_rows.append(row)
            metric_p_values.append(p_value)
        adjusted = holm_adjust(metric_p_values)
        for row, adj in zip(metric_rows, adjusted):
            row["holm_p_value"] = adj
            row["significant_0_05"] = bool(adj < 0.05)
            rows.append(row)
    tests = pd.DataFrame(rows)
    tests.to_csv(output_path, index=False, encoding="utf-8-sig")
    return tests


def make_average_ranks(per_seed: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for metric, direction in METRICS.items():
        pivot = per_seed.pivot(index="seed", columns="method", values=metric).reindex(columns=METHODS)
        ranks = pivot.rank(axis=1, ascending=(direction == "min"), method="average")
        for method in METHODS:
            rows.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "method": method,
                    "average_rank": float(ranks[method].mean()),
                    "rank_std": float(ranks[method].std(ddof=1)),
                }
            )
    rank_df = pd.DataFrame(rows)
    rank_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return rank_df


def plot_boxplots(per_seed: pd.DataFrame, output_path: Path) -> None:
    metrics = ["Average_objective", "HV", "IGD", "PAM"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2))
    for ax, metric in zip(axes.flat, metrics):
        data = [per_seed[per_seed["method"] == method][metric].to_numpy(dtype=float) for method in METHODS]
        box = ax.boxplot(data, patch_artist=True, labels=["KC-NSGA-II", "NSGA-II", "NSGA-III", "MOEA/D"])
        for patch, method in zip(box["boxes"], METHODS):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.28)
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_individual_trajectories(history: pd.DataFrame, output_path: Path) -> None:
    metrics = [("hv", "Hypervolume"), ("igd", "IGD")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in METHODS:
            sub = history[history["method"] == method]
            for _, run in sub.groupby("seed"):
                ax.plot(run["generation"], run[metric], color=COLORS[method], alpha=0.10, linewidth=0.55)
            mean = sub.groupby("generation")[metric].mean()
            ax.plot(mean.index, mean.values, color=COLORS[method], linewidth=1.75, label=method)
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.grid(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def risk_label(pass_count: int, total: int) -> str:
    ratio = pass_count / max(total, 1)
    if ratio >= 0.85:
        return "low"
    if ratio >= 0.60:
        return "medium"
    return "high"


def make_risk_audit(
    results_dir: Path,
    per_seed: pd.DataFrame,
    metrics: pd.DataFrame,
    tests: pd.DataFrame,
    ranks: pd.DataFrame,
) -> None:
    run_count = int(per_seed["seed"].nunique())
    summary = metrics.set_index("method").reindex(METHODS)
    checks = []

    checks.append(("runs_count", run_count >= 20, f"{run_count} independent runs detected."))

    best_counts = 0
    for metric, direction in METRICS.items():
        values = summary[f"{metric}_mean"]
        best_method = values.idxmin() if direction == "min" else values.idxmax()
        best_counts += int(best_method == OURS)
    checks.append(
        (
            "not_overclaiming_all_metrics",
            best_counts < len(METRICS),
            f"{OURS} is best on {best_counts}/{len(METRICS)} audited metrics.",
        )
    )

    moead_worst = 0
    for metric, direction in {"Average_objective": "min", "HV": "max", "IGD": "min"}.items():
        values = summary[f"{metric}_mean"]
        worst = values.idxmax() if direction == "min" else values.idxmin()
        moead_worst += int(worst == "MOEA/D")
    local_strengths = []
    if "Final_F3_CTA_mean" in summary.columns:
        f3_values = summary["Final_F3_CTA_mean"]
        if f3_values.rank(method="min").loc["MOEA/D"] <= 2:
            local_strengths.append("top-2 F3-CTA")
    if "Convergence_Generation_mean" in summary.columns:
        conv_values = summary["Convergence_Generation_mean"]
        if conv_values.idxmin() == "MOEA/D":
            local_strengths.append("fastest mean convergence generation")
    checks.append(
        (
            "moead_tradeoff_not_hidden",
            bool(local_strengths),
            f"MOEA/D is worst on {moead_worst}/3 key set-quality metrics, but shows local strength: {', '.join(local_strengths) or 'none'}.",
        )
    )

    key_tests = tests[tests["metric"].isin(["HV", "IGD", "PAM"])]
    sig_rate = float(key_tests["significant_0_05"].mean()) if not key_tests.empty else 0.0
    checks.append(("key_statistics", sig_rate >= 0.5, f"Holm-significant comparison rate on HV/IGD/PAM is {sig_rate:.2f}."))

    kcsr_values = metrics.set_index("method").get("KCSR_mean")
    if kcsr_values is not None:
        spread = float(kcsr_values.max() - kcsr_values.min())
        checks.append(("kcsr_not_core_claim", spread < 1e-9, "KCSR has no method-level spread; report only as feasibility evidence."))

    selected_config = results_dir / "selected_baseline_config.json"
    checks.append(("baseline_sensitivity_documented", selected_config.exists(), f"Selected baseline config file exists: {selected_config.exists()}."))

    pass_count = sum(int(ok) for _, ok, _ in checks)
    label = risk_label(pass_count, len(checks))
    lines = [
        "# Reviewer Risk Audit",
        "",
        f"- Overall residual risk: **{label}**",
        f"- Passed checks: {pass_count}/{len(checks)}",
        "",
        "## Checks",
        "",
    ]
    for name, ok, note in checks:
        status = "PASS" if ok else "WARN"
        lines.append(f"- {status} `{name}`: {note}")

    lines.extend(
        [
            "",
            "## Conservative Claim",
            "",
            "Use a robust-overall claim centered on HV, IGD, PAM, convergence speed, and statistical evidence. Do not claim universal single-objective dominance.",
            "",
            "## Average Rank Snapshot",
            "",
            ranks.pivot(index="metric", columns="method", values="average_rank").to_markdown(floatfmt=".3f"),
        ]
    )
    (results_dir / "reviewer_risk_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def analyze_results(results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    per_seed = pd.read_csv(results_dir / "optimization_metrics_by_seed.csv")
    metrics = pd.read_csv(results_dir / "optimization_metrics.csv")
    history = pd.read_csv(results_dir / "convergence_history.csv")

    tests = make_statistical_tests(per_seed, results_dir / "statistical_tests.csv")
    ranks = make_average_ranks(per_seed, results_dir / "average_rank_summary.csv")
    plot_boxplots(per_seed, results_dir / "per_seed_metric_boxplots.png")
    plot_individual_trajectories(history, results_dir / "individual_trajectories_hv_igd.png")
    make_risk_audit(results_dir, per_seed, metrics, tests, ranks)
    print("Reviewer robustness analysis written to", results_dir)


if __name__ == "__main__":
    analyze_results(Path(__file__).resolve().parent / "results")
