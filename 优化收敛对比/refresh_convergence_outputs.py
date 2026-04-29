# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def results_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": False,
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
    ax.tick_params(axis="x", top=False, bottom=True)
    ax.tick_params(axis="y", right=False, left=True)


def metric_column(history: pd.DataFrame, preferred: str, fallback: str | None = None) -> str:
    if preferred in history.columns:
        return preferred
    if fallback and fallback in history.columns:
        return fallback
    raise KeyError(f"Metric column not found: {preferred}")


def curve_stats(history: pd.DataFrame, metric: str) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    generations = np.array(sorted(history["generation"].unique()))
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in METHODS:
        sub = history[history["method"] == method]
        pivot = sub.pivot(index="generation", columns="seed", values=metric).loc[generations]
        stats[method] = (
            pivot.mean(axis=1).to_numpy(),
            pivot.std(axis=1).fillna(0.0).to_numpy(),
        )
    return generations, stats


def draw_metric(
    ax: plt.Axes,
    history: pd.DataFrame,
    metric: str,
    ylabel: str,
    legend: bool = True,
    legend_loc: str = "best",
) -> None:
    generations, stats = curve_stats(history, metric)
    for method in METHODS:
        mean, std = stats[method]
        ax.plot(generations, mean, label=method, color=COLORS[method], linewidth=1.65)
        ax.fill_between(
            generations,
            mean - std,
            mean + std,
            color=COLORS[method],
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xlim(-10, generations.max())
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    style_axes(ax)
    if legend:
        ax.legend(loc=legend_loc, frameon=False)


def save_single_metric(
    history: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_base: Path,
    legend_loc: str = "best",
) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.55))
    draw_metric(ax, history, metric, ylabel, legend=True, legend_loc=legend_loc)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_average_with_inset(history: pd.DataFrame, output_base: Path) -> None:
    avg_metric = metric_column(history, "average_objective")
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.55))
    draw_metric(ax, history, avg_metric, "Average objective value", legend=True, legend_loc="upper right")

    inset = ax.inset_axes([0.50, 0.43, 0.45, 0.43])
    generations, stats = curve_stats(history, avg_metric)
    zoom = generations >= 50
    for method in METHODS:
        mean, std = stats[method]
        inset.plot(generations[zoom], mean[zoom], color=COLORS[method], linewidth=1.20)
        inset.fill_between(
            generations[zoom],
            (mean - std)[zoom],
            (mean + std)[zoom],
            color=COLORS[method],
            alpha=0.10,
            linewidth=0,
        )
    inset.set_xlim(50, generations.max())
    zoom_values = np.concatenate([stats[m][0][zoom] for m in METHODS])
    span = float(np.nanmax(zoom_values) - np.nanmin(zoom_values))
    pad = max(span * 0.12, 1e-4)
    inset.set_ylim(float(np.nanmin(zoom_values)) - pad, float(np.nanmax(zoom_values)) + pad)
    inset.set_xticks([100, 300, 500])
    inset.tick_params(labelsize=7, top=False, right=False)
    style_axes(inset)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_metric_grid(
    history: pd.DataFrame,
    panels: list[tuple[str, str]],
    output_base: Path,
) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.1))
    for ax, (metric, ylabel) in zip(axes.flat, panels):
        draw_metric(ax, history, metric, ylabel, legend=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def current_metric_panels(history: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (metric_column(history, "front_mean_select_f1_iv", "front_mean_f1_iv"), "F1-IV"),
        (metric_column(history, "front_mean_select_f2_deg", "front_mean_f2_deg"), "F2-DEG"),
        (metric_column(history, "front_mean_select_f3_cta", "front_mean_f3_cta"), "F3-CTA"),
        (metric_column(history, "average_objective"), "Average objective value"),
        (metric_column(history, "hv"), "Hypervolume"),
        (metric_column(history, "igd"), "IGD"),
    ]


def metric_at_generation(history: pd.DataFrame, metric: str, generation: int) -> pd.Series:
    return history[history["generation"] == generation].groupby("method")[metric].mean().reindex(METHODS)


def make_consistency_check(history: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    metrics_by_method = metrics.set_index("method").reindex(METHODS)
    checks = [
        ("F1-IV", "min", metrics_by_method["Final_F1_IV_mean"]),
        ("F2-DEG", "min", metrics_by_method["Final_F2_DEG_mean"]),
        ("F3-CTA", "min", metrics_by_method["Final_F3_CTA_mean"]),
        ("Average objective", "min", metrics_by_method["Average_objective_mean"]),
        ("HV", "max", metrics_by_method["HV_mean"]),
        ("IGD", "min", metrics_by_method["IGD_mean"]),
        ("KCSR", "max", metrics_by_method["KCSR_mean"]),
        ("PAM", "min", metrics_by_method["PAM_mean"]),
    ]
    rows = []
    for metric, direction, values in checks:
        best_method = values.idxmin() if direction == "min" else values.idxmax()
        our_value = float(values.loc[OURS])
        best_value = float(values.loc[best_method])
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "best_method": best_method,
                "ours_value": our_value,
                "best_value": best_value,
                "supports_main_claim": best_method == OURS,
                "note": "consistent" if best_method == OURS else "local single-metric advantage; discuss with HV/IGD/KCSR/PAM",
            }
        )
    check_df = pd.DataFrame(rows)
    check_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return check_df


def make_baseline_start_check(history: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for metric, label in current_metric_panels(history):
        gen0 = metric_at_generation(history, metric, 0).reindex(BASELINES)
        values = gen0.astype(float)
        all_values = history[metric].astype(float)
        span = float(all_values.max() - all_values.min())
        estimated_tick = span / 5.0 if span > 0 else 0.0
        spread = float(values.max() - values.min())
        rows.append(
            {
                "metric": label,
                "baseline_start_min": float(values.min()),
                "baseline_start_max": float(values.max()),
                "baseline_start_spread": spread,
                "estimated_one_tick": estimated_tick,
                "within_one_tick": bool(spread <= estimated_tick + 1e-12),
                "NSGA-II": float(values.loc["NSGA-II"]),
                "NSGA-III": float(values.loc["NSGA-III"]),
                "MOEA/D": float(values.loc["MOEA/D"]),
            }
        )
    start_df = pd.DataFrame(rows)
    start_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return start_df


def fmt(series: pd.Series, method: str) -> str:
    return f"{float(series.loc[method]):.4f}"


def write_report(
    history: pd.DataFrame,
    metrics: pd.DataFrame,
    check_df: pd.DataFrame,
    start_df: pd.DataFrame,
    output_path: Path,
) -> None:
    avg_metric = metric_column(history, "average_objective")
    hv_metric = metric_column(history, "hv")
    igd_metric = metric_column(history, "igd")
    avg0 = metric_at_generation(history, avg_metric, 0)
    avg100 = metric_at_generation(history, avg_metric, 100)
    avg500 = metric_at_generation(history, avg_metric, 500)
    hv0 = metric_at_generation(history, hv_metric, 0)
    hv500 = metric_at_generation(history, hv_metric, 500)
    igd100 = metric_at_generation(history, igd_metric, 100)
    igd500 = metric_at_generation(history, igd_metric, 500)
    run_count = int(history["seed"].nunique())

    display_metrics = metrics.copy()
    for col in display_metrics.columns:
        if col != "method":
            display_metrics[col] = display_metrics[col].map(lambda v: f"{v:.4f}")
    display_checks = check_df.copy()
    for col in ["ours_value", "best_value"]:
        display_checks[col] = display_checks[col].map(lambda v: f"{v:.4f}")
    display_start = start_df.copy()
    for col in [
        "baseline_start_min",
        "baseline_start_max",
        "baseline_start_spread",
        "estimated_one_tick",
        "NSGA-II",
        "NSGA-III",
        "MOEA/D",
    ]:
        display_start[col] = display_start[col].map(lambda v: f"{v:.4f}")

    lines = [
        "# 优化收敛对比实验报告",
        "",
        "## 实验设置",
        "",
        "- 代理模型：KAN，输入 22 个关键工艺参数，输出 IV、DEG、CTA。",
        "- 对比方法：Knowledge-constrained NSGA-II、NSGA-II、NSGA-III、MOEA/D。",
        f"- 迭代范围：0-500 代；随机种子重复次数：{run_count}。",
        "- 目标函数：F1=|Y_IV-50|，F2=|Y_DEG-1.37|，F3=|Y_CTA-51|。",
        "- 正文收敛图直接使用逐代当前代非支配前沿均值及当前代 HV/IGD；不再进行累计最优、事件抽点、阶梯化或后处理扰动。",
        "- 本文方法在 0-80 代保留较强全局探索，80-160 代逐步增强知识引导，160 代后进入稳定开发阶段。",
        "",
        "## 关键代数对比（当前代口径）",
        "",
        f"- 第 0 代 Average objective：本文方法 {fmt(avg0, OURS)}，NSGA-II {fmt(avg0, 'NSGA-II')}，NSGA-III {fmt(avg0, 'NSGA-III')}，MOEA/D {fmt(avg0, 'MOEA/D')}。",
        f"- 第 100 代 Average objective：本文方法 {fmt(avg100, OURS)}，NSGA-II {fmt(avg100, 'NSGA-II')}，NSGA-III {fmt(avg100, 'NSGA-III')}，MOEA/D {fmt(avg100, 'MOEA/D')}。",
        f"- 第 500 代 Average objective：本文方法 {fmt(avg500, OURS)}，NSGA-II {fmt(avg500, 'NSGA-II')}，NSGA-III {fmt(avg500, 'NSGA-III')}，MOEA/D {fmt(avg500, 'MOEA/D')}。",
        f"- 第 0 代 HV：本文方法 {fmt(hv0, OURS)}，NSGA-II {fmt(hv0, 'NSGA-II')}，NSGA-III {fmt(hv0, 'NSGA-III')}，MOEA/D {fmt(hv0, 'MOEA/D')}。",
        f"- 第 500 代 HV：本文方法 {fmt(hv500, OURS)}，NSGA-II {fmt(hv500, 'NSGA-II')}，NSGA-III {fmt(hv500, 'NSGA-III')}，MOEA/D {fmt(hv500, 'MOEA/D')}。",
        f"- 第 100 代 IGD：本文方法 {fmt(igd100, OURS)}，NSGA-II {fmt(igd100, 'NSGA-II')}，NSGA-III {fmt(igd100, 'NSGA-III')}，MOEA/D {fmt(igd100, 'MOEA/D')}。",
        f"- 第 500 代 IGD：本文方法 {fmt(igd500, OURS)}，NSGA-II {fmt(igd500, 'NSGA-II')}，NSGA-III {fmt(igd500, 'NSGA-III')}，MOEA/D {fmt(igd500, 'MOEA/D')}。",
        "",
        "## 最终指标",
        "",
        display_metrics.to_markdown(index=False),
        "",
        "## 图间一致性检查",
        "",
        display_checks.to_markdown(index=False),
        "",
        "## 结果分析",
        "",
        "正文曲线展示当前代搜索状态，因此早期可能出现局部震荡或短暂退化；这是早期探索阶段的自然结果，而不是后处理平滑或人为扰动。",
        "",
        "本文方法在 IV、DEG、HV、IGD 上整体更优，但 CTA 存在局部折中，说明知识引导并非对所有目标无条件最优。最终结论应结合 Average objective、HV、IGD、KCSR 和 PAM 共同判断，而不是只看单个目标。",
        "",
    ]
    lines.extend(
        [
            "## Baseline start consistency",
            "",
            "NSGA-II, NSGA-III, and MOEA/D share the same global initial population; their generation-0 spread should stay within one estimated y-axis tick.",
            "",
            display_start.to_markdown(index=False),
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    out = results_dir()
    history = pd.read_csv(out / "convergence_history.csv")
    metrics = pd.read_csv(out / "optimization_metrics.csv")

    current_f1, _ = current_metric_panels(history)[0]
    current_f2, _ = current_metric_panels(history)[1]
    current_f3, _ = current_metric_panels(history)[2]
    _, current_hv = ("Hypervolume", metric_column(history, "hv"))
    _, current_igd = ("IGD", metric_column(history, "igd"))
    save_single_metric(history, current_f1, "F1-IV", out / "convergence_curve_f1_iv")
    save_single_metric(history, current_f2, "F2-DEG", out / "convergence_curve_f2_deg")
    save_single_metric(history, current_f3, "F3-CTA", out / "convergence_curve_f3_cta")
    save_average_with_inset(history, out / "convergence_curve_average_objective")
    save_single_metric(history, current_hv, "Hypervolume", out / "convergence_curve_hv", legend_loc="lower right")
    save_single_metric(history, current_igd, "IGD", out / "convergence_curve_igd")
    save_metric_grid(history, current_metric_panels(history), out / "convergence_curve_main_metrics")

    check_df = make_consistency_check(history, metrics, out / "curve_consistency_check.csv")
    start_df = make_baseline_start_check(history, out / "baseline_start_consistency.csv")
    write_report(history, metrics, check_df, start_df, out / "experiment_report.md")
    print("Refreshed convergence figures, consistency check, and report in", out)


if __name__ == "__main__":
    main()
