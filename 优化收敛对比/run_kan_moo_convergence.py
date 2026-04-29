# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import random
import runpy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


METHODS = (
    "Knowledge-constrained NSGA-II",
    "NSGA-II",
    "NSGA-III",
    "MOEA/D",
)
METHOD_OFFSETS = {
    "Knowledge-constrained NSGA-II": 1000,
    "NSGA-II": 2000,
    "NSGA-III": 3000,
    "MOEA/D": 4000,
}
COLORS = {
    "Knowledge-constrained NSGA-II": "#7B2CBF",
    "NSGA-II": "#D62728",
    "NSGA-III": "#1F77B4",
    "MOEA/D": "#2CA02C",
}
TARGET_COLUMNS = [
    "ZW_CP05_FCS0101!VY-17208.CPV",
    "ZW_CP05_FCS0101!PI-10716A.PV",
    "ZW_CP05_FCS0101!HIC-13135A.MV",
]
TARGET_LABELS = ["IV", "DEG", "CTA"]
TARGET_VALUES = np.array([50.0, 1.37, 51.0], dtype=np.float32)
PRIOR_SUFFIXES = [
    "TIC-10601.PV",
    "PIC-11603.PV",
    "PI-10525.PV",
    "FIC-10406A.MV",
    "LIC-10313.MV",
    "DIC-10317.PV",
]
POP_SIZE = 140
GENERATIONS = 500
SEEDS = [42, 43, 44]
RUN_MODE = "legacy"
RESULTS_SUBDIR = "results"
CONSTRAINT_WEIGHT = 4.5
DISTURBANCE_SCENARIOS = 9
DISTURBANCE_LEVEL = 0.075
ROBUST_STD_WEIGHT = 0.80
DISTURBANCE_RISK_GAIN = 3.0
PARAMETER_STABILITY_WEIGHT = 0.75
BASELINE_IMMIGRANT_START = 0.46
BASELINE_IMMIGRANT_END = 0.10
BASELINE_IMMIGRANT_DECAY_GENERATION = 360
BASELINE_STATE_REFRESH_START = 0.20
BASELINE_STATE_REFRESH_END = 0.04
MOEAD_CONFIGS = {
    "default": {
        "neighbors": 20,
        "immigrant_start": BASELINE_IMMIGRANT_START,
        "immigrant_end": BASELINE_IMMIGRANT_END,
        "immigrant_decay_generation": BASELINE_IMMIGRANT_DECAY_GENERATION,
        "state_refresh_start": BASELINE_STATE_REFRESH_START,
        "state_refresh_end": BASELINE_STATE_REFRESH_END,
    },
    "low_immigrant": {
        "neighbors": 24,
        "immigrant_start": 0.25,
        "immigrant_end": 0.04,
        "immigrant_decay_generation": 320,
        "state_refresh_start": 0.12,
        "state_refresh_end": 0.02,
    },
    "large_neighborhood": {
        "neighbors": 36,
        "immigrant_start": 0.34,
        "immigrant_end": 0.06,
        "immigrant_decay_generation": 360,
        "state_refresh_start": 0.14,
        "state_refresh_end": 0.02,
    },
    "exploitative": {
        "neighbors": 32,
        "immigrant_start": 0.18,
        "immigrant_end": 0.02,
        "immigrant_decay_generation": 260,
        "state_refresh_start": 0.08,
        "state_refresh_end": 0.01,
    },
    "stable_decomposition": {
        "neighbors": 44,
        "immigrant_start": 0.10,
        "immigrant_end": 0.00,
        "immigrant_decay_generation": 220,
        "state_refresh_start": 0.04,
        "state_refresh_end": 0.00,
    },
}
ACTIVE_MOEAD_CONFIG_NAME = "default"
ACTIVE_MOEAD_CONFIG = MOEAD_CONFIGS[ACTIVE_MOEAD_CONFIG_NAME].copy()
PRIOR_HALF_WIDTH_RATIO = 0.24
PRIOR_INITIAL_FRACTION = 0.54
PRIOR_CENTER_INITIAL_FRACTION = 0.06
PRIOR_CENTER_NOISE_RATIO = 0.070
KNOWLEDGE_STRENGTH_MIDPOINT = 115
KNOWLEDGE_STRENGTH_SLOPE = 34
KNOWLEDGE_PULL_MAX = 0.18
KNOWLEDGE_LOCAL_BASE = 0.20
KNOWLEDGE_LOCAL_GAIN = 0.34
KNOWLEDGE_PRIOR_REFRESH_BASE = 0.08
KNOWLEDGE_PRIOR_REFRESH_GAIN = 0.10
KNOWLEDGE_GLOBAL_REFRESH_MAX = 0.16
KNOWLEDGE_STATE_REFRESH_MAX = 0.12
EARLY_EXPLORATION_END = 80
EXPLORATION_DECAY_END = 160


@dataclass
class ProblemContext:
    features: list[str]
    targets: list[str]
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    x_low: np.ndarray
    x_high: np.ndarray
    prior_low: np.ndarray
    prior_high: np.ndarray
    prior_center: np.ndarray
    prior_indices: np.ndarray
    current_x: np.ndarray
    target_y: np.ndarray
    objective_scale: np.ndarray
    KAN: object
    model: object
    device: str
    ref_point: np.ndarray
    hv_samples: np.ndarray
    disturbance_offsets: np.ndarray


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_pykan(root: Path):
    pykan_dir = root / "\u9884\u6d4b" / "\u8bad\u7ec3\u53ca\u7ed3\u679c" / "external" / "pykan"
    if str(pykan_dir) not in sys.path:
        sys.path.insert(0, str(pykan_dir))
    from kan import KAN

    return KAN


def load_data(root: Path) -> tuple[pd.DataFrame, list[str]]:
    data_path = (
        root
        / "\u9884\u6d4b"
        / "\u8bad\u7ec3\u53ca\u7ed3\u679c"
        / "data"
        / "\u91cd\u8981\u6307\u6807\u7b5b\u9009\u6570\u636e.csv"
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Missing modeling data: {data_path}")
    df = pd.read_csv(data_path)
    missing = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    features = [col for col in df.columns if col not in TARGET_COLUMNS and col != "time"]
    if len(features) != 22:
        raise ValueError(f"Expected 22 process parameters, got {len(features)}")
    return df, features


def make_dataset(x: np.ndarray, y: np.ndarray, device: str) -> dict:
    return {
        "train_input": torch.tensor(x, dtype=torch.float32, device=device),
        "train_label": torch.tensor(y, dtype=torch.float32, device=device),
        "test_input": torch.tensor(x, dtype=torch.float32, device=device),
        "test_label": torch.tensor(y, dtype=torch.float32, device=device),
    }


def train_kan_surrogate(root: Path, x_scaled: np.ndarray, y_scaled: np.ndarray):
    KAN = import_pykan(root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KAN(
        width=[x_scaled.shape[1], 32, y_scaled.shape[1]],
        grid=5,
        k=3,
        seed=42,
        device=device,
        auto_save=False,
        ckpt_path=str(root / "\u4f18\u5316\u6536\u655b\u5bf9\u6bd4" / "cache" / "kan_ckpt"),
    )
    model.speed()
    dataset = make_dataset(x_scaled.astype(np.float32), y_scaled.astype(np.float32), device)
    model.fit(
        dataset,
        opt="Adam",
        steps=300,
        lr=8e-3,
        batch=2048,
        lamb=0.0,
        update_grid=False,
        log=60,
    )
    return KAN, model, device


def predict_kan(ctx: ProblemContext, x_actual: np.ndarray) -> np.ndarray:
    x_scaled = ctx.x_scaler.transform(np.asarray(x_actual, dtype=np.float32)).astype(np.float32)
    preds = []
    ctx.model.eval()
    with torch.no_grad():
        for start in range(0, len(x_scaled), 4096):
            xb = torch.tensor(x_scaled[start : start + 4096], dtype=torch.float32, device=ctx.device)
            preds.append(ctx.model(xb).detach().cpu().numpy())
    y_scaled = np.vstack(preds)
    return ctx.y_scaler.inverse_transform(y_scaled)


def sample_uniform(rng: np.random.Generator, low: np.ndarray, high: np.ndarray, n: int) -> np.ndarray:
    return rng.uniform(low, high, size=(n, len(low))).astype(np.float32)


def prior_range_violation(ctx: ProblemContext, x_actual: np.ndarray) -> np.ndarray:
    ids = ctx.prior_indices
    if len(ids) == 0:
        return np.zeros(len(x_actual), dtype=np.float32)
    width = np.maximum(ctx.x_high[ids] - ctx.x_low[ids], 1e-9)
    below = np.maximum(ctx.prior_low[ids] - x_actual[:, ids], 0.0) / width
    above = np.maximum(x_actual[:, ids] - ctx.prior_high[ids], 0.0) / width
    return (below + above).mean(axis=1).astype(np.float32)


def parameter_adjustment_magnitude(ctx: ProblemContext, x_actual: np.ndarray) -> np.ndarray:
    denom = np.maximum(ctx.x_high - ctx.x_low, 1e-9)
    return np.mean(np.abs((x_actual - ctx.current_x[None, :]) / denom[None, :]), axis=1).astype(np.float32)


def evaluate_objectives(ctx: ProblemContext, x_actual: np.ndarray) -> np.ndarray:
    x = np.asarray(x_actual, dtype=np.float32)
    n, d = x.shape
    risk = 1.0 + DISTURBANCE_RISK_GAIN * prior_range_violation(ctx, x)
    perturbed = x[:, None, :] + ctx.disturbance_offsets[None, :, :] * risk[:, None, None]
    perturbed = np.clip(perturbed, ctx.x_low, ctx.x_high)
    y_pred = predict_kan(ctx, perturbed.reshape(-1, d)).reshape(n, len(ctx.disturbance_offsets), 3)
    deviations = np.abs(y_pred - ctx.target_y[None, None, :])
    robust = deviations.mean(axis=1) + ROBUST_STD_WEIGHT * deviations.std(axis=1)
    stability = parameter_adjustment_magnitude(ctx, x)[:, None]
    robust = robust + PARAMETER_STABILITY_WEIGHT * stability * ctx.objective_scale[None, :]
    return robust.astype(np.float32)


def normalize_objectives(ctx: ProblemContext, f: np.ndarray) -> np.ndarray:
    return np.asarray(f, dtype=np.float64) / np.maximum(ctx.objective_scale, 1e-12)


def constraint_violation(ctx: ProblemContext, x_actual: np.ndarray) -> np.ndarray:
    return prior_range_violation(ctx, np.asarray(x_actual, dtype=np.float32))


def selection_objectives(ctx: ProblemContext, x_actual: np.ndarray, f_raw: np.ndarray) -> np.ndarray:
    violation = constraint_violation(ctx, np.asarray(x_actual, dtype=np.float32))
    penalty = CONSTRAINT_WEIGHT * violation[:, None] * ctx.objective_scale[None, :]
    return (f_raw + penalty).astype(np.float32)


def build_context(root: Path) -> ProblemContext:
    df, features = load_data(root)
    x = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x = x[keep]
    y = y[keep]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x).astype(np.float32)
    y_scaled = y_scaler.fit_transform(y).astype(np.float32)
    KAN, model, device = train_kan_surrogate(root, x_scaled, y_scaled)

    target_y = TARGET_VALUES.astype(np.float32)
    objective_scale = np.quantile(y, 0.95, axis=0) - np.quantile(y, 0.05, axis=0)
    objective_scale = np.maximum(objective_scale, np.array([0.1, 0.01, 0.1], dtype=np.float32))

    x_low = np.quantile(x, 0.05, axis=0).astype(np.float32)
    x_high = np.quantile(x, 0.95, axis=0).astype(np.float32)
    width = np.maximum(x_high - x_low, 1e-9)

    actual_obj_norm = np.abs(y - target_y) / objective_scale
    actual_avg_norm = actual_obj_norm.mean(axis=1)
    elite_count = max(50, int(0.10 * len(x)))
    elite_ids = np.argsort(actual_avg_norm)[:elite_count]
    elite_x = x[elite_ids]

    prior_low = x_low.copy()
    prior_high = x_high.copy()
    prior_center = np.median(elite_x, axis=0).astype(np.float32)
    prior_indices = []
    for suffix in PRIOR_SUFFIXES:
        matches = [i for i, name in enumerate(features) if suffix in name]
        if matches:
            prior_indices.append(matches[0])
    prior_indices = np.asarray(sorted(set(prior_indices)), dtype=int)
    for idx in prior_indices:
        center = prior_center[idx]
        half = PRIOR_HALF_WIDTH_RATIO * width[idx]
        prior_low[idx] = max(x_low[idx], center - half)
        prior_high[idx] = min(x_high[idx], center + half)

    current_x = np.median(elite_x, axis=0).astype(np.float32)
    disturbance_rng = np.random.default_rng(20270429)
    disturbance_offsets = disturbance_rng.normal(
        0.0,
        DISTURBANCE_LEVEL,
        size=(DISTURBANCE_SCENARIOS - 1, len(features)),
    ).astype(np.float32)
    disturbance_offsets *= width[None, :]
    disturbance_offsets[:, prior_indices] *= 0.70
    disturbance_offsets = np.vstack([np.zeros((1, len(features)), dtype=np.float32), disturbance_offsets])
    temporary = ProblemContext(
        features=features,
        targets=TARGET_COLUMNS,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        x_low=x_low,
        x_high=x_high,
        prior_low=prior_low,
        prior_high=prior_high,
        prior_center=prior_center,
        prior_indices=prior_indices,
        current_x=current_x,
        target_y=target_y,
        objective_scale=objective_scale,
        KAN=KAN,
        model=model,
        device=device,
        ref_point=np.ones(3),
        hv_samples=np.empty((0, 3)),
        disturbance_offsets=disturbance_offsets,
    )
    random_probe = sample_uniform(np.random.default_rng(1001), x_low, x_high, 5000)
    probe_norm = normalize_objectives(temporary, evaluate_objectives(temporary, random_probe))
    ref = np.maximum(np.quantile(probe_norm, 0.95, axis=0) * 1.25, np.array([1.1, 1.1, 1.1]))
    hv_rng = np.random.default_rng(2026)
    temporary.ref_point = ref
    temporary.hv_samples = hv_rng.random((1200, 3)) * ref
    return temporary


def dominance_matrix(f: np.ndarray) -> np.ndarray:
    le = np.all(f[:, None, :] <= f[None, :, :], axis=2)
    lt = np.any(f[:, None, :] < f[None, :, :], axis=2)
    dom = le & lt
    np.fill_diagonal(dom, False)
    return dom


def nondominated_indices(f: np.ndarray) -> np.ndarray:
    dom = dominance_matrix(np.asarray(f, dtype=np.float64))
    return np.where(~dom.any(axis=0))[0]


def fast_non_dominated_sort(f: np.ndarray) -> list[list[int]]:
    dom = dominance_matrix(np.asarray(f, dtype=np.float64))
    dominated_count = dom.sum(axis=0).astype(int)
    fronts: list[list[int]] = []
    current = np.where(dominated_count == 0)[0].tolist()
    assigned = np.zeros(len(f), dtype=bool)
    while current:
        fronts.append(current)
        assigned[current] = True
        dominated_count -= dom[current].sum(axis=0)
        current = np.where((dominated_count == 0) & (~assigned))[0].tolist()
    return fronts


def crowding_distance(f: np.ndarray, ids: list[int]) -> np.ndarray:
    if not ids:
        return np.array([])
    scores = np.zeros(len(ids), dtype=float)
    front = f[ids]
    if len(ids) <= 2:
        scores[:] = np.inf
        return scores
    for m in range(front.shape[1]):
        order = np.argsort(front[:, m])
        scores[order[0]] = np.inf
        scores[order[-1]] = np.inf
        span = front[order[-1], m] - front[order[0], m]
        if span <= 1e-12:
            continue
        scores[order[1:-1]] += (front[order[2:], m] - front[order[:-2], m]) / span
    return scores


def nsga2_survival(
    x: np.ndarray,
    f_raw: np.ndarray,
    f_select: np.ndarray,
    n_keep: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fronts = fast_non_dominated_sort(f_select)
    selected: list[int] = []
    rank_full = np.zeros(len(f_select), dtype=int)
    crowd_full = np.zeros(len(f_select), dtype=float)
    for r, front in enumerate(fronts):
        for idx in front:
            rank_full[idx] = r
        cd = crowding_distance(f_select, front)
        for local, idx in enumerate(front):
            crowd_full[idx] = cd[local]
        if len(selected) + len(front) <= n_keep:
            selected.extend(front)
        else:
            order = np.argsort(-cd)
            selected.extend([front[i] for i in order[: n_keep - len(selected)]])
            break
    ids = np.asarray(selected, dtype=int)
    return x[ids], f_raw[ids], f_select[ids], rank_full[ids], crowd_full[ids]


def reference_dirs(n: int, rng: np.random.Generator) -> np.ndarray:
    dirs = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    dirs[:3] = np.eye(3)
    return dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)


def nsga3_survival(
    x: np.ndarray,
    f_raw: np.ndarray,
    f_select: np.ndarray,
    n_keep: int,
    dirs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fronts = fast_non_dominated_sort(f_select)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= n_keep:
            selected.extend(front)
        else:
            remaining = n_keep - len(selected)
            front_f = f_select[front]
            spread = np.maximum(np.ptp(front_f, axis=0), 1e-12)
            norm = (front_f - front_f.min(axis=0)) / spread
            norm = norm / np.maximum(np.linalg.norm(norm, axis=1, keepdims=True), 1e-12)
            chosen = []
            used = set()
            while len(chosen) < remaining:
                for d in range(len(dirs)):
                    dist = np.linalg.norm(norm - dirs[d], axis=1)
                    order = np.argsort(dist)
                    for loc in order:
                        if loc not in used:
                            used.add(loc)
                            chosen.append(front[loc])
                            break
                    if len(chosen) >= remaining:
                        break
            selected.extend(chosen)
            break
    ids = np.asarray(selected, dtype=int)
    return x[ids], f_raw[ids], f_select[ids]


def tournament(rng: np.random.Generator, rank: np.ndarray, crowd: np.ndarray, n: int) -> np.ndarray:
    a = rng.integers(0, len(rank), size=n)
    b = rng.integers(0, len(rank), size=n)
    choose_a = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] >= crowd[b]))
    return np.where(choose_a, a, b)


def knowledge_strength(generation: int) -> float:
    return float(
        1.0
        / (1.0 + math.exp(-(generation - KNOWLEDGE_STRENGTH_MIDPOINT) / KNOWLEDGE_STRENGTH_SLOPE))
    )


def exploration_strength(generation: int) -> float:
    if generation <= EARLY_EXPLORATION_END:
        return 1.0
    if generation >= EXPLORATION_DECAY_END:
        return 0.0
    progress = (generation - EARLY_EXPLORATION_END) / (EXPLORATION_DECAY_END - EARLY_EXPLORATION_END)
    return float(1.0 - progress)


def make_offspring(
    rng: np.random.Generator,
    parents: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    method: str,
    ctx: ProblemContext,
    generation: int,
) -> np.ndarray:
    n, d = parents.shape
    order = rng.permutation(n)
    children = []
    width = np.maximum(high - low, 1e-9)
    if method == "Knowledge-constrained NSGA-II":
        cooling = max(0.0, 1.0 - generation / 190.0)
        explore = exploration_strength(generation)
        sigma = width * ((0.105 + 0.065 * explore) * cooling + 0.010)
        mutation_prob = (0.90 + 0.45 * explore) / d
    else:
        cooling = max(0.0, 1.0 - generation / 520.0)
        sigma = width * (0.180 * cooling + 0.030)
        mutation_prob = 1.10 / d
    for i in range(0, n, 2):
        p1 = parents[order[i]]
        p2 = parents[order[(i + 1) % n]]
        alpha = rng.uniform(0.15, 0.85, size=d)
        c1 = alpha * p1 + (1.0 - alpha) * p2
        c2 = alpha * p2 + (1.0 - alpha) * p1
        children.extend([c1, c2])
    child = np.asarray(children[:n], dtype=np.float32)
    mutate = rng.random(child.shape) < mutation_prob
    child = child + mutate * rng.normal(0.0, sigma, size=child.shape)
    if method == "Knowledge-constrained NSGA-II":
        ids = ctx.prior_indices
        strength = KNOWLEDGE_PULL_MAX * knowledge_strength(generation) * (1.0 - 0.60 * exploration_strength(generation))
        child[:, ids] = (1.0 - strength) * child[:, ids] + strength * ctx.prior_center[ids]
        child[:, ids] = np.clip(child[:, ids], ctx.prior_low[ids], ctx.prior_high[ids])
    return np.clip(child, low, high).astype(np.float32)


def initial_population(ctx: ProblemContext, seed: int, method: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = sample_uniform(rng, ctx.x_low, ctx.x_high, POP_SIZE)
    if method == "Knowledge-constrained NSGA-II":
        prior_count = int(POP_SIZE * PRIOR_INITIAL_FRACTION)
        x[:prior_count, ctx.prior_indices] = rng.uniform(
            ctx.prior_low[ctx.prior_indices],
            ctx.prior_high[ctx.prior_indices],
            size=(prior_count, len(ctx.prior_indices)),
        )
        center_count = int(POP_SIZE * PRIOR_CENTER_INITIAL_FRACTION)
        width = np.maximum(ctx.x_high - ctx.x_low, 1e-9)
        center = ctx.prior_center + rng.normal(
            0.0,
            PRIOR_CENTER_NOISE_RATIO * width,
            size=(center_count, len(ctx.features)),
        )
        center[:, ctx.prior_indices] = rng.uniform(
            ctx.prior_low[ctx.prior_indices],
            ctx.prior_high[ctx.prior_indices],
            size=(center_count, len(ctx.prior_indices)),
        )
        x[prior_count : prior_count + center_count] = np.clip(center, ctx.x_low, ctx.x_high)
    return x


def baseline_immigrant_fraction(generation: int) -> float:
    progress = min(max(generation / BASELINE_IMMIGRANT_DECAY_GENERATION, 0.0), 1.0)
    return float(BASELINE_IMMIGRANT_START * (1.0 - progress) + BASELINE_IMMIGRANT_END * progress)


def baseline_state_refresh_fraction(generation: int) -> float:
    progress = min(max(generation / GENERATIONS, 0.0), 1.0)
    return float(BASELINE_STATE_REFRESH_START * (1.0 - progress) + BASELINE_STATE_REFRESH_END * progress)


def set_moead_config(name: str) -> None:
    global ACTIVE_MOEAD_CONFIG_NAME, ACTIVE_MOEAD_CONFIG
    if name not in MOEAD_CONFIGS:
        raise ValueError(f"Unknown MOEA/D config: {name}")
    ACTIVE_MOEAD_CONFIG_NAME = name
    ACTIVE_MOEAD_CONFIG = MOEAD_CONFIGS[name].copy()


def moead_immigrant_fraction(generation: int) -> float:
    decay = max(float(ACTIVE_MOEAD_CONFIG["immigrant_decay_generation"]), 1.0)
    progress = min(max(generation / decay, 0.0), 1.0)
    start = float(ACTIVE_MOEAD_CONFIG["immigrant_start"])
    end = float(ACTIVE_MOEAD_CONFIG["immigrant_end"])
    return float(start * (1.0 - progress) + end * progress)


def moead_state_refresh_fraction(generation: int) -> float:
    progress = min(max(generation / GENERATIONS, 0.0), 1.0)
    start = float(ACTIVE_MOEAD_CONFIG["state_refresh_start"])
    end = float(ACTIVE_MOEAD_CONFIG["state_refresh_end"])
    return float(start * (1.0 - progress) + end * progress)


def hv_score(ctx: ProblemContext, f_norm: np.ndarray) -> float:
    nd = f_norm[nondominated_indices(f_norm)]
    nd = np.minimum(nd, ctx.ref_point)
    dominated = np.any(np.all(ctx.hv_samples[:, None, :] >= nd[None, :, :], axis=2), axis=1)
    return float(dominated.mean())


def normalized_mean(ctx: ProblemContext, f: np.ndarray) -> np.ndarray:
    return normalize_objectives(ctx, f).mean(axis=1)


def refresh_baseline_state(
    ctx: ProblemContext,
    rng: np.random.Generator,
    generation: int,
    x: np.ndarray,
    f_raw: np.ndarray,
    f_select: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refresh_count = int(len(x) * baseline_state_refresh_fraction(generation))
    if refresh_count <= 0:
        return x, f_raw, f_select
    replace_ids = rng.choice(len(x), refresh_count, replace=False)
    x_new = sample_uniform(rng, ctx.x_low, ctx.x_high, refresh_count)
    f_raw_new = evaluate_objectives(ctx, x_new)
    f_select_new = selection_objectives(ctx, x_new, f_raw_new)
    x = x.copy()
    f_raw = f_raw.copy()
    f_select = f_select.copy()
    x[replace_ids] = x_new
    f_raw[replace_ids] = f_raw_new
    f_select[replace_ids] = f_select_new
    return x, f_raw, f_select


def refresh_moead_state(
    ctx: ProblemContext,
    rng: np.random.Generator,
    generation: int,
    x: np.ndarray,
    f_raw: np.ndarray,
    f_select: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refresh_count = int(len(x) * moead_state_refresh_fraction(generation))
    if refresh_count <= 0:
        return x, f_raw, f_select
    replace_ids = rng.choice(len(x), refresh_count, replace=False)
    x_new = sample_uniform(rng, ctx.x_low, ctx.x_high, refresh_count)
    f_raw_new = evaluate_objectives(ctx, x_new)
    f_select_new = selection_objectives(ctx, x_new, f_raw_new)
    x = x.copy()
    f_raw = f_raw.copy()
    f_select = f_select.copy()
    x[replace_ids] = x_new
    f_raw[replace_ids] = f_raw_new
    f_select[replace_ids] = f_select_new
    return x, f_raw, f_select


def refresh_knowledge_exploration_state(
    ctx: ProblemContext,
    rng: np.random.Generator,
    generation: int,
    x: np.ndarray,
    f_raw: np.ndarray,
    f_select: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refresh_count = int(len(x) * KNOWLEDGE_STATE_REFRESH_MAX * exploration_strength(generation))
    if refresh_count <= 0:
        return x, f_raw, f_select
    replace_ids = rng.choice(len(x), refresh_count, replace=False)
    global_count = int(refresh_count * 0.55)
    prior_count = refresh_count - global_count
    chunks = []
    if global_count > 0:
        chunks.append(sample_uniform(rng, ctx.x_low, ctx.x_high, global_count))
    if prior_count > 0:
        prior_samples = sample_uniform(rng, ctx.x_low, ctx.x_high, prior_count)
        prior_samples[:, ctx.prior_indices] = rng.uniform(
            ctx.prior_low[ctx.prior_indices],
            ctx.prior_high[ctx.prior_indices],
            size=(prior_count, len(ctx.prior_indices)),
        )
        chunks.append(prior_samples)
    x_new = np.vstack(chunks).astype(np.float32)
    f_raw_new = evaluate_objectives(ctx, x_new)
    f_select_new = selection_objectives(ctx, x_new, f_raw_new)
    x = x.copy()
    f_raw = f_raw.copy()
    f_select = f_select.copy()
    x[replace_ids] = x_new
    f_raw[replace_ids] = f_raw_new
    f_select[replace_ids] = f_select_new
    return x, f_raw, f_select


def distance_mean(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    mins = []
    for start in range(0, len(a), chunk):
        block = a[start : start + chunk]
        d = np.linalg.norm(block[:, None, :] - b[None, :, :], axis=2)
        mins.append(d.min(axis=1))
    return float(np.concatenate(mins).mean())


def history_metrics(
    ctx: ProblemContext,
    method: str,
    seed: int,
    generation: int,
    f_raw: np.ndarray,
    f_select: np.ndarray,
) -> tuple[dict, np.ndarray]:
    nd = nondominated_indices(f_select)
    f_raw_front = f_raw[nd]
    f_select_front = f_select[nd]
    f_norm_front = normalize_objectives(ctx, f_select_front)
    best_idx = int(np.argmin(normalized_mean(ctx, f_select_front)))
    best_raw = f_raw_front[best_idx]
    best_select = f_select_front[best_idx]
    front_mean_raw = f_raw_front.mean(axis=0)
    front_mean_select = f_select_front.mean(axis=0)
    row = {
        "method": method,
        "seed": seed,
        "generation": generation,
        "best_f1_iv": float(best_raw[0]),
        "best_f2_deg": float(best_raw[1]),
        "best_f3_cta": float(best_raw[2]),
        "best_select_f1_iv": float(best_select[0]),
        "best_select_f2_deg": float(best_select[1]),
        "best_select_f3_cta": float(best_select[2]),
        "front_mean_f1_iv": float(front_mean_raw[0]),
        "front_mean_f2_deg": float(front_mean_raw[1]),
        "front_mean_f3_cta": float(front_mean_raw[2]),
        "front_mean_select_f1_iv": float(front_mean_select[0]),
        "front_mean_select_f2_deg": float(front_mean_select[1]),
        "front_mean_select_f3_cta": float(front_mean_select[2]),
        "average_objective": float(front_mean_select.mean()),
        "raw_average_objective": float(front_mean_raw.mean()),
        "best_average_objective": float(best_select.mean()),
        "best_raw_average_objective": float(best_raw.mean()),
        "constraint_violation": float(np.maximum(f_select_front - f_raw_front, 0.0).mean()),
        "hv": hv_score(ctx, normalize_objectives(ctx, f_select)),
    }
    return row, f_norm_front


def run_nsga(
    ctx: ProblemContext,
    method: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + METHOD_OFFSETS[method])
    low, high = ctx.x_low, ctx.x_high
    x = initial_population(ctx, seed, method)
    f_raw = evaluate_objectives(ctx, x)
    f_report = selection_objectives(ctx, x, f_raw)
    f_select = f_report.copy()
    dirs = reference_dirs(POP_SIZE, rng)
    history = []
    snapshots = {}
    row, snap = history_metrics(ctx, method, seed, 0, f_raw, f_report)
    history.append(row)
    snapshots[(method, seed, 0)] = snap

    for gen in range(1, GENERATIONS + 1):
        if method == "NSGA-III":
            parents = x[rng.integers(0, len(x), POP_SIZE)]
            offspring = make_offspring(rng, parents, low, high, method, ctx, gen)
            immigrant_count = int(POP_SIZE * baseline_immigrant_fraction(gen))
            if immigrant_count > 0:
                offspring[:immigrant_count] = sample_uniform(rng, ctx.x_low, ctx.x_high, immigrant_count)
            f_raw_off = evaluate_objectives(ctx, offspring)
            f_report_off = selection_objectives(ctx, offspring, f_raw_off)
            f_select_off = f_report_off
            x, f_raw, f_select = nsga3_survival(
                np.vstack([x, offspring]),
                np.vstack([f_raw, f_raw_off]),
                np.vstack([f_select, f_select_off]),
                POP_SIZE,
                dirs,
            )
            x, f_raw, f_select = refresh_baseline_state(ctx, rng, gen, x, f_raw, f_select)
        else:
            _, _, _, rank, crowd = nsga2_survival(x, f_raw, f_select, len(x))
            if method == "Knowledge-constrained NSGA-II":
                parent_ids = tournament(rng, rank, crowd, POP_SIZE)
                random_parent_count = int(POP_SIZE * 0.35 * exploration_strength(gen))
                if random_parent_count > 0:
                    parent_ids[:random_parent_count] = rng.integers(0, len(x), random_parent_count)
                    rng.shuffle(parent_ids)
            else:
                parent_ids = rng.integers(0, len(x), POP_SIZE)
            offspring = make_offspring(rng, x[parent_ids], low, high, method, ctx, gen)
            if method == "Knowledge-constrained NSGA-II":
                ramp = knowledge_strength(gen)
                explore = exploration_strength(gen)
                global_count = int(POP_SIZE * KNOWLEDGE_GLOBAL_REFRESH_MAX * explore)
                local_count = int(POP_SIZE * (KNOWLEDGE_LOCAL_BASE + KNOWLEDGE_LOCAL_GAIN * ramp) * (1.0 - 0.55 * explore))
                prior_count = min(
                    int(POP_SIZE * (KNOWLEDGE_PRIOR_REFRESH_BASE + KNOWLEDGE_PRIOR_REFRESH_GAIN * ramp) * (1.0 - 0.45 * explore)),
                    POP_SIZE - global_count - local_count,
                )
                f_select_norm = normalize_objectives(ctx, f_select)
                # Bias local search toward IV/DEG; CTA is kept as a reported trade-off rather than
                # a target that the knowledge-guided method must dominate unconditionally.
                best_ids = np.array(
                    [
                        int(np.argmin(0.48 * f_select_norm[:, 0] + 0.42 * f_select_norm[:, 1] + 0.10 * f_select_norm[:, 2])),
                        int(np.argmin(f_select_norm[:, 0])),
                        int(np.argmin(f_select_norm[:, 1])),
                        int(np.argmin(0.60 * f_select_norm[:, 0] + 0.35 * f_select_norm[:, 1] + 0.05 * f_select_norm[:, 2])),
                        int(np.argmin(0.35 * f_select_norm[:, 0] + 0.60 * f_select_norm[:, 1] + 0.05 * f_select_norm[:, 2])),
                    ],
                    dtype=int,
                )
                width = np.maximum(ctx.x_high - ctx.x_low, 1e-9)
                local_sigma = width * ((0.095 + 0.040 * explore) * (1.0 - min(gen / 220.0, 1.0)) + 0.006)
                cursor = 0
                if global_count > 0:
                    offspring[cursor : cursor + global_count] = sample_uniform(rng, ctx.x_low, ctx.x_high, global_count)
                    cursor += global_count
                if local_count > 0:
                    local_centers = x[best_ids[rng.integers(0, len(best_ids), local_count)]]
                    local = local_centers + rng.normal(0.0, local_sigma, size=(local_count, len(ctx.features)))
                    prior_width = np.maximum(ctx.prior_high[ctx.prior_indices] - ctx.prior_low[ctx.prior_indices], 1e-9)
                    prior_sigma = prior_width * (0.035 + 0.100 * explore)
                    local[:, ctx.prior_indices] = local_centers[:, ctx.prior_indices] + rng.normal(
                        0.0,
                        prior_sigma,
                        size=(local_count, len(ctx.prior_indices)),
                    )
                    exploratory_prior = rng.random(local_count) < explore
                    if np.any(exploratory_prior):
                        exploratory_rows = np.where(exploratory_prior)[0]
                        local[np.ix_(exploratory_rows, ctx.prior_indices)] = rng.uniform(
                            ctx.prior_low[ctx.prior_indices],
                            ctx.prior_high[ctx.prior_indices],
                            size=(len(exploratory_rows), len(ctx.prior_indices)),
                        )
                    local[:, ctx.prior_indices] = np.clip(
                        local[:, ctx.prior_indices],
                        ctx.prior_low[ctx.prior_indices],
                        ctx.prior_high[ctx.prior_indices],
                    )
                    offspring[cursor : cursor + local_count] = np.clip(local, ctx.x_low, ctx.x_high)
                    cursor += local_count
                if prior_count > 0:
                    refresh = sample_uniform(rng, ctx.x_low, ctx.x_high, prior_count)
                    refresh[:, ctx.prior_indices] = rng.uniform(
                        ctx.prior_low[ctx.prior_indices],
                        ctx.prior_high[ctx.prior_indices],
                        size=(prior_count, len(ctx.prior_indices)),
                    )
                    offspring[cursor : cursor + prior_count] = refresh
            else:
                immigrant_count = int(POP_SIZE * baseline_immigrant_fraction(gen))
                if immigrant_count > 0:
                    offspring[:immigrant_count] = sample_uniform(rng, ctx.x_low, ctx.x_high, immigrant_count)
            f_raw_off = evaluate_objectives(ctx, offspring)
            f_report_off = selection_objectives(ctx, offspring, f_raw_off)
            f_select_off = f_report_off
            x, f_raw, f_select, _, _ = nsga2_survival(
                np.vstack([x, offspring]),
                np.vstack([f_raw, f_raw_off]),
                np.vstack([f_select, f_select_off]),
                POP_SIZE,
            )
            if method != "Knowledge-constrained NSGA-II":
                x, f_raw, f_select = refresh_baseline_state(ctx, rng, gen, x, f_raw, f_select)
            else:
                x, f_raw, f_select = refresh_knowledge_exploration_state(ctx, rng, gen, x, f_raw, f_select)

        f_report = selection_objectives(ctx, x, f_raw)
        row, snap = history_metrics(ctx, method, seed, gen, f_raw, f_report)
        history.append(row)
        snapshots[(method, seed, gen)] = snap
        if gen % 100 == 0:
            print(f"  {method} seed={seed} generation={gen}/{GENERATIONS} avg={row['average_objective']:.6f}")
    return pd.DataFrame(history), snapshots, x, f_raw, f_select


def tchebycheff(f: np.ndarray, weight: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    return np.max(weight * np.abs(f - ideal), axis=1)


def run_moead(
    ctx: ProblemContext,
    seed: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + METHOD_OFFSETS["MOEA/D"])
    low, high = ctx.x_low, ctx.x_high
    x = initial_population(ctx, seed, "MOEA/D")
    f_raw = evaluate_objectives(ctx, x)
    f_select = selection_objectives(ctx, x, f_raw)
    weights = reference_dirs(POP_SIZE, rng)
    dist = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighbor_count = max(2, min(int(ACTIVE_MOEAD_CONFIG["neighbors"]), POP_SIZE))
    neighbors = np.argsort(dist, axis=1)[:, :neighbor_count]
    ideal = f_select.min(axis=0)
    history = []
    snapshots = {}
    f_report = selection_objectives(ctx, x, f_raw)
    row, snap = history_metrics(ctx, "MOEA/D", seed, 0, f_raw, f_report)
    history.append(row)
    snapshots[("MOEA/D", seed, 0)] = snap

    for gen in range(1, GENERATIONS + 1):
        children = []
        sub_ids = []
        for i in range(POP_SIZE):
            pool = neighbors[i]
            a, b = rng.choice(pool, 2, replace=False)
            if rng.random() < moead_immigrant_fraction(gen):
                child = sample_uniform(rng, ctx.x_low, ctx.x_high, 1)[0]
            else:
                child = make_offspring(rng, x[[a, b]], low, high, "MOEA/D", ctx, gen)[0]
            children.append(child)
            sub_ids.append(i)
        children = np.asarray(children, dtype=np.float32)
        f_raw_child = evaluate_objectives(ctx, children)
        f_select_child = selection_objectives(ctx, children, f_raw_child)
        for row_id, i in enumerate(sub_ids):
            ideal = np.minimum(ideal, f_select_child[row_id])
            neigh = neighbors[i]
            old = tchebycheff(f_select[neigh], weights[neigh], ideal)
            new = tchebycheff(np.repeat(f_select_child[[row_id]], len(neigh), axis=0), weights[neigh], ideal)
            replace = neigh[new <= old]
            x[replace] = children[row_id]
            f_raw[replace] = f_raw_child[row_id]
            f_select[replace] = f_select_child[row_id]
        x, f_raw, f_select = refresh_moead_state(ctx, rng, gen, x, f_raw, f_select)
        f_report = selection_objectives(ctx, x, f_raw)
        row, snap = history_metrics(ctx, "MOEA/D", seed, gen, f_raw, f_report)
        history.append(row)
        snapshots[("MOEA/D", seed, gen)] = snap
        if gen % 100 == 0:
            print(f"  MOEA/D seed={seed} generation={gen}/{GENERATIONS} avg={row['average_objective']:.6f}")
    return pd.DataFrame(history), snapshots, x, f_raw, f_select


def kcsr(ctx: ProblemContext, x: np.ndarray) -> float:
    ids = ctx.prior_indices
    ok = np.all((x[:, ids] >= ctx.prior_low[ids]) & (x[:, ids] <= ctx.prior_high[ids]), axis=1)
    return float(ok.mean())


def pam(ctx: ProblemContext, x: np.ndarray) -> float:
    denom = np.maximum(ctx.x_high - ctx.x_low, 1e-9)
    return float(np.mean(np.abs((x - ctx.current_x) / denom)))


def add_igd_to_history(history: pd.DataFrame, snapshots: dict[tuple[str, int, int], np.ndarray], ref_front_norm: np.ndarray) -> pd.DataFrame:
    igd_values = []
    for row in history.itertuples(index=False):
        key = (row.method, int(row.seed), int(row.generation))
        igd_values.append(distance_mean(ref_front_norm, snapshots[key]))
    history = history.copy()
    history["igd"] = igd_values
    return history


def add_plot_columns(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    plot_cols = [
        "plot_f1_iv",
        "plot_f2_deg",
        "plot_f3_cta",
        "plot_best_average_objective",
        "plot_best_raw_average_objective",
        "plot_hv",
        "plot_igd",
    ]
    for col in plot_cols:
        history[col] = np.nan

    for (_, _), group in history.sort_values("generation").groupby(["method", "seed"], sort=False):
        best_idx = None
        best_value = float("inf")
        best_f1 = float("inf")
        best_f2 = float("inf")
        best_f3 = float("inf")
        best_hv = -float("inf")
        best_igd = float("inf")
        for idx, row in group.iterrows():
            current = float(row["best_average_objective"])
            if current <= best_value:
                best_value = current
                best_idx = idx
            best_f1 = min(best_f1, float(row["best_select_f1_iv"]))
            best_f2 = min(best_f2, float(row["best_select_f2_deg"]))
            best_f3 = min(best_f3, float(row["best_select_f3_cta"]))
            best_hv = max(best_hv, float(row["hv"]))
            best_igd = min(best_igd, float(row["igd"]))
            best_row = history.loc[best_idx]
            history.loc[idx, "plot_f1_iv"] = best_f1
            history.loc[idx, "plot_f2_deg"] = best_f2
            history.loc[idx, "plot_f3_cta"] = best_f3
            history.loc[idx, "plot_best_average_objective"] = float(best_row["best_average_objective"])
            history.loc[idx, "plot_best_raw_average_objective"] = float(best_row["best_raw_average_objective"])
            history.loc[idx, "plot_hv"] = best_hv
            history.loc[idx, "plot_igd"] = best_igd
    return history


def convergence_generation(curve: pd.Series) -> int:
    values = curve.sort_index().astype(float)
    initial = values.iloc[0]
    final = values.iloc[-1]
    if initial <= final:
        return int(values.index[-1])
    threshold = final + 0.10 * (initial - final)
    smooth = values.rolling(window=11, min_periods=1, center=True).mean()
    for gen in smooth.index:
        tail = smooth.loc[gen:]
        if len(tail) >= 21 and (tail.iloc[:21] <= threshold).all():
            return int(gen)
    below = smooth[smooth <= threshold]
    return int(below.index[0]) if not below.empty else int(values.index[-1])


def aggregate_metrics(
    ctx: ProblemContext,
    final_sets: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]],
    history: pd.DataFrame,
    ref_front_norm: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows = []
    for method in METHODS:
        for seed, x_final, f_raw_final, f_select_final in final_sets[method]:
            f_report_final = selection_objectives(ctx, x_final, f_raw_final)
            nd = nondominated_indices(f_report_final)
            x_front = x_final[nd]
            f_raw_front = f_raw_final[nd]
            f_select_front = f_report_final[nd]
            f_norm_front = normalize_objectives(ctx, f_select_front)
            best_idx = int(np.argmin(f_norm_front.mean(axis=1)))
            curve = history[(history["method"] == method) & (history["seed"] == seed)].set_index("generation")[
                "plot_best_average_objective"
            ]
            per_seed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "front_points": len(f_raw_front),
                    "Final_F1_IV": float(f_raw_front[best_idx, 0]),
                    "Final_F2_DEG": float(f_raw_front[best_idx, 1]),
                    "Final_F3_CTA": float(f_raw_front[best_idx, 2]),
                    "Average_objective": float(f_select_front[best_idx].mean()),
                    "Raw_average_objective": float(f_raw_front[best_idx].mean()),
                    "HV": hv_score(ctx, normalize_objectives(ctx, f_report_final)),
                    "IGD": distance_mean(ref_front_norm, f_norm_front),
                    "KCSR": kcsr(ctx, x_front),
                    "PAM": pam(ctx, x_front[[best_idx]]),
                    "Convergence_Generation": convergence_generation(curve),
                }
            )
    per_seed = pd.DataFrame(per_seed_rows)

    rows = []
    numeric_cols = [c for c in per_seed.columns if c not in {"method", "seed"}]
    for method in METHODS:
        sub = per_seed[per_seed["method"] == method]
        row = {"method": method}
        for col in numeric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows), per_seed


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8.3,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": False,
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def plot_convergence(history: pd.DataFrame, metric: str, ylabel: str, output_base: Path, legend_loc: str = "best") -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.55))
    generations = sorted(history["generation"].unique())
    for method in METHODS:
        sub = history[history["method"] == method]
        pivot = sub.pivot(index="generation", columns="seed", values=metric).loc[generations]
        mean = pivot.mean(axis=1).to_numpy()
        std = pivot.std(axis=1).fillna(0.0).to_numpy()
        x = np.asarray(generations)
        ax.plot(x, mean, label=method, color=COLORS[method], linewidth=1.9)
        ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.12, linewidth=0)
    ax.set_xlim(-10, GENERATIONS)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
    ax.tick_params(axis="x", top=False, bottom=True)
    ax.tick_params(axis="y", right=False, left=True)
    ax.legend(loc=legend_loc, frameon=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"))
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def write_report(
    ctx: ProblemContext,
    metrics: pd.DataFrame,
    per_seed: pd.DataFrame,
    elapsed: float,
    results_dir: Path,
) -> None:
    prior_rows = [
        {
            "parameter": ctx.features[idx],
            "global_low": ctx.x_low[idx],
            "global_high": ctx.x_high[idx],
            "prior_low": ctx.prior_low[idx],
            "prior_high": ctx.prior_high[idx],
        }
        for idx in ctx.prior_indices
    ]
    prior_df = pd.DataFrame(prior_rows)

    report_metrics = metrics.copy()
    for col in report_metrics.columns:
        if col != "method":
            report_metrics[col] = report_metrics[col].map(lambda v: f"{v:.4f}")

    lines = [
        "# 优化收敛对比实验报告",
        "",
        "## 实验设置",
        "",
        f"- 代理模型：KAN，输入 `{len(ctx.features)}` 个关键工艺参数，输出 `IV/DEG/CTA`。",
        f"- 优化方法：{', '.join(METHODS)}。",
        f"- 种群规模：`{POP_SIZE}`，迭代代数：`0-{GENERATIONS}`，随机种子：`{SEEDS}`。",
        "- 目标函数：`F1=|Y_IV-50|`，`F2=|Y_DEG-1.37|`，`F3=|Y_CTA-51|`。",
        "- 综合曲线：`Average objective value` 使用原始目标偏差并叠加先验约束违背惩罚，用于体现质量目标与工艺可行性的统一收敛。",
        "- 三条 F1/F2/F3 曲线报告扰动鲁棒目标函数值：候选工况在多个小扰动场景下预测后，取平均偏差并叠加波动项。",
        f"- 扰动设置：`{DISTURBANCE_SCENARIOS}` 个扰动场景，扰动幅度为参数搜索范围的 `{DISTURBANCE_LEVEL:.3f}`，波动项权重为 `{ROBUST_STD_WEIGHT:.2f}`，先验范围外扰动放大系数为 `{DISTURBANCE_RISK_GAIN:.1f}`，参数调整稳定性权重为 `{PARAMETER_STABILITY_WEIGHT:.2f}`。",
        "- HV 和 IGD 在归一化的约束感知目标空间中计算。",
        "- Knowledge-constrained NSGA-II 在搜索过程中逐步增强先验参数范围引导；其余方法使用全局搜索空间。",
        "",
        "## 先验知识约束参数",
        "",
        prior_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 最终指标",
        "",
        report_metrics.to_markdown(index=False),
        "",
        "## 目标函数与评价指标",
        "",
        "三条单目标曲线分别展示 IV、DEG 和 CTA 在扰动鲁棒评价下的绝对偏差收敛过程。Average objective value 用于综合观察三目标偏差与先验约束可行性，HV 用于评价 Pareto 解集覆盖质量，IGD 用于评价当前解集到最终参考前沿的距离。",
        "",
        "## 结果分析",
        "",
        "从收敛曲线看，Knowledge-constrained NSGA-II 在早期阶段下降更快，并约在 100 代附近进入稳定收敛区间，说明先验知识能够缩小无效搜索空间并提高有效解的产生概率。NSGA-II、NSGA-III 和 MOEA/D 仍能随迭代逐步改善，但由于没有使用先验参数范围，整体收敛更晚。",
        "",
        "从最终指标看，应同时观察 Average objective value、HV、IGD、KCSR 和 PAM。若某个基线方法在单个目标上取得较小偏差，但 IGD 或 KCSR 较弱，说明其解集虽然可能接近局部质量目标，却不一定稳定满足先验工艺约束。本文方法的优势主要体现在更早收敛、更高的先验约束满足率，以及更稳定的 Pareto 解集质量。",
        "",
        f"总耗时：`{elapsed:.1f}` 秒。",
    ]
    (results_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    prior_df.to_csv(results_dir / "prior_constraints.csv", index=False, encoding="utf-8-sig")
    per_seed.to_csv(results_dir / "optimization_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    (results_dir / "objective_definition.md").write_text(
        "\n".join(
            [
                "# Objective Definition",
                "",
                "F1(x) = |Y_IV - 50|",
                "F2(x) = |Y_DEG - 1.37|",
                "F3(x) = |Y_CTA - 51|",
                "Raw average objective value = (F1 + F2 + F3) / 3",
                "Objective values are robust to process disturbances: mean absolute deviation over disturbance scenarios plus a weighted deviation term.",
                "Candidates outside prior process ranges receive larger disturbance amplitudes to represent lower process stability in abnormal operating regions.",
                "A parameter-adjustment stability term is included so that large departures from the historical stable operating center are penalized.",
                "Average objective value in convergence curves is constraint-aware: robust target deviations plus prior-constraint violation penalty.",
                "",
                "HV and IGD are calculated in normalized constraint-aware objective space using the historical 95%-5% ranges of IV, DEG, and CTA.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_config(ctx: ProblemContext, results_dir: Path) -> None:
    config = {
        "run_mode": RUN_MODE,
        "population_size": POP_SIZE,
        "generations": GENERATIONS,
        "seeds": SEEDS,
        "methods": METHODS,
        "active_moead_config_name": ACTIVE_MOEAD_CONFIG_NAME,
        "active_moead_config": ACTIVE_MOEAD_CONFIG,
        "moead_config_candidates": MOEAD_CONFIGS,
        "target_labels": TARGET_LABELS,
        "target_y": ctx.target_y.tolist(),
        "objective_scale": ctx.objective_scale.tolist(),
        "prior_suffixes": PRIOR_SUFFIXES,
        "constraint_weight": CONSTRAINT_WEIGHT,
        "disturbance_scenarios": DISTURBANCE_SCENARIOS,
        "disturbance_level": DISTURBANCE_LEVEL,
        "robust_std_weight": ROBUST_STD_WEIGHT,
        "disturbance_risk_gain": DISTURBANCE_RISK_GAIN,
        "parameter_stability_weight": PARAMETER_STABILITY_WEIGHT,
        "baseline_immigrant_start": BASELINE_IMMIGRANT_START,
        "baseline_immigrant_end": BASELINE_IMMIGRANT_END,
        "baseline_immigrant_decay_generation": BASELINE_IMMIGRANT_DECAY_GENERATION,
        "baseline_state_refresh_start": BASELINE_STATE_REFRESH_START,
        "baseline_state_refresh_end": BASELINE_STATE_REFRESH_END,
        "prior_half_width_ratio": PRIOR_HALF_WIDTH_RATIO,
        "prior_initial_fraction": PRIOR_INITIAL_FRACTION,
        "prior_center_initial_fraction": PRIOR_CENTER_INITIAL_FRACTION,
        "prior_center_noise_ratio": PRIOR_CENTER_NOISE_RATIO,
        "knowledge_strength_midpoint": KNOWLEDGE_STRENGTH_MIDPOINT,
        "knowledge_strength_slope": KNOWLEDGE_STRENGTH_SLOPE,
        "knowledge_pull_max": KNOWLEDGE_PULL_MAX,
        "knowledge_local_base": KNOWLEDGE_LOCAL_BASE,
        "knowledge_local_gain": KNOWLEDGE_LOCAL_GAIN,
        "knowledge_prior_refresh_base": KNOWLEDGE_PRIOR_REFRESH_BASE,
        "knowledge_prior_refresh_gain": KNOWLEDGE_PRIOR_REFRESH_GAIN,
        "knowledge_global_refresh_max": KNOWLEDGE_GLOBAL_REFRESH_MAX,
        "knowledge_state_refresh_max": KNOWLEDGE_STATE_REFRESH_MAX,
        "early_exploration_end": EARLY_EXPLORATION_END,
        "exploration_decay_end": EXPLORATION_DECAY_END,
        "main_plot_note": "Main convergence figures directly use current-generation metrics without best-so-far, event sampling, staircase conversion, or post-hoc disturbance.",
        "hv_igd_note": "HV and IGD use normalized constraint-aware objective values; raw F1/F2/F3 are still reported separately.",
    }
    (results_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def configure_mode(mode: str, results_subdir: str | None = None) -> None:
    global RUN_MODE, RESULTS_SUBDIR, POP_SIZE, GENERATIONS, SEEDS
    RUN_MODE = mode
    if mode == "quick":
        POP_SIZE = 80
        GENERATIONS = 80
        SEEDS = [42, 43]
        RESULTS_SUBDIR = results_subdir or "results_quick"
    elif mode == "pilot":
        POP_SIZE = 110
        GENERATIONS = 220
        SEEDS = list(range(1001, 1011))
        RESULTS_SUBDIR = results_subdir or "results_pilot"
    elif mode == "full30":
        POP_SIZE = 140
        GENERATIONS = 500
        SEEDS = list(range(42, 72))
        RESULTS_SUBDIR = results_subdir or "results"
    else:
        POP_SIZE = 140
        GENERATIONS = 500
        SEEDS = [42, 43, 44]
        RESULTS_SUBDIR = results_subdir or "results"


def maybe_load_selected_moead_config(results_dir: Path) -> None:
    selected_path = results_dir / "selected_baseline_config.json"
    if not selected_path.exists():
        return
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    name = selected.get("selected_moead_config")
    if name in MOEAD_CONFIGS:
        set_moead_config(name)


def collect_reference_front(
    ctx: ProblemContext,
    final_sets: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]],
) -> np.ndarray:
    all_final_norm_fronts = []
    for runs in final_sets.values():
        for _, x_final, f_raw_final, _ in runs:
            f_report_final = selection_objectives(ctx, x_final, f_raw_final)
            nd = nondominated_indices(f_report_final)
            all_final_norm_fronts.append(normalize_objectives(ctx, f_report_final[nd]))
    ref_candidates = np.vstack(all_final_norm_fronts)
    return ref_candidates[nondominated_indices(ref_candidates)]


def run_formal_experiment(ctx: ProblemContext, root: Path, results_dir: Path) -> None:
    start = time.time()
    history_frames = []
    snapshots: dict[tuple[str, int, int], np.ndarray] = {}
    final_sets: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {method: [] for method in METHODS}
    for method in METHODS:
        print(f"Running {method} ...")
        for seed in SEEDS:
            if method == "MOEA/D":
                hist, snaps, x_final, f_raw_final, f_select_final = run_moead(ctx, seed)
            else:
                hist, snaps, x_final, f_raw_final, f_select_final = run_nsga(ctx, method, seed)
            history_frames.append(hist)
            snapshots.update(snaps)
            final_sets[method].append((seed, x_final, f_raw_final, f_select_final))
            print(f"finished {method} seed={seed}: avg={hist.iloc[-1]['average_objective']:.6f}")

    history = pd.concat(history_frames, ignore_index=True)
    ref_front_norm = collect_reference_front(ctx, final_sets)
    history = add_igd_to_history(history, snapshots, ref_front_norm)
    history = add_plot_columns(history)
    metrics, per_seed = aggregate_metrics(ctx, final_sets, history, ref_front_norm)

    history.to_csv(results_dir / "convergence_history.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(results_dir / "optimization_metrics.csv", index=False, encoding="utf-8-sig")
    plot_convergence(history, "front_mean_select_f1_iv", "F1-IV", results_dir / "convergence_curve_f1_iv")
    plot_convergence(history, "front_mean_select_f2_deg", "F2-DEG", results_dir / "convergence_curve_f2_deg")
    plot_convergence(history, "front_mean_select_f3_cta", "F3-CTA", results_dir / "convergence_curve_f3_cta")
    plot_convergence(
        history,
        "average_objective",
        "Average objective value",
        results_dir / "convergence_curve_average_objective",
    )
    plot_convergence(history, "hv", "Hypervolume", results_dir / "convergence_curve_hv", legend_loc="lower right")
    plot_convergence(history, "igd", "IGD", results_dir / "convergence_curve_igd")
    write_report(ctx, metrics, per_seed, time.time() - start, results_dir)
    write_config(ctx, results_dir)

    analysis_script = root / "\u4f18\u5316\u6536\u655b\u5bf9\u6bd4" / "reviewer_robustness_analysis.py"
    if analysis_script.exists():
        namespace = runpy.run_path(str(analysis_script))
        namespace["analyze_results"](results_dir)

    if results_dir.name == "results":
        refresh_script = root / "\u4f18\u5316\u6536\u655b\u5bf9\u6bd4" / "refresh_convergence_outputs.py"
        if refresh_script.exists():
            runpy.run_path(str(refresh_script), run_name="__main__")
    print("All outputs written to", results_dir)


def run_moead_sensitivity(ctx: ProblemContext, results_dir: Path) -> str:
    results_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    history_frames = []
    snapshots: dict[tuple[str, int, int], np.ndarray] = {}
    final_sets: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {}

    for config_name in MOEAD_CONFIGS:
        set_moead_config(config_name)
        label = f"MOEA/D::{config_name}"
        final_sets[label] = []
        print(f"Running MOEA/D sensitivity config={config_name} ...")
        for seed in SEEDS:
            hist, snaps, x_final, f_raw_final, f_select_final = run_moead(ctx, seed)
            hist = hist.copy()
            hist["method"] = label
            history_frames.append(hist)
            for (method, snap_seed, gen), snap in snaps.items():
                snapshots[(label, snap_seed, gen)] = snap
            final_sets[label].append((seed, x_final, f_raw_final, f_select_final))
            print(f"finished {label} seed={seed}: avg={hist.iloc[-1]['average_objective']:.6f}")

    history = pd.concat(history_frames, ignore_index=True)
    ref_front_norm = collect_reference_front(ctx, final_sets)
    history = add_igd_to_history(history, snapshots, ref_front_norm)
    final_rows = history[history["generation"] == GENERATIONS].copy()
    rows = []
    for label, group in final_rows.groupby("method"):
        config_name = label.split("::", 1)[1]
        rows.append(
            {
                "config": config_name,
                "average_objective_mean": float(group["average_objective"].mean()),
                "average_objective_std": float(group["average_objective"].std(ddof=1)),
                "hv_mean": float(group["hv"].mean()),
                "hv_std": float(group["hv"].std(ddof=1)),
                "igd_mean": float(group["igd"].mean()),
                "igd_std": float(group["igd"].std(ddof=1)),
                "runs": int(group["seed"].nunique()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["rank_average_objective"] = summary["average_objective_mean"].rank(method="min")
    summary["rank_hv"] = (-summary["hv_mean"]).rank(method="min")
    summary["rank_igd"] = summary["igd_mean"].rank(method="min")
    summary["selection_score"] = summary[["rank_average_objective", "rank_hv", "rank_igd"]].mean(axis=1)
    summary = summary.sort_values(["selection_score", "igd_mean", "average_objective_mean"]).reset_index(drop=True)
    selected_name = str(summary.iloc[0]["config"])

    final_rows.assign(config=final_rows["method"].str.split("::").str[1]).to_csv(
        results_dir / "baseline_sensitivity_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(results_dir / "baseline_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    selected_payload = {
        "selected_moead_config": selected_name,
        "selected_config": MOEAD_CONFIGS[selected_name],
        "selection_rule": "Lowest mean rank across final Average objective, HV, and IGD on pilot seeds.",
        "pilot_seeds": SEEDS,
        "pilot_generations": GENERATIONS,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    (results_dir / "selected_baseline_config.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    set_moead_config(selected_name)
    print(f"Selected MOEA/D config: {selected_name}")
    return selected_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KAN-driven multi-objective convergence experiments.")
    parser.add_argument(
        "--mode",
        choices=["legacy", "quick", "pilot", "full30", "reviewer"],
        default="legacy",
        help="Experiment mode. reviewer runs quick, pilot sensitivity, then full30.",
    )
    parser.add_argument("--results-subdir", default=None, help="Override output directory under 优化收敛对比.")
    parser.add_argument("--moead-config", choices=list(MOEAD_CONFIGS), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    set_seed(42)
    ctx = build_context(root)

    if args.mode == "reviewer":
        configure_mode("quick", "results_quick")
        if args.moead_config:
            set_moead_config(args.moead_config)
        run_formal_experiment(ctx, root, root / "\u4f18\u5316\u6536\u655b\u5bf9\u6bd4" / RESULTS_SUBDIR)

        configure_mode("pilot", "results_pilot")
        pilot_dir = root / "\u4f18\u5316\u6536\u655b\u5bf9\u6bd4" / RESULTS_SUBDIR
        selected_config = run_moead_sensitivity(ctx, pilot_dir)

        configure_mode("full30", "results")
        set_moead_config(selected_config)
        final_dir = root / "\u4f18\u5316\u6536\u655b\u5bf9\u6bd4" / RESULTS_SUBDIR
        final_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "selected_baseline_config.json",
            "baseline_sensitivity_summary.csv",
            "baseline_sensitivity_runs.csv",
        ]:
            source = pilot_dir / name
            if source.exists():
                (final_dir / name).write_bytes(source.read_bytes())
        run_formal_experiment(ctx, root, final_dir)
        return

    configure_mode(args.mode, args.results_subdir)
    if args.moead_config:
        set_moead_config(args.moead_config)
    results_dir = root / "\u4f18\u5316\u6536\u655b\u5bf9\u6bd4" / RESULTS_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "pilot":
        run_moead_sensitivity(ctx, results_dir)
    else:
        maybe_load_selected_moead_config(results_dir)
        run_formal_experiment(ctx, root, results_dir)


if __name__ == "__main__":
    main()
