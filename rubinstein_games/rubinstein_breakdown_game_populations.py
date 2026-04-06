#!/usr/bin/env python3
"""
rubinstein_breakdown_game_populations.py

A genuine interacting population game built on Rubinstein bargaining with breakdown risk.

Core idea
---------
We simulate many independent bargaining pairs, but now pairs INTERACT via
evolutionary imitation:

1. Each pair i has parameters
       delta_A_i, delta_B_i, p_i
   and current continuation values
       V_A_i, V_B_i.

2. Within each macro-epoch, every pair runs Rubinstein-with-breakdown
   two-step dynamics for `micro_steps` updates:
       V_A(next) = 1 - (1-p) delta_B + (1-p)^2 delta_A delta_B V_A
       V_B(next) = 1 - (1-p) delta_A + (1-p)^2 delta_A delta_B V_B

3. After the micro dynamics, each pair gets a score.
   We use a simple "successful pair" criterion:
       - balanced split is good
       - faster contraction is good

   If A proposes first, the actual accepted split is:
       A_share = V_A
       B_share = 1 - V_A

   We define:
       balance_score = 1 - |A_share - B_share|
       speed_score   = 1 - ((1-p)^2 delta_A delta_B)

   and total score:
       score = w_balance * balance_score + w_speed * speed_score

4. Interaction:
   The bottom-performing fraction of pairs imitate the top-performing fraction.
   Learners partially copy elite parameters:
       theta_i <- (1-eta) theta_i + eta theta_elite + mutation
   for theta in {delta_A, delta_B, p_breakdown}

   Learners also partially copy elite continuation values.

This turns the independent-pairs model into a genuine population game:
the future of one pair depends on the success of other pairs.

Outputs
-------
- terminal summary printed to CLI
- optional CSV
- optional static 2x2 plot
- optional animated GIF showing the moving continuation-value cloud by epoch

Example
-------
python rubinstein_breakdown_game_populations.py \
  --num-pairs 1000 \
  --epochs 80 \
  --micro-steps 20 \
  --plot-out rb_pop_game.png \
  --animate-out rb_pop_game.gif \
  --csv-out rb_pop_game.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def validate_args(args: argparse.Namespace) -> None:
    if args.num_pairs <= 1:
        raise ValueError("--num-pairs must be > 1.")
    if args.epochs < 0:
        raise ValueError("--epochs must be nonnegative.")
    if args.micro_steps < 0:
        raise ValueError("--micro-steps must be nonnegative.")
    if not (0.0 < args.delta_min < 1.0 and 0.0 < args.delta_max < 1.0):
        raise ValueError("delta bounds must lie strictly between 0 and 1.")
    if args.delta_min >= args.delta_max:
        raise ValueError("Require delta_min < delta_max.")
    if not (0.0 <= args.p_min < 1.0 and 0.0 <= args.p_max < 1.0):
        raise ValueError("p bounds must satisfy 0 <= p < 1.")
    if args.p_min >= args.p_max:
        raise ValueError("Require p_min < p_max.")
    if args.v_min >= args.v_max:
        raise ValueError("Require v_min < v_max.")
    if not (0.0 < args.top_frac < 1.0):
        raise ValueError("--top-frac must lie in (0,1).")
    if not (0.0 < args.bottom_frac < 1.0):
        raise ValueError("--bottom-frac must lie in (0,1).")
    if args.top_frac + args.bottom_frac > 1.0:
        raise ValueError("--top-frac + --bottom-frac must be <= 1.")
    if not (0.0 <= args.imitation_rate <= 1.0):
        raise ValueError("--imitation-rate must lie in [0,1].")
    if args.mutation_std < 0.0:
        raise ValueError("--mutation-std must be nonnegative.")
    if args.value_mutation_std < 0.0:
        raise ValueError("--value-mutation-std must be nonnegative.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive.")
    if args.weight_balance < 0 or args.weight_speed < 0:
        raise ValueError("score weights must be nonnegative.")
    if args.weight_balance == 0 and args.weight_speed == 0:
        raise ValueError("At least one score weight must be positive.")


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def sample_population(
    num_pairs: int,
    delta_min: float,
    delta_max: float,
    p_min: float,
    p_max: float,
    v_min: float,
    v_max: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    delta_a = rng.uniform(delta_min, delta_max, size=num_pairs)
    delta_b = rng.uniform(delta_min, delta_max, size=num_pairs)
    p_breakdown = rng.uniform(p_min, p_max, size=num_pairs)
    v_a = rng.uniform(v_min, v_max, size=num_pairs)
    v_b = rng.uniform(v_min, v_max, size=num_pairs)
    return delta_a, delta_b, p_breakdown, v_a, v_b


def exact_fixed_points(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    survival = 1.0 - p_breakdown
    denom = 1.0 - (survival ** 2) * delta_a * delta_b
    v_a_star = (1.0 - survival * delta_b) / denom
    v_b_star = (1.0 - survival * delta_a) / denom
    return v_a_star, v_b_star


def contraction_factor(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
) -> np.ndarray:
    return ((1.0 - p_breakdown) ** 2) * delta_a * delta_b


def two_step_update(
    v_a: np.ndarray,
    v_b: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    survival = 1.0 - p_breakdown
    contr = contraction_factor(delta_a, delta_b, p_breakdown)
    next_v_a = 1.0 - survival * delta_b + contr * v_a
    next_v_b = 1.0 - survival * delta_a + contr * v_b
    return next_v_a, next_v_b


def run_micro_dynamics(
    v_a: np.ndarray,
    v_b: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
    micro_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = v_a.copy()
    y = v_b.copy()
    for _ in range(micro_steps):
        x, y = two_step_update(x, y, delta_a, delta_b, p_breakdown)
    return x, y


def compute_scores(
    v_a: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
    weight_balance: float,
    weight_speed: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_share = v_a
    b_share = 1.0 - v_a
    balance_score = 1.0 - np.abs(a_share - b_share)
    speed_score = 1.0 - contraction_factor(delta_a, delta_b, p_breakdown)
    score = weight_balance * balance_score + weight_speed * speed_score
    return score, balance_score, speed_score


def interact_by_imitation(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
    v_a: np.ndarray,
    v_b: np.ndarray,
    score: np.ndarray,
    rng: np.random.Generator,
    top_frac: float,
    bottom_frac: float,
    imitation_rate: float,
    mutation_std: float,
    value_mutation_std: float,
    delta_min: float,
    delta_max: float,
    p_min: float,
    p_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = score.size
    elite_count = max(1, int(np.floor(top_frac * n)))
    learner_count = max(1, int(np.floor(bottom_frac * n)))

    order = np.argsort(score)
    learners = order[:learner_count]
    elites = order[-elite_count:]

    chosen_elites = rng.choice(elites, size=learner_count, replace=True)

    new_delta_a = delta_a.copy()
    new_delta_b = delta_b.copy()
    new_p = p_breakdown.copy()
    new_v_a = v_a.copy()
    new_v_b = v_b.copy()

    l = learners
    e = chosen_elites
    eta = imitation_rate

    new_delta_a[l] = (1.0 - eta) * delta_a[l] + eta * delta_a[e] + rng.normal(0.0, mutation_std, size=learner_count)
    new_delta_b[l] = (1.0 - eta) * delta_b[l] + eta * delta_b[e] + rng.normal(0.0, mutation_std, size=learner_count)
    new_p[l] = (1.0 - eta) * p_breakdown[l] + eta * p_breakdown[e] + rng.normal(0.0, mutation_std, size=learner_count)

    new_v_a[l] = (1.0 - eta) * v_a[l] + eta * v_a[e] + rng.normal(0.0, value_mutation_std, size=learner_count)
    new_v_b[l] = (1.0 - eta) * v_b[l] + eta * v_b[e] + rng.normal(0.0, value_mutation_std, size=learner_count)

    new_delta_a = np.clip(new_delta_a, delta_min, delta_max)
    new_delta_b = np.clip(new_delta_b, delta_min, delta_max)
    new_p = np.clip(new_p, p_min, p_max)
    new_v_a = clip01(new_v_a)
    new_v_b = clip01(new_v_b)

    return new_delta_a, new_delta_b, new_p, new_v_a, new_v_b


def simulate_population_game(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
    v_a: np.ndarray,
    v_b: np.ndarray,
    epochs: int,
    micro_steps: int,
    weight_balance: float,
    weight_speed: float,
    top_frac: float,
    bottom_frac: float,
    imitation_rate: float,
    mutation_std: float,
    value_mutation_std: float,
    delta_min: float,
    delta_max: float,
    p_min: float,
    p_max: float,
    seed: int,
    keep_history: bool,
) -> dict:
    rng = np.random.default_rng(seed + 10_000)

    n = delta_a.size
    history_v_a = []
    history_v_b = []
    history_star_a = []
    history_star_b = []
    history_mean_score = []
    history_mean_balance = []
    history_mean_speed = []
    history_mean_contr = []

    initial_v_a = v_a.copy()
    initial_v_b = v_b.copy()
    initial_delta_a = delta_a.copy()
    initial_delta_b = delta_b.copy()
    initial_p = p_breakdown.copy()

    if keep_history:
        history_v_a.append(v_a.copy())
        history_v_b.append(v_b.copy())
        star_a, star_b = exact_fixed_points(delta_a, delta_b, p_breakdown)
        history_star_a.append(star_a.copy())
        history_star_b.append(star_b.copy())

    mean_score = np.nan
    mean_balance = np.nan
    mean_speed = np.nan

    for _ in range(epochs):
        v_a, v_b = run_micro_dynamics(v_a, v_b, delta_a, delta_b, p_breakdown, micro_steps)
        score, balance_score, speed_score = compute_scores(
            v_a=v_a,
            delta_a=delta_a,
            delta_b=delta_b,
            p_breakdown=p_breakdown,
            weight_balance=weight_balance,
            weight_speed=weight_speed,
        )

        mean_score = float(score.mean())
        mean_balance = float(balance_score.mean())
        mean_speed = float(speed_score.mean())

        history_mean_score.append(mean_score)
        history_mean_balance.append(mean_balance)
        history_mean_speed.append(mean_speed)
        history_mean_contr.append(float(contraction_factor(delta_a, delta_b, p_breakdown).mean()))

        delta_a, delta_b, p_breakdown, v_a, v_b = interact_by_imitation(
            delta_a=delta_a,
            delta_b=delta_b,
            p_breakdown=p_breakdown,
            v_a=v_a,
            v_b=v_b,
            score=score,
            rng=rng,
            top_frac=top_frac,
            bottom_frac=bottom_frac,
            imitation_rate=imitation_rate,
            mutation_std=mutation_std,
            value_mutation_std=value_mutation_std,
            delta_min=delta_min,
            delta_max=delta_max,
            p_min=p_min,
            p_max=p_max,
        )

        if keep_history:
            history_v_a.append(v_a.copy())
            history_v_b.append(v_b.copy())
            star_a, star_b = exact_fixed_points(delta_a, delta_b, p_breakdown)
            history_star_a.append(star_a.copy())
            history_star_b.append(star_b.copy())

    final_star_a, final_star_b = exact_fixed_points(delta_a, delta_b, p_breakdown)
    final_score, final_balance_score, final_speed_score = compute_scores(
        v_a=v_a,
        delta_a=delta_a,
        delta_b=delta_b,
        p_breakdown=p_breakdown,
        weight_balance=weight_balance,
        weight_speed=weight_speed,
    )

    result = {
        "initial_delta_a": initial_delta_a,
        "initial_delta_b": initial_delta_b,
        "initial_p": initial_p,
        "initial_v_a": initial_v_a,
        "initial_v_b": initial_v_b,
        "final_delta_a": delta_a,
        "final_delta_b": delta_b,
        "final_p": p_breakdown,
        "final_v_a": v_a,
        "final_v_b": v_b,
        "final_star_a": final_star_a,
        "final_star_b": final_star_b,
        "final_score": final_score,
        "final_balance_score": final_balance_score,
        "final_speed_score": final_speed_score,
        "history_mean_score": np.asarray(history_mean_score),
        "history_mean_balance": np.asarray(history_mean_balance),
        "history_mean_speed": np.asarray(history_mean_speed),
        "history_mean_contr": np.asarray(history_mean_contr),
    }

    if keep_history:
        result["history_v_a"] = np.asarray(history_v_a)
        result["history_v_b"] = np.asarray(history_v_b)
        result["history_star_a"] = np.asarray(history_star_a)
        result["history_star_b"] = np.asarray(history_star_b)

    return result


def print_summary(result: dict, sample_rows: int) -> None:
    final_delta_a = result["final_delta_a"]
    final_delta_b = result["final_delta_b"]
    final_p = result["final_p"]
    final_v_a = result["final_v_a"]
    final_v_b = result["final_v_b"]
    final_star_a = result["final_star_a"]
    final_star_b = result["final_star_b"]
    final_score = result["final_score"]
    final_balance_score = result["final_balance_score"]
    final_speed_score = result["final_speed_score"]

    err_a = np.abs(final_v_a - final_star_a)
    err_b = np.abs(final_v_b - final_star_b)
    contr = contraction_factor(final_delta_a, final_delta_b, final_p)

    print("\n=== Rubinstein breakdown population game ===")
    print(f"Number of pairs                       = {final_delta_a.size}")
    print(f"Final delta_A mean / min / max       = {final_delta_a.mean():.6f} / {final_delta_a.min():.6f} / {final_delta_a.max():.6f}")
    print(f"Final delta_B mean / min / max       = {final_delta_b.mean():.6f} / {final_delta_b.min():.6f} / {final_delta_b.max():.6f}")
    print(f"Final p mean / min / max             = {final_p.mean():.6f} / {final_p.min():.6f} / {final_p.max():.6f}")
    print(f"Final contraction mean / min / max   = {contr.mean():.6f} / {contr.min():.6f} / {contr.max():.6f}")
    print(f"Final mean score                     = {final_score.mean():.6f}")
    print(f"Final mean balance score             = {final_balance_score.mean():.6f}")
    print(f"Final mean speed score               = {final_speed_score.mean():.6f}")
    print(f"Final |V_A - V_A*| mean / max        = {err_a.mean():.6e} / {err_a.max():.6e}")
    print(f"Final |V_B - V_B*| mean / max        = {err_b.mean():.6e} / {err_b.max():.6e}")

    count = min(sample_rows, final_delta_a.size)
    print("\n=== Sample final rows ===")
    print(
        f"{'pair':>6}  {'dA':>8}  {'dB':>8}  {'p':>8}  "
        f"{'V_A':>10}  {'V_B':>10}  {'V_A*':>10}  {'V_B*':>10}  "
        f"{'score':>10}  {'err_A':>10}  {'err_B':>10}"
    )
    print("-" * 118)
    for i in range(count):
        print(
            f"{i:>6}  {final_delta_a[i]:>8.4f}  {final_delta_b[i]:>8.4f}  {final_p[i]:>8.4f}  "
            f"{final_v_a[i]:>10.6f}  {final_v_b[i]:>10.6f}  {final_star_a[i]:>10.6f}  {final_star_b[i]:>10.6f}  "
            f"{final_score[i]:>10.6f}  {abs(final_v_a[i] - final_star_a[i]):>10.3e}  {abs(final_v_b[i] - final_star_b[i]):>10.3e}"
        )


def write_csv(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = zip(
        result["initial_delta_a"],
        result["initial_delta_b"],
        result["initial_p"],
        result["initial_v_a"],
        result["initial_v_b"],
        result["final_delta_a"],
        result["final_delta_b"],
        result["final_p"],
        result["final_v_a"],
        result["final_v_b"],
        result["final_star_a"],
        result["final_star_b"],
        result["final_score"],
        result["final_balance_score"],
        result["final_speed_score"],
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id",
            "initial_delta_A", "initial_delta_B", "initial_p_breakdown",
            "initial_V_A", "initial_V_B",
            "final_delta_A", "final_delta_B", "final_p_breakdown",
            "final_V_A", "final_V_B",
            "final_V_A_star", "final_V_B_star",
            "final_score", "final_balance_score", "final_speed_score",
        ])
        for idx, row in enumerate(rows):
            writer.writerow([idx, *[f"{x:.16f}" for x in row]])


def make_static_plot(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    initial_v_a = result["initial_v_a"]
    initial_v_b = result["initial_v_b"]
    final_v_a = result["final_v_a"]
    final_v_b = result["final_v_b"]
    final_star_a = result["final_star_a"]
    final_star_b = result["final_star_b"]
    final_delta_a = result["final_delta_a"]
    final_delta_b = result["final_delta_b"]
    final_score = result["final_score"]
    mean_score = result["history_mean_score"]
    mean_contr = result["history_mean_contr"]

    err_a = np.abs(final_v_a - final_star_a)
    err_b = np.abs(final_v_b - final_star_b)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(initial_v_a, initial_v_b, s=12, alpha=0.35)
    ax.set_title("Initial continuation-value cloud")
    ax.set_xlabel("A: V_A initial")
    ax.set_ylabel("B: V_B initial")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(final_v_a, final_v_b, s=12, alpha=0.35, label="Final cloud")
    ax.scatter(final_star_a, final_star_b, s=12, alpha=0.20, marker="x", label="Current equilibrium cloud")
    ax.set_title("Final cloud vs current equilibrium cloud")
    ax.set_xlabel("A: final V_A")
    ax.set_ylabel("B: final V_B")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(final_delta_a, final_delta_b, s=12, alpha=0.35)
    ax.set_title("Final parameter cloud")
    ax.set_xlabel("delta_A")
    ax.set_ylabel("delta_B")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(np.arange(1, mean_score.size + 1), mean_score, label="Mean score")
    ax.plot(np.arange(1, mean_contr.size + 1), mean_contr, label="Mean contraction")
    ax.set_title("Population-level evolution across epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Rubinstein Breakdown Population Game", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_animation(
    path: Path,
    result: dict,
    frame_step: int,
    fps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    history_v_a = result["history_v_a"]
    history_v_b = result["history_v_b"]
    history_star_a = result["history_star_a"]
    history_star_b = result["history_star_b"]

    frames = list(range(0, history_v_a.shape[0], frame_step))
    if frames[-1] != history_v_a.shape[0] - 1:
        frames.append(history_v_a.shape[0] - 1)

    x_all = np.concatenate([history_v_a.ravel(), history_star_a.ravel()])
    y_all = np.concatenate([history_v_b.ravel(), history_star_b.ravel()])

    x_min, x_max = float(x_all.min()), float(x_all.max())
    y_min, y_max = float(y_all.min()), float(y_all.max())
    x_pad = max(0.02, 0.05 * (x_max - x_min + 1e-12))
    y_pad = max(0.02, 0.05 * (y_max - y_min + 1e-12))

    fig, ax = plt.subplots(figsize=(8, 7))
    eq = ax.scatter(history_star_a[0], history_star_b[0], s=14, alpha=0.20, marker="x", label="Epoch equilibrium cloud")
    sc = ax.scatter(history_v_a[0], history_v_b[0], s=14, alpha=0.35, label="Moving cloud")
    title = ax.set_title("Population game | epoch 0")
    ax.set_xlabel("A continuation value")
    ax.set_ylabel("B continuation value")
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, alpha=0.3)
    ax.legend()

    def update(frame_idx: int):
        ep = frames[frame_idx]
        sc.set_offsets(np.column_stack((history_v_a[ep], history_v_b[ep])))
        eq.set_offsets(np.column_stack((history_star_a[ep], history_star_b[ep])))
        title.set_text(f"Population game | epoch {ep}")
        return sc, eq, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interacting Rubinstein breakdown population game."
    )
    parser.add_argument("--num-pairs", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--micro-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--delta-min", type=float, default=0.51)
    parser.add_argument("--delta-max", type=float, default=0.99)
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument("--p-max", type=float, default=0.4)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=1.0)

    parser.add_argument("--top-frac", type=float, default=0.20)
    parser.add_argument("--bottom-frac", type=float, default=0.20)
    parser.add_argument("--imitation-rate", type=float, default=0.50)
    parser.add_argument("--mutation-std", type=float, default=0.02)
    parser.add_argument("--value-mutation-std", type=float, default=0.02)

    parser.add_argument("--weight-balance", type=float, default=0.70)
    parser.add_argument("--weight-speed", type=float, default=0.30)

    parser.add_argument("--plot-out", type=str, default="")
    parser.add_argument("--animate-out", type=str, default="")
    parser.add_argument("--csv-out", type=str, default="")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--sample-rows", type=int, default=10)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 1

    keep_history = bool(args.animate_out)

    delta_a, delta_b, p_breakdown, v_a, v_b = sample_population(
        num_pairs=args.num_pairs,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        p_min=args.p_min,
        p_max=args.p_max,
        v_min=args.v_min,
        v_max=args.v_max,
        seed=args.seed,
    )

    result = simulate_population_game(
        delta_a=delta_a,
        delta_b=delta_b,
        p_breakdown=p_breakdown,
        v_a=v_a,
        v_b=v_b,
        epochs=args.epochs,
        micro_steps=args.micro_steps,
        weight_balance=args.weight_balance,
        weight_speed=args.weight_speed,
        top_frac=args.top_frac,
        bottom_frac=args.bottom_frac,
        imitation_rate=args.imitation_rate,
        mutation_std=args.mutation_std,
        value_mutation_std=args.value_mutation_std,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        p_min=args.p_min,
        p_max=args.p_max,
        seed=args.seed,
        keep_history=keep_history,
    )

    print_summary(result, sample_rows=args.sample_rows)

    if args.plot_out:
        make_static_plot(Path(args.plot_out), result)
        print(f"\nPlot written to: {args.plot_out}")

    if args.csv_out:
        write_csv(Path(args.csv_out), result)
        print(f"CSV written to: {args.csv_out}")

    if args.animate_out:
        make_animation(Path(args.animate_out), result, frame_step=args.frame_step, fps=args.fps)
        print(f"Animation written to: {args.animate_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
