#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def validate_args(args: argparse.Namespace) -> None:
    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be positive.")
    if args.steps < 0:
        raise ValueError("--steps must be nonnegative.")
    if args.snapshot_step < 0:
        raise ValueError("--snapshot-step must be nonnegative.")
    if args.snapshot_step > args.steps:
        raise ValueError("--snapshot-step cannot exceed --steps.")
    if not (0.0 < args.delta_min < 1.0 and 0.0 < args.delta_max < 1.0):
        raise ValueError("delta_min and delta_max must both lie strictly between 0 and 1.")
    if args.delta_min >= args.delta_max:
        raise ValueError("Require delta_min < delta_max.")
    if not (0.0 <= args.p_min < 1.0 and 0.0 <= args.p_max < 1.0):
        raise ValueError("p_min and p_max must satisfy 0 <= p < 1.")
    if args.p_min >= args.p_max:
        raise ValueError("Require p_min < p_max.")
    if args.v_min >= args.v_max:
        raise ValueError("Require v_min < v_max.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive.")


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
    v_a0 = rng.uniform(v_min, v_max, size=num_pairs)
    v_b0 = rng.uniform(v_min, v_max, size=num_pairs)
    return delta_a, delta_b, p_breakdown, v_a0, v_b0


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


def two_step_update(
    v_a: np.ndarray,
    v_b: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    survival = 1.0 - p_breakdown
    contraction = (survival ** 2) * delta_a * delta_b
    next_v_a = 1.0 - survival * delta_b + contraction * v_a
    next_v_b = 1.0 - survival * delta_a + contraction * v_b
    return next_v_a, next_v_b


def simulate_population(
    v_a0: np.ndarray,
    v_b0: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
    steps: int,
    snapshot_step: int,
    keep_history: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    v_a = v_a0.copy()
    v_b = v_b0.copy()

    snap_v_a = v_a.copy()
    snap_v_b = v_b.copy()

    history_a = None
    history_b = None
    if keep_history:
        history_a = np.empty((steps + 1, v_a0.size), dtype=float)
        history_b = np.empty((steps + 1, v_b0.size), dtype=float)
        history_a[0] = v_a0
        history_b[0] = v_b0

    for k in range(1, steps + 1):
        v_a, v_b = two_step_update(v_a, v_b, delta_a, delta_b, p_breakdown)
        if keep_history:
            history_a[k] = v_a
            history_b[k] = v_b
        if k == snapshot_step:
            snap_v_a = v_a.copy()
            snap_v_b = v_b.copy()

    if snapshot_step == 0:
        snap_v_a = v_a0.copy()
        snap_v_b = v_b0.copy()

    return v_a0, v_b0, snap_v_a, snap_v_b, v_a, v_b, history_a, history_b


def summarize_population(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
    v_a0: np.ndarray,
    v_b0: np.ndarray,
    v_a_final: np.ndarray,
    v_b_final: np.ndarray,
    v_a_star: np.ndarray,
    v_b_star: np.ndarray,
) -> None:
    err_a = np.abs(v_a_final - v_a_star)
    err_b = np.abs(v_b_final - v_b_star)
    contraction = ((1.0 - p_breakdown) ** 2) * delta_a * delta_b

    print("\n=== Population Rubinstein breakdown game ===")
    print(f"Number of pairs                    = {delta_a.size}")
    print(f"delta_A mean / min / max          = {delta_a.mean():.6f} / {delta_a.min():.6f} / {delta_a.max():.6f}")
    print(f"delta_B mean / min / max          = {delta_b.mean():.6f} / {delta_b.min():.6f} / {delta_b.max():.6f}")
    print(f"p_breakdown mean / min / max      = {p_breakdown.mean():.6f} / {p_breakdown.min():.6f} / {p_breakdown.max():.6f}")
    print(f"Initial V_A mean / min / max      = {v_a0.mean():.6f} / {v_a0.min():.6f} / {v_a0.max():.6f}")
    print(f"Initial V_B mean / min / max      = {v_b0.mean():.6f} / {v_b0.min():.6f} / {v_b0.max():.6f}")
    print(f"Contraction mean / min / max      = {contraction.mean():.6f} / {contraction.min():.6f} / {contraction.max():.6f}")
    print(f"Final |V_A - V_A*| mean / max     = {err_a.mean():.6e} / {err_a.max():.6e}")
    print(f"Final |V_B - V_B*| mean / max     = {err_b.mean():.6e} / {err_b.max():.6e}")


def print_sample_rows(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
    v_a0: np.ndarray,
    v_b0: np.ndarray,
    v_a_final: np.ndarray,
    v_b_final: np.ndarray,
    v_a_star: np.ndarray,
    v_b_star: np.ndarray,
    num_rows: int,
) -> None:
    count = min(num_rows, delta_a.size)
    err_a = np.abs(v_a_final - v_a_star)
    err_b = np.abs(v_b_final - v_b_star)

    print("\n=== Sample pair rows ===")
    print(
        f"{'pair':>6}  {'delta_A':>8}  {'delta_B':>8}  {'p':>8}  "
        f"{'V_A(0)':>10}  {'V_B(0)':>10}  {'V_A(T)':>10}  {'V_B(T)':>10}  "
        f"{'V_A*':>10}  {'V_B*':>10}  {'err_A':>10}  {'err_B':>10}"
    )
    print("-" * 130)

    for i in range(count):
        print(
            f"{i:>6}  {delta_a[i]:>8.4f}  {delta_b[i]:>8.4f}  {p_breakdown[i]:>8.4f}  "
            f"{v_a0[i]:>10.6f}  {v_b0[i]:>10.6f}  {v_a_final[i]:>10.6f}  {v_b_final[i]:>10.6f}  "
            f"{v_a_star[i]:>10.6f}  {v_b_star[i]:>10.6f}  {err_a[i]:>10.3e}  {err_b[i]:>10.3e}"
        )


def write_csv(
    path: Path,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    p_breakdown: np.ndarray,
    v_a0: np.ndarray,
    v_b0: np.ndarray,
    snap_v_a: np.ndarray,
    snap_v_b: np.ndarray,
    v_a_final: np.ndarray,
    v_b_final: np.ndarray,
    v_a_star: np.ndarray,
    v_b_star: np.ndarray,
    snapshot_step: int,
    steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair_id",
                "delta_A",
                "delta_B",
                "p_breakdown",
                "V_A_initial",
                "V_B_initial",
                f"V_A_step_{snapshot_step}",
                f"V_B_step_{snapshot_step}",
                f"V_A_step_{steps}",
                f"V_B_step_{steps}",
                "V_A_star",
                "V_B_star",
                "abs_err_A_final",
                "abs_err_B_final",
            ]
        )

        for i in range(delta_a.size):
            writer.writerow(
                [
                    i,
                    f"{delta_a[i]:.16f}",
                    f"{delta_b[i]:.16f}",
                    f"{p_breakdown[i]:.16f}",
                    f"{v_a0[i]:.16f}",
                    f"{v_b0[i]:.16f}",
                    f"{snap_v_a[i]:.16f}",
                    f"{snap_v_b[i]:.16f}",
                    f"{v_a_final[i]:.16f}",
                    f"{v_b_final[i]:.16f}",
                    f"{v_a_star[i]:.16f}",
                    f"{v_b_star[i]:.16f}",
                    f"{abs(v_a_final[i] - v_a_star[i]):.16e}",
                    f"{abs(v_b_final[i] - v_b_star[i]):.16e}",
                ]
            )


def make_plot(
    plot_path: Path,
    v_a0: np.ndarray,
    v_b0: np.ndarray,
    snap_v_a: np.ndarray,
    snap_v_b: np.ndarray,
    v_a_final: np.ndarray,
    v_b_final: np.ndarray,
    v_a_star: np.ndarray,
    v_b_star: np.ndarray,
    steps: int,
    snapshot_step: int,
    alpha: float,
    point_size: float,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    err_a = np.abs(v_a_final - v_a_star)
    err_b = np.abs(v_b_final - v_b_star)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(v_a0, v_b0, s=point_size, alpha=alpha)
    ax.set_title("Initial continuation-value cloud")
    ax.set_xlabel("A: V_A(0)")
    ax.set_ylabel("B: V_B(0)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(snap_v_a, snap_v_b, s=point_size, alpha=alpha)
    ax.set_title(f"Intermediate cloud at step {snapshot_step}")
    ax.set_xlabel(f"A: V_A({snapshot_step})")
    ax.set_ylabel(f"B: V_B({snapshot_step})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(v_a_final, v_b_final, s=point_size, alpha=alpha, label="Final cloud")
    ax.scatter(v_a_star, v_b_star, s=point_size, alpha=alpha, marker="x", label="Equilibrium cloud")
    ax.set_title(f"Final cloud vs equilibrium cloud at step {steps}")
    ax.set_xlabel(f"A: V_A({steps})")
    ax.set_ylabel(f"B: V_B({steps})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(err_a, err_b, s=point_size, alpha=alpha)
    ax.set_title(f"Distance-to-equilibrium cloud at step {steps}")
    ax.set_xlabel(r"$|V_A(T)-V_A^*|$")
    ax.set_ylabel(r"$|V_B(T)-V_B^*|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Population Rubinstein Breakdown Game (two-step dynamics)", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_animation(
    animate_path: Path,
    history_a: np.ndarray,
    history_b: np.ndarray,
    v_a_star: np.ndarray,
    v_b_star: np.ndarray,
    frame_step: int,
    fps: int,
    alpha: float,
    point_size: float,
) -> None:
    animate_path.parent.mkdir(parents=True, exist_ok=True)

    frames = list(range(0, history_a.shape[0], frame_step))
    if frames[-1] != history_a.shape[0] - 1:
        frames.append(history_a.shape[0] - 1)

    x_all = np.concatenate([history_a.ravel(), v_a_star.ravel()])
    y_all = np.concatenate([history_b.ravel(), v_b_star.ravel()])

    x_min, x_max = float(x_all.min()), float(x_all.max())
    y_min, y_max = float(y_all.min()), float(y_all.max())

    x_pad = max(0.02, 0.05 * (x_max - x_min + 1e-12))
    y_pad = max(0.02, 0.05 * (y_max - y_min + 1e-12))

    fig, ax = plt.subplots(figsize=(8, 7))
    eq = ax.scatter(v_a_star, v_b_star, s=point_size, alpha=0.20, marker="x", label="Equilibrium cloud")
    sc = ax.scatter(history_a[0], history_b[0], s=point_size, alpha=alpha, label="Moving cloud")
    title = ax.set_title("Population Rubinstein Breakdown Game | step 0")
    ax.set_xlabel("A continuation value")
    ax.set_ylabel("B continuation value")
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, alpha=0.3)
    ax.legend()

    def update(frame_idx: int):
        step = frames[frame_idx]
        offsets = np.column_stack((history_a[step], history_b[step]))
        sc.set_offsets(offsets)
        title.set_text(f"Population Rubinstein Breakdown Game | step {step}")
        return sc, title, eq

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(animate_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Population Rubinstein breakdown dynamics with heterogeneous pairs."
    )
    parser.add_argument("--num-pairs", type=int, default=1000, help="Number of independent bargaining pairs. Default: 1000")
    parser.add_argument("--steps", type=int, default=100, help="Number of two-step updates. Default: 100")
    parser.add_argument("--snapshot-step", type=int, default=20, help="Intermediate step used in the 2x2 plot. Default: 20")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument("--delta-min", type=float, default=0.51, help="Lower bound for Uniform sampling of both deltas. Default: 0.51")
    parser.add_argument("--delta-max", type=float, default=0.99, help="Upper bound for Uniform sampling of both deltas. Default: 0.99")
    parser.add_argument("--p-min", type=float, default=0.0, help="Lower bound for Uniform sampling of p_breakdown. Default: 0.0")
    parser.add_argument("--p-max", type=float, default=0.4, help="Upper bound for Uniform sampling of p_breakdown. Default: 0.4")
    parser.add_argument("--v-min", type=float, default=0.0, help="Lower bound for Uniform sampling of initial values. Default: 0.0")
    parser.add_argument("--v-max", type=float, default=1.0, help="Upper bound for Uniform sampling of initial values. Default: 1.0")
    parser.add_argument("--plot-out", type=str, default="", help="Optional output PNG path for the 2x2 plot.")
    parser.add_argument("--csv-out", type=str, default="", help="Optional CSV output path.")
    parser.add_argument("--animate-out", type=str, default="", help="Optional animated GIF output path.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the GIF. Default: 8")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every kth simulation step in the movie. Default: 1")
    parser.add_argument("--sample-rows", type=int, default=10, help="How many sample pair rows to print. Default: 10")
    parser.add_argument("--alpha", type=float, default=0.35, help="Scatter transparency for the plots. Default: 0.35")
    parser.add_argument("--point-size", type=float, default=14.0, help="Scatter point size. Default: 14.0")
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

    delta_a, delta_b, p_breakdown, v_a0, v_b0 = sample_population(
        num_pairs=args.num_pairs,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        p_min=args.p_min,
        p_max=args.p_max,
        v_min=args.v_min,
        v_max=args.v_max,
        seed=args.seed,
    )

    v_a_star, v_b_star = exact_fixed_points(delta_a, delta_b, p_breakdown)

    v_a0_plot, v_b0_plot, snap_v_a, snap_v_b, v_a_final, v_b_final, history_a, history_b = simulate_population(
        v_a0=v_a0,
        v_b0=v_b0,
        delta_a=delta_a,
        delta_b=delta_b,
        p_breakdown=p_breakdown,
        steps=args.steps,
        snapshot_step=args.snapshot_step,
        keep_history=keep_history,
    )

    summarize_population(
        delta_a=delta_a,
        delta_b=delta_b,
        p_breakdown=p_breakdown,
        v_a0=v_a0_plot,
        v_b0=v_b0_plot,
        v_a_final=v_a_final,
        v_b_final=v_b_final,
        v_a_star=v_a_star,
        v_b_star=v_b_star,
    )

    print_sample_rows(
        delta_a=delta_a,
        delta_b=delta_b,
        p_breakdown=p_breakdown,
        v_a0=v_a0_plot,
        v_b0=v_b0_plot,
        v_a_final=v_a_final,
        v_b_final=v_b_final,
        v_a_star=v_a_star,
        v_b_star=v_b_star,
        num_rows=args.sample_rows,
    )

    if args.plot_out:
        make_plot(
            plot_path=Path(args.plot_out),
            v_a0=v_a0_plot,
            v_b0=v_b0_plot,
            snap_v_a=snap_v_a,
            snap_v_b=snap_v_b,
            v_a_final=v_a_final,
            v_b_final=v_b_final,
            v_a_star=v_a_star,
            v_b_star=v_b_star,
            steps=args.steps,
            snapshot_step=args.snapshot_step,
            alpha=args.alpha,
            point_size=args.point_size,
        )
        print(f"\nPlot written to: {args.plot_out}")

    if args.csv_out:
        write_csv(
            path=Path(args.csv_out),
            delta_a=delta_a,
            delta_b=delta_b,
            p_breakdown=p_breakdown,
            v_a0=v_a0_plot,
            v_b0=v_b0_plot,
            snap_v_a=snap_v_a,
            snap_v_b=snap_v_b,
            v_a_final=v_a_final,
            v_b_final=v_b_final,
            v_a_star=v_a_star,
            v_b_star=v_b_star,
            snapshot_step=args.snapshot_step,
            steps=args.steps,
        )
        print(f"CSV written to: {args.csv_out}")

    if args.animate_out:
        if history_a is None or history_b is None:
            raise RuntimeError("Animation requested but history was not retained.")
        make_animation(
            animate_path=Path(args.animate_out),
            history_a=history_a,
            history_b=history_b,
            v_a_star=v_a_star,
            v_b_star=v_b_star,
            frame_step=args.frame_step,
            fps=args.fps,
            alpha=args.alpha,
            point_size=args.point_size,
        )
        print(f"Animation written to: {args.animate_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
