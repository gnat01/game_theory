#!/usr/bin/env python3
"""
rubinstein_dynamics_full.py

Numerically evolve the continuation-value dynamical system for the
alternating-offers Rubinstein bargaining model, compare against the
exact fixed-point solution, optionally write CSV, and optionally
visualize the CSV using matplotlib.

Model:
    V_A = 1 - delta_B * V_B
    V_B = 1 - delta_A * V_A

One-step map:
    T(V_A, V_B) = (1 - delta_B * V_B, 1 - delta_A * V_A)

Two-step same-role map:
    V_A(next same role) = 1 - delta_B + delta_A * delta_B * V_A
    V_B(next same role) = 1 - delta_A + delta_A * delta_B * V_B

Exact fixed point:
    V_A* = (1 - delta_B) / (1 - delta_A * delta_B)
    V_B* = (1 - delta_A) / (1 - delta_A * delta_B)

Usage examples:
    python rubinstein_dynamics_full.py --delta_A 0.9 --delta_B 0.8
    python rubinstein_dynamics_full.py --delta_A 0.95 --delta_B 0.85 --V_A0 0.3 --V_B0 0.7 --steps 30 --csv-out history.csv
    python rubinstein_dynamics_full.py --delta_A 0.95 --delta_B 0.85 --steps 30 --csv-out history.csv --plot-out convergence.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def validate_discount_factor(name: str, value: float) -> None:
    """Ensure discount factor lies strictly between 0 and 1."""
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must satisfy 0 < {name} < 1, got {value}.")


def exact_fixed_point(delta_a: float, delta_b: float) -> Tuple[float, float]:
    """
    Compute the exact fixed point (V_A*, V_B*).

    V_A* = (1 - delta_B) / (1 - delta_A * delta_B)
    V_B* = (1 - delta_A) / (1 - delta_A * delta_B)
    """
    denom = 1.0 - delta_a * delta_b
    if math.isclose(denom, 0.0, rel_tol=0.0, abs_tol=1e-15):
        raise ZeroDivisionError("Denominator 1 - delta_A * delta_B is too close to zero.")
    v_a_star = (1.0 - delta_b) / denom
    v_b_star = (1.0 - delta_a) / denom
    return v_a_star, v_b_star


def one_step_map(v_a: float, v_b: float, delta_a: float, delta_b: float) -> Tuple[float, float]:
    """
    Apply the one-step alternating map:
        T(V_A, V_B) = (1 - delta_B * V_B, 1 - delta_A * V_A)
    """
    next_v_a = 1.0 - delta_b * v_b
    next_v_b = 1.0 - delta_a * v_a
    return next_v_a, next_v_b


def two_step_same_role_map(v_a: float, v_b: float, delta_a: float, delta_b: float) -> Tuple[float, float]:
    """
    Apply the two-step same-role map:
        V_A' = 1 - delta_B + delta_A * delta_B * V_A
        V_B' = 1 - delta_A + delta_A * delta_B * V_B
    """
    next_v_a = 1.0 - delta_b + delta_a * delta_b * v_a
    next_v_b = 1.0 - delta_a + delta_a * delta_b * v_b
    return next_v_a, next_v_b


def evolve_one_step(
    v_a0: float,
    v_b0: float,
    delta_a: float,
    delta_b: float,
    steps: int,
) -> List[Tuple[int, float, float]]:
    """Evolve the one-step map for the requested number of steps."""
    history: List[Tuple[int, float, float]] = [(0, v_a0, v_b0)]
    v_a, v_b = v_a0, v_b0
    for t in range(1, steps + 1):
        v_a, v_b = one_step_map(v_a, v_b, delta_a, delta_b)
        history.append((t, v_a, v_b))
    return history


def evolve_two_step(
    v_a0: float,
    v_b0: float,
    delta_a: float,
    delta_b: float,
    steps: int,
) -> List[Tuple[int, float, float]]:
    """Evolve the two-step same-role contraction for the requested number of steps."""
    history: List[Tuple[int, float, float]] = [(0, v_a0, v_b0)]
    v_a, v_b = v_a0, v_b0
    for k in range(1, steps + 1):
        v_a, v_b = two_step_same_role_map(v_a, v_b, delta_a, delta_b)
        history.append((k, v_a, v_b))
    return history


def format_float(x: float) -> str:
    """Compact numeric formatting for terminal output."""
    return f"{x:.10f}"


def print_exact_solution(delta_a: float, delta_b: float, v_a_star: float, v_b_star: float) -> None:
    """Print the exact symbolic/numeric equilibrium results."""
    print("\n=== Exact fixed point ===")
    print(f"delta_A = {delta_a}")
    print(f"delta_B = {delta_b}")
    print(f"V_A* = (1 - delta_B) / (1 - delta_A * delta_B) = {format_float(v_a_star)}")
    print(f"V_B* = (1 - delta_A) / (1 - delta_A * delta_B) = {format_float(v_b_star)}")

    a_first_share = v_a_star
    b_first_share = 1.0 - a_first_share

    print("\nIf A moves first, the accepted split is:")
    print(f"A gets x_A* = V_A* = {format_float(a_first_share)}")
    print(f"B gets 1 - x_A*       = {format_float(b_first_share)}")

    b_rejection_value = delta_b * v_b_star
    print("\nConsistency check:")
    print("B's value from rejecting A's offer and waiting one round = delta_B * V_B*")
    print(f"delta_B * V_B* = {format_float(b_rejection_value)}")
    print(f"1 - V_A*       = {format_float(1.0 - v_a_star)}")
    print(f"Difference      = {abs((1.0 - v_a_star) - b_rejection_value):.3e}")


def print_history_one_step(
    history: List[Tuple[int, float, float]],
    v_a_star: float,
    v_b_star: float,
    max_rows: int,
) -> None:
    """Print selected rows from one-step evolution."""
    print("\n=== One-step alternating map: T(V_A, V_B) = (1 - delta_B V_B, 1 - delta_A V_A) ===")
    print("This may oscillate by proposer role, but it converges to the same fixed point.")
    print()
    print(
        f"{'t':>4}  {'V_A(t)':>14}  {'V_B(t)':>14}  {'|V_A-V_A*|':>14}  {'|V_B-V_B*|':>14}"
    )
    print("-" * 70)

    rows_to_show = history if len(history) <= max_rows else history[: max_rows - 1] + [history[-1]]

    for t, v_a, v_b in rows_to_show:
        err_a = abs(v_a - v_a_star)
        err_b = abs(v_b - v_b_star)
        print(
            f"{t:>4}  {format_float(v_a):>14}  {format_float(v_b):>14}  "
            f"{err_a:>14.3e}  {err_b:>14.3e}"
        )

    if len(history) > max_rows:
        print(" ...")
        print(f"(showing first {max_rows - 1} rows and final row only)")


def print_history_two_step(
    history: List[Tuple[int, float, float]],
    v_a_star: float,
    v_b_star: float,
    max_rows: int,
    delta_a: float,
    delta_b: float,
) -> None:
    """Print selected rows from two-step evolution and compare decay ratio."""
    contraction = delta_a * delta_b

    print("\n=== Two-step same-role contraction ===")
    print("Map:")
    print("  V_A(next) = 1 - delta_B + delta_A*delta_B*V_A")
    print("  V_B(next) = 1 - delta_A + delta_A*delta_B*V_B")
    print(f"Expected contraction factor = delta_A * delta_B = {format_float(contraction)}")
    print()
    print(
        f"{'k':>4}  {'V_A(k)':>14}  {'V_B(k)':>14}  {'|V_A-V_A*|':>14}  "
        f"{'|V_B-V_B*|':>14}  {'errA ratio':>12}"
    )
    print("-" * 90)

    rows_to_show = history if len(history) <= max_rows else history[: max_rows - 1] + [history[-1]]

    prev_err_a = None
    for k, v_a, v_b in rows_to_show:
        err_a = abs(v_a - v_a_star)
        err_b = abs(v_b - v_b_star)
        ratio_str = "N/A"
        if prev_err_a is not None and prev_err_a > 0:
            ratio_str = f"{err_a / prev_err_a:.6f}"
        print(
            f"{k:>4}  {format_float(v_a):>14}  {format_float(v_b):>14}  "
            f"{err_a:>14.3e}  {err_b:>14.3e}  {ratio_str:>12}"
        )
        prev_err_a = err_a

    if len(history) > max_rows:
        print(" ...")
        print(f"(showing first {max_rows - 1} rows and final row only)")


def final_checks(
    one_step_hist: List[Tuple[int, float, float]],
    two_step_hist: List[Tuple[int, float, float]],
    v_a_star: float,
    v_b_star: float,
) -> None:
    """Print final numerical comparison to the exact solution."""
    _, v_a_1, v_b_1 = one_step_hist[-1]
    _, v_a_2, v_b_2 = two_step_hist[-1]

    print("\n=== Final numerical comparison ===")
    print("One-step map final state:")
    print(f"  V_A(final) = {format_float(v_a_1)}")
    print(f"  V_B(final) = {format_float(v_b_1)}")
    print(f"  |V_A(final) - V_A*| = {abs(v_a_1 - v_a_star):.6e}")
    print(f"  |V_B(final) - V_B*| = {abs(v_b_1 - v_b_star):.6e}")

    print("\nTwo-step map final state:")
    print(f"  V_A(final) = {format_float(v_a_2)}")
    print(f"  V_B(final) = {format_float(v_b_2)}")
    print(f"  |V_A(final) - V_A*| = {abs(v_a_2 - v_a_star):.6e}")
    print(f"  |V_B(final) - V_B*| = {abs(v_b_2 - v_b_star):.6e}")


def write_csv(
    path: Path,
    one_step_hist: List[Tuple[int, float, float]],
    two_step_hist: List[Tuple[int, float, float]],
    v_a_star: float,
    v_b_star: float,
) -> None:
    """
    Write both histories to CSV for inspection.

    Columns:
        map_type, step, V_A, V_B, exact_V_A_star, exact_V_B_star, abs_err_A, abs_err_B
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["map_type", "step", "V_A", "V_B", "exact_V_A_star", "exact_V_B_star", "abs_err_A", "abs_err_B"]
        )

        for step, v_a, v_b in one_step_hist:
            writer.writerow(
                ["one_step", step, f"{v_a:.16f}", f"{v_b:.16f}", f"{v_a_star:.16f}", f"{v_b_star:.16f}",
                 f"{abs(v_a - v_a_star):.16e}", f"{abs(v_b - v_b_star):.16e}"]
            )

        for step, v_a, v_b in two_step_hist:
            writer.writerow(
                ["two_step", step, f"{v_a:.16f}", f"{v_b:.16f}", f"{v_a_star:.16f}", f"{v_b_star:.16f}",
                 f"{abs(v_a - v_a_star):.16e}", f"{abs(v_b - v_b_star):.16e}"]
            )


def read_csv_for_plot(path: Path) -> dict:
    """
    Read the CSV and structure it for plotting.

    Returns:
        {
            "one_step": {"step": [...], "V_A": [...], "V_B": [...], "err_A": [...], "err_B": [...], ...},
            "two_step": {...}
        }
    """
    data = {
        "one_step": {"step": [], "V_A": [], "V_B": [], "err_A": [], "err_B": [], "exact_V_A_star": None, "exact_V_B_star": None},
        "two_step": {"step": [], "V_A": [], "V_B": [], "err_A": [], "err_B": [], "exact_V_A_star": None, "exact_V_B_star": None},
    }

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            map_type = row["map_type"]
            if map_type not in data:
                continue
            data[map_type]["step"].append(int(row["step"]))
            data[map_type]["V_A"].append(float(row["V_A"]))
            data[map_type]["V_B"].append(float(row["V_B"]))
            data[map_type]["err_A"].append(float(row["abs_err_A"]))
            data[map_type]["err_B"].append(float(row["abs_err_B"]))
            data[map_type]["exact_V_A_star"] = float(row["exact_V_A_star"])
            data[map_type]["exact_V_B_star"] = float(row["exact_V_B_star"])

    return data


def make_plot_from_csv(csv_path: Path, plot_path: Path) -> None:
    """
    Read the CSV and create a single matplotlib figure with four panels:
    1) one-step V_A, V_B versus exact values
    2) one-step absolute errors
    3) two-step V_A, V_B versus exact values
    4) two-step absolute errors
    """
    data = read_csv_for_plot(csv_path)

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    one = data["one_step"]
    two = data["two_step"]

    ax = axes[0, 0]
    ax.plot(one["step"], one["V_A"], marker="o", linewidth=1.5, label="One-step V_A")
    ax.plot(one["step"], one["V_B"], marker="s", linewidth=1.5, label="One-step V_B")
    ax.axhline(one["exact_V_A_star"], linestyle="--", linewidth=1.2, label="Exact V_A*")
    ax.axhline(one["exact_V_B_star"], linestyle=":", linewidth=1.2, label="Exact V_B*")
    ax.set_title("One-step map: values")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(one["step"], one["err_A"], marker="o", linewidth=1.5, label="|V_A - V_A*|")
    ax.plot(one["step"], one["err_B"], marker="s", linewidth=1.5, label="|V_B - V_B*|")
    ax.set_title("One-step map: absolute errors")
    ax.set_xlabel("Step")
    ax.set_ylabel("Absolute error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(two["step"], two["V_A"], marker="o", linewidth=1.5, label="Two-step V_A")
    ax.plot(two["step"], two["V_B"], marker="s", linewidth=1.5, label="Two-step V_B")
    ax.axhline(two["exact_V_A_star"], linestyle="--", linewidth=1.2, label="Exact V_A*")
    ax.axhline(two["exact_V_B_star"], linestyle=":", linewidth=1.2, label="Exact V_B*")
    ax.set_title("Two-step map: values")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(two["step"], two["err_A"], marker="o", linewidth=1.5, label="|V_A - V_A*|")
    ax.plot(two["step"], two["err_B"], marker="s", linewidth=1.5, label="|V_B - V_B*|")
    ax.set_title("Two-step map: absolute errors")
    ax.set_xlabel("Step")
    ax.set_ylabel("Absolute error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Rubinstein bargaining dynamics: CSV visualization", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Evolve Rubinstein bargaining continuation values, compare to exact equilibrium, write CSV, and visualize the CSV."
    )
    parser.add_argument("--delta_A", type=float, required=True, help="Discount factor for player A, with 0 < delta_A < 1.")
    parser.add_argument("--delta_B", type=float, required=True, help="Discount factor for player B, with 0 < delta_B < 1.")
    parser.add_argument("--V_A0", type=float, default=0.0, help="Initial value for V_A. Default: 0.0")
    parser.add_argument("--V_B0", type=float, default=0.0, help="Initial value for V_B. Default: 0.0")
    parser.add_argument("--steps", type=int, default=20, help="Number of evolution steps. Default: 20")
    parser.add_argument("--max_rows", type=int, default=12, help="Maximum number of rows to print from each history table. Default: 12")
    parser.add_argument("--csv-out", type=str, default="", help="Optional path to CSV output containing both histories.")
    parser.add_argument("--plot-out", type=str, default="", help="Optional path to a PNG plot visualizing the CSV.")
    return parser


def main() -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_discount_factor("delta_A", args.delta_A)
        validate_discount_factor("delta_B", args.delta_B)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    if args.steps < 0:
        print("Input error: --steps must be nonnegative.", file=sys.stderr)
        return 1

    try:
        v_a_star, v_b_star = exact_fixed_point(args.delta_A, args.delta_B)
    except ZeroDivisionError as exc:
        print(f"Numerical error: {exc}", file=sys.stderr)
        return 1

    one_step_hist = evolve_one_step(args.V_A0, args.V_B0, args.delta_A, args.delta_B, args.steps)
    two_step_hist = evolve_two_step(args.V_A0, args.V_B0, args.delta_A, args.delta_B, args.steps)

    print_exact_solution(args.delta_A, args.delta_B, v_a_star, v_b_star)
    print_history_one_step(one_step_hist, v_a_star, v_b_star, args.max_rows)
    print_history_two_step(two_step_hist, v_a_star, v_b_star, args.max_rows, args.delta_A, args.delta_B)
    final_checks(one_step_hist, two_step_hist, v_a_star, v_b_star)

    csv_path = None
    if args.csv_out:
        csv_path = Path(args.csv_out)
        write_csv(csv_path, one_step_hist, two_step_hist, v_a_star, v_b_star)
        print(f"\nCSV written to: {csv_path}")

    if args.plot_out:
        plot_path = Path(args.plot_out)
        if csv_path is None:
            default_csv = plot_path.with_suffix(".csv")
            write_csv(default_csv, one_step_hist, two_step_hist, v_a_star, v_b_star)
            csv_path = default_csv
            print(f"CSV written to: {csv_path}")
        make_plot_from_csv(csv_path, plot_path)
        print(f"Plot written to: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
