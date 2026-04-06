#!/usr/bin/env python3
"""
rubinstein_dynamics_breakdown_risk.py

Rubinstein alternating-offers bargaining with exogenous breakdown risk.

After every rejection:
    - with probability p_breakdown, bargaining ends and both players get 0
    - with probability 1 - p_breakdown, bargaining continues to the next round

Continuation-value recursions:
    V_A = 1 - (1-p) * delta_B * V_B
    V_B = 1 - (1-p) * delta_A * V_A

Exact fixed point:
    V_A* = [1 - (1-p) delta_B] / [1 - (1-p)^2 delta_A delta_B]
    V_B* = [1 - (1-p) delta_A] / [1 - (1-p)^2 delta_A delta_B]

If A moves first, A's accepted share is:
    x_A* = V_A*

B's accepted share is:
    1 - x_A* = (1-p) delta_B V_B*

This script:
    1. computes the exact fixed point
    2. evolves the one-step map
    3. evolves the two-step same-role map
    4. prints tables and final errors
    5. writes CSV
    6. reads the CSV back in
    7. plots convergence using matplotlib

Example:
    python rubinstein_dynamics_breakdown_risk.py --delta_A 0.9 --delta_B 0.8 --p_breakdown 0.2
    python rubinstein_dynamics_breakdown_risk.py --delta_A 0.95 --delta_B 0.85 --p_breakdown 0.15 --V_A0 0.3 --V_B0 0.7 --steps 30 --csv-out breakdown_history.csv --plot-out breakdown_plot.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def validate_discount_factor(name: str, value: float) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must satisfy 0 < {name} < 1, got {value}.")


def validate_breakdown_probability(value: float) -> None:
    if not (0.0 <= value < 1.0):
        raise ValueError(f"p_breakdown must satisfy 0 <= p_breakdown < 1, got {value}.")


def exact_fixed_point(delta_a: float, delta_b: float, p_breakdown: float) -> Tuple[float, float]:
    """
    Exact fixed point for Rubinstein bargaining with breakdown risk.

        V_A* = [1 - (1-p) delta_B] / [1 - (1-p)^2 delta_A delta_B]
        V_B* = [1 - (1-p) delta_A] / [1 - (1-p)^2 delta_A delta_B]
    """
    survival = 1.0 - p_breakdown
    denom = 1.0 - (survival ** 2) * delta_a * delta_b
    if math.isclose(denom, 0.0, rel_tol=0.0, abs_tol=1e-15):
        raise ZeroDivisionError("Denominator is too close to zero.")
    v_a_star = (1.0 - survival * delta_b) / denom
    v_b_star = (1.0 - survival * delta_a) / denom
    return v_a_star, v_b_star


def one_step_map(
    v_a: float,
    v_b: float,
    delta_a: float,
    delta_b: float,
    p_breakdown: float,
) -> Tuple[float, float]:
    """
    One-step alternating map:
        T(V_A, V_B) = (1 - (1-p) delta_B V_B, 1 - (1-p) delta_A V_A)
    """
    survival = 1.0 - p_breakdown
    next_v_a = 1.0 - survival * delta_b * v_b
    next_v_b = 1.0 - survival * delta_a * v_a
    return next_v_a, next_v_b


def two_step_same_role_map(
    v_a: float,
    v_b: float,
    delta_a: float,
    delta_b: float,
    p_breakdown: float,
) -> Tuple[float, float]:
    """
    Two-step same-role map:
        V_A(next) = 1 - (1-p) delta_B + (1-p)^2 delta_A delta_B V_A
        V_B(next) = 1 - (1-p) delta_A + (1-p)^2 delta_A delta_B V_B
    """
    survival = 1.0 - p_breakdown
    next_v_a = 1.0 - survival * delta_b + (survival ** 2) * delta_a * delta_b * v_a
    next_v_b = 1.0 - survival * delta_a + (survival ** 2) * delta_a * delta_b * v_b
    return next_v_a, next_v_b


def evolve_one_step(
    v_a0: float,
    v_b0: float,
    delta_a: float,
    delta_b: float,
    p_breakdown: float,
    steps: int,
) -> List[Tuple[int, float, float]]:
    history: List[Tuple[int, float, float]] = [(0, v_a0, v_b0)]
    v_a, v_b = v_a0, v_b0
    for t in range(1, steps + 1):
        v_a, v_b = one_step_map(v_a, v_b, delta_a, delta_b, p_breakdown)
        history.append((t, v_a, v_b))
    return history


def evolve_two_step(
    v_a0: float,
    v_b0: float,
    delta_a: float,
    delta_b: float,
    p_breakdown: float,
    steps: int,
) -> List[Tuple[int, float, float]]:
    history: List[Tuple[int, float, float]] = [(0, v_a0, v_b0)]
    v_a, v_b = v_a0, v_b0
    for k in range(1, steps + 1):
        v_a, v_b = two_step_same_role_map(v_a, v_b, delta_a, delta_b, p_breakdown)
        history.append((k, v_a, v_b))
    return history


def format_float(x: float) -> str:
    return f"{x:.10f}"


def print_exact_solution(
    delta_a: float,
    delta_b: float,
    p_breakdown: float,
    v_a_star: float,
    v_b_star: float,
) -> None:
    survival = 1.0 - p_breakdown
    print("\n=== Exact fixed point: Rubinstein with breakdown risk ===")
    print(f"delta_A      = {delta_a}")
    print(f"delta_B      = {delta_b}")
    print(f"p_breakdown  = {p_breakdown}")
    print(f"survival prob= {survival}")
    print()
    print("Formulas:")
    print("  V_A* = [1 - (1-p) delta_B] / [1 - (1-p)^2 delta_A delta_B]")
    print("  V_B* = [1 - (1-p) delta_A] / [1 - (1-p)^2 delta_A delta_B]")
    print()
    print(f"V_A* = {format_float(v_a_star)}")
    print(f"V_B* = {format_float(v_b_star)}")

    a_first_share = v_a_star
    b_first_share = 1.0 - a_first_share
    b_reject_value = survival * delta_b * v_b_star

    print("\nIf A moves first:")
    print(f"Theoretical ideal A share = {format_float(a_first_share)}")
    print(f"Theoretical ideal B share = {format_float(b_first_share)}")
    print(f"B rejection continuation  = {format_float(b_reject_value)}")
    print(f"Consistency gap           = {abs(b_first_share - b_reject_value):.3e}")


def print_history_one_step(
    history: List[Tuple[int, float, float]],
    v_a_star: float,
    v_b_star: float,
    max_rows: int,
) -> None:
    print("\n=== One-step alternating map with breakdown risk ===")
    print("T(V_A, V_B) = (1 - (1-p) delta_B V_B, 1 - (1-p) delta_A V_A)")
    print()
    print(f"{'t':>4}  {'V_A(t)':>14}  {'V_B(t)':>14}  {'|V_A-V_A*|':>14}  {'|V_B-V_B*|':>14}")
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
    p_breakdown: float,
) -> None:
    survival = 1.0 - p_breakdown
    contraction = (survival ** 2) * delta_a * delta_b

    print("\n=== Two-step same-role contraction with breakdown risk ===")
    print("Map:")
    print("  V_A(next) = 1 - (1-p) delta_B + (1-p)^2 delta_A delta_B V_A")
    print("  V_B(next) = 1 - (1-p) delta_A + (1-p)^2 delta_A delta_B V_B")
    print(f"Expected contraction factor = (1-p)^2 delta_A delta_B = {format_float(contraction)}")
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
    delta_a: float,
    delta_b: float,
    p_breakdown: float,
) -> None:
    survival = 1.0 - p_breakdown

    theoretical_a_share_if_a_proposes = v_a_star
    theoretical_b_share_if_a_proposes = 1.0 - v_a_star
    theoretical_b_share_via_continuation = survival * delta_b * v_b_star

    _, v_a_1, v_b_1 = one_step_hist[-1]
    _, v_a_2, v_b_2 = two_step_hist[-1]

    actual_a_share_one = v_a_1
    actual_b_share_one = 1.0 - v_a_1

    actual_a_share_two = v_a_2
    actual_b_share_two = 1.0 - v_a_2

    print("\n=== Final numerical comparison ===")
    print("Theoretical ideal accepted split when A proposes first:")
    print(f"  Ideal A share = {format_float(theoretical_a_share_if_a_proposes)}")
    print(f"  Ideal B share = {format_float(theoretical_b_share_if_a_proposes)}")
    print(f"  Ideal B share via (1-p) * delta_B * V_B* = {format_float(theoretical_b_share_via_continuation)}")
    print(f"  Ideal-share consistency gap = {abs(theoretical_b_share_if_a_proposes - theoretical_b_share_via_continuation):.6e}")

    print("\nOne-step map final state:")
    print(f"  V_A(final) = {format_float(v_a_1)}")
    print(f"  V_B(final) = {format_float(v_b_1)}")
    print(f"  |V_A(final) - V_A*| = {abs(v_a_1 - v_a_star):.6e}")
    print(f"  |V_B(final) - V_B*| = {abs(v_b_1 - v_b_star):.6e}")
    print("  Actual accepted split if A proposes first, using the final iterate:")
    print(f"    ACTUAL A share = {format_float(actual_a_share_one)}")
    print(f"    ACTUAL B share = {format_float(actual_b_share_one)}")
    print(f"    Distance of ACTUAL A share from ideal A share = {abs(actual_a_share_one - theoretical_a_share_if_a_proposes):.6e}")
    print(f"    Distance of ACTUAL B share from ideal B share = {abs(actual_b_share_one - theoretical_b_share_if_a_proposes):.6e}")

    print("\nTwo-step map final state:")
    print(f"  V_A(final) = {format_float(v_a_2)}")
    print(f"  V_B(final) = {format_float(v_b_2)}")
    print(f"  |V_A(final) - V_A*| = {abs(v_a_2 - v_a_star):.6e}")
    print(f"  |V_B(final) - V_B*| = {abs(v_b_2 - v_b_star):.6e}")
    print("  Actual accepted split if A proposes first, using the final iterate:")
    print(f"    ACTUAL A share = {format_float(actual_a_share_two)}")
    print(f"    ACTUAL B share = {format_float(actual_b_share_two)}")
    print(f"    Distance of ACTUAL A share from ideal A share = {abs(actual_a_share_two - theoretical_a_share_if_a_proposes):.6e}")
    print(f"    Distance of ACTUAL B share from ideal B share = {abs(actual_b_share_two - theoretical_b_share_if_a_proposes):.6e}")


def write_csv(
    path: Path,
    one_step_hist: List[Tuple[int, float, float]],
    two_step_hist: List[Tuple[int, float, float]],
    v_a_star: float,
    v_b_star: float,
    delta_a: float,
    delta_b: float,
    p_breakdown: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "map_type",
                "step",
                "V_A",
                "V_B",
                "exact_V_A_star",
                "exact_V_B_star",
                "delta_A",
                "delta_B",
                "p_breakdown",
                "abs_err_A",
                "abs_err_B",
            ]
        )

        for step, v_a, v_b in one_step_hist:
            writer.writerow(
                [
                    "one_step",
                    step,
                    f"{v_a:.16f}",
                    f"{v_b:.16f}",
                    f"{v_a_star:.16f}",
                    f"{v_b_star:.16f}",
                    f"{delta_a:.16f}",
                    f"{delta_b:.16f}",
                    f"{p_breakdown:.16f}",
                    f"{abs(v_a - v_a_star):.16e}",
                    f"{abs(v_b - v_b_star):.16e}",
                ]
            )

        for step, v_a, v_b in two_step_hist:
            writer.writerow(
                [
                    "two_step",
                    step,
                    f"{v_a:.16f}",
                    f"{v_b:.16f}",
                    f"{v_a_star:.16f}",
                    f"{v_b_star:.16f}",
                    f"{delta_a:.16f}",
                    f"{delta_b:.16f}",
                    f"{p_breakdown:.16f}",
                    f"{abs(v_a - v_a_star):.16e}",
                    f"{abs(v_b - v_b_star):.16e}",
                ]
            )


def read_csv_for_plot(path: Path) -> Dict[str, Dict[str, List[float]]]:
    data: Dict[str, Dict[str, List[float]]] = {
        "one_step": {"step": [], "V_A": [], "V_B": [], "err_A": [], "err_B": []},
        "two_step": {"step": [], "V_A": [], "V_B": [], "err_A": [], "err_B": []},
    }
    meta: Dict[str, float] = {}

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            map_type = row["map_type"]
            data[map_type]["step"].append(int(row["step"]))
            data[map_type]["V_A"].append(float(row["V_A"]))
            data[map_type]["V_B"].append(float(row["V_B"]))
            data[map_type]["err_A"].append(float(row["abs_err_A"]))
            data[map_type]["err_B"].append(float(row["abs_err_B"]))
            meta["exact_V_A_star"] = float(row["exact_V_A_star"])
            meta["exact_V_B_star"] = float(row["exact_V_B_star"])
            meta["delta_A"] = float(row["delta_A"])
            meta["delta_B"] = float(row["delta_B"])
            meta["p_breakdown"] = float(row["p_breakdown"])

    return {"series": data, "meta": meta}  # type: ignore[return-value]


def make_plot_from_csv(csv_path: Path, plot_path: Path) -> None:
    parsed = read_csv_for_plot(csv_path)
    data = parsed["series"]  # type: ignore[index]
    meta = parsed["meta"]    # type: ignore[index]

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    one = data["one_step"]
    two = data["two_step"]
    exact_v_a = meta["exact_V_A_star"]
    exact_v_b = meta["exact_V_B_star"]
    delta_a = meta["delta_A"]
    delta_b = meta["delta_B"]
    p_breakdown = meta["p_breakdown"]

    ax = axes[0, 0]
    ax.plot(one["step"], one["V_A"], marker="o", linewidth=1.5, label="One-step V_A")
    ax.plot(one["step"], one["V_B"], marker="s", linewidth=1.5, label="One-step V_B")
    ax.axhline(exact_v_a, linestyle="--", linewidth=1.2, label="Exact V_A*")
    ax.axhline(exact_v_b, linestyle=":", linewidth=1.2, label="Exact V_B*")
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
    ax.axhline(exact_v_a, linestyle="--", linewidth=1.2, label="Exact V_A*")
    ax.axhline(exact_v_b, linestyle=":", linewidth=1.2, label="Exact V_B*")
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

    fig.suptitle(
        f"Rubinstein bargaining with breakdown risk | delta_A={delta_a:.3f}, delta_B={delta_b:.3f}, p={p_breakdown:.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rubinstein bargaining with breakdown risk: exact solution, dynamics, tables, CSV, and plots."
    )
    parser.add_argument("--delta_A", type=float, required=True, help="Discount factor for player A, with 0 < delta_A < 1.")
    parser.add_argument("--delta_B", type=float, required=True, help="Discount factor for player B, with 0 < delta_B < 1.")
    parser.add_argument("--p_breakdown", type=float, required=True, help="Breakdown probability after rejection, with 0 <= p_breakdown < 1.")
    parser.add_argument("--V_A0", type=float, default=0.0, help="Initial value for V_A. Default: 0.0")
    parser.add_argument("--V_B0", type=float, default=0.0, help="Initial value for V_B. Default: 0.0")
    parser.add_argument("--steps", type=int, default=20, help="Number of evolution steps. Default: 20")
    parser.add_argument("--max_rows", type=int, default=12, help="Maximum number of rows printed in each table. Default: 12")
    parser.add_argument("--csv-out", type=str, default="", help="Optional CSV output path.")
    parser.add_argument("--plot-out", type=str, default="", help="Optional plot PNG output path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_discount_factor("delta_A", args.delta_A)
        validate_discount_factor("delta_B", args.delta_B)
        validate_breakdown_probability(args.p_breakdown)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    if args.steps < 0:
        print("Input error: --steps must be nonnegative.", file=sys.stderr)
        return 1

    try:
        v_a_star, v_b_star = exact_fixed_point(args.delta_A, args.delta_B, args.p_breakdown)
    except ZeroDivisionError as exc:
        print(f"Numerical error: {exc}", file=sys.stderr)
        return 1

    one_step_hist = evolve_one_step(
        args.V_A0,
        args.V_B0,
        args.delta_A,
        args.delta_B,
        args.p_breakdown,
        args.steps,
    )
    two_step_hist = evolve_two_step(
        args.V_A0,
        args.V_B0,
        args.delta_A,
        args.delta_B,
        args.p_breakdown,
        args.steps,
    )

    print_exact_solution(args.delta_A, args.delta_B, args.p_breakdown, v_a_star, v_b_star)
    print_history_one_step(one_step_hist, v_a_star, v_b_star, args.max_rows)
    print_history_two_step(two_step_hist, v_a_star, v_b_star, args.max_rows, args.delta_A, args.delta_B, args.p_breakdown)
    final_checks(one_step_hist, two_step_hist, v_a_star, v_b_star, args.delta_A, args.delta_B, args.p_breakdown)

    csv_path = None
    if args.csv_out:
        csv_path = Path(args.csv_out)
        write_csv(
            csv_path,
            one_step_hist,
            two_step_hist,
            v_a_star,
            v_b_star,
            args.delta_A,
            args.delta_B,
            args.p_breakdown,
        )
        print(f"\nCSV written to: {csv_path}")

    if args.plot_out:
        plot_path = Path(args.plot_out)
        if csv_path is None:
            csv_path = plot_path.with_suffix(".csv")
            write_csv(
                csv_path,
                one_step_hist,
                two_step_hist,
                v_a_star,
                v_b_star,
                args.delta_A,
                args.delta_B,
                args.p_breakdown,
            )
            print(f"CSV written to: {csv_path}")
        make_plot_from_csv(csv_path, plot_path)
        print(f"Plot written to: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
