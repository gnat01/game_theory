# rubinstein_games

Yes: `rubinstein_games` is a sensible directory name for this little project.

This script does four things in one place:

1. Computes the exact Rubinstein fixed point.
2. Evolves the one-step alternating map.
3. Evolves the two-step same-role contraction.
4. Writes CSV and visualizes that CSV using matplotlib.

## Exact formulas

\[
V_A^*=\frac{1-\delta_B}{1-\delta_A\delta_B}, \qquad
V_B^*=\frac{1-\delta_A}{1-\delta_A\delta_B}.
\]

If A moves first, A's equilibrium share is

\[
x_A^* = V_A^*,
\]

and B gets

\[
1-x_A^* = \delta_B V_B^*.
\]

## Run

Basic:
```bash
python rubinstein_dynamics_full.py --delta_A 0.9 --delta_B 0.8
```

Write CSV:
```bash
python rubinstein_dynamics_full.py --delta_A 0.95 --delta_B 0.85 --V_A0 0.3 --V_B0 0.7 --steps 30 --csv-out history.csv
```

Write CSV and plot:
```bash
python rubinstein_dynamics_full.py --delta_A 0.95 --delta_B 0.85 --V_A0 0.3 --V_B0 0.7 --steps 30 --csv-out history.csv --plot-out convergence.png
```

If you pass `--plot-out` but not `--csv-out`, the script auto-generates a CSV alongside the plot.


---

## Breakdown-risk extension

Added script:

- `rubinstein_dynamics_breakdown_risk.py`

### Model

After every rejection:
- with probability `p_breakdown`, bargaining ends and both players get 0
- with probability `1 - p_breakdown`, bargaining continues

The continuation-value recursions become:

\[
V_A = 1 - (1-p)\delta_B V_B,
\qquad
V_B = 1 - (1-p)\delta_A V_A.
\]

Exact fixed point:

\[
V_A^*=\frac{1-(1-p)\delta_B}{1-(1-p)^2\delta_A\delta_B},
\qquad
V_B^*=\frac{1-(1-p)\delta_A}{1-(1-p)^2\delta_A\delta_B}.
\]

If A moves first, the accepted split is:

\[
x_A^* = V_A^*,
\qquad
1-x_A^* = (1-p)\delta_B V_B^*.
\]

### What the script does

- computes exact values
- evolves the one-step alternating map
- evolves the two-step same-role contraction
- prints tables and final errors
- writes CSV
- reads the CSV back in
- plots convergence with matplotlib

### Run

Basic:
```bash
python rubinstein_dynamics_breakdown_risk.py --delta_A 0.9 --delta_B 0.8 --p_breakdown 0.2
```

With CSV and plot:
```bash
python rubinstein_dynamics_breakdown_risk.py --delta_A 0.95 --delta_B 0.85 --p_breakdown 0.15 --V_A0 0.3 --V_B0 0.7 --steps 30 --csv-out breakdown_history.csv --plot-out breakdown_plot.png
```

If you pass `--plot-out` without `--csv-out`, the script auto-generates the CSV and then plots it.
