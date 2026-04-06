# rubinstein_games

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

---

## Population Rubinstein breakdown game

Added scripts:

- `rubinstein_breakdown_game.py`
- `rubinstein_breakdown_game_animated.py`

### Idea

We simulate a **large population of independent bargaining pairs** with heterogeneous parameters:

\[
\delta_A^{(i)}, \delta_B^{(i)} \sim \mathrm{Uniform}(0.51, 0.99),
\qquad
p^{(i)} \sim \mathrm{Uniform}(p_{\min}, p_{\max}),
\]

\[
V_A^{(i)}(0), V_B^{(i)}(0) \sim \mathrm{Uniform}(0,1).
\]

Each pair evolves under the **two-step same-role Rubinstein-with-breakdown dynamics**:

\[
V_A^{(i)}(k+1)
=
1-(1-p_i)\delta_{B,i}
+
(1-p_i)^2 \delta_{A,i}\delta_{B,i} V_A^{(i)}(k),
\]

\[
V_B^{(i)}(k+1)
=
1-(1-p_i)\delta_{A,i}
+
(1-p_i)^2 \delta_{A,i}\delta_{B,i} V_B^{(i)}(k).
\]

Each pair converges to its own fixed point:

\[
V_A^{(i)*}, \quad V_B^{(i)*}.
\]

---

## Static visualization (2×2 plot)

The base script produces:

1. Initial continuation-value cloud  
2. Intermediate cloud  
3. Final cloud vs equilibrium cloud  
4. Distance-to-equilibrium cloud (log scale)  

### Run

```bash
python rubinstein_breakdown_game.py --num-pairs 1000 --steps 100 --seed 42 --plot-out rb_game.png
```

With CSV:
```bash
python rubinstein_breakdown_game.py --num-pairs 1000 --steps 100 --seed 42 --plot-out rb_game.png --csv-out rb_game.csv
```

---

## Animated visualization (movie of cloud evolution)

This produces a **moving cloud animation (GIF)** showing how the continuation-value cloud evolves over time and converges to equilibrium.

### Run (animation + plot + CSV)

```bash
python rubinstein_breakdown_game_animated.py \
  --num-pairs 1000 \
  --steps 100 \
  --snapshot-step 20 \
  --seed 42 \
  --plot-out rb_game.png \
  --csv-out rb_game.csv \
  --animate-out rb_game.gif
```

### Animation-only run

```bash
python rubinstein_breakdown_game_animated.py \
  --num-pairs 1500 \
  --steps 100 \
  --seed 42 \
  --animate-out rb_game.gif
```

### Controls

- `--animate-out` → enables GIF  
- `--fps` → frames per second  
- `--frame-step` → skip steps to control speed/size  

---

## Notes

- All plots use **continuation values** \((V_A, V_B)\), not actual shares  
- The cloud is genuinely **2D**  
- Final distribution reflects **heterogeneous equilibria**  
- Convergence speed depends on  
  \[
  (1-p)^2 \delta_A \delta_B
  \]


---

## Interacting population game

Added script:

- `rubinstein_breakdown_game_populations.py`

### Idea

We now move from independent pairs to a **genuine population game** via **evolutionary imitation**.

Each epoch:

1. Every pair runs `micro_steps` of Rubinstein-with-breakdown two-step dynamics.
2. Each pair is scored based on outcomes.
3. Bottom-performing pairs imitate top-performing pairs (with noise).

### Scoring

If A proposes first:

\[
A\text{-share} = V_A, \quad B\text{-share} = 1 - V_A
\]

Define:

\[
\text{balance score} = 1 - |A - B|
\]

\[
\text{speed score} = 1 - (1-p)^2 \delta_A \delta_B
\]

\[
\text{total score} = w_{balance} \cdot balance + w_{speed} \cdot speed
\]

### Interaction rule

Learners (bottom fraction) imitate elites (top fraction):

\[
\theta_i \leftarrow (1-\eta)\theta_i + \eta \theta_{elite} + noise
\]

for

\[
\theta \in \{\delta_A, \delta_B, p\}
\]

Continuation values are also partially copied.

### Run

```bash
python rubinstein_breakdown_game_populations.py --num-pairs 1000 --epochs 80 --micro-steps 20 --plot-out rb_pop_game.png --animate-out rb_pop_game.gif --csv-out rb_pop_game.csv
```

Heavier run:

```bash
python rubinstein_breakdown_game_populations.py --num-pairs 3000 --epochs 120 --micro-steps 20 --top-frac 0.2 --bottom-frac 0.2 --imitation-rate 0.5 --mutation-std 0.02 --plot-out rb_pop_game.png --animate-out rb_pop_game.gif --csv-out rb_pop_game.csv
```

### Controls

- `--top-frac`, `--bottom-frac`
- `--imitation-rate`
- `--mutation-std`, `--value-mutation-std`
- `--weight-balance`, `--weight-speed`

### Interpretation

This is an **evolutionary bargaining system**:

- bargaining happens within pairs
- success is evaluated at pair level
- successful strategies spread through imitation

The continuation-value cloud now evolves due to **both contraction AND population interaction**.
