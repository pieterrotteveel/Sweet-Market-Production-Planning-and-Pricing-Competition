# ILP + Coarse-to-Fine Price & Production Planner

This project combines **integer linear programming (ILP)** with a **coarse-to-fine (C2F) line search** to pick prices and build a profit‑maximizing production & sales plan. It is implemented in `project_notebook.ipynb` using PuLP, pandas, and NumPy.

> Summary: (1) model demand per product and channel as linear in price, (2) search prices with a simple but efficient coarse-to-fine routine, while (3) solving an ILP that chooses batches and allocations subject to discrete constraints. Results are written to `results/`.

---

## Contents
- [Quick start](#quick-start)
- [Data inputs](#data-inputs)
- [How it works](#how-it-works)
  - [Demand model](#demand-model)
  - [Coarse-to-Fine price search](#coarse-to-fine-price-search)
  - [ILP formulation](#ilp-formulation)
- [Mathematics behind ILP & C2F](#mathematics-behind-ilp--c2f)
  - [ILP: variables, objective, constraints](#ilp-variables-objective-constraints)
  - [Demand curve and revenue shape](#demand-curve-and-revenue-shape)
  - [Why coarse-to-fine works here](#why-coarse-to-fine-works-here)
- [Code examples](#code-examples)
  - [Minimal ILP in PuLP](#minimal-ilp-in-pulp)
  - [Coarse-to-Fine price search](#coarse-to-fine-price-search-code)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
# 1) Create an environment (Python >= 3.10 recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install pulp pandas numpy matplotlib sympy ipython

# 3) Open the notebook
jupyter notebook project_notebook.ipynb
```

---

## Data inputs
Place the following CSV files under `data/` (the notebook expects these paths):

- `ingredients.csv` – catalog of ingredients and unit costs
- `bill_of_materials.csv` – per‑product recipe/ingredient usage
- `cakes.csv` – list of SKUs/products (also used for batch sizes and minimum production quantities if provided)
- `channels.csv` – list of sales channels/markets
- `instructor_demand_competition.csv` – demand parameters per product × channel
- `price_table_template.csv` – initial/seed prices per product × channel
- `wages_energy.csv` – wage rates and oven/energy or other operating cost parameters

> Column names may vary slightly; the notebook shows how these are merged. Adjust the data‑loading cells if needed.

---

## How it works

### Demand model
We assume a **linear demand** curve per product `i` and channel `j`:

```
Demand_ij(p) = max(0, alpha_ij - beta_ij * p)
```

- `alpha_ij` (intercept) and `beta_ij` (slope) come from `instructor_demand_competition.csv` and are arranged into `alpha_pivot`, `beta_pivot` tables.
- Prices are stored in `price_pivot` (initialized from `price_table_template.csv`).

### Coarse-to-Fine price search
We optimize prices by iterating over each product × channel and performing a **1‑D coarse‑to‑fine line search** around the current price while holding other prices fixed:

1) **Bounds:** `compute_price_bounds(i, j, cur)` sets a reasonable low/high range around the current price `cur`.
2) **Profit oracle:** `make_profit_fn_for_pair(i, j, best_prices_df)` builds a function `f(p)` that plugs price `p` for item `(i,j)` into the model and solves the ILP to return the resulting total profit.
3) **Search:** `coarse_to_fine_maximize(f, lo, hi, steps)` evaluates `f` on a grid with progressively smaller step sizes (e.g., `5.0`, then `1.0`, then `0.25`). At each stage it keeps the best point and shrinks the bracket around it.

Pseudocode (simplified):
```python
x_best, f_best = lo, f(lo)
for step in [5.0, 1.0, 0.25]:
    xs = np.arange(lo, hi + 1e-9, step)
    vals = [f(x) for x in xs]
    k = int(np.argmax(vals))
    x_best, f_best = xs[k], vals[k]
    lo, hi = max(lo, x_best - step), min(hi, x_best + step)
return x_best, f_best
```
The notebook writes the suggested prices to `results/best_prices_corse_to_fine.csv` (historical misspelling `corse`).

### ILP formulation
For any fixed price table, we solve an **integer linear program** to choose batches and allocate sales to channels to **maximize profit**.

**Decision variables (typical):**
- `b_i` (integer >= 0): number of production batches for product `i`.
- `y_ij` (integer >= 0): units of product `i` produced/allocated to channel `j`.
- `s_ij` (integer >= 0): units of product `i` sold in channel `j` (cannot exceed both demand and allocation).
- `z_i` (binary 0/1): “make” indicator for product `i` (used with Big‑M to enforce minimums only if we produce the item).

**Objective (maximize):**
```
Total_Profit = sum_{i,j} price_ij * s_ij
               - (ingredient + labor + packaging + oven/energy + transport costs)
```

**Key constraints (illustrative):**
- **Batching:** `sum_j y_ij = batch_size_i * b_i` (for each product `i`).
- **Sales <= allocation:** `s_ij <= y_ij` for all `(i,j)`.
- **Demand cap (linear demand):** `s_ij <= max(0, alpha_ij - beta_ij * price_ij)`.
- **Optional minimums via Big‑M:** if `min_prod[i] > 0` with binary `z_i` and large bound `M_i`:
  - `sum_j y_ij >= min_prod_i * z_i`
  - `sum_j y_ij <= M_i * z_i`

---

## Mathematics behind ILP & C2F

### ILP: variables, objective, constraints
An ILP chooses **integer** decisions to optimize a **linear** objective under **linear** constraints. In compact form:

```
maximize   c^T x
subject to A x <= b
           x_k integer (some or all)
```

Solvers use LP relaxations to get an **upper bound**, then branch on integer variables and prune subproblems (branch‑and‑bound, often with cutting planes). Keeping Big‑M values tight and adding realistic bounds helps a lot.

### Demand curve and revenue shape
For a single `(i,j)` pair with a fixed allocation `y_ij`:

```
Demand_ij(p) = max(0, alpha_ij - beta_ij * p)
Sold_ij(p)   = min( y_ij, Demand_ij(p) )
Revenue_ij(p)= p * Sold_ij(p)
```
- If `Demand_ij(p) >= y_ij`, then `Revenue_ij(p) = p * y_ij` (linear in price).
- If demand binds, `Revenue_ij(p) = p*(alpha_ij - beta_ij*p) = alpha_ij*p - beta_ij*p^2` (concave quadratic).

So each pair is **linear then concave** (piecewise‑concave). Summing pairs preserves that shape. When the ILP re‑optimizes allocations and batches at each price, the per‑coordinate profit curve is typically **unimodal** in practice.

### Why coarse-to-fine works here
Let `f(p)` be the **profit returned by the ILP** when the current coordinate’s price is `p` and all other prices are fixed. Because each coordinate’s curve tends to be unimodal, we can sample on a coarse grid, keep the best bracket, then refine the step. This derivative‑free approach is robust to non‑smoothness from integrality.

Rule of thumb: with grid step `delta` on `[a,b]`, your best sample is within about `delta` of the true maximizer. Shrinking `delta` geometrically (e.g., `5 -> 1 -> 0.25`) converges quickly with few ILP solves.

---

## Code examples
Below are minimal, copy‑pasteable snippets that mirror the notebook structure (toy data so they run instantly).

### Minimal ILP in PuLP
```python
import pulp as pl
import numpy as np

# --- Toy data ---
products = ["cake"]
channels = ["shop", "online"]

alpha = {("cake","shop"): 120, ("cake","online"): 90}
beta  = {("cake","shop"): 1.2,  ("cake","online"): 1.0}
price = {("cake","shop"): 30.0, ("cake","online"): 28.0}

batch_size = {"cake": 40}
min_prod   = {"cake": 0}
unit_cost  = {"cake": 12.0}   # aggregate per unit cost

pairs = [(i,j) for i in products for j in channels]

# --- ILP builder/solver ---
def solve_ilp(prices):
    prob = pl.LpProblem("Planner", pl.LpMaximize)

    # Decision vars
    b = {i: pl.LpVariable(f"b_{i}", lowBound=0, cat="Integer") for i in products}
    z = {i: pl.LpVariable(f"z_{i}", lowBound=0, upBound=1, cat="Binary") for i in products}
    y = {ij: pl.LpVariable(f"y_{ij[0]}_{ij[1]}", lowBound=0, cat="Integer") for ij in pairs}
    s = {ij: pl.LpVariable(f"s_{ij[0]}_{ij[1]}", lowBound=0, cat="Integer") for ij in pairs}

    # Objective: revenue - simple cost (per-unit cost × units produced)
    revenue = pl.lpSum(prices[ij] * s[ij] for ij in pairs)
    prod_units = {i: pl.lpSum(y[i,j] for j in channels) for i in products}
    cost = pl.lpSum(unit_cost[i] * prod_units[i] for i in products)
    prob += revenue - cost

    # Batching
    for i in products:
        prob += prod_units[i] == batch_size[i] * b[i]

    # Sales <= allocation
    for i in products:
        for j in channels:
            prob += s[i,j] <= y[i,j]

    # Demand caps at current prices
    for i in products:
        for j in channels:
            cap = max(0, alpha[i,j] - beta[i,j] * prices[i,j])
            prob += s[i,j] <= cap

    # Optional minimums via Big-M
    for i in products:
        M = 10_000  # tighten if you know a better bound
        prob += prod_units[i] >= min_prod.get(i, 0) * z[i]
        prob += prod_units[i] <= M * z[i]

    prob.solve(pl.PULP_CBC_CMD(msg=False))
    return pl.value(prob.objective)

# Run once
profit_now = solve_ilp(price)
print(f"Profit at current prices: {profit_now:.2f}")
```

### Coarse-to-Fine price search (code)
```python
import numpy as np

# 1-D coarse-to-fine maximizer
def coarse_to_fine_maximize(f, lo, hi, steps=(5.0, 1.0, 0.25)):
    x_best = lo
    f_best = f(x_best)
    for step in steps:
        xs = np.arange(lo, hi + 1e-9, step)
        vals = [f(x) for x in xs]
        k = int(np.argmax(vals))
        x_best, f_best = xs[k], vals[k]
        lo, hi = max(lo, x_best - step), min(hi, x_best + step)
    return x_best, f_best

# Profit oracle that tweaks a single price and resolves the ILP
def profit_with_price_change(item, channel, base_prices):
    def f(p):
        prices = dict(base_prices)
        prices[(item, channel)] = float(p)
        return solve_ilp(prices)
    return f

# Example: optimize shop price while holding others fixed
f = profit_with_price_change("cake", "shop", price)
best_p, best_val = coarse_to_fine_maximize(f, lo=20.0, hi=45.0, steps=(5.0, 1.0, 0.25))
print(f"Best shop price ~= {best_p:.2f}, estimated profit = {best_val:.2f}")

# Coordinate-wise sweep (small demo)
for _ in range(2):  # a couple of passes is usually enough
    for (i,j) in pairs:
        f = profit_with_price_change(i, j, price)
        p_star, _ = coarse_to_fine_maximize(f, lo=20.0, hi=45.0)
        price[(i,j)] = p_star
```

> These snippets are intentionally small. The notebook generalizes the same pattern across many products, channels, and richer cost structures.

---

## Outputs
After running the notebook, you should see CSVs under `results/` such as:

- `plan.csv` – production & sales plan at the current prices
- `batches.csv` – batch counts per product
- `costs.csv` – cost breakdown at the current prices
- `best_prices_corse_to_fine.csv` – suggested per‑channel prices from C2F search
- `plan_prices_coarse_to_fine.csv` – plan at the suggested prices
- `batches_prices_coarse_to_fine.csv` – batches at the suggested prices
- `costs_prices_coarse_to_fine.csv` – cost breakdown at the suggested prices

---

## Configuration
- **Bounds & steps** for the price search: controlled in the cells defining `compute_price_bounds(...)` and the `steps` list
- **Batch sizes / minimums**: read from `cakes.csv` (adjust column names if yours differ)
- **Cost parameters**: edit `wages_energy.csv` and ingredient prices in `ingredients.csv`
- **Solver**: PuLP will pick an available solver; you can install CBC/HiGHS for performance

---

## Troubleshooting
- `ImportError: pulp not found` -> `pip install pulp`
- **Infeasible model** -> check that demand is non‑negative at your price bounds; relax minimums or widen bounds
- **Slow solve** -> tighten Big‑M, add upper bounds to variables, and make sure units and costs are in realistic ranges
- **Weird prices** -> adjust `steps` or bounds; coarse‑to‑fine is robust, but bad bounds can trap local plateaus
- **CSV column mismatch** -> open the data‑loading cells and align column names to your files

---
