# ILP  and Coarse-to-Fine Price & Production Planner

This project picks prices and builds a production & sales plan to maximize profit.

It uses:

* **Integer Linear Programming (ILP)** to decide how much to make and sell
* A simple **coarse-to-fine (C2F)** search to tune prices

All code is in `project_notebook.ipynb` and uses **PuLP**, **pandas**, and **NumPy**.

---

## What the project does (in plain English)

1. **Estimate demand** for each product in each sales channel based on price
2. **Search over possible prices** with a coarse-to-fine grid (start rough, then zoom in)
3. **Solve an ILP** at each price choice to decide:

   * how many batches to make
   * how many units to send to each channel
   * how many units to sell

The best prices and plans are saved in the `results/` folder.

---

## Quick start

```bash
# 1) Create and activate a virtual environment (Python >= 3.10)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install pulp pandas numpy matplotlib sympy ipython

# 3) Open the notebook
jupyter notebook project_notebook.ipynb
```

---

## Data inputs

Put these CSV files in the `data/` folder (paths are hard-coded in the notebook):

* `ingredients.csv` – ingredients and unit costs
* `bill_of_materials.csv` – how much of each ingredient each product uses
* `cakes.csv` – list of products/SKUs (may also hold batch size & minimum production)
* `channels.csv` – sales channels (e.g. shop, online, wholesale)
* `instructor_demand_competition.csv` – demand parameters per product × channel
* `price_table_template.csv` – starting prices per product × channel
* `wages_energy.csv` – wages, energy and other operating cost parameters

If your column names differ, adjust the data-loading cells in the notebook.

---

## How it works (high level)

### 1. Demand model

For each product and channel, demand **goes down when price goes up**.
The notebook reads the demand parameters from `instructor_demand_competition.csv` and stores them in tables (`alpha_pivot`, `beta_pivot`).
Current prices are stored in `price_pivot`.

You don’t need to touch the formulas unless you want to change how demand is modeled.

---

### 2. Coarse-to-Fine price search

For each product × channel:

1. Start from the current price.
2. Build a **“profit function”** that:

   * plugs in a candidate price
   * solves the ILP
   * returns the resulting total profit
3. Search over a **grid of prices**:

   * first with a big step (e.g. 5.0)
   * then zoom in around the best value with smaller steps (e.g. 1.0, then 0.25)

In short: **try a few prices, keep the best area, zoom in, repeat**.

The final suggested prices are saved as:

* `results/best_prices_corse_to_fine.csv`

  > Note: filename keeps the original “corse” typo for compatibility.

---

### 3. ILP (Integer Linear Program)

For any fixed set of prices, the ILP decides:

* **How many batches** of each product to produce
* **How many units** to send to each channel
* **How many units** can actually be sold

It maximizes **profit = revenue – costs**, where costs can include:

* ingredients
* labor
* packaging
* energy/oven
* transport

Typical decision variables:

* `b_i` – number of batches of product *i* (integer)
* `y_ij` – units of product *i* allocated to channel *j* (integer)
* `s_ij` – units actually sold in channel *j* (integer, limited by demand & allocation)
* `z_i` – binary “we produce this product or not” (for optional minimum production rules)

Key constraints (simplified):

* Production is in whole **batches**
* You can’t **sell more than you produce**
* You can’t **sell more than demand**
* Optional: if a product is produced at all, it may need to meet a **minimum quantity**

You can see and edit this ILP in the notebook if you want to change the logic.

---

## Outputs

After running the notebook, you should see files in `results/`, such as:

* `plan.csv` – final production & sales plan at current/specified prices
* `batches.csv` – batch counts per product
* `costs.csv` – cost breakdown
* `best_prices_corse_to_fine.csv` – best prices found by the search
* `plan_prices_coarse_to_fine.csv` – plan at the best prices
* `batches_prices_coarse_to_fine.csv` – batches at the best prices
* `costs_prices_coarse_to_fine.csv` – cost breakdown at the best prices

---

## Changing settings

* **Price search bounds & steps**

  * Change `compute_price_bounds(...)` and the list of steps passed to the coarse-to-fine search.

* **Batch sizes and minimums**

  * Come from `cakes.csv` (edit the CSV or adjust the column names in the notebook).

* **Cost parameters**

  * Edit `wages_energy.csv` and ingredient prices in `ingredients.csv`.

* **Solver**

  * PuLP will use a default solver.
  * For faster solves, install and configure CBC or HiGHS.

---

## Troubleshooting

* **`ImportError: No module named 'pulp'`**
  → Run `pip install pulp`

* **Model infeasible**
  → Check that demand is non-negative in your price range.
  → Relax minimum production or widen price bounds.

* **Solves are slow**
  → Use smaller Big-M values, add sensible upper bounds, and keep units realistic.

* **Strange or unrealistic prices**
  → Narrow or shift the price bounds; adjust the steps of the coarse-to-fine search.

* **CSV column mismatch errors**
  → Open the data-loading cells and align the column names to your actual CSVs.
