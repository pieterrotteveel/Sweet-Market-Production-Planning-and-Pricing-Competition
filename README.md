# Sweet Market: Production Planning and Pricing (INDE 302)

A competition where each team runs a small artisan cake shop producing and selling ten cake types across three channels. Your goal is to choose investments, set prices, and plan production to maximize profit.

- Due date: November 20, 2025
- Deliverables: model formulations, Python solutions, analysis, and comparison (details below)

---

## Repository Structure
- Final Project Description.pdf — Full problem statement
- project_notebook.ipynb — Starter notebook for data, modeling, and analysis
- data/
  - bill_of_materials.csv — Ingredient usage per unit of cake (BOM)
  - cakes.csv — Batch size, oven time per batch, prep/pack minutes per unit, packaging cost, minimum units
  - channels.csv — Channel transport cost per unit and service capacity per week
  - ingredients.csv — Ingredient unit and unit cost
  - instructor_demand_competition.csv — Demand parameters α and β for cake–channel pairs
  - wages_energy.csv — Wages, oven rental/cost, project budget

---

## Business Context
Three sales channels with distinct costs and capacities:
- Local Shop: higher prices, moderate demand
- Supermarket: lower prices, higher volume
- Online Local: mid-range prices, delivery/service fees

Production stages and costs:
- Preparation (labor, minutes per unit)
- Baking (oven batches, minutes per batch; oven rental and energy cost)
- Decoration (if applicable in prep time)
- Packaging (materials + labor minutes per unit)

---

## Project Stages

### Stage 1 — Initial Investment (Budget: $4,000)
Decide weekly capacities subject to the budget:
- Raw materials (ingredients inventory)
- Labor hours for preparation/decoration and packaging
- Oven time (baking capacity in minutes per week)

Budget constraint ensures total investment ≤ $4,000.

### Stage 2 — Pricing and Demand
Demand for cake i in channel j is:

D_ij = α_ij − β_ij · P_ij

where α_ij, β_ij come from instructor_demand_competition.csv and P_ij is the price you set. Typically, Supermarket requires lower prices to stimulate volume.

### Stage 3 — ILP Formulation (Basic)
Formulate an ILP for the weekly production plan. To keep linearity, treat prices P_ij as inputs (try several price sets), then solve the ILP to get the optimal plan for each price set.

Decision variables (weekly):
- y_ij ≥ 0 integer: units of cake i produced for channel j
- s_ij ≥ 0 integer: units of cake i sold in channel j
- b_i ≥ 0 integer: number of batches produced for cake i

Parameters:
- P_ij: chosen price for cake i, channel j (treated as input in the basic model)
- D_ij = α_ij − β_ij P_ij: demand at chosen price

Objective (maximize profit):

max Π = Σ_{i,j} P_ij · s_ij − C_ingredients − C_labor − C_utilities − C_transportation

Key costs (from data):
- Ingredients via BOM × ingredient unit cost
- Labor: prep/pack minutes × wage rates
- Oven: rental + energy per hour × oven minutes used
- Transport: per-unit transport cost by channel
- Packaging: per-unit packaging material cost

Core constraints:
- Sales vs demand and production (min condition via inequalities):
  - s_ij ≤ D_ij
  - s_ij ≤ y_ij
  - s_ij ≥ 0
- Channel service capacity (channels.csv): Σ_i s_ij ≤ service_cap_j
- Batch production: y_i• = Σ_j y_ij = b_i × batch_size_i and b_i integer
- Minimum units if produced: if b_i ≥ 1 then y_i• ≥ minimum_units_if_made_i (use big-M with a binary if needed; can be deferred to Stage 4)
- Stage capacities (from your investment decisions):
  - Prep minutes: Σ_i (prep_min_per_unit_i × y_i•) ≤ prep_minutes_capacity
  - Oven minutes: Σ_i (oven_min_per_batch_i × b_i) ≤ oven_minutes_capacity
  - Pack minutes: Σ_i (pack_min_per_unit_i × y_i•) ≤ pack_minutes_capacity
- Ingredient limits: Σ_i (usage_{i,k} × y_i•) ≤ inventory_k for each ingredient k
- Non-negativity and integrality for y_ij, s_ij, b_i

Note: If you instead keep P_ij as variables, P_ij·s_ij is bilinear (nonlinear). The basic phase treats P_ij as inputs to preserve linearity.

### Stage 4 — Logical Constraints (Full)
1) At least two channels per cake type:
- Introduce z_ij ∈ {0,1} indicating whether cake i is sold in channel j
- Link sales to activation, e.g., s_ij ≤ U_ij · z_ij (choose a valid U_ij upper bound)
- Enforce Σ_j z_ij ≥ 2 for each cake i

2) Supermarket supply threshold:
- For each cake i in Supermarket channel J=SM, either 0 units or at least 15 units
- Use big-M: 15 · z_i,SM ≤ s_i,SM ≤ U_i,SM · z_i,SM with z_i,SM ∈ {0,1}

### Stage 5 — Report and Deliverables
1. Basic Formulation: ILP without logical constraints (variables, objective, constraints)
2. Basic Solution: Python solution for several price sets; choose best plan and explain price search strategy
3. Analysis of Outputs: Cost and revenue breakdowns by cake and channel (tables/plots)
4. Full Formulation: ILP including logical constraints
5. Full Solution: Python solution for the full model; report good prices and plan
6. Comparison and Conclusion: Compare basic vs full models and insights

### Competition and Evaluation
- Correct, consistent formulation
- Sound investment and pricing strategy
- Quality of analysis and discussion
- Final achieved profit (leaderboard)

---

## Data Files (details)
| File | Description |
| --- | --- |
| data/bill_of_materials.csv | Ingredient quantities per cake unit (columns match ingredient names) |
| data/ingredients.csv | Ingredient units and unit_cost_usd |
| data/cakes.csv | Batch sizes; oven minutes per batch; prep/pack minutes per unit; packaging_cost_per_unit_usd; minimum_units_if_made |
| data/channels.csv | channel, transport_cost_per_unit_usd, service_cap_per_week |
| data/instructor_demand_competition.csv | α and β parameters per cake–channel pair (linear demand) |
| data/wages_energy.csv | Wage rates, oven rental/cost per hour, and overall budget_usd |

---

## Getting Started
1) Open project_notebook.ipynb and run the first two sections to load data and view shapes
2) Build a cost model (BOM + ingredients + wages/oven + packaging + transport)
3) Choose several price sets P_ij and compute D_ij = α_ij − β_ij P_ij
4) Solve the basic ILP for each price set and select the best profit plan
5) Add the logical constraints and repeat

Tip: Keep a reproducible workflow (clear sections, fixed random seeds, and a requirements list if needed).

---

## Modeling Tips
- Separate variable vs. fixed costs and document assumptions
- Use sensible upper bounds U_ij for big-M constraints (e.g., channel capacity or demand at minimum feasible price)
- Validate units and time bases (minutes vs hours) and conversions for costs
- Perform sensitivity analyses on ingredient costs, capacities, and price bounds

---

## References
- See Final Project Description.pdf for full narrative and requirements
- All datasets are in the data/ directory