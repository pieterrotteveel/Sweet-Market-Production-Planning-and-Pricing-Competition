#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaned and corrected version.
Major fixes:
- Correct data file paths (use data/ directory).
- Dynamic dimensions (N_I, N_J, N_K) based on loaded data.
- Correct cost sign (use additive costs, not subtractive) in objective and price range floor.
- Fix ingredient usage orientation and budget ingredient term.
- Robust wage/energy parameter extraction (no length assert).
- Guard against division by zero for beta when computing price upper bound.
- Fix budget constraint expression (remove erroneous matrix multiplication in lpSum).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pulp as pl

# ------------------------------------------------------------------
# Data loading (robust paths)
# ------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'

# Demand parameters (alpha, beta)
demand_df = pd.read_csv(DATA / 'instructor_demand_competition.csv')
demand_df['channel'] = demand_df['channel'].str.lower().str.strip()

# Channels master
df_channels = pd.read_csv(DATA / 'channels.csv')
df_channels['channel'] = df_channels['channel'].str.lower().str.strip()
channels = df_channels['channel'].tolist()

# Cakes master
cakes_df = pd.read_csv(DATA / 'cakes.csv')
# Use cake_name from demand file for consistency if present
cakes = demand_df['cake_name'].drop_duplicates().tolist()

# Pivot alpha, beta aligned with cakes × channels
A = (demand_df.pivot_table(index='cake_name', columns='channel', values='alpha', aggfunc='mean')
               .reindex(index=cakes, columns=channels))
B = (demand_df.pivot_table(index='cake_name', columns='channel', values='beta', aggfunc='mean')
               .reindex(index=cakes, columns=channels))
A_arr = A.to_numpy(dtype=float)
B_arr = B.to_numpy(dtype=float)

# Bill of materials (usage per cake per ingredient)
bom_df = pd.read_csv(DATA / 'bill_of_materials.csv')
# Normalize column names for identifier filtering
identifier_tokens = {"cake_id", "cake_name", "cake name"}
# If cake_name present, align by cake_name; else assume ordering matches 'cakes'
if 'cake_name' in bom_df.columns:
    usage_df_raw = bom_df.set_index('cake_name').reindex(cakes)
elif 'cake_id' in bom_df.columns and 'cake_name' in demand_df.columns:
    # Map cake_id to cake_name via cakes_df if possible
    if 'cake_id' in cakes_df.columns and 'cake_name' in cakes_df.columns:
        id_to_name = cakes_df.set_index('cake_id')['cake_name']
        bom_df['__cake_name__'] = bom_df['cake_id'].map(id_to_name)
        usage_df_raw = bom_df.set_index('__cake_name__').reindex(cakes)
    else:
        usage_df_raw = bom_df.iloc[:len(cakes)]
else:
    usage_df_raw = bom_df.iloc[:len(cakes)]
# Keep only numeric ingredient columns (prevents string conversion errors like cake names)
usage_df = usage_df_raw.select_dtypes(include='number').fillna(0)
if usage_df.empty:
    raise ValueError('No numeric ingredient columns detected in bill_of_materials.csv.')
usage = usage_df.to_numpy(dtype=float)  # shape: (N_I, N_K)
# Update ingredient_cols to numeric columns actually used
ingredient_cols = usage_df.columns.tolist()
# Ingredients master (unit cost)
ingred_df = pd.read_csv(DATA / 'ingredients.csv')
unit_cost = ingred_df.set_index('ingredient')['unit_cost_usd']
# Align unit_cost with usage columns
unit_cost_vec = unit_cost.reindex(ingredient_cols).fillna(0).to_numpy(dtype=float)

# Wages & energy
wages_energy_df = pd.read_csv(DATA / 'wages_energy.csv')
we = wages_energy_df.set_index('parameter')['value']
prep_wage = float(we.get('prep_wage_usd_per_hour', 0.0))
oven_wage = float(we.get('oven_wage_usd_per_hour', 0.0))
pack_wage = float(we.get('pack_wage_usd_per_hour', 0.0))
oven_rental_per_hr = float(we.get('oven_rental_usd_per_hour', 0.0))
oven_cost_per_hr = float(we.get('oven_cost_usd_per_hour', 0.0))
budget = float(we.get('budget_usd', 4000.0))

# Transport costs & service caps
transportation_cost = df_channels.set_index('channel')['transport_cost_per_unit_usd'].reindex(channels).to_numpy(dtype=float)
service_cap = df_channels.set_index('channel')['service_cap_per_week'].reindex(channels).to_numpy(dtype=float)

# Cake operational parameters (align with cakes list)
cakes_info = cakes_df.set_index('cake_id') if 'cake_id' in cakes_df.columns else cakes_df.copy()
# Attempt to map cake_name to rows if cake_name present
if 'cake_name' in cakes_df.columns:
    cakes_info = cakes_df.set_index('cake_name').reindex(cakes)

batch_size = cakes_info['batch_size_units'].to_numpy(dtype=float)
oven_minutes = cakes_info['oven_min_per_batch'].to_numpy(dtype=float)
prep_time = cakes_info['prep_min_per_unit'].to_numpy(dtype=float)
packaging_time = cakes_info['pack_min_per_unit'].to_numpy(dtype=float)
packaging_cost = cakes_info['packaging_cost_per_unit_usd'].to_numpy(dtype=float)
minimum_units = cakes_info['minimum_units_if_made'].fillna(0).to_numpy(dtype=float)

# ------------------------------------------------------------------
# Dimensions
# ------------------------------------------------------------------
N_I = len(cakes)
N_J = len(channels)
N_K = usage.shape[1]
Ir = range(N_I)
Jr = range(N_J)
Kr = range(N_K)

# ------------------------------------------------------------------
# Derived per-unit costs
# ------------------------------------------------------------------
cost_ingredients = usage @ unit_cost_vec  # (N_I,)

to_hours = lambda m: m / 60.0
oven_hr_per_unit = to_hours(oven_minutes) / batch_size
prep_labor_cost_per_unit = to_hours(prep_time) * prep_wage
oven_labor_cost_per_unit = oven_hr_per_unit * oven_wage
packaging_labor_cost_per_unit = to_hours(packaging_time) * pack_wage
oven_rental_cost_per_unit = oven_hr_per_unit * oven_rental_per_hr
oven_energy_cost_per_unit = oven_hr_per_unit * oven_cost_per_hr
packaging_materials_cost_per_unit = packaging_cost

labor_cost = prep_labor_cost_per_unit + oven_labor_cost_per_unit + packaging_labor_cost_per_unit
utilities_cost = oven_rental_cost_per_unit + oven_energy_cost_per_unit + packaging_materials_cost_per_unit

# ------------------------------------------------------------------
# Solver with specified price matrix
# ------------------------------------------------------------------

def solve_lp_with_prices(P_mat: np.ndarray):
    """Solve MILP given price matrix P_mat[i,j]."""
    m = pl.LpProblem('Max_Profit', pl.LpMaximize)
    # Decision vars
    Y = pl.LpVariable.dicts('Y', Ir, 0, None, pl.LpInteger)
    b = pl.LpVariable.dicts('b', Ir, 0, None, pl.LpInteger)
    y = pl.LpVariable.dicts('y', (Ir, Jr), 0, None, pl.LpInteger)
    s = pl.LpVariable.dicts('s', (Ir, Jr), 0, None, pl.LpInteger)
    # Ingredient purchase vars
    Q = pl.LpVariable.dicts('Q', Kr, 0, None, pl.LpContinuous)
    # Labor / oven time vars
    H_prep = pl.LpVariable('H_prep', 0, None, pl.LpContinuous)
    H_pack = pl.LpVariable('H_pack', 0, None, pl.LpContinuous)
    T_oven = pl.LpVariable('T_oven', 0, None, pl.LpContinuous)

    # Objective
    # Revenue - production costs (ingredients, labor, utilities, transport) - transport costs on sold units
    # Transport cost applies to sold units (s), other per-unit costs apply to produced units (y)
    cost_per_unit_matrix = np.zeros((N_I, N_J))
    for i in Ir:
        for j in Jr:
            cost_per_unit_matrix[i, j] = (cost_ingredients[i] + labor_cost[i] + utilities_cost[i] + transportation_cost[j])
    m += pl.lpSum(P_mat[i, j] * s[i][j] - cost_per_unit_matrix[i, j] * y[i][j] for i in Ir for j in Jr)

    # Capacity (resource purchase) constraints
    m += pl.lpSum(prep_time[i] * Y[i] for i in Ir) <= H_prep
    m += pl.lpSum(packaging_time[i] * Y[i] for i in Ir) <= H_pack
    m += pl.lpSum(oven_minutes[i] * b[i] for i in Ir) <= T_oven

    # Production linking
    for i in Ir:
        m += Y[i] == batch_size[i] * b[i]
        m += Y[i] == pl.lpSum(y[i][j] for j in Jr)
        if minimum_units[i] > 0:
            m += Y[i] >= minimum_units[i]

    # Ingredient usage ≤ purchased
    for k in Kr:
        m += pl.lpSum(usage[i, k] * Y[i] for i in Ir) <= Q[k]

    # Channel service caps
    for j in Jr:
        m += pl.lpSum(s[i][j] for i in Ir) <= service_cap[j]

    # Demand & sales linking
    for i in Ir:
        for j in Jr:
            demand_ij = max(A_arr[i, j] - B_arr[i, j] * P_mat[i, j], 0.0)
            m += s[i][j] <= demand_ij
            m += s[i][j] <= y[i][j]

    # Budget constraint (cost of purchased resources + ingredients)
    m += ((prep_wage / 60.0) * H_prep + (pack_wage / 60.0) * H_pack + ((oven_wage / 60.0) + (oven_rental_per_hr / 60.0)) * T_oven + pl.lpSum(unit_cost_vec[k] * Q[k] for k in Kr)) <= budget

    # Solve
    m.solve(pl.PULP_CBC_CMD(msg=False))
    res = {
        'status': pl.LpStatus[m.status],
        'objective': pl.value(m.objective),
        'Y': np.array([Y[i].value() for i in Ir]),
        'b': np.array([b[i].value() for i in Ir]),
        's': np.array([[s[i][j].value() for j in Jr] for i in Ir]),
        'y': np.array([[y[i][j].value() for j in Jr] for i in Ir]),
        'Q': np.array([Q[k].value() for k in Kr]),
        'H_prep': H_prep.value(),
        'H_pack': H_pack.value(),
        'T_oven': T_oven.value(),
    }
    return res

# ------------------------------------------------------------------
# Price ranges generation
# ------------------------------------------------------------------
P_ranges = [[None for _ in Jr] for _ in Ir]
for i in Ir:
    for j in Jr:
        # Floor at full unit cost (excluding transport on unsold units) but include transport as conservative floor
        p0 = cost_ingredients[i] + labor_cost[i] + utilities_cost[i] + transportation_cost[j]
        a = A_arr[i, j]
        b = B_arr[i, j]
        if b <= 0 or a <= 0:
            # fallback upper bound
            p1 = p0 * 2.0 + 1.0
        else:
            p1 = a / b
        lo, hi = sorted((max(0.0, p0), max(0.01, p1)))
        if hi <= lo:
            hi = lo + 0.01
        P_ranges[i][j] = [lo, hi]

# ------------------------------------------------------------------
# Sampling scenarios
# ------------------------------------------------------------------

def sample_P_matrix(P_ranges, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    P_mat = np.empty((N_I, N_J), dtype=float)
    for i in Ir:
        for j in Jr:
            lo, hi = P_ranges[i][j]
            if hi < lo:
                lo, hi = hi, lo
            P_mat[i, j] = rng.uniform(lo, hi)
    return P_mat


def run_price_scenarios(P_ranges, runs, seed=None):
    rng = np.random.default_rng(seed)
    best = {'objective': -np.inf, 'status': None, 'prices': None, 'vars': None}
    for _ in range(runs):
        P_mat = sample_P_matrix(P_ranges, rng)
        res = solve_lp_with_prices(P_mat)
        if res['status'] == 'Optimal' and res['objective'] > best['objective']:
            best = {'objective': res['objective'], 'status': res['status'], 'prices': P_mat, 'vars': res}
    return best

# ------------------------------------------------------------------
# Execute scenarios
# ------------------------------------------------------------------
RUNS = 10000
SEED = None
best = run_price_scenarios(P_ranges, runs=RUNS, seed=SEED)
print('\n=== BEST SCENARIO ===')
print('Status:', best['status'])
print('Best profit:', best['objective'])
print('Best price matrix P:\n', best['prices'])
vars_best = best['vars']
print('Y:', vars_best['Y'])
print('b:', vars_best['b'])
print('s:\n', vars_best['s'])
print('y:\n', vars_best['y'])
print('Q:', vars_best['Q'])
print('H_prep:', vars_best['H_prep'], 'H_pack:', vars_best['H_pack'], 'T_oven:', vars_best['T_oven'])



































