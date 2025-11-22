#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 13:31:22 2025

@author: ritamariaelchaer
"""

import pandas as pd
import numpy as np
import pulp as pl
import random

# LOAD ALL DATA
df_demand = pd.read_csv("data/instructor_demand_competition.csv")
df_bom    = pd.read_csv("data/bill_of_materials (1).csv")
df_chan   = pd.read_csv("data/channels.csv")
df_ing    = pd.read_csv("data/ingredients.csv")
df_cakes  = pd.read_csv("data/cakes.csv")
df_wages  = pd.read_csv("data/wages_energy.csv")


# CLEAN DATA + ALIGN INGREDIENT NAMES WITH BOM COLUMNS
bom_cols = [
    "Banana","Butter","Chocolate","CocoaPowder","CoconutFlakes",
    "CreamCheese","Dairy","Eggs_each","Flavorings","Flour",
    "Leavening","Lemon","Sugar","VegetableOil","Walnuts"
]

# map ingredient → unit cost
cost_map = dict(zip(df_ing["ingredient"], df_ing["unit_cost_usd"]))

# fix naming difference
cost_map["Eggs_each"] = cost_map["Eggs"]

# build ingredient cost vector
c_ing_vec = [cost_map[col] for col in bom_cols]

# ingredient requirements
ing_req = df_bom[bom_cols].astype(float).values  # 10×15

# ingredient cost per unit
ing_cost = ing_req @ np.array(c_ing_vec)


# EXTRACT WAGES / OVEN COSTS
w_prep = df_wages.loc[df_wages['parameter']=="prep_wage_usd_per_hour","value"].iloc[0]
w_oven = df_wages.loc[df_wages['parameter']=="oven_wage_usd_per_hour","value"].iloc[0]
w_pack = df_wages.loc[df_wages['parameter']=="pack_wage_usd_per_hour","value"].iloc[0]
r_oven = df_wages.loc[df_wages['parameter']=="oven_rental_usd_per_hour","value"].iloc[0]
e_oven = df_wages.loc[df_wages['parameter']=="oven_cost_usd_per_hour","value"].iloc[0]
budget = df_wages.loc[df_wages['parameter']=="budget_usd","value"].iloc[0]


# COSTS PER UNIT (no constraints)
t_prep_hour = df_cakes["prep_min_per_unit"]/60 
t_pack_hour = df_cakes["pack_min_per_unit"]/60 
c_pack_mat = df_cakes["packaging_cost_per_unit_usd"]
t_oven_hour = df_cakes["oven_min_per_batch"]/60
minp = df_cakes["minimum_units_if_made"]
S_batch = df_cakes ["batch_size_units"]


pack_and_prep_cost_per_cake = t_prep_hour * w_prep + t_pack_hour * w_pack + c_pack_mat 
ing_cost_per_cake = ing_cost
oven_cost_per_batch = t_oven_hour * (w_oven + r_oven + e_oven)

#total_cost_excl_trans = ing_cost + prep_cost + pack_labor_cost + oven_cost + pack_mat_cost

# transport cost per channel
c_trans = df_chan["transport_cost_per_unit_usd"].values  # [local, supermarket, online]
cap_service = df_chan["service_cap_per_week"].values

# DEMAND PARAMETERS A, B
df_demand["channel"] = df_demand["channel"].str.strip().str.lower()
channels_order = ["local","supermarket","online"]

A_df = df_demand.pivot_table(index="ID", columns="channel", values="alpha")
B_df = df_demand.pivot_table(index="ID", columns="channel", values="beta")

cake_ids = df_cakes["cake_id"].tolist()

A = A_df.reindex(index=cake_ids, columns=channels_order).values
B = B_df.reindex(index=cake_ids, columns=channels_order).values

N_I = 10
N_J = 3
N_K = 15
I = range(N_I)
J = range(N_J)
K = range(N_K)



# ================================
def solve_LP_with_prices (P_mat):
 
    # demand = alpha – beta * price
    D = np.maximum(A - B * P_mat, 0)


# ============================
# BUILD BASIC ILP (NO LOGIC)
# ============================
    m = pl.LpProblem("MaxProfit_NoLogic", pl.LpMaximize)

    # decision variables: sales only
    s = pl.LpVariable.dicts("s", (I,J), lowBound=0, cat="Integer")
    y = pl.LpVariable.dicts("y", (I,J), lowBound=0, cat="Integer")
    Ysum = pl.LpVariable.dicts("Y", I, lowBound=0, cat="Integer")
    b = pl.LpVariable.dicts("b", I, lowBound=0, cat="Integer")
    
    zA = pl.LpVariable.dicts("zA", I, lowBound=0, upBound=1, cat="Binary")
    
    Cap_prep = pl.LpVariable(name='Cap_prep', lowBound=0, cat='Continuous')
    Cap_pack = pl.LpVariable(name='Cap_pack', lowBound=0, cat='Continuous')
    Cap_oven = pl.LpVariable(name='Cap_oven', lowBound=0, cat='Continuous')
    Cap_ing  = pl.LpVariable.dicts(name='Cap_per_ing', indices=K, lowBound=0, cat='Continuous')
    
    #binary variables for the new logical constraints 
    zB = pl.LpVariable.dicts("zB", (I,J), lowBound=0, upBound=1, cat="Binary")

    


    # objective
    m += pl.lpSum(P_mat[i][j] * s[i][j] 
   - (pack_and_prep_cost_per_cake[i] + ing_cost_per_cake[i]) * y[i][j] 
   - oven_cost_per_batch[i] * b[i] 
   - c_trans[j] * y[i][j] for i in I for j in J), "Total Profit"

    # constraints: 
    #1) Cap constraints 
    for i in I:
        m += Ysum[i] == pl.lpSum(y[i][j] for j in J), f"def_Y_{i}"

    m += pl.lpSum(t_prep_hour[i] * Ysum[i] for i in I) <= Cap_prep, "Prep_Capacity"
    m += pl.lpSum(t_pack_hour[i] * Ysum[i] for i in I) <= Cap_pack, "Pack_Capacity"
    m += pl.lpSum(t_oven_hour[i] * b[i]     for i in I) <= Cap_oven, "Oven_Capacity"
    
    #2) production batch chanel linking:
    for i in I:
         m += pl.lpSum(y[i][j] for j in J) <= S_batch[i] * b[i], f"BatchCap_{i}"
         
    M_1 = int(sum (cap_service))
   
    for i in I:
        m += pl.lpSum(y[i][j] for j in J) <= M_1 * zA[i], f"MinProd_up_{i}"
        m += pl.lpSum(y[i][j] for j in J) >= minp[i] - M_1 * (1-zA[i]), f"MinProd_low_{i}" #ou 1-z?
        
    #3)Ingredients constraint:
    for k in K:
        m += pl.lpSum(ing_req[i][k] * Ysum[i] for i in I) <= Cap_ing[k], f"IngrCap_{k}"
        
    #4)Channel Capacity:
    for j in J:
        m += pl.lpSum(s[i][j] for i in I) <= cap_service[j], f"Cap_{j}"
        
        
    #5)Demand and Sales Constraint:
    for i in I:
        for j in J:
            m += s[i][j] <= D[i][j]
            m += s[i][j] <= y[i][j]
            
            
    #6)Budget Constraint:
    m += (w_prep * Cap_prep
                + w_pack * Cap_pack
                + (w_oven + r_oven + e_oven) * Cap_oven
                + pl.lpSum(c_ing_vec[k] * Cap_ing[k] for k in K)
                + pl.lpSum(y[i][j] * c_pack_mat[i] for i in I for j in J)
            ) <= budget, "Budget"
    
    
    # additional logical consraints:
    # at least two channels per cake type 
    for i in I: 
        m += pl.lpSum(zB[i][j] for j in J) >= 2
    
    M_3 = int(max(cap_service))
    for i in I:
        for j in J:
            m += s[i][j] <= M_3 * zB[i][j]
            
            
    # supply treshold activation: 
    for i in I:
        m += s[i][1] >= 15 * zB[i][1]
        
    #M_5 = int(max(cap_service))
   # for i in I:
       # m += s[i][1] <= M_5 * zC[i][1] --> they are redundant with the constraint above (mentioned in report)

     



    # SOLVE
    status = m.solve(pl.PULP_CBC_CMD(msg=False))
    status_str = pl.LpStatus[status]
    
    # objective value (might be None if infeasible/unbounded)
    obj_val = pl.value(m.objective)
    
    # you can keep these prints if you want
    print("Solver status:", status_str)
    print("Profit =", obj_val)
    
    # Return a small result dict (no need to print all variables here)
    return {
        "status": status_str,
        "objective": obj_val,
        "model": m
}

    
    
    
rows, cols = 10, 3  # better: rows, cols = N_I, N_J
Plow = [[0.0 for _ in range(cols)] for _ in range(rows)]

def f(i, j):
    return (ing_cost_per_cake [i] + pack_and_prep_cost_per_cake[i] + c_trans[j] + (oven_cost_per_batch[i]/S_batch[i]))
      

for i in range(rows):
    for j in range(cols):
        Plow[i][j] = f(i, j)


        
## P high
rows, cols = 10, 3
Phigh = [[0.0 for _ in range(cols)] for _ in range(rows)]

def f(i, j):
    return (A[i][j] / B[i][j]) 

for i in range(rows):
    for j in range(cols):
        Phigh[i][j] = f(i, j)



# ---------------------------------------------------
# 3) Scenario sampling and LP solve
# ---------------------------------------------------



def P_mat_generator(lo, hi):
    rows, cols = 10, 3  # same as N_I, N_J
    P_mat = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            a = float(lo[i][j])
            b = float(hi[i][j])
            low, high = (a, b) if a <= b else (b, a)
            P_mat[i][j] = round(random.uniform(low, high), 2)
    return P_mat




#LOOOOOP:
# =========================================================
# RANDOM SEARCH OVER PRICE MATRICES TO FIND BEST PROFIT
# =========================================================

N_SAMPLES = 105000 # number of random price matrices to test

best_obj = float("-inf")
best_P = None
best_res = None

for t in range(N_SAMPLES):
    # sample a candidate price matrix within [Plow, Phigh]
    P_try = np.array(P_mat_generator(Plow, Phigh), dtype=float)

    # solve the ILP with these prices
    res = solve_LP_with_prices(P_try)

    # keep only truly optimal solves with a valid objective
    if res and res.get("status") == "Optimal" and res.get("objective") is not None:
        obj = float(res["objective"])
        if obj > best_obj:
            best_obj = obj
            best_P = P_try.copy()
            best_res = res
            print(f"[{t+1}/{N_SAMPLES}] New best objective: {best_obj:.2f}")

if best_P is not None:
    
    print('Lower Bound of Prices: ', Plow)
    print('Higher Bound of Prices: ', Phigh)

    cakes = df_cakes["cake_id"].tolist()
    channels = channels_order  

    best_P_df = pd.DataFrame(best_P, index=cakes, columns=channels)
    best_P_df.to_csv("best_price_matrix.csv", index=True)

    print("\nBest price matrix:")
    print(best_P_df)

    print("\n========== BEST RESULT ==========")
    print(f"Best objective (profit): {best_obj:.2f}")

    print("\nDecision variables for BEST solution:")
    for v in best_res["model"].variables():
        if v.varValue not in (0, None):
            print(v.name, "=", v.varValue)

else:
    print("\nNo optimal solution found in the sampled price matrices.")
