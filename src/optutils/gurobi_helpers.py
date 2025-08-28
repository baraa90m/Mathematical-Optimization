from typing import Iterable, Mapping, Hashable, Optional
import gurobipy as gp
from gurobipy import GRB, quicksum
from itertools import product
from gurobipy import multidict, tuplelist



def delta(t, q):
    """
    define discount factor for the profit made in time period t
    Args:
        t: time period
        q: constant interest rate
    """
    return (1/(1 + q) ** t)


def define_variables(
        model,
        blocks: Iterable[Hashable],
        periods: Iterable[Hashable],
        plants: Iterable[Hashable] = (),
        stockpiles: Iterable[Hashable] = (),
        waste_dumps: Iterable[Hashable] = (),
        x_binary: bool = True,
        aggregate = False
):
    """
    Create decision variables for the Natural Formulation (NF) with stockpiles.

    Args:
        model: Gurobi model.
        blocks, periods, plants, stockpiles, waste_dumps: index sets.
        x_binary: if False, x is continuous in [0,1] (useful for LP relaxations).
        aggregate: if True, create AT-specific variables.


    Base variables (always created):
        x[i,t]   (binary)    — 1 if block i is mined by end of period t (cumulative).
        y[i,t]   ∈ [0,1]     — fraction of block i processed in period t.
        z_p[i,t,p] ∈ [0,1]   — fraction of block i sent directly to plant p in period t.
        z_s[i,t,s] ∈ [0,1]   — fraction of block i sent to stockpile s in period t.
        oreSP[p,s,t]   ≥ 0   — ore quantity sent from stockpile s to plant p in period t.
        oreS_rem[s,t]  ≥ 0   — ore quantity remaining in stockpile s during period t.
        metalSP[p,s,t] ≥ 0   — metal quantity sent from stockpile s to plant p in period t.
        metalS_rem[s,t]≥ 0   — metal quantity remaining in stockpile s during period t.

    AT-specific variables (only when aggregate=True):
        z_sp[i,t,s,p]     ∈ [0,1]            — fraction of block i routed (via s) to plant p in t.
        z_ss[i,t,s]       ∈ [0,1]            — fraction of block i remaining in stockpile s in t.
        f_t[t]            ∈ [0,1]            — out-fraction for period t.
    Notes:
        • z_* variables are fractions per block; ore*/metal* are absolute quantities (e.g., tonnes).
        • Return order matches the tuple below—keep the caller’s unpacking consistent.

    Returns:
        if aggregate=True  (AT order):
            (x, y, z_p, z_s, oreSP, oreS_rem, metalSP, metalS_rem, z_ss, z_sp, f_t)
        else:
            (x, y, z_p, z_s, oreSP, oreS_rem, metalSP, metalS_rem)
    """

    # --- Helper ---
    global block_periods_stockpiles_plants
    block_periods = tuplelist(product(blocks, periods))
    block_periods_plants = tuplelist(product(blocks, periods, plants))
    block_periods_stockpiles = tuplelist(product(blocks, periods, stockpiles))
    plants_stockpiles_periods = tuplelist(product(plants, stockpiles, periods))
    stockpile_periods = tuplelist(product(stockpiles, periods))

    # Only needed if aggregate=True
    if aggregate:
        block_periods_stockpiles_plants = tuplelist(product(blocks, periods, stockpiles, plants))

    # Decision variables
    # x[i,t] = 1 if block i is mined by (end of) period t  (cumulative)
    if x_binary:
        x = model.addVars(block_periods, vtype=GRB.BINARY, name="x")
    else:
        x = model.addVars(block_periods, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")

    # y[i,t] ∈ [0,1], fraction of block i processed in period t
    y = model.addVars(block_periods, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")

    # z[i,t,p] ∈ [0,1], fraction of block i sent directly for processing in period t
    z_p = model.addVars(block_periods_plants, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z_p")

    # z[i,t,s] ∈ [0,1], fraction of block i sent to stockpile in period t
    z_s = model.addVars(block_periods_stockpiles, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z_s")

    # ore from stockpile --> plant
    oreSP = model.addVars(plants_stockpiles_periods, vtype=GRB.CONTINUOUS, lb=0, name="oreSP")

    # ore remaining in stockpile
    oreS_rem = model.addVars(stockpile_periods, vtype=GRB.CONTINUOUS, lb=0, name="oreS_rem")

    # metal from stockpile --> plant
    metalSP = model.addVars(plants_stockpiles_periods, vtype=GRB.CONTINUOUS, lb=0, name="metalSP")

    # metal remaining in stockpile
    metalS_rem = model.addVars(stockpile_periods, vtype=GRB.CONTINUOUS, lb=0, name="metalS_rem")

    # --- New AT-specific variables ---
    z_sp = z_ss = f_t = None
    if aggregate:
        # z[i,t,s,p] ∈ [0,1], fraction of block i sent from the stockpile s for processing in the plant p during time period t
        z_sp = model.addVars(block_periods_stockpiles_plants, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z_sp")

        # z[i,t,s] ∈ [0,1], fraction of block i remaining in the stockpile s in time period t
        z_ss = model.addVars(block_periods_stockpiles, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z_ss")

        # f_t ∈ [0,1], out-fractions variable for each time period t
        f_t = model.addVars(periods, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="f_t")

    if aggregate:
        return (
            x, y, z_p, z_s, oreSP, oreS_rem, metalSP, metalS_rem, z_ss, z_sp, f_t
        )
    else:
        return x, y, z_p, z_s, oreSP, oreS_rem, metalSP, metalS_rem

def define_objective(
        model: gp.Model,
        y, z_p, oreSP, metalSP,
        blocks, plants, stockpiles, periods,
        O, A, R,                 # dict-like: block -> ore, metal, rock
        c: float,                # revenue per metal unit
        proc_cost: float,        # processing cost per ore unit
        m: float,                # mining cost per rock unit
        #delta,                   # discount: callable like delta(t) or delta(t, q)
        q=None                   # optional extra passed to delta if needed
) -> None:
    """
    Set the NF objective: maximize discounted net value over periods.

    For each period t:
      aP(t) = Σ_{p,s} metalSP[p,s,t]               (reclaimed metal)
      oP(t) = Σ_{p,s} oreSP[p,s,t]                  (reclaimed ore)
      dir_metal(t) = Σ_i A[i] Σ_p z_p[i,t,p]       (direct metal)
      dir_ore(t)   = Σ_i O[i] Σ_p z_p[i,t,p]       (direct ore)
      mined_rock(t)= Σ_i R[i] * y[i,t]             (mined rock)

    Objective:
      maximize Σ_t δ(t[,q]) * [ c*(aP(t)+dir_metal(t))
                                - proc_cost*(oP(t)+dir_ore(t))
                                - m*mined_rock(t) ]

    Notes:
      • Pass `delta` as a function, e.g. `lambda t: 1/(1+r)**(t-1)`.
      • This builds a linear objective (δ must return a numeric scalar).
      • Does not return a value; it calls model.setObjective(...).
    """

    def aP(t):
        # metal reclaimed from stockpile to plants in period t
        return gp.quicksum(metalSP[p, s, t] for p in plants for s in stockpiles)

    def oP(t):
        # ore reclaimed from stockpiles to plants in period t
        return gp.quicksum(oreSP[p, s, t] for p in plants for s in stockpiles)

    def dir_metal(t):
        # direct-to-plant metal in period t = sum_i A[i] * sum_p z_p[i,t,p]
        return gp.quicksum(A[i] * gp.quicksum(z_p[i, t, p] for p in plants) for i in blocks)

    def dir_ore(t):
        # direct-to-plant ore in t = sum_i O[i] * sum_p z_p[i,t,p]
        return gp.quicksum(O[i] * gp.quicksum(z_p[i, t, pl] for pl in plants) for i in blocks)

    def mined_rock(t):
        return gp.quicksum(R[i] * y[i, t] for i in blocks)

    obj = gp.quicksum(
        delta(t, q) * (
            c * (aP(t) + dir_metal(t))
            - proc_cost * (oP(t) + dir_ore(t))
            - m * mined_rock(t)
        )
        for t in periods
    )

    model.setObjective(obj, GRB.MAXIMIZE)


def define_constraints(
    model, x, y, z_p, z_s, oreS_rem, metalS_rem, oreSP, metalSP,
    blocks, plants, stockpiles, time_periods,
    O, A, R,                  # dict-like by block: ore, metal, rock
    P,                        # precedence: dict i -> iterable of predecessors j
    mining_capacity,          # dict-like by t
    processing_capacity,      # dict-like by t
    enforce_mixing=False,     # True => add (11), False => skip
    aggregate=False,
    enforce_mixing_aggregate=False,
    z_ss=None, z_sp=None, f_t=None,
):
    """
    Add Natural Formulation (NF) constraints to the model.

    Indices/sets:
        blocks, plants, stockpiles, time_periods (ordered; t0=min, tT=max).
    Parameters:
        O[i], A[i], R[i] – ore, metal, rock per block i.
        P[i] – predecessors j of block i (cumulative precedence).
        mining_capacity[t], processing_capacity[t] – per-period limits.

    Variables:
        x[i,t] ∈ {0,1}     – cumulative completion (1 if i done by end of t).
        y[i,t] ∈ [0,1]     – fraction of block i mined/processed in period t.
        z_p[i,t,p] ∈ [0,1] – fraction of i sent directly to plant p in t.
        z_s[i,t,s] ∈ [0,1] – fraction of i sent to stockpile s in t.
        oreS_rem[s,t] ≥ 0  – ore remaining in stockpile s during t.
        metalS_rem[s,t] ≥ 0– metal remaining in stockpile s during t.
        oreSP[p,s,t] ≥ 0   – ore reclaimed from s to plant p in t.
        metalSP[p,s,t] ≥ 0 – metal reclaimed from s to plant p in t.

    Constraints added:
      (1) cumulative:            x[i,t-1] ≤ x[i,t]
      (2) link x–y:              x[i,t] ≤ Σ_{τ≤t} y[i,τ]
      (3) mined at most once:    Σ_t y[i,t] ≤ 1
      (4) precedence (cum.):     Σ_{τ≤t} y[i,τ] ≤ x[j,t]   ∀ j ∈ P(i)
      (5) split feasibility:     Σ_p z_p[i,t,p] + Σ_s z_s[i,t,s] ≤ y[i,t]
      (6) ore sp balance:        oS[s,t-1] + Σ_i O[i] z_s[i,t-1,s] = oS[s,t] + Σ_p oreSP[p,s,t]
      (7) metal sp balance:      aS[s,t-1] + Σ_i A[i] z_s[i,t-1,s] = aS[s,t] + Σ_p metalSP[p,s,t]
      (8) boundaries:            oS[s,t0]=aS[s,t0]=oS[s,tT]=aS[s,tT]=0;  Σ_p oreSP[p,s,t0]=Σ_p metalSP[p,s,t0]=0
      (9) mining capacity:       Σ_i R[i] y[i,t] ≤ mining_capacity[t]
     (10) processing capacity:   Σ_i O[i] Σ_p z_p[i,t,p] + Σ_{p,s} oreSP[p,s,t] ≤ processing_capacity[t]
     (11) homogeneous mixing:    aP_t (oS_t + oP_t) = oP_t (aS_t + aP_t),
                                 where oP_t=Σ_{p,s} oreSP[p,s,t], aP_t=Σ_{p,s} metalSP[p,s,t],
                                       oS_t=Σ_s oreS_rem[s,t],    aS_t=Σ_s metalS_rem[s,t]
                                 (bilinear; requires model.Params.NonConvex = 2)

    Notes:
      • Assumes time_periods are numeric (uses t-1). If not, build a prev[t] mapping.
      • (8) boundary equalities could be implemented as variable fixings (UB=0) for stronger presolve.
      • (11) is nonconvex; set `model.Params.NonConvex = 2` before optimize.

    Side effects:
    """

    # --- Helper---
    t0 = min(time_periods)
    tT = max(time_periods)

    # The variables (z_sp, z_ss, f_t) are mandatory in the AT case
    if aggregate and (z_sp is None or z_ss is None or f_t is None):
        raise ValueError("aggregate=True requires z_sp, z_ss, and f_t to be provided.")


    # 1) cumulative completion
    model.addConstrs(
        (x[i, t - 1] <= x[i, t] for i in blocks for t in time_periods if t > 1), name="cumulative")

    # 2) completion implies sufficient cumulative mining
    model.addConstrs(
        (x[i, t] <= quicksum(y[i, tau] for tau in time_periods if tau <= t)
         for i in blocks for t in time_periods), name="link_xy")

    # 3) Each block can be mined at most once
    model.addConstrs((quicksum(y[i, t] for t in time_periods) <= 1 for i in blocks), name="frac_leq1")

    # 4) Precedence feasibility (j before i): cumulative precedence
    # For every predecessor j ∈ P(i), block i’s cumulative mined fraction up to t
    # cannot exceed j’s completion status at t. This enforces that i does not advance
    # ahead of any immediate predecessor (i waits until j is completely mined).

    model.addConstrs((quicksum(y[i, tau] for tau in time_periods if tau <= t) <= x[j, t]
                      for i in P for j in P[i] for t in time_periods), name="precedence")

    # 5) processing_fraction + stockpile_fraction <= mined_fraction
    # The sum of fractions sent for processing and to stockpile cannot exceed what was mined
    model.addConstrs((quicksum(z_p[i, t, p] for p in plants) + quicksum(z_s[i, t, s] for s in stockpiles) <= y[i, t]
                      for i in blocks for t in time_periods), name="split_mined_to_routing")

    # 6) ore stockpile balance: o^s_{t-1} + sum_i O_i z^s_{i,t-1} = o^s_t + o^P_t
    model.addConstrs(
        (oreS_rem[s, t - 1] + quicksum(O[i] * z_s[i, t - 1, s] for i in blocks) == oreS_rem[s, t] + quicksum(
            oreSP[p, s, t] for p in plants)
         for s in stockpiles for t in time_periods if t > t0),
        name="sp_bal_ore"
    )

    # 7) metal stockpile balance: a^s_{t-1} + sum_i A_i z^s_{i,t-1} = a^s_t + a^P_t
    model.addConstrs(
        (metalS_rem[s, t - 1] + quicksum(A[i] * z_s[i, t - 1, s] for i in blocks) == metalS_rem[s, t] + quicksum(
            metalSP[p, s, t] for p in plants)
         for s in stockpiles for t in time_periods if t > t0),
        name="sp_bal_metal"
    )

    # 8) boundary condition: empty at start and end, no reclaim in period 1
    model.addConstrs((oreS_rem[s, t0] == 0 for s in stockpiles), name="sp_init_ore")
    model.addConstrs((oreS_rem[s, tT] == 0 for s in stockpiles), name="sp_end_ore")
    model.addConstrs((metalS_rem[s, t0] == 0 for s in stockpiles), name="sp_init_metal")
    model.addConstrs((metalS_rem[s, tT] == 0 for s in stockpiles), name="sp_end_metal")
    # o^P_1 = a^P_1 = 0  (no reclaim at t=1)
    model.addConstrs(
        (quicksum(oreSP[p, s, t0] for p in plants) == 0 for s in stockpiles),
        name="no_reclaim_ore_t1"
    )
    model.addConstrs(
        (quicksum(metalSP[p, s, t0] for p in plants) == 0 for s in stockpiles),
        name="no_reclaim_metal_t1"
    )

    # 9) mining capacity
    model.addConstrs((quicksum(R[i] * y[i, t] for i in blocks) <= mining_capacity[t]
                      for t in time_periods),
                     name="mining_cap")

    # 10) processing capacity
    model.addConstrs(
        (quicksum(O[i] * quicksum(z_p[i, t, p] for p in plants) for i in blocks)
         + quicksum(oreSP[p, s, t] for p in plants for s in stockpiles)
         <= processing_capacity[t]
         for t in time_periods),
        name="processing_cap")

    # 11) homogeneous-mixing ratio constraint (nonconvex)
    if enforce_mixing:
        model.Params.NonConvex = 2
        for t in time_periods:
            oP_t = quicksum(oreSP[p, s, t] for p in plants for s in stockpiles)
            aP_t = quicksum(metalSP[p, s, t] for p in plants for s in stockpiles)
            oS_t = quicksum(oreS_rem[s, t] for s in stockpiles)
            aS_t = quicksum(metalS_rem[s, t] for s in stockpiles)
            model.addQConstr(aP_t * (oS_t + oP_t) == oP_t * (aS_t + aP_t),
                             name=f"mix_ratio_allS[{t}]")


    # --- New AT-specific constraints ---
    if aggregate:

        assert z_sp is not None and z_ss is not None and f_t is not None
        # (12) stockpile balance for all i, s, t>=2: z_ss[i,t-1,s] + z_s[i,t-1,s] = z_ss[i,t,s] + z_sp[i,t,s,p]
        model.addConstrs(
            (z_ss[i, t - 1, s] + z_s[i, t - 1, s] == z_ss[i, t, s] + quicksum(z_sp[i, t, s, p] for p in plants)
             for i in blocks for s in stockpiles for t in time_periods[1:]), name="stockpile_balance"

        )

        # (13) stockpile initializing:
        for i in blocks:
            for s in stockpiles:
                model.addConstr(z_ss[i, t0, s] == 0, name=f"zss_start[{i},{s}]")
                model.addConstr(z_ss[i, tT, s] == 0, name=f"zss_end[{i},{s}]")
                model.addConstr(quicksum(z_sp[i, t0, s, p] for p in plants) == 0, name=f"zsp_start[{i},{s}]")

        # (14)  ore remaining in stockpile s during period t:
        #       oreS_rem[s,t] = sum_i O[i] * z_ss[i,t,s]
        model.addConstrs(
            (oreS_rem[s, t] == quicksum(O[i] * z_ss[i, t, s] for i in blocks) for s in stockpiles for t in
             time_periods),
            name="oreS_rem_def")

        # (15)  ore sent from stockpile s to plant p during period t:
        #       oreSP[p,s,t] = sum_i O[i] * z_sp[i,t,s,p]
        model.addConstrs(
            (quicksum(oreSP[p, s, t] for p in plants) == quicksum(
                O[i] * quicksum(z_sp[i, t, s, p] for p in plants) for i in blocks)
             for s in stockpiles for t in time_periods),
            name="oreSP_def")
        # (16) metal remaining in stockpile s during period t:
        #      metalS_rem[] = sum_i A[i] * z_ss[i,t,s]
        model.addConstrs(
            (metalS_rem[s, t] == quicksum(A[i] * z_ss[i, t, s] for i in blocks) for s in stockpiles for t in
             time_periods),
            name="metalS_rem_def")

        # (17) metal sent from stockpile s to plant p during period t:
        #      metalSP[s,t] = sum_i A[i] * z_sp[i,t,s]
        model.addConstrs(
            (quicksum(metalSP[p, s, t] for p in plants) == quicksum(
                A[i] * quicksum(z_sp[i, t, s, p] for p in plants) for i in blocks)
             for s in stockpiles for t in time_periods),
            name="metalSP_def")
        if enforce_mixing_aggregate:
            # (18) Mixing per aggregate i, period t, stockpile s:
            #      (sum_p z_sp[i,t,s,p]) * (1 - f_t[t]) == z_ss[i,t,s] * f_t[t]
            model.addConstrs(
                (quicksum(z_sp[i, t, s, p] for p in plants) * (1 - f_t[t])
                 == z_ss[i, t, s] * f_t[t] for i in blocks for t in time_periods for s in stockpiles),
                name="f_t_def"
            )
