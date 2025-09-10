from enum import CONTINUOUS

from pyomo.environ import ( ConcreteModel, Set, Param, Var, Binary, NonNegativeReals, Reals,\
                            Constraint, Objective, maximize, summation, value, RangeSet)
from pyomo.core.base.PyomoModel import Model as PyomoModel
from optutils.gurobi_helpers import delta


def build_model(time_data, blocks, plants, stockpiles, waste_dumps, P_pred, parameters):
    """
    Define a Concrete Model
    """
    M = ConcreteModel()

    # scalars
    q = parameters["annual_discount"]
    proc_cost = parameters["cost_processing"]
    m = parameters["cost_mining"]
    c = parameters["price_metal"]

    # -------- Sets --------
    def add_sets(M: PyomoModel) -> None:

        M.T = Set(initialize=sorted(time_data.keys()), doc='time_periods')
        M.I = Set(initialize=sorted(blocks.keys()), doc='blocks')
        M.P = Set(initialize=sorted(plants), doc='plants')
        M.S = Set(initialize=sorted(stockpiles), doc='stockpiles')
        M.W = Set(initialize=sorted(waste_dumps), doc='waste dumps')
        M.Pred = Set(M.I, within=M.I,
                     initialize=lambda M, i: list(sorted(P_pred.get(i, []))),
                     doc='blocks with predecessors')

    add_sets(M)

    # ------ Parameters ------
    def add_parameters(M: PyomoModel) -> None:

        # capacities
        M.mining_cap     = Param(M.T, initialize={t: time_data[t][0] for t in M.T}, within=Reals)
        M.processing_cap = Param(M.T, initialize={t: time_data[t][1] for t in M.T}, within=Reals)
        M.discount       = Param(M.T, initialize={t: delta(t, q) for t in M.T}, within=Reals)

        # block tonnages
        M.R = Param(M.I, initialize={b: blocks[b][0] for b in M.I}, within=Reals, doc='rock')
        M.O = Param(M.I, initialize={i: blocks[i][1] for i in M.I}, within=Reals, doc='ore')
        M.A = Param(M.I, initialize={i: blocks[i][2] for i in M.I}, within=Reals, doc='metal')

        # scalar economic params
        M.m_cost  = Param(initialize=m, doc='mining cost')
        M.p_cost  = Param(initialize=proc_cost, doc='processing cost')
        M.price_m = Param(initialize=c, doc='sales price of metal')

    add_parameters(M)

    return M

def add_variables(M: PyomoModel, x_binary: bool = True, aggregate = False) -> None:
    """
    Decision Variables
    """

    # x[i,t] ∈ {0,1}: 1 if block i is completed (cumulative) by end of period t
    if x_binary:
        M.x = Var(M.I, M.T, domain=Binary)
    else:
        M.x = Var(M.I, M.T, domain=Reals, bounds=(0,1))

    # y[i,t] ∈ [0,1]: fraction of block i that is mined in period t
    M.y = Var(M.I, M.T, bounds=(0, 1))

    # z_p[i,t,p] ∈ [0,1]: fraction of block i sent directly to plant p in period t
    M.z_p = Var(M.I, M.T, M.P, bounds=(0, 1))

    # z_s[i,t,s] ∈ [0,1]: fraction of block i sent to stockpile s in period t
    M.z_s = Var(M.I, M.T, M.S, bounds=(0, 1))

    # oreSP[p,s,t] ≥ 0 : reclaimed ore flow from stockpile s to plant p in period t
    M.oreSP = Var(M.P, M.S, M.T, domain=NonNegativeReals)

    # oreS_rem[s,t] ≥ 0 : ore remaining in stockpile s at end of period t
    M.oreS_rem = Var(M.S, M.T, domain=NonNegativeReals)

    # metalSP[p,s,t] ≥ 0 : reclaimed metal flow (associated with oreSP) to plant
    M.metalSP = Var(M.P, M.S, M.T, domain=NonNegativeReals)

    # metalS_rem[s,t] ≥ 0 : metal content remaining in stockpile s at end of t
    M.metalS_rem = Var(M.S, M.T, domain=NonNegativeReals)

    # --- New AT-specific variables ---
    #z_sp = z_ss = f_t = None
    if aggregate:
        # z[i,t,s,p] ∈ [0,1], fraction of block i sent from the stockpile s for processing in the plant p during time period t
        M.z_sp = Var(M.I, M.T, M.S, M.P, domain=Reals, bounds=(0,1))

        # z[i,t,s] ∈ [0,1], fraction of block i remaining in the stockpile s in time period t
        M.z_ss = Var(M.I, M.T, M.S, domain=Reals, bounds=(0,1))

        # f_t ∈ [0,1], out-fractions variable for each time period t
        M.f_t = Var(M.T, domain=Reals, bounds=(0, 1))

def add_constraints(M: PyomoModel, aggregate = False,
                    enforce_mixing = False,
                    enforce_mixing_aggregate = False) -> None:
    """
    Constraints
    """

    # first and last period
    t0 = min(M.T)
    tT = max(M.T)

    # (1)  Cumulative completion: x[i,t-1] ≤ x[i,t]
    def cumulative_rule(M, i, t):
        if t == t0:
            return Constraint.Skip
        return M.x[i, t - 1] <= M.x[i, t]

    M.Cumulative = Constraint(M.I, M.T, rule=cumulative_rule)

    # (2) Completion implies sufficient cumulative mining: x[i,t] ≤ sum_{τ ≤ t} y[i,τ]
    def link_xy_rule(M, i, t):
        return M.x[i, t] <= sum(M.y[i, tau] for tau in M.T if tau <= t)

    M.LinkXY = Constraint(M.I, M.T, rule=link_xy_rule)

    # (3) Each block mined at most once: sum_t y[i,t] ≤ 1
    def mined_once_rule(M, i):
        return sum(M.y[i, t] for t in M.T) <= 1.0

    M.MinedOnce = Constraint(M.I, rule=mined_once_rule)

    # (4) Precedence: for all i with predecessors j in P_pred[i],
    #     sum_{τ ≤ t} y[i,τ] ≤ x[j,t]  (i cannot advance ahead of j)
    def presedence_rule(M, i, j, t):
        # Apply only if i has predecessor j
        return sum(M.y[i, tau] for tau in M.T if tau <= t) <= M.x[j, t]

    # Build an index for (i, j, t) where j in P_pred[i]
    M.PrecedenceIndex = Set(dimen=3, initialize=lambda M: {(i, j, t) for i in M.I for j in M.Pred[i] for t in M.T})
    M.Precedence = Constraint(M.PrecedenceIndex, rule=lambda M, i, j, t: presedence_rule(M, i, j, t))

    # (5) Splitting mined fraction into routing: sum_p z_p + sum_s z_s ≤ y[i,t]
    def split_rule(M, i, t):
        return (
                sum(M.z_p[i, t, p] for p in M.P)  # part sent tp plants
                + sum(M.z_s[i, t, s] for s in M.S)  # part sent to stockpiles
                <= M.y[i, t]  # the summation cannot exceed mined fraction
        )

    M.SplitRouting = Constraint(M.I, M.T, rule=split_rule)

    # (6) Ore stockpile balance (for t > t0):
    #     oreS_rem[s,t-1] + sum_i O[i]*z_s[i,t-1,s] = oreS_rem[s,t] + sum_p oreSP[p,s,t]
    def ore_balance_rule(M, s, t):
        if t == t0:
            return Constraint.Skip
        return (
                M.oreS_rem[s, t - 1]
                + sum(M.O[i] * M.z_s[i, t - 1, s] for i in M.I)
                == M.oreS_rem[s, t]
                + sum(M.oreSP[p, s, t] for p in M.P)
        )

    M.OreBalance = Constraint(M.S, M.T, rule=ore_balance_rule)

    # (7) Metal stockpile balance (for t > t0):
    #     metalS_rem[s,t-1] + sum_i A[i]*z_s[i,t-1,s] = metalS_rem[s,t] + sum_p metalSP[p,s,t]
    def metal_balance_rule(M, s, t):
        if t == t0:
            return Constraint.Skip
        return (
                M.metalS_rem[s, t - 1]
                + sum(M.A[i] * M.z_s[i, t - 1, s] for i in M.I)
                == M.metalS_rem[s, t]
                + sum(M.metalSP[p, s, t] for p in M.P)
        )

    M.MetalBalance = Constraint(M.S, M.T, rule=metal_balance_rule)

    # (8) Boundary conditions and no reclaim in period 1
    #     - Empty stockpile at start and end: oreS/metalS at t0 and tT are 0
    #     - No reclaim at t0: sum_p oreSP[p, s, t0] = sum_p metalSP[p, s, t0] = 0
    def ore_init_rule(M, s):
        return M.oreS_rem[s, t0] == 0.0

    def ore_end_rule(M, s):
        return M.oreS_rem[s, tT] == 0.0

    def metal_init_rule(M, s):
        return M.metalS_rem[s, t0] == 0.0

    def metal_end_rule(M, s):
        return M.metalS_rem[s, tT] == 0.0

    def no_reclaim_ore_t0_rule(M, s):
        return sum(M.oreSP[p, s, t0] for p in M.P) == 0.0

    def no_reclaim_metal_t0_rule(M, s):
        return sum(M.metalSP[p, s, t0] for p in M.P) == 0.0

    M.OreInit = Constraint(M.S, rule=ore_init_rule)
    M.OreEnd = Constraint(M.S, rule=ore_end_rule)
    M.MetalInit = Constraint(M.S, rule=metal_init_rule)
    M.MetalEnd = Constraint(M.S, rule=metal_end_rule)
    M.NoReclaimOre_t0 = Constraint(M.S, rule=no_reclaim_ore_t0_rule)
    M.NoReclaimMetal_t0 = Constraint(M.S, rule=no_reclaim_metal_t0_rule)

    # (9) Mining capacity: sum_i R[i]*y[i,t] ≤ mining_cap[t]
    def mining_cap_rule(M, t):
        return (
                sum(M.R[i] * M.y[i, t] for i in M.I) <= M.mining_cap[t]
        )

    M.MiningCap = Constraint(M.T, rule=mining_cap_rule)

    # (10) Processing capacity:
    #      sum_i O[i]*sum_p z_p[i,t,p] + sum_{p,s} oreSP[p,s,t] ≤ processing_cap[t]
    def processing_cap_rule(M, t):
        direct_ore = sum(M.O[i] * sum(M.z_p[i, t, p] for p in M.P) for i in M.I)
        reclaim_ore = sum(M.oreSP[p, s, t] for p in M.P for s in M.S)
        return direct_ore + reclaim_ore <= M.processing_cap[t]

    M.ProcessingCap = Constraint(M.T, rule=processing_cap_rule)

    # (11) Homogeneous-mixing ration constraint (nonconvex)
    #   For each t:
    #     aP_t * (oS_t + oP_t) == oP_t * (aS_t + aP_t)
    # where
    #   oP_t = sum_{p,s} oreSP[p,s,t],     aP_t = sum_{p,s} metalSP[p,s,t]
    #   oS_t = sum_s oreS_rem[s,t],        aS_t = sum_s metalS_rem[s,t]
    def mix_ratio_rule(M, t):
        aP_t = sum(M.metalSP[p, s, t] for p in M.P for s in M.S)
        aS_t = sum(M.metalS_rem[s, t] for s in M.S)
        oP_t = sum(M.oreSP[p, s, t] for p in M.P for s in M.S)
        oS_t = sum(M.oreS_rem[s, t] for s in M.S)
        return aP_t * (oP_t + oS_t) == oP_t * (aP_t + aS_t)

    if enforce_mixing:
        M.MixRatio = Constraint(M.T, rule=mix_ratio_rule)

    # --- New AT-specific constraints ---
    if aggregate:

        # (12) stockpile balance for all i, s, t>=2: z_ss[i,t-1,s] + z_s[i,t-1,s] = z_ss[i,t,s] + z_sp[i,t,s,p]
        def stockpile_balance_rule(M, i, s, t):
            if t == min(M.T):  # skip first period
                return Constraint.Skip
            return (
                    M.z_ss[i, t - 1, s] + M.z_s[i, t - 1, s]
                    == M.z_ss[i, t, s] + sum(M.z_sp[i, t, s, p] for p in M.P)
            )
        M.stockpile_balance = Constraint(M.I, M.S, M.T, rule=stockpile_balance_rule)

        # (13) stockpile initializing:
        def zss_start_rule(M, i, s):
            return M.z_ss[i, t0, s] == 0.0
        def zss_end_rule(M, i, s):
            return M.z_ss[i, tT, s] == 0.0
        def zsp_start_rule(M, i, s):
            return sum(M.z_sp[i, t0, s, p] for p in M.P) == 0.0

        M.ZssStart = Constraint(M.I, M.S, rule=zss_start_rule)
        M.ZssEnd = Constraint(M.I, M.S, rule=zss_end_rule)
        M.ZspStart = Constraint(M.I, M.S, rule=zsp_start_rule)

        # (14)  ore remaining in stockpile s during period t:
        #       oreS_rem[s,t] = sum_i O[i] * z_ss[i,t,s]
        def ore_rem_rule(M, s, t):
            return M.oreS_rem[s, t] == sum(M.O[i] * M.z_ss[i, t, s] for i in M.I)

        M.OreS_rem_def = Constraint(M.S, M.T, rule=ore_rem_rule)

        # (15)  ore sent from stockpile s to plant p during period t:
        #       oreSP[p,s,t] = sum_i O[i] * z_sp[i,t,s,p]
        def oreSP_def_rule(M, p, s, t):
            return M.oreSP[p, s, t] == sum(M.O[i] * M.z_sp[i, t, s, p] for i in M.I)

        M.OreSP_def = Constraint(M.P, M.S, M.T, rule=oreSP_def_rule)

        # (16) metal remaining in stockpile s during period t:
        #      metalS_rem[] = sum_i A[i] * z_ss[i,t,s]
        def metal_rem_rule(M, s, t):
            return M.metalS_rem[s, t] == sum(M.A[i] * M.z_ss[i, t, s] for i in M.I)

        M.MetalS_rem_def = Constraint(M.S, M.T, rule=metal_rem_rule)

        # (17) metal sent from stockpile s to plant p during period t:
        #      metalSP[s,t] = sum_i A[i] * z_sp[i,t,s]
        def metalSP_def_rule(M, p, s, t):
            return M.metalSP[p, s, t] == sum(M.A[i] * M.z_sp[i, t, s, p] for i in M.I)

        M.MetalSP_def = Constraint(M.P, M.S, M.T, rule=metalSP_def_rule)


        # (18) Mixing per aggregate i, period t, stockpile s:
        #      (sum_p z_sp[i,t,s,p]) * (1 - f_t[t]) == z_ss[i,t,s] * f_t[t]
        def mixing_rule(M, i, t, s):
            return sum(M.z_sp[i, t, s, p] for p in M.P) * (1 - M.f_t[t]) == M.z_ss[i, t, s] * M.f_t[t]

        if enforce_mixing_aggregate:
            M.F_t_def = Constraint(M.I, M.T, M.S, rule=mixing_rule)


def add_objective(M: PyomoModel) -> None:
    """
    Objective Function
    """
    # Profit in period t (discounted):
    #  price * (aP_t + direct_metal_t)
    #  - proc_cost * (oP_t + direct_ore_t)
    #  - mining_cost * mined_rock_t
    def period_terms(M, t):
        # reclaimed flows
        aP_t = sum(M.metalSP[p, s, t] for p in M.P for s in M.S)
        oP_t = sum(M.oreSP[p, s, t]   for p in M.P for s in M.S)
        # direct-to-plant flows from current mining
        dir_metal_t = sum(M.A[i] * sum(M.z_p[i, t, p] for p in M.P) for i in M.I)
        dir_ore_t   = sum(M.O[i] * sum(M.z_p[i, t, p] for p in M.P) for i in M.I)
        # mined rock in t
        mined_rock_t = sum(M.R[i] * M.y[i, t] for i in M.I)
        # discounted contribution
        return M.discount[t] * (M.price_m * (aP_t + dir_metal_t)
                               - M.p_cost * (oP_t + dir_ore_t)
                               - M.m_cost * (mined_rock_t))
    M.Obj = Objective(expr=sum(period_terms(M, t) for t in M.T), sense=maximize)