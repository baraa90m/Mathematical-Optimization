import re
import pandas as pd
from typing import Dict, Tuple, Optional
import gurobipy

def read_txt(path):

    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, val = line.split()[:1][0], line.split()[-1]
            try:
                out[key] = int(val) if val.isdigit() else float(val)
            except ValueError:
                out[key] = val
    return out


def parse_arcs_head_format(text, blocks):
    """
        Expect lines like: [14,7,23] -> Block 14 has predecessors {7,23}.
        Any missing blocks get an empty predecessor quantity.
    """

    P = {j: set() for j in blocks}
    for line in text:
        line = line.strip()
        if not line:
            continue
        nums = [int(x) for x in re.findall(r"-?\d+", line)]
        if not nums:
            continue
        j, preds = nums[0], nums[1:]
        P[j].update(preds)
    return P

def td_to_df(model, td, cols=None, tol=1e-9):
    """
    Convert a gurobipy.tupledict (Vars or Constrs) to a tidy DataFrame.
    """
    if not td:
        return pd.DataFrame(columns=(cols or []))

    items = list(td.items())
    keys = [k if isinstance(k, tuple) else (k,) for k, _ in items]
    objs = [v.X for _, v in items]

    data = [(*k, v) for k, v in zip(keys, objs) if v >= tol]

    return pd.DataFrame(data, columns=cols+["Val"])


def print_solution(model, aggregate=False, y=None, z_p=None, z_s=None, tol=1e-9):

    global z_p_filtered, z_s_filtered

    if model.SolCount == 0:
        raise RuntimeError("No solution available.")

    # Check if variables exist in the model
    all_vars = model.getVars()
    var_names = [var.VarName for var in all_vars]

    # Filter only variables that exist in the model
    y_filtered = {}
    if not y:
        assert y is not None
    for key, var in y.items():
        # if var in all_vars:
        y_filtered[key] = var

    # AT-specific
    if aggregate:

        assert z_p is not None and z_s is not None

        z_p_filtered = {}
        for key, var in z_p.items():
            # if var in all_vars:
            z_p_filtered[key] = var

        z_s_filtered = {}
        for key, var in z_s.items():
            # if var in all_vars:
            z_s_filtered[key] = var

    # Use filtered dictionaries
    ydf = td_to_df(model, y_filtered, ["Block", "Period"], tol)
    ydf.rename(columns={"Val": "Mine"}, inplace=True)
    yact = ydf[ydf["Mine"] > tol][["Block", "Period", "Mine"]]  # active (i,t)

    zpdf = td_to_df(model, z_p_filtered, ["Block", "Period", "Plant"], tol)
    zpdf.rename(columns={"Val":"Proc"}, inplace=True)

    zsdf = td_to_df(model, z_s_filtered, ["Block", "Period", "Stock"], tol)
    zsdf.rename(columns={"Val": "Pile"}, inplace=True)

    # Cross-join on (Block,Period) via two merges (creates (i,t,p,s) for nonzeros)
    df = yact.merge(zpdf, on=["Block","Period"], how="left") \
             .merge(zsdf, on=["Block","Period"], how="left")

    df[["Stock"]] = df[["Stock"]].fillna("-")
    df[["Plant"]] = df[["Plant"]].fillna("-")
    df[["Proc","Pile"]] = df[["Proc","Pile"]].fillna(0.0)
    df = df.sort_values(["Block","Period","Plant","Stock"], kind="stable")
    df.index = [""] * len(df)

    return df

def _print_solution(model,
                    y: Dict[Tuple, "gurobipy.Var"],
                    z_p:Optional[Dict[Tuple, "gurobipy.Var"]] = None ,
                    z_s:Optional[Dict[Tuple, "gurobipy.Var"]] = None,
                    aggregate: bool = False,
                    tol: float = 1e-9,
                    y_cols=("Block", "Period"),
                    z_p_cols=("Block", "Period", "Plant"),
                    z_s_cols=("Block", "Period", "Stock"),
                    ) -> pd.DataFrame:
    """
    Build a tidy summary DataFrame from solution variables.

    Parameters
    ----------
    model : gurobipy.Model
        Optimized model with a feasible solution (SolCount > 0).
    y, z_p, z_s : dict-like of tuple -> gurobipy.Var
        Variable containers (tupledict/dict). `y` is required.
        `z_p` and `z_s` are optional unless `aggregate=True`.
    aggregate : bool
        If True, both z_p and z_s must be provided (asserted).
    tol : float
        Small threshold to treat near-zeros as zero.
    y_cols, z_p_cols, z_s_cols : tuple[str, ...]
        Column names for keys of y, z_p, z_s respectively.

    Returns
    -------
    pd.DataFrame
        Tidy table with columns:
        y_cols + ['Mine'] plus optional z_p_cols[-1] as 'Plant' & 'Proc',
        and z_s_cols[-1] as 'Stock' & 'Pile'.
    """

    # --- Asserions ---
    assert hasattr(model, "SolCount"), "model must be a Gurobi model with attribute 'SolCount'."
    assert model.SolCount > 0, "No solution available (model.SolCount == 0). optimize the model first."
    assert y is not None and isinstance(y_cols, (tuple, list)) and len(y_cols) >= 1, \
        "Argument 'y' is required and y_cols must be a non-empty tuple/list."

    if aggregate:
        assert z_p is not None and z_s is not None, \
        "aggregate = True requires both z_p and z_s to be provided."

    # --- helper ---
    model_varnames = {v.VarName for v in model.getVars()}

    def _filter_vars(td: Optional[Dict[Tuple, "gurobipy.Var"]]) -> Dict[Tuple, "gurobipy.Var"]:
        if td is None:
            return {}
        out = {}
        for k, v in td.items():
            # if VarName missing, keep it; otherwise ensure it's in the model.
            vname = getattr(v, "VarName", None)
            if (vname is None) or (vname in model_varnames):
                out[k] = v
        return out

    y_filtered = _filter_vars(y)
    z_p_filtered = _filter_vars(z_p)
    z_s_filtered = _filter_vars(z_s)

    assert len(y_filtered) > 0, "No entries in 'y' after filtering. Check variable construction or names."
    if aggregate:
        assert len(z_p_filtered) > 0, "No entries in 'z_p' after filtering (aggregate=True)."
        assert len(z_s_filtered) > 0, "No entries in 'z_s' after filtering (aggregate=True)."

    # Convert tupledicts to tidy DFs (expects your td_to_df utility)
    ydf = td_to_df(model, y_filtered, list(y_cols), tol)
    if "Val" in ydf.columns:
        ydf.rename(columns={"Val": "Mine"}, inplace=True)
    # Only active (i,t) pairs
    yact_cols = list(y_cols) + ["Mine"]
    yact = ydf[ydf["Mine"] > tol][yact_cols]

    # z_p
    if z_p_filtered:
        zpdf = td_to_df(model, z_p_filtered, list(z_p_cols), tol)
        if "Val" in zpdf.columns:
            zpdf.rename(columns={"Val": "Proc"}, inplace=True)

    # z_s
    if z_s_filtered:
        zsdf = td_to_df(model, z_s_filtered, list(z_s_cols), tol)
        if "Val" in zsdf.columns:
            zsdf.rename(columns={"Val": "Pile"}, inplace=True)

    # Merge (left joins keep only active (i,t) from y)
    merge_keys = list(y_cols)
    df = (
        yact.merge(zpdf, on=merge_keys, how="left")
            .merge(zsdf, on=merge_keys, how="left")
    )

    # Fill and sort (work even if optional pieces are absent)
    for col in ("Stock", "Plant"):
        if col in df.columns:
            df[[col]] = df[[col]].fillna("-")
    for col in ("Proc", "Pile"):
        if col in df.columns:
            df[[col]] = df[[col]].fillna(0.0)

    sort_cols = list(y_cols) + ["Plant", "Stock"]
    df = df.sort_values(sort_cols, kind="stable")
    df.index = [""] * len(df)

    assert len(df) > 0, (
        f"No active rows above tol={tol}. "
        "Either the solution is all ~0, or variable extraction/columns mismatch."
    )

    return df

