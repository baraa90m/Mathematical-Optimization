# Gurobi Tutorials — Mathematical Optimization

This folder contains Jupyter notebooks and small models using **Gurobi**.

## Folder layout
gurobi/
└─ tutorials/
├─ notebooks/ # runnable notebooks
├─ data/ # input data (ignored by git)
├─ models/ # small .py model scripts (optional)
└─ assets/ # figures, images


## Prerequisites
- Python ≥ 3.9
- Gurobi installed and licensed (e.g. 12.x)
- `gurobipy` available in your environment

> Tip (Windows): verify with
> ```python
> import gurobipy as gp; print(gp.gurobi.version())
> ```

## Install shared utilities (`optutils`)
At the **repo root**:
```bash
pip install -e .