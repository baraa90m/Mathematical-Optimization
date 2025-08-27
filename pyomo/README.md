# Pyomo Tutorials — Mathematical Optimization

This area contains Jupyter notebooks and examples using **Pyomo** (LP/MIP/NLP).  
Shared helpers live in the top-level package **`optutils`** so notebooks stay clean.

## Folder layout
pyomo/
└─ tutorials/
├─ notebooks/ # runnable notebooks
├─ data/ # input data (ignored by git)
├─ models/ # small .py model scripts (optional)
└─ assets/ # figures, images


## Prerequisites
- Python ≥ 3.9
- `pyomo` installed
- Solver(s):
  - LP/MIP: `glpk` / `cbc` / commercial (`gurobi`, `cplex`)
  - NLP: `ipopt` (for nonlinear problems)


## Install shared utilities (`optutils`)
At the **repo root** (one time per environment):
```bash
pip install -e .