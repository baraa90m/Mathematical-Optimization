from typing import Iterable, Mapping, Hashable, Optional
from graphviz import Digraph
import gurobipy as gp
from gurobipy import GRB, quicksum
from itertools import product
from gurobipy import multidict, tuplelist

def draw_problem(
        blocks: Iterable[Hashable],
        plants: Iterable[Hashable] = (),
        stockpiles: Iterable[Hashable] = (),
        waste_dumps: Iterable[Hashable] = (),
        R: Optional[Mapping[Hashable, float]] = None,
        O: Optional[Mapping[Hashable, float]] = None,
        A: Optional[Mapping[Hashable, float]] = None
) -> Digraph:
    """
    Build a simple flows diagram (Graphviz) for blocks → {stockpiles|plants|waste}.

    Args:
        blocks: Block IDs.
        plants: Plant IDs.
        stockpiles: Stockpile IDs.
        waste_dumps: Waste dump IDs.
        R, O, A: (optional) dicts mapping block -> value; shown on block labels.

    Returns:
        graphviz.Digraph ready to .render() or display in Jupyter.
    """
    blocks_list = list(blocks or [])
    plants_list = list(plants or [])
    stockpiles_list = list(stockpiles or [])
    waste_dumps_list = list(waste_dumps or [])

    dot = Digraph (
        graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'newrank': 'true', 'size': '10,12', 'nodesep': '0.8', 'ranksep': '1.0'},
        node_attr={'fontsize': '12', 'fontname': 'Arial'},
        edge_attr={'color': 'gray', 'penwidth': '0.8'},
        name="MineFlow"
    )

    # === Nodes ===
    for i in blocks_list:
        dot.node(
            f"b{i}",
            label=f"Block {i}\nR={R[i]}  O={O[i]}  A={A[i]}",
            shape='box', fontcolor='blue'
        )

    for idx, s in enumerate(stockpiles_list, start=1):
        dot.node(
            f"s{s}",
            label=f"Stockpile {idx}\n",
            shape='doublecircle', fontcolor='blue'
        )

    for idx, w in enumerate(waste_dumps_list, start=1):
        dot.node(
            f"w{w}",
            label=f"Waste dump {idx}",
            shape='invhouse', fontcolor='blue'
        )

    for idx, p in enumerate(plants_list, start=1):
        dot.node(
            f"p{p}",
            label=f"Plant {idx}",
            shape='invhouse', fontcolor='blue'
        )

    # === Edges ===
    # Block -> Stockpile edges
    for i in blocks_list:
        for s in stockpiles_list:
            dot.edge(f"b{i}", f"s{s}")

    # Block -> Waste dump edges
    for i in blocks_list:
        for w in waste_dumps_list:
            dot.edge(f"b{i}", f"w{w}")

    # Block -> Plant edges
    for i in blocks_list:
        for p in plants_list:
            dot.edge(f"b{i}", f"p{p}")
    # Stockpile -> Plant edges
    for s in stockpiles_list:
        for p in plants_list:
            dot.edge(f"s{s}", f"p{p}")
    # Stockpile -> Waste dump edges
    for s in stockpiles_list:
        for w in waste_dumps_list:
            dot.edge(f"s{s}", f"w{w}")
    return dot


