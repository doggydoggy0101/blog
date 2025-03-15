import os
import copy

from src.least_squares import GaussNewton
from src.loader import read_graph_g2o
from src.visualize import plot_graph


DATA_NAME = "INTEL"
SOLVE_EUCLIDEAN = True
SOLVE_LIE = True


def main():
    graph = read_graph_g2o(os.path.join("data", f"input_{DATA_NAME}_g2o.g2o"))
    graphs = [copy.deepcopy(graph)]
    titles = ["Initial graph"]

    pgo = GaussNewton(max_iteration=1000, tolerance=1e-4)

    if SOLVE_EUCLIDEAN:
        optimized_Euclidean_graph = pgo.solve(
            copy.deepcopy(graph), gradType="Euclidean"
        )
        graphs.append(optimized_Euclidean_graph)
        titles.append("Optimized by Euclidean gradient")

    if SOLVE_LIE:
        optimized_Lie_graph = pgo.solve(copy.deepcopy(graph), gradType="Lie")
        graphs.append(optimized_Lie_graph)
        titles.append("Optimized by Lie derivative")

    plot_graph(graphs, titles, os.path.join("fig", f"{DATA_NAME}.png"))


if __name__ == "__main__":
    main()
