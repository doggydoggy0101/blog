import os
import numpy as np
import matplotlib.pyplot as plt


TITLE_FONTSIZE = 24
COLOR_LIST = ["#e74c3c", "#3498db", "#3498db"]
NODE_KWARG = {
    "marker": "o",
    "edgecolor": "black",
    "linewidth": 1,
    "s": 20,
    "zorder": 1,
}
EDGE_KWARG = {
    "color": "#95a5a6",
    "ls": "-",  # linestyle
    "linewidth": 1,
    "zorder": 0,
}
CUSTOM_CONFIG = {
    "figure.dpi": 200,
    "font.family": "serif",
    "font.serif": "times",
    "mathtext.fontset": "dejavuserif",
    "text.usetex": True,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "grid.color": "#d6dbdf",
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

plt.rcParams.update(CUSTOM_CONFIG)


def plot_graph(graphs, titles, save_path):
    if not isinstance(graphs, list):
        graphs = [graphs]
    if not isinstance(titles, list):
        titles = [titles]

    fig, ax = plt.subplots(1, len(graphs), figsize=(8 * len(graphs), 6))

    if len(graphs) == 1:
        ax = [ax]

    for i, graph in enumerate(graphs):
        poses = []
        for nodeId in graph.nodes:
            index = graph.lut[nodeId]
            pose = graph.x[index : index + 3]
            poses.append(pose)
        poses = np.stack(poses, axis=0)
        ax[i].scatter(poses[:, 0], poses[:, 1], facecolor=COLOR_LIST[i], **NODE_KWARG)

        edges_fromNode = np.array(
            [
                graph.x[graph.lut[edge.fromNode] : graph.lut[edge.fromNode] + 3]
                for edge in graph.edges
            ]
        )
        edges_toNode = np.array(
            [
                graph.x[graph.lut[edge.toNode] : graph.lut[edge.toNode] + 3]
                for edge in graph.edges
            ]
        )
        ax[i].plot(
            np.column_stack((edges_fromNode[:, 0], edges_toNode[:, 0])).T,
            np.column_stack((edges_fromNode[:, 1], edges_toNode[:, 1])).T,
            **EDGE_KWARG,
        )
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_axis_off()
        ax[i].set_title(titles[i], fontsize=TITLE_FONTSIZE)

    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
