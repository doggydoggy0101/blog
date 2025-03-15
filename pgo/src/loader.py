import numpy as np
from collections import namedtuple


class Graph:
    def __init__(self, x, nodes, edges, lut):
        self.nodes = nodes
        self.edges = edges
        self.x = x  # all node variables as a vector
        self.lut = lut  # look up table for a node's starting index in vector x


def read_graph(file_path):
    nodes = {}
    edges = []
    Edge = namedtuple("Edge", ["fromNode", "toNode", "measurement", "information"])
    with open(file_path, "r") as file:
        for line in file:
            data = line.split()

            if data[0] == "VERTEX_SE2":
                nodeId = int(data[1])
                pose = np.array(data[2:5], dtype=np.float32)  # x, y, theta
                nodes[nodeId] = pose

            elif data[0] == "EDGE_SE2":
                fromNode = int(data[1])
                toNode = int(data[2])
                measurement = np.array(data[3:6], dtype=np.float32)  # x, y, theta
                uppertri = np.array(data[6:12], dtype=np.float32)
                information = np.array(
                    [
                        [uppertri[0], uppertri[1], uppertri[2]],
                        [uppertri[1], uppertri[3], uppertri[4]],
                        [uppertri[2], uppertri[4], uppertri[5]],
                    ]
                )
                edge = Edge(fromNode, toNode, measurement, information)
                edges.append(edge)

            else:
                print("VERTEX/EDGE type not defined (see `src/loader.py`)")
                exit(1)

    x = []
    lut = {}
    index = 0
    for nodeId in nodes:
        lut.update({nodeId: index})
        index += len(nodes[nodeId])
        x.append(nodes[nodeId])
    x = np.concatenate(x, axis=0)

    graph = Graph(x, nodes, edges, lut)
    print(
        "Loaded graph with {} nodes and {} edges".format(
            len(graph.nodes), len(graph.edges)
        ),
        end="\n\n",
    )

    return graph
