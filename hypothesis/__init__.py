from networkx import DiGraph, gnm_random_graph, find_cycle, exception

from hypothesis.networks import RestrictedWeight, RestrictedBias, NeuralMotif, NeuralNetwork
from hypothesis.operations import Monitor, prepare_data, prepare_motifs
from hypothesis.similarity import calculate_similarity, maximum_minimum_search, minimum_search
from hypothesis.robustness import intervene_entrances, calculate_gradients


acyclic_motifs = {
    "collider": [DiGraph([(1, 3, {"weight": +1}), (2, 3, {"weight": +1})]),
                 DiGraph([(1, 3, {"weight": +1}), (2, 3, {"weight": -1})]),
                 DiGraph([(1, 3, {"weight": -1}), (2, 3, {"weight": -1})])],
    "fork": [DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": +1})]),
             DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": -1})]),
             DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": -1})])],
    "chain": [DiGraph([(1, 2, {"weight": +1}), (2, 3, {"weight": +1})]),
              DiGraph([(1, 2, {"weight": +1}), (2, 3, {"weight": -1})]),
              DiGraph([(1, 2, {"weight": -1}), (2, 3, {"weight": +1})]),
              DiGraph([(1, 2, {"weight": -1}), (2, 3, {"weight": -1})])],
    "coherent-loop": [DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": +1}), (2, 3, {"weight": +1})]),
                      DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": +1}), (2, 3, {"weight": -1})]),
                      DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": -1}), (2, 3, {"weight": +1})]),
                      DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": -1}), (2, 3, {"weight": -1})])],
    "incoherent-loop": [DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": +1}), (2, 3, {"weight": +1})]),
                        DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": +1}), (2, 3, {"weight": -1})]),
                        DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": -1}), (2, 3, {"weight": +1})]),
                        DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": -1}), (2, 3, {"weight": -1})])]
}


def generate_network(vertex_number, arc_number):
    network = None

    while True:
        try:
            network = gnm_random_graph(n=vertex_number, m=arc_number, directed=True)
            find_cycle(G=network)
        except exception.NetworkXNoCycle:
            break

    return network
