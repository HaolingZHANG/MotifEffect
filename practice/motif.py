from itertools import combinations, permutations
from networkx import DiGraph
from numpy import zeros, array, all, sqrt

acyclic_motifs = {
    "incoherent-loop": [DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": +1}), (2, 3, {"weight": +1})]),
                        DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": +1}), (2, 3, {"weight": -1})]),
                        DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": -1}), (2, 3, {"weight": +1})]),
                        DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": -1}), (2, 3, {"weight": -1})])],
    "coherent-loop": [DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": +1}), (2, 3, {"weight": +1})]),
                      DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": +1}), (2, 3, {"weight": -1})]),
                      DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": -1}), (2, 3, {"weight": +1})]),
                      DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": -1}), (2, 3, {"weight": -1})])],
    "collider": [DiGraph([(1, 3, {"weight": +1}), (2, 3, {"weight": +1})]),
                 DiGraph([(1, 3, {"weight": +1}), (2, 3, {"weight": -1})]),
                 DiGraph([(1, 3, {"weight": -1}), (2, 3, {"weight": +1})]),
                 DiGraph([(1, 3, {"weight": -1}), (2, 3, {"weight": -1})])],
}


def motif_matrix_to_fingerprint(motif_matrix):
    """
    Change the type from the motif to the fingerprint for saving.

    :param motif_matrix: the matrix of motif.
    :type motif_matrix: numpy.ndarray

    :return: motif fingerprint.
    :type: tuple
    """
    return tuple(motif_matrix.reshape(-1).tolist())


def fingerprint_to_motif_matrix(fingerprint):
    """
    Change the type from the fingerprint to the motif for motif calculating.

    :param fingerprint: motif fingerprint.
    :type fingerprint: tuple

    :return: motif matrix.
    :rtype: numpy.ndarray
    """
    search_size = int(sqrt(len(fingerprint)))

    return array(list(fingerprint)).reshape(search_size, search_size)


def motif_matrix_to_connections(motif_matrix):
    """
    Change the type from the motif matrix to the graph connections for agent training.

    :param motif_matrix: the matrix of motif.
    :type motif_matrix: numpy.ndarray

    :return: connections.
    :rtype: list
    """
    connections = []

    for tail_index in range(len(motif_matrix)):
        for head_index in range(len(motif_matrix[tail_index])):
            if motif_matrix[tail_index][head_index] != 0:
                connections += [tail_index, head_index, motif_matrix[tail_index][head_index]]

    return connections


def connections_to_motif_matrix(connections, search_size):
    """
    Change the type from the graph connections to the motif matrix for motif calculating.

    :param connections: graph connections.
    :type connections: list

    :param search_size: size of search.
    :type search_size: int

    :return: motif matrix.
    :rtype: numpy.ndarray
    """
    motif_matrix = [[0 for _ in range(search_size)] for _ in range(search_size)]

    for index in range(0, len(connections), 3):
        motif_matrix[connections[index]][connections[index + 1]] = connections[index + 2]

    return array(motif_matrix)


def is_same_motif(motif_1, motif_2):
    """
    Judge whether the two motifs are the same.

    :param motif_1: motif one.
    :type motif_1: numpy.ndarray

    :param motif_2: motif another.
    :type motif_2: numpy.ndarray

    :return: same judgement.
    :rtype: bool
    """
    if motif_1.shape != motif_2.shape:
        return False

    for rule in permutations([_ for _ in range(len(motif_1))], len(motif_1)):
        restructured_motif = zeros(shape=motif_2.shape)
        for tail_index in range(len(motif_1)):
            for head_index in range(len(motif_1[tail_index])):
                restructured_motif[rule[tail_index], rule[head_index]] = motif_1[tail_index, head_index]

        if all(restructured_motif == motif_2):
            return True

    return False


def obtain_motif(adjacency_matrix, combination, search_size):
    """
    Obtain a candidate motif of specific nodes in the adjacency matrix.

    :param adjacency_matrix: the adjacency matrix of the method.
    :type adjacency_matrix: numpy.ndarray

    :param combination: selected vertices in the adjacency matrix.
    :type combination: list or tuple

    :param search_size: size of search, which is same as the length of combination.
    :type search_size: int

    :return: obtained motif.
    :rtype: numpy.ndarray
    """
    motif = zeros(shape=(search_size, search_size))

    for index_1, node_id_1 in enumerate(combination):
        for index_2, node_id_2 in enumerate(combination):
            if adjacency_matrix[node_id_1, node_id_2] > 0:
                motif[index_1, index_2] = 1
            elif adjacency_matrix[node_id_1, node_id_2] < 0:
                motif[index_1, index_2] = -1

    return motif


def compliance_motif_specification(motif):
    """
    Test the rationality of the obtained motif.

    :param motif: an obtained motif.
    :type motif: numpy.ndarray

    :return: rationality of the motif.
    :rtype: bool
    """
    for node_index in range(len(motif)):
        if all(motif[node_index] == 0) and all(motif[:, node_index] == 0):
            return False

    return True


def collect_motifs(adjacency_matrix, search_size=3):
    """
    Collect all the rational motifs from the adjacency matrix.

    :param adjacency_matrix: the adjacency matrix of the method.
    :type adjacency_matrix: numpy.ndarray

    :param search_size: size of search.
    :type search_size: int

    :return: collected motif set.
    :rtype: dict
    """
    collector, saved_motif_matrices = {}, []
    for combination in combinations([node_id for node_id in range(len(adjacency_matrix))], search_size):
        motif = obtain_motif(adjacency_matrix=adjacency_matrix, combination=combination, search_size=search_size)
        if compliance_motif_specification(motif):
            is_found = False
            for saved_motif in saved_motif_matrices:
                if is_same_motif(saved_motif, motif):
                    is_found = True
                    collector[motif_matrix_to_fingerprint(saved_motif)] += 1
                    break
            if not is_found:
                saved_motif_matrices.append(motif)
                collector[motif_matrix_to_fingerprint(motif)] = 1

    results = []
    for fingerprint, count in collector.items():
        results.append((fingerprint_to_motif_matrix(fingerprint), count))

    return results


def count_motifs_from_adjacency_matrix(matrix, search_size, reference_motifs=None):
    """
    Count the rational motif frequencies from a given adjacency matrix.

    :param matrix: adjacency matrix of the neural network.
    :type matrix: numpy.ndarray

    :param search_size: size of search.
    :type search_size: int

    :param reference_motifs: reference motifs for order.
    :type reference_motifs: numpy.ndarray or list

    :return: motif frequencies.
    :rtype: numpy.ndarray
    """
    if reference_motifs is None:
        reference_motifs = []
        for motif_type in ["incoherent-loop", "coherent-loop", "collider"]:
            for acyclic_motif in acyclic_motifs[motif_type]:
                motif = zeros(shape=(3, 3), dtype=int)
                for former, latter in acyclic_motif.edges:
                    motif[former - 1, latter - 1] = acyclic_motif.get_edge_data(former, latter)["weight"]
                reference_motifs.append(motif)

    counts = zeros(shape=(len(reference_motifs),), dtype=int)
    for observed_motif, count in collect_motifs(adjacency_matrix=matrix, search_size=search_size):
        for reference_index, reference_motif in enumerate(reference_motifs):
            if is_same_motif(observed_motif, reference_motif):
                counts[reference_index] += count

    return counts


def detect_motifs_from_adjacency_matrix(matrix, search_size, detected_motifs):
    """
    Detect motifs from a given adjacency matrix.

    :param matrix: adjacency matrix of the neural network.
    :type matrix: numpy.ndarray

    :param search_size: size of search.
    :type search_size: int

    :param detected_motifs: motifs with the given search size to be detected.
    :type detected_motifs: list, tuple, or numpy.ndarray

    :return: detection flag.
    :rtype: bool
    """
    for observed_motif, _ in collect_motifs(adjacency_matrix=matrix, search_size=search_size):
        for detected_motif in detected_motifs:
            if is_same_motif(observed_motif, detected_motif):
                return True

    return False
