from enum import Enum
from itertools import combinations, permutations
from networkx import erdos_renyi_graph
from numpy import zeros, array, all, percentile, mean, std, sqrt, maximum
from random import choice


class GraphType(Enum):
    zo = 0  # zero and one
    pn = 1  # positive and negative


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


def find_same_motifs(motif_matrices):
    """
    Find same motifs in the motif list.

    :param motif_matrices: motif list.
    :type motif_matrices: numpy.ndarray or list

    :return: same groups.
    :rtype: list
    """
    same_pairs = []

    for (index_1, index_2) in combinations([_ for _ in range(len(motif_matrices))], 2):
        if is_same_motif(motif_matrices[index_1], motif_matrices[index_2]):
            same_pairs.append((index_1, index_2))

    if len(same_pairs) > 0:
        groups = [set(same_pairs[0])]
        for graph_index_1, graph_index_2 in same_pairs[1:]:
            found = False
            for group_index in range(len(groups)):
                if graph_index_1 in groups[group_index] or graph_index_2 in groups[group_index]:
                    groups[group_index].add(graph_index_1)
                    groups[group_index].add(graph_index_2)
                    found = True
                    break
            if not found:
                groups.append({graph_index_1, graph_index_2})

        same_groups = []
        for group in groups:
            same_groups.append(sorted(list(group)))

        return same_groups

    return None


def obtain_motif(adjacency_matrix, combination, search_size, graph_type=GraphType.pn, limits=None):
    """
    Obtain a candidate motif of specific nodes in the adjacency matrix.

    :param adjacency_matrix: the adjacency matrix of the method.
    :type adjacency_matrix: numpy.ndarray

    :param combination: selected vertices in the adjacency matrix.
    :type combination: list or tuple

    :param search_size: size of search, which is same as the length of combination.
    :type search_size: int

    :param graph_type: type of graph.
    :type graph_type: grace1.motifs.GraphType

    :param limits: bounds for collecting motifs.
    :type limits: list

    :return: obtained motif.
    :rtype: numpy.ndarray
    """
    motif = zeros(shape=(search_size, search_size))

    for index_1, node_id_1 in enumerate(combination):
        for index_2, node_id_2 in enumerate(combination):
            if graph_type == GraphType.zo:
                if adjacency_matrix[node_id_1][node_id_2] != 0:
                    if limits is None:
                        motif[index_1, index_2] = 1
                    elif limits[0] <= adjacency_matrix[node_id_1][node_id_2] <= limits[1]:
                        motif[index_1, index_2] = 1
            elif graph_type == GraphType.pn:
                if adjacency_matrix[node_id_1][node_id_2] > 0:
                    if limits is None:
                        motif[index_1, index_2] = 1
                    elif adjacency_matrix[node_id_1][node_id_2] <= limits[1]:
                        motif[index_1, index_2] = 1
                elif adjacency_matrix[node_id_1][node_id_2] < 0:
                    if limits is None:
                        motif[index_1, index_2] = -1
                    elif limits[0] <= adjacency_matrix[node_id_1][node_id_2]:
                        motif[index_1, index_2] = -1
            else:
                raise ValueError("No such graph type.")

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


def collect_motifs(adjacency_matrix, search_size=3, graph_type=GraphType.zo, pruning=False):
    """
    Collect all the rational motifs from the adjacency matrix.

    :param adjacency_matrix: the adjacency matrix of the method.
    :type adjacency_matrix: numpy.ndarray

    :param search_size: size of search.
    :type search_size: int

    :param graph_type: type of graph.
    :type graph_type: grace1.motifs.GraphType

    :param pruning: pruning the arrows by box plot.
    :type pruning: bool

    :return: collected motif set.
    :rtype: dict
    """
    limits = None
    if pruning:
        values = array(adjacency_matrix).reshape(-1)
        if len(values[values != 0]) > 0:
            percentiles = percentile(values[values != 0], [25, 75])
            iqr = percentiles[1] - percentiles[0]
            limits = [percentiles[0] - iqr * 1.5, percentiles[1] + iqr * 1.5]

    collector, saved_motif_matrices = {}, []
    for combination in combinations([node_id for node_id in range(len(adjacency_matrix))], search_size):
        motif = obtain_motif(adjacency_matrix=adjacency_matrix, combination=combination, search_size=search_size,
                             graph_type=graph_type, limits=limits)
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


def count_motifs_from_matrices(matrices, search_size, graph_type, pruning, reference_motifs=None):
    """
    Statistics the rational motif frequencies from matrix groups.

    :param matrices: adjacency matrices of agents.
    :type matrices: list or numpy.ndarray

    :param search_size: size of search.
    :type search_size: int

    :param graph_type: type of graph.
    :type graph_type: grace1.motifs.GraphType

    :param pruning: pruning the arrows by box plot.
    :type pruning: bool

    :param reference_motifs: reference motifs for order.
    :type reference_motifs: numpy.ndarray or None

    :return: motif frequency collector, and the collected motifs if there is.
    :rtype: tuple or numpy.ndarray
    """
    collectors = []
    for matrix in matrices:
        collectors.append(collect_motifs(adjacency_matrix=matrix, search_size=search_size,
                                         graph_type=graph_type, pruning=pruning))

    collected_motifs = []
    for collector in collectors:
        for motif, _ in collector:
            is_found = False
            for collected_motif in collected_motifs:
                if is_same_motif(collected_motif, motif):
                    is_found = True
                    break
            if not is_found:
                collected_motifs.append(motif)

    count_collector = []
    for collector in collectors:
        counts = [0 for _ in range(len(collected_motifs))]
        for motif, count in collector:
            for motif_index, collected_motif in enumerate(collected_motifs):
                if is_same_motif(collected_motif, motif):
                    counts[motif_index] += count
        count_collector.append(counts)

    if reference_motifs is not None:
        motif_mappings = []
        for reference_motif in reference_motifs:
            for index, motif in enumerate(collected_motifs):
                if is_same_motif(reference_motif, motif):
                    motif_mappings.append(index)
                    break
        new_count_collector = []
        for counts in count_collector:
            new_counts = [0 for _ in range(len(reference_motifs))]
            for motif_index, motif_mapping in enumerate(motif_mappings):
                new_counts[motif_index] = counts[motif_mapping]
            new_count_collector.append(new_counts)

        return array(new_count_collector, dtype=int)
    else:
        return array(count_collector, dtype=int), collected_motifs


def generate_random_graphs(random_graph_count, variable_number, graph_type, probability=0.5, verbose=False):
    """
    Generate adjacency matrix of random graphs.

    :param random_graph_count: count for generating random graphs.
    :type random_graph_count: int

    :param variable_number: variable number of the reference adjacency matrix.
    :type variable_number: int

    :param graph_type: type of graph.
    :type graph_type: grace1.motifs.GraphType

    :param probability: probability for arc creation.
    :type probability: float

    :param verbose: need to print log.
    :type verbose: bool

    :return: random adjacency matrices.
    :rtype: list
    """
    random_matrices = []

    if verbose:
        print("Generate random graph.")

    for random_id in range(random_graph_count):
        graph = erdos_renyi_graph(n=variable_number, p=probability, directed=True)
        random_matrix = zeros(shape=(variable_number, variable_number))
        for tail, head in graph.edges:
            if graph_type == GraphType.zo:
                random_matrix[tail, head] = 1
            elif graph_type == GraphType.pn:
                random_matrix[tail, head] = choice([-1, 1])
            else:
                raise ValueError("No such graph type!")

        random_matrices.append(random_matrix)

    return random_matrices


def calculate_z_scores(reference_motifs, adjacency_matrix, random_graph_count, search_size,
                       graph_type=GraphType.zo, pruning=False, verbose=False):
    """
    Calculate z-scores from the adjacency matrix.

    :param reference_motifs: reference motifs used for the order of statistical results.
    :type reference_motifs: numpy.ndarray

    :param adjacency_matrix: the adjacency matrix of the method.
    :type adjacency_matrix: numpy.ndarray

    :param random_graph_count: count for generating random graphs.
    :type random_graph_count: int

    :param search_size: size of search.
    :type search_size: int

    :param graph_type: type of graph.
    :type graph_type: grace1.motifs.GraphType

    :param pruning: pruning the arrows by box plot.
    :type pruning: bool

    :param verbose: need to print log.
    :type verbose: bool

    :return: z-scores collector.
    :rtype: list
    """
    original_collector = collect_motifs(adjacency_matrix, search_size, graph_type, pruning)
    original_counts = [0 for _ in range(len(reference_motifs))]
    for motif, value in original_collector:
        is_found = False
        for index, reference_motif in enumerate(reference_motifs):
            if is_same_motif(motif, reference_motif):
                is_found = True
                original_counts[index] += value
                break
        if not is_found:
            raise ValueError("The types of motif in \"reference_motifs\" are incomplete!")

    random_count_groups = []
    random_matrices = generate_random_graphs(random_graph_count, variable_number=len(adjacency_matrix),
                                             graph_type=graph_type, probability=0.5, verbose=verbose)
    if verbose:
        print("Calculate the motif frequency of random graphs.")

    for random_index, random_matrix in enumerate(random_matrices):
        random_collector = collect_motifs(random_matrix, search_size, graph_type, pruning)
        random_counts = [0 for _ in range(len(reference_motifs))]
        for random_motif, count in random_collector:
            for index, reference_motif in enumerate(reference_motifs):
                if is_same_motif(random_motif, reference_motif):
                    random_counts[index] += count

        random_count_groups.append(random_counts)

    original_counts = array(original_counts).T
    random_count_groups = array(random_count_groups).T
    score_collector = []
    for index, (motif, _) in enumerate(original_collector):
        source = original_counts[index]
        mean_value = mean(random_count_groups[index])
        std_value = maximum(std(random_count_groups[index]), 1e-10)
        score_collector.append((motif, (source - mean_value) / std_value))

    return score_collector
