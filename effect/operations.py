from itertools import product
from numpy import array, zeros, expand_dims, swapaxes, repeat, vstack, sqrt, min, mean, sum, abs
from torch import cat, linspace, meshgrid, unsqueeze, squeeze

from effect.networks import NeuralMotif


def prepare_data(value_range, points=101):
    """
    Prepare database through the range of variable and sampling points.

    :param value_range: range of variable.
    :type value_range: tuple

    :param points: sampling points.
    :type points: int

    :return: database.
    :rtype: torch.Tensor
    """
    x, y = linspace(value_range[0], value_range[1], points), linspace(value_range[0], value_range[1], points)
    x, y = meshgrid((x, y), indexing="ij")
    x, y = unsqueeze(x.reshape(-1), dim=1), unsqueeze(y.reshape(-1), dim=1)

    data = cat((x, y), dim=1)

    return data


def prepare_motifs(motif_type, motif_index, activations, aggregations, sample, weights=None, biases=None):
    """
    Prepare motif based on the selected parameters.

    :param motif_type: type of 3-node network motif.
    :type motif_type: str

    :param motif_index: index of network motif (1 ~ 4).
    :type motif_index: int

    :param activations: activation function list.
    :type activations: tuple or list

    :param aggregations: aggregation function list.
    :type aggregations: tuple or list

    :param sample: generation number.
    :type sample: int

    :param weights: established weights.
    :type weights: tuple, list, or None

    :param biases: established biases.
    :type biases: tuple, list, or None

    :return: generated motif list.
    :rtype: list
    """
    motifs = []

    for sample_index in range(sample):
        if weights is not None and biases is not None:
            motif = None
            for flags in product([1.0, -1.0], repeat=len(weights)):
                try:
                    adjusted_weights = [flag * weight for flag, weight in zip(flags, weights)]
                    motif = NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                        activations=activations, aggregations=aggregations,
                                        weights=adjusted_weights, biases=biases)
                    break
                except ValueError:
                    pass

        else:
            motif = NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                activations=activations, aggregations=aggregations,
                                weights=None, biases=None)

        motifs.append(motif)

    return motifs


def calculate_landscape(value_range, points, motif):
    """
    Calculate the output landscape of the selected motif.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param motif: 3-node network motif in the artificial neural network.
    :type motif: effect.networks.NeuralMotif

    :return: output landscape.
    :rtype: numpy.ndarray
    """
    return motif(prepare_data(value_range=value_range, points=points)).reshape(points, points).detach().numpy()


def calculate_gradients(value_range, points, motif):
    """
    Calculate the gradient matrix of the selected motif.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param motif: 3-node network motif in the artificial neural network.
    :type motif: effect.networks.NeuralMotif

    :return: gradient matrix.
    :rtype: numpy.ndarray
    """
    sources = prepare_data(value_range=value_range, points=points)
    sources.requires_grad = True
    targets = motif(sources)
    gradients = zeros(shape=(points ** 2,))

    for index in range(points ** 2):
        # noinspection PyArgumentList
        squeeze(targets[index]).backward(retain_graph=True)
        values = sources.grad[index].detach().numpy()
        gradients[index] = sqrt(sum(values ** 2))

    return gradients.reshape(points, points)


def generate_qualified_motifs(motif_type, motif_index, activations, aggregations, weight_groups, bias_groups,
                              value_range, points, minimum_difference=None):
    """
    Generate qualified motif with specific requirements.

    :param motif_type: type of motif, i.e. "incoherent-loop", "coherent-loop", or "collider".
    :type motif_type: str

    :param motif_index: index (or sub-type) of motif, i.e. 1, 2, 3, or 4.
    :type motif_index: int

    :param activation: activation functions, i.e. "tanh", "sigmoid", or "relu".
    :type activation: list

    :param aggregations: aggregation functions, i.e. "sum" or "max".
    :type aggregations: list

    :param weight_groups: used weight value groups.
    :type weight_groups: list

    :param bias_groups: used bias value groups.
    :type bias_groups: list

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param minimum_difference: minimum difference (L1 loss) between qualified motifs.
    :type minimum_difference: float

    :return: qualified motifs and their corresponding output landscapes.
    :rtype: list, numpy.ndarray
    """
    signal_groups, saved_motifs = None, []
    for weights in product(*weight_groups):
        for biases in product(*bias_groups):
            motif = NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                activations=activations, aggregations=aggregations,
                                weights=weights, biases=biases)
            signals = calculate_landscape(value_range=value_range, points=points, motif=motif)
            signals = expand_dims(signals.reshape(-1), axis=0)
            if signal_groups is not None:
                if minimum_difference is not None:
                    if min(mean(abs(signal_groups - signals), axis=1)) > minimum_difference:
                        saved_motifs.append(motif)
                        signal_groups = vstack((signal_groups, signals))
                else:
                    saved_motifs.append(motif)
                    signal_groups = vstack((signal_groups, signals))
            else:
                saved_motifs.append(motif)
                signal_groups = signals

    return saved_motifs, signal_groups


def calculate_motif_differences(landscapes):
    """
    Calculate difference (L1 loss) between motif landscapes.

    :param landscapes: landscapes of given motifs.
    :type landscapes: numpy.ndarray

    :return: differences.
    :rtype: numpy.ndarray
    """
    former = repeat(expand_dims(landscapes, 0), len(landscapes), 0)
    latter = swapaxes(former.copy(), 0, 1)

    return mean(abs(former - latter), axis=2)


def calculate_population_differences(source_landscapes, target_landscapes):
    """
    Calculate minimum differences (L1 loss) between two motif populations.

    :param source_landscapes: motif landscapes in the source motif population.
    :type source_landscapes: numpy.ndarray

    :param target_landscapes: motif landscapes in the target (reference) motif population.
    :type target_landscapes: numpy.ndarray

    :return: minimum differences between two motif populations.
    :rtype numpy.ndarray
    """
    results, source_landscapes = [], expand_dims(source_landscapes, axis=1)

    for source in source_landscapes:
        matrix = abs(target_landscapes - source)
        distances = mean(matrix, axis=1)
        results.append(min(distances))

    return array(results)
