"""
@Author      : Haoling Zhang
@Description : Data processing related to neural motif
"""
from itertools import product
from numpy import ndarray, array, zeros, ones, expand_dims, vstack
from numpy import min, mean, max, abs, sum, sqrt, power, cumproduct
from torch import Tensor, cat, linspace, meshgrid, unsqueeze, squeeze
from typing import Tuple, Union

from effect import Monitor
from effect.networks import NeuralMotif


def prepare_data(value_range: tuple,
                 points: int = 41) \
        -> Tensor:
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
    # noinspection PyTypeChecker
    x, y = meshgrid((x, y), indexing="ij")
    x, y = unsqueeze(x.reshape(-1), dim=1), unsqueeze(y.reshape(-1), dim=1)

    data = cat((x, y), dim=1)

    return data


def prepare_motifs(motif_type: str,
                   motif_index: int,
                   activations: Union[tuple, list],
                   aggregations: Union[tuple, list],
                   sample: int, weights: Union[tuple, list, None] = None,
                   biases: Union[tuple, list, None] = None,
                   weight_bound: Union[tuple, None] = None,
                   bias_bound: Union[tuple, None] = None) \
        -> list:
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

    :param weight_bound: bound of weight.
    :type weight_bound: tuple or None

    :param bias_bound: bound of bias.
    :type bias_bound: tuple or None

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
                                        weights=adjusted_weights, biases=biases,
                                        weight_bound=weight_bound, bias_bound=bias_bound)
                    break
                except ValueError:
                    pass

        else:
            motif = NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                activations=activations, aggregations=aggregations,
                                weight_bound=weight_bound, bias_bound=bias_bound)

        motifs.append(motif)

    return motifs


def calculate_landscape(value_range: tuple,
                        points: int,
                        motif: NeuralMotif) \
        -> ndarray:
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


def calculate_gradients(value_range: tuple,
                        points: int,
                        motif: NeuralMotif) \
        -> ndarray:
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


def generate_motifs(motif_type: str,
                    motif_index: int,
                    activations: Union[tuple, list],
                    aggregations: Union[tuple, list],
                    weight_groups: Union[tuple, list],
                    bias_groups: Union[tuple, list],
                    value_range: tuple,
                    points: int,
                    minimum_difference=None) \
        -> Tuple[list, ndarray]:
    """
    Generate qualified motif with specific requirements.

    :param motif_type: type of motif, i.e. "incoherent-loop", "coherent-loop", or "collider".
    :type motif_type: str

    :param motif_index: index (or subtype) of motif, i.e. 1, 2, 3, or 4.
    :type motif_index: int

    :param activations: activation function list, including "tanh", "sigmoid", or "relu".
    :type activations: tuple or list

    :param aggregations: aggregation function list, including "sum" or "max".
    :type aggregations: tuple or list

    :param weight_groups: used weight value groups.
    :type weight_groups: tuple or list

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
    signal_groups, saved_motifs, monitor = None, [], Monitor()
    count, total = 0, int(cumproduct([len(v) for v in weight_groups] + [len(v) for v in bias_groups])[-1])
    for weights in product(*weight_groups):
        for biases in product(*bias_groups):
            count += 1
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
            monitor(count, total, extra={"saved": len(saved_motifs)})

    return saved_motifs, signal_groups


def generate_outputs(motif_type: str,
                     motif_index: int,
                     activations: Union[tuple, list],
                     aggregations: Union[tuple, list],
                     weight_groups: Union[tuple, list],
                     bias_groups: Union[tuple, list],
                     value_range: tuple,
                     points: int,
                     verbose: bool = False) \
        -> Tuple[ndarray, ndarray]:
    """
    Generate all output landscapes and the corresponding parameters based on the given parameter domain.

    :param motif_type: type of motif, i.e. "incoherent-loop", "coherent-loop", or "collider".
    :type motif_type: str

    :param motif_index: index (or subtype) of motif, i.e. 1, 2, 3, or 4.
    :type motif_index: int

    :param activations: activation functions, i.e. "tanh", "sigmoid", or "relu".
    :type activations: tuple or list

    :param aggregations: aggregation functions, i.e. "sum" or "max".
    :type aggregations: tuple or list

    :param weight_groups: used weight value groups.
    :type weight_groups: tuple or list

    :param bias_groups: used bias value groups.
    :type bias_groups: tuple or list

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param verbose: need to show process log.
    :type verbose: bool

    :return: parameter list and output signal landscape list.
    :rtype: numpy.ndarray, numpy.ndarray
    """
    monitor, parameters, landscapes = Monitor(), [], []
    count, total = 0, int(cumproduct([len(v) for v in weight_groups] + [len(v) for v in bias_groups])[-1])
    for weights in product(*weight_groups):
        for biases in product(*bias_groups):
            count += 1
            motif = NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                activations=activations, aggregations=aggregations,
                                weights=weights, biases=biases)
            signals = calculate_landscape(value_range=value_range, points=points, motif=motif)
            signals = expand_dims(signals.reshape(-1), axis=0)
            parameters.append([weight.value() for weight in motif.w] + [bias.value() for bias in motif.b])
            landscapes.append(signals.reshape(-1).tolist())
            if verbose:
                monitor(count, total)

    return array(parameters), array(landscapes)


def calculate_differences(landscapes_1: ndarray,
                          landscapes_2: Union[ndarray, None] = None,
                          norm_type: str = "L-2",
                          verbose: bool = False) \
        -> ndarray:
    """
    Calculate differences between motif landscapes.

    :param landscapes_1: landscapes of given motifs.
    :type landscapes_1: numpy.ndarray

    :param landscapes_2: other landscapes of given motifs.
    :type landscapes_2: numpy.ndarray or None

    :param norm_type: norm type, including "L-1", "L-2", and "L-inf".
    :type norm_type: str

    :param verbose: need to show process log.
    :type verbose: bool

    :return: differences.
    :rtype: numpy.ndarray
    """
    differences, terminal, monitor = -ones(shape=len(landscapes_1)), len(landscapes_1), Monitor()

    if landscapes_2 is None:
        if norm_type == "L-1":
            for current, landscape in enumerate(landscapes_1):
                result = sum(abs(landscapes_1 - expand_dims(landscape, axis=0)), axis=1)
                result[current] = max(result) + 1
                differences[current] = min(result)

                if verbose:
                    monitor(current + 1, terminal)

        elif norm_type == "L-2":
            for current, landscape in enumerate(landscapes_1):
                result = sqrt(sum(power(landscapes_1 - expand_dims(landscape, axis=0), 2), axis=1))
                result[current] = max(result) + 1
                differences[current] = min(result)

                if verbose:
                    monitor(current + 1, terminal)

        elif norm_type == "L-inf":
            for current, landscape in enumerate(landscapes_1):
                result = max(abs(landscapes_1 - expand_dims(landscape, axis=0)), axis=1).astype(float)
                result[current] = max(result) + 1
                differences[current] = min(result)

                if verbose:
                    monitor(current + 1, terminal)

        else:
            raise ValueError("No such norm type!")

    else:
        if norm_type == "L-1":
            for current, landscape in enumerate(landscapes_1):
                result = sum(abs(landscapes_2 - expand_dims(landscape, axis=0)), axis=1)
                differences[current] = min(result)

                if verbose:
                    monitor(current + 1, terminal)

        elif norm_type == "L-2":
            for current, landscape in enumerate(landscapes_1):
                result = sqrt(sum(power(landscapes_2 - expand_dims(landscape, axis=0), 2), axis=1))
                differences[current] = min(result)

                if verbose:
                    monitor(current + 1, terminal)

        elif norm_type == "L-inf":
            for current, landscape in enumerate(landscapes_1):
                result = max(abs(landscapes_2 - expand_dims(landscape, axis=0)), axis=1).astype(float)
                differences[current] = min(result)

                if verbose:
                    monitor(current + 1, terminal)

        else:
            raise ValueError("No such norm type!")

    return differences
