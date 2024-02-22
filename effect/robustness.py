"""
@Author      : Haoling Zhang
@Description : Definition of robustness
"""
from numpy import ndarray, zeros, abs, min, mean, median, max, sqrt, vstack

from effect.networks import NeuralMotif
from effect.operations import calculate_landscape


def evaluate_propagation(value_range: tuple,
                         points: int,
                         motif: NeuralMotif,
                         compute_type: str = "max") \
        -> ndarray:
    """
    Evaluate the error propagation through the selected motif.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param motif: 3-node network motif in the artificial neural network.
    :type motif: effect.networks.NeuralMotif

    :param compute_type: type to evaluating the error propagation, including "max", "mean", "median", and "min".
    :type compute_type: str

    :return: propagation matrix.
    :rtype: numpy.ndarray
    """
    output = calculate_landscape(value_range=value_range, points=points, motif=motif)
    propagation = zeros(shape=(points, points))

    if max(output) - min(output) > 0.0:
        for index_1 in range(points):
            for index_2 in range(points):
                values = ((output[index_1:, index_2:] - output[:points - index_1, :points - index_2]).reshape(-1),
                          (output[index_1:, :points - index_2] - output[:points - index_1, index_2:]).reshape(-1),
                          (output[:points - index_1, index_2:] - output[index_1:, :points - index_2]).reshape(-1),
                          (output[:points - index_1, :points - index_2] - output[index_1:, index_2:]).reshape(-1))

                if compute_type == "max":
                    propagation[index_1, index_2] = max(abs(vstack(values))) / (max(output) - min(output))
                elif compute_type == "mean":
                    propagation[index_1, index_2] = mean(abs(vstack(values))) / (max(output) - min(output))
                elif compute_type == "median":
                    propagation[index_1, index_2] = median(abs(vstack(values))) / (max(output) - min(output))
                elif compute_type == "min":
                    propagation[index_1, index_2] = min(abs(vstack(values))) / (max(output) - min(output))
                else:
                    raise ValueError("No such computing type!")

    return propagation


def estimate_lipschitz(value_range: tuple,
                       points: int,
                       output: ndarray,
                       norm_type: str = "L-2") \
        -> float:
    """
    Estimate the Lipschitz constant of the output signals produced by selected motif.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param output: output signals produced by the artificial neural network.
    :type output: numpy.ndarray

    :param norm_type: norm type, including "L-1", "L-2", and "L-inf".
    :type norm_type: str

    :return: estimated Lipschitz constant.
    :rtype: float
    """
    value_interval, constant = (value_range[1] - value_range[0]) / (points - 1), 0.0

    if max(output) - min(output) > 0.0:
        for index_1 in range(points):
            for index_2 in range(points):
                if index_1 + index_2 > 0:
                    value_1, value_2 = index_1 * value_interval, index_2 * value_interval
                    values = ((output[index_1:, index_2:] - output[:points - index_1, :points - index_2]).reshape(-1),
                              (output[index_1:, :points - index_2] - output[:points - index_1, index_2:]).reshape(-1),
                              (output[:points - index_1, index_2:] - output[index_1:, :points - index_2]).reshape(-1),
                              (output[:points - index_1, :points - index_2] - output[index_1:, index_2:]).reshape(-1))
                    maximum_output_difference = max(vstack(values))

                    if norm_type == "L-1":
                        input_difference = value_1 + value_2
                    elif norm_type == "L-2":
                        input_difference = sqrt(value_1 ** 2 + value_2 ** 2)
                    elif norm_type == "L-inf":
                        input_difference = max([value_1, value_2])
                    else:
                        raise ValueError("No such norm type!")

                    constant = max([constant, maximum_output_difference / input_difference])

    return constant


def estimate_lipschitz_by_motif(value_range: tuple,
                                points: int,
                                motif: NeuralMotif,
                                norm_type: str = "L-2") \
        -> float:
    """
    Estimate the Lipschitz constant of the selected motif.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param motif: 3-node network motif in the artificial neural network.
    :type motif: effect.networks.NeuralMotif

    :param norm_type: norm type, including "L-1", "L-2", and "L-inf".
    :type norm_type: str

    :return: estimated Lipschitz constant.
    :rtype: float
    """
    output = calculate_landscape(value_range=value_range, points=points, motif=motif)

    return estimate_lipschitz(value_range=value_range, points=points, output=output, norm_type=norm_type)
