from numpy import array, zeros, ones, linspace, abs, sum, min, mean, median, max, sqrt, vstack

from effect.operations import Monitor, calculate_landscape


def calculate_rugosity(value_range, points, motif):
    """
    Calculate rugosity index of the output landscape through the selected motif.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param motif: 3-node network motif in the artificial neural network.
    :type motif: hypothesis.networks.NeuralMotif

    :return: rugosity index.
    :rtype: float
    """
    value_size = value_range[1] - value_range[0]
    output = calculate_landscape(value_range=value_range, points=points, motif=motif)

    temp_0, temp_1 = zeros(shape=((points - 1) ** 2,)), ones(shape=((points - 1) ** 2,)) * value_size / (points - 1)

    locations = array([vstack((temp_0, temp_0, output[0:points - 1, 0:points - 1].reshape(-1))).T,
                       vstack((temp_0, temp_1, output[1:points - 0, 0:points - 1].reshape(-1))).T,
                       vstack((temp_1, temp_1, output[1:points - 0, 1:points - 0].reshape(-1))).T,
                       vstack((temp_1, temp_0, output[0:points - 1, 1:points - 0].reshape(-1))).T])
    lengths = array([sqrt(sum((locations[0] - locations[1]) ** 2, axis=1)),
                     sqrt(sum((locations[1] - locations[2]) ** 2, axis=1)),
                     sqrt(sum((locations[2] - locations[3]) ** 2, axis=1)),
                     sqrt(sum((locations[3] - locations[0]) ** 2, axis=1))])

    semiperimeters = sum(lengths, axis=0) * 0.5
    surface_area = sum(sqrt((semiperimeters - lengths[0]) * (semiperimeters - lengths[1]) *
                            (semiperimeters - lengths[2]) * (semiperimeters - lengths[3])))

    return float(surface_area / (value_size ** 2.0))


# noinspection PyArgumentList,PyTypeChecker
def evaluate_propagation(value_range, points, motif, compute_type="max", verbose=False):
    """
    Evaluate the error propagation through the selected motif.

    :param value_range:

    :param points:

    :param motif:

    :param compute_type:

    :param verbose:

    :return:
    """
    output = calculate_landscape(value_range=value_range, points=points, motif=motif)
    propagation, monitor, current = zeros(shape=(points, points)), Monitor(), 0
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

                if verbose:
                    monitor.output(current + 1, points ** 2)
                    current += 1

    return propagation
