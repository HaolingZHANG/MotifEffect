from datetime import datetime
from itertools import product
from numpy import zeros, sqrt
from torch import cat, linspace, meshgrid, unsqueeze, squeeze

from effect.networks import NeuralMotif


class Monitor(object):

    def __init__(self):
        """
        Initialize the monitor to identify the task progress.
        """
        self.last_time = None

    def __call__(self, current_state, total_state, extra=None):
        """
        Output the current state of process.

        :param current_state: current state of process.
        :type current_state: int

        :param total_state: total state of process.
        :type total_state: int

        :param extra: extra vision information if required.
        :type extra: dict
        """
        if self.last_time is None:
            self.last_time = datetime.now()

        if current_state == 0:
            return

        position = int(current_state / total_state * 100)

        string = "|"

        for index in range(0, 100, 5):
            if position >= index:
                string += "█"
            else:
                string += " "

        string += "|"

        pass_time = (datetime.now() - self.last_time).total_seconds()
        wait_time = int(pass_time * (total_state - current_state) / current_state)

        string += " " * (3 - len(str(position))) + str(position) + "% ("

        string += " " * (len(str(total_state)) - len(str(current_state))) + str(current_state) + "/" + str(total_state)

        if current_state < total_state:
            minute, second = divmod(wait_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") wait " + "%04d:%02d:%02d" % (hour, minute, second)
        else:
            minute, second = divmod(pass_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") used " + "%04d:%02d:%02d" % (hour, minute, second)

        if extra is not None:
            string += " " + str(extra).replace("\'", "").replace("{", "(").replace("}", ")") + "."
        else:
            string += "."

        print("\r" + string, end="", flush=True)

        if current_state >= total_state:
            self.last_time = None
            print()


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


def calculate_gradients(value_range, points, motif, verbose=False):
    """
    Calculate the gradient matrix of the selected motif.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param motif: 3-node network motif in the artificial neural network.
    :type motif: effect.networks.NeuralMotif

    :param verbose: need to show process log.
    :type verbose: bool

    :return: gradient matrix.
    :rtype: numpy.ndarray
    """
    sources = prepare_data(value_range=value_range, points=points)
    sources.requires_grad = True
    targets = motif(sources)
    gradients, monitor = zeros(shape=(points ** 2,)), Monitor()

    if verbose:
        print("estimate the gradient in the current resolution of the landscape.")

    for index in range(points ** 2):
        # noinspection PyArgumentList
        squeeze(targets[index]).backward(retain_graph=True)
        values = sources.grad[index].detach().numpy()
        gradients[index] = sqrt(sum(values ** 2))
        if verbose:
            monitor(index + 1, points ** 2, extra={"source": gradients[index]})

    return gradients.reshape(points, points)
