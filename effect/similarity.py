from copy import deepcopy
from numpy import array, argmin, ptp, min
from torch import optim, nn
from warnings import filterwarnings

from effect.operations import Monitor, prepare_data

filterwarnings(action="ignore", category=UserWarning)


def maximum_minimum_loss_search(value_range, points, source_motif, target_motifs,
                                learn_rate, loss_threshold, check_threshold, iteration_thresholds, verbose=False):
    """
    Find the maximum-minimum L1 loss (as the representation capacity bound) between source motif and target motifs.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param source_motif: source motif (incoherent loop in this work).
    :type source_motif: effect.networks.NeuralMotif

    :param target_motifs: target motifs (collider in this work).
    :type target_motifs: list

    :param learn_rate: learning rate to train each target motif as the source motif.
    :type learn_rate: float

    :param loss_threshold: training termination condition, range (maximum - minimum) threshold of last n losses.
    :type loss_threshold: float

    :param check_threshold: check loss number, combining with the parameter "loss_threshold".
    :type check_threshold: int

    :param iteration_thresholds: maximum iteration of training source motif and target motif.
    :type iteration_thresholds: tuple

    :param verbose: need to show process log.
    :type verbose: bool

    :return: training motif_collection (motif sets, losses during training, Lipschitz constants, rugosity indices).
    :rtype: list, list, list, list
    """
    record = {"motifs": [], "losses": []}
    input_signals = prepare_data(value_range=value_range, points=points)
    optimizer, criterion, monitor = optim.Adam(source_motif.parameters(), lr=learn_rate), nn.L1Loss(), Monitor()
    source_iterations, target_iterations = iteration_thresholds

    for iteration in range(source_iterations):
        if verbose:
            print("*" * 80)
            print("ITERATION " + str(iteration + 1))
            print("-" * 80)
            print("We train the target motifs (to approach the source motif) using the gradient descent.")

        target_loss_record = []
        for target_index in range(len(target_motifs)):
            motifs, losses = minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                                 source_motif=source_motif, target_motif=target_motifs[target_index],
                                                 loss_threshold=loss_threshold, check_threshold=check_threshold,
                                                 iteration_threshold=target_iterations, verbose=False)
            target_motifs[target_index] = motifs[-1]
            target_loss_record.append(losses[-1])
            if verbose:
                monitor(target_index + 1, len(target_motifs))

        choice = argmin(target_loss_record)  # choose the most similar target motif.
        target_motif, target_loss = target_motifs[choice], min(target_loss_record)

        if verbose:
            print("The loss of each trained target motif is")
            print(array([list(range(1, len(target_loss_record) + 1)), target_loss_record]))
            print("We find the most similar \"" + target_motif.t + " " + str(target_motif.i) + "\" motif is")
            print(target_motif)
            print(str(choice + 1) + "-th motif in the target motifs " +
                  "with the minimum error %.6f.\n" % float(target_loss))

        source_output_signals, target_output_signals = source_motif(input_signals), target_motif(input_signals)
        source_loss = criterion(source_output_signals, target_output_signals)
        optimizer.zero_grad()
        (-source_loss).backward(retain_graph=True)  # gradient ascent.
        optimizer.step()
        source_motif.restrict()

        record["motifs"].append((deepcopy(source_motif), deepcopy(target_motif)))
        record["losses"].append(float(criterion(source_motif(input_signals), target_motif(input_signals))))

        if verbose:
            print("Reversely, we train the source motif (to away from the targets motifs) using the gradient ascent.")
            print(source_motif)
            print("\nNow, the maximum-minimum loss is %.6f.\n" % record["losses"][-1])

        if iteration > check_threshold and ptp(record["losses"][-check_threshold:]) < loss_threshold:
            break

    return record["motifs"], record["losses"]


def minimum_loss_search(value_range, points, source_motif, target_motif,
                        learn_rate, loss_threshold, check_threshold, iteration_threshold, verbose=True):
    """
    Train the target motif to achieve the source motif and find the minimum L1 loss between the two motifs.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param source_motif: source motif as the reference (incoherent loop in this work).
    :type source_motif: effect.networks.NeuralMotif

    :param target_motif: target motif should be trained (collider in this work).
    :type target_motif: effect.networks.NeuralMotif

    :param learn_rate: learning rate to train each target motif as the source motif.
    :type learn_rate: float

    :param loss_threshold: training termination condition, range (maximum - minimum) threshold of last n losses.
    :type loss_threshold: float

    :param check_threshold: check loss number, combining with the parameter "loss_threshold".
    :type check_threshold: int

    :param iteration_threshold: maximum iteration of training the target motif.
    :type iteration_threshold: int

    :param verbose: need to show process log.
    :type verbose: bool

    :return: training motif_collection (trained motifs and losses during training).
    :rtype: list, list, list, list
    """
    record, monitor = {"motifs": [], "losses": []}, Monitor()
    optimizer, criterion = optim.Adam(target_motif.parameters(), lr=learn_rate), nn.L1Loss()

    input_signals = prepare_data(value_range=value_range, points=points)
    source_output_signals = source_motif(input_signals)

    for iteration in range(iteration_threshold):
        target_output_signals = target_motif(input_signals)
        loss = criterion(source_output_signals, target_output_signals)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        target_motif.restrict()

        record["motifs"].append(deepcopy(target_motif))
        record["losses"].append(float(loss))

        if verbose:
            monitor(iteration + 1, iteration_threshold, extra={"loss": record["losses"][-1]})

        if iteration > check_threshold and ptp(record["losses"][-check_threshold:]) < loss_threshold:
            if verbose:
                monitor(iteration_threshold, iteration_threshold, extra={"loss": record["losses"][-1]})
            break

    return record["motifs"], record["losses"]
