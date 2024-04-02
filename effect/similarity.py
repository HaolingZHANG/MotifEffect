"""
@Author      : Haoling Zhang
@Description : Definition of similarity
"""
from copy import deepcopy

from numpy import array, argmin
from torch import optim, nn
from typing import Tuple
from warnings import filterwarnings

from effect import Monitor
from effect.networks import NeuralMotif
from effect.operations import prepare_data

filterwarnings(action="ignore", category=UserWarning)


def execute_escape_processes(motif_pairs: list,
                             value_range: tuple,
                             points: int,
                             learn_rate: float,
                             thresholds: tuple,
                             verbose: bool = False) \
        -> list:
    """
    Execute the escape process for multiple pairs of an escape motif and several catch motifs.

    :param motif_pairs: list of pairs of source and target motifs.
    :type motif_pairs: list

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param learn_rate: learning rate to train each target motif as the source motif.
    :type learn_rate: float

    :param thresholds: maximum iteration of training source motif and target motif.
    :type thresholds: tuple

    :param verbose: need to show process log.
    :type verbose: bool

    :return: records of the escape processes.
    :rtype: list
    """
    records, monitor = [], Monitor()
    for sample_index, (escaper, catchers) in enumerate(motif_pairs):
        record = maximum_minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                             escaper=escaper, catchers=catchers, thresholds=thresholds)
        records.append(record)

        if verbose:
            monitor(sample_index + 1, len(motif_pairs))

    return records


def execute_catch_processes(references: list,
                            catchers: list,
                            value_range: tuple,
                            points: int,
                            learn_rate: float,
                            threshold: int,
                            verbose: bool = False) \
        -> list:
    """
    Execute the catching process for referenced motifs and several catch motifs.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param references: source motifs (incoherent loop or coherent loop in this work).
    :type references: list

    :param catchers: target motifs (collider in this work).
    :type catchers: list

    :param learn_rate: learning rate to train each target motif as the source motif.
    :type learn_rate: float

    :param threshold: maximum iteration of training the target motif.
    :type threshold: int

    :param verbose: need to show process log.
    :type verbose: bool

    :return: records of the catching processes.
    :rtype: list
    """
    records, monitor = [], Monitor()
    for sample_index, reference in enumerate(references):
        saved_motif, saved_loss = None, None
        for catcher in catchers:
            motifs, losses = minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                                 escaper=reference, catcher=deepcopy(catcher), threshold=threshold)
            losses = array(losses)
            location = argmin(losses)
            motif, loss = motifs[location], losses[location]
            if saved_loss is None or loss < saved_loss:
                saved_motif, saved_loss = motif, loss

        records.append((saved_motif, saved_loss))

        if verbose:
            monitor(sample_index + 1, len(references))

    return records


def maximum_minimum_loss_search(value_range: tuple,
                                points: int,
                                escaper: NeuralMotif,
                                catchers: list,
                                learn_rate: float,
                                thresholds: tuple,
                                verbose: bool = False) \
        -> Tuple[list, list]:
    """
    Find the maximum-minimum L2 loss (as the representation capacity bound) between source motif and target motifs.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param escaper: source motif (incoherent loop or coherent loop in this work).
    :type escaper: effect.networks.NeuralMotif

    :param catchers: target motifs (collider in this work).
    :type catchers: list

    :param learn_rate: learning rate to train each target motif as the source motif.
    :type learn_rate: float

    :param thresholds: maximum iteration of training source motif and target motif.
    :type thresholds: tuple

    :param verbose: need to show process log.
    :type verbose: bool

    :return: training results (motif sets and their losses during training).
    :rtype: list, list
    """
    record = {"motifs": [], "losses": []}
    input_signals = prepare_data(value_range=value_range, points=points)
    optimizer, criterion = optim.Adam(escaper.parameters(), lr=learn_rate), nn.MSELoss()
    source_iterations, target_iterations = thresholds

    for iteration in range(source_iterations):
        if verbose:
            print("*" * 80)
            print("ITERATION " + str(iteration + 1))
            print("-" * 80)
            print("We train the target motifs (to approach the source motif) using the gradient descent.")

        target_motifs, target_loss_record = [], []
        for target_index in range(len(catchers)):
            motifs, losses = minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                                 escaper=escaper, catcher=catchers[target_index],
                                                 threshold=target_iterations)
            target_motifs.append(motifs[-1])
            target_loss_record.append(losses[-1])

        choice = argmin(target_loss_record)  # choose the most similar target motif.
        target_motif, target_loss = target_motifs[choice], target_loss_record[choice]

        if verbose:
            print("The loss of each trained target motif is")
            print(array([list(range(1, len(target_loss_record) + 1)), target_loss_record]))
            print("We find the most similar \"" + target_motif.t + " " + str(target_motif.i) + "\" motif is")
            print(target_motif)
            print(str(choice + 1) + "-th motif in the target motifs " +
                  "with the minimum error %.6f.\n" % float(target_loss))

        source_output_signals, target_output_signals = escaper(input_signals), target_motif(input_signals)
        source_loss = criterion(source_output_signals, target_output_signals)
        optimizer.zero_grad()
        (-source_loss).backward(retain_graph=True)  # gradient ascent.
        optimizer.step()
        escaper.restrict()

        record["motifs"].append((deepcopy(escaper), deepcopy(target_motif)))
        record["losses"].append(float(criterion(escaper(input_signals), target_motif(input_signals))))

        if verbose:
            print("Reversely, we train the source motif (to away from the targets motifs) using the gradient ascent.")
            print(escaper)
            print("\nNow, the maximum-minimum loss is %.6f.\n" % record["losses"][-1])

    return record["motifs"], record["losses"]


def minimum_loss_search(value_range: tuple,
                        points: int,
                        escaper: NeuralMotif,
                        catcher: NeuralMotif,
                        learn_rate: float,
                        threshold: int) \
        -> Tuple[list, list]:
    """
    Train the target motif to achieve the source motif and find the minimum L2 loss between the two motifs.

    :param value_range: definition field of two input signals.
    :type value_range: tuple

    :param points: number of equidistant sampling in the definition field.
    :type points: int

    :param escaper: source motif as the reference (incoherent/coherent loop in this work).
    :type escaper: effect.networks.NeuralMotif

    :param catcher: target motif should be trained (collider in this work).
    :type catcher: effect.networks.NeuralMotif

    :param learn_rate: learning rate to train each target motif as the source motif.
    :type learn_rate: float

    :param threshold: maximum iteration of training the target motif.
    :type threshold: int

    :return: training motif_collection (trained motifs and losses during training).
    :rtype: list, list
    """
    record = {"motifs": [], "losses": []}
    optimizer, criterion = optim.Adam(catcher.parameters(), lr=learn_rate), nn.MSELoss()

    input_signals = prepare_data(value_range=value_range, points=points)
    source_output_signals = escaper(input_signals)

    for iteration in range(threshold):
        target_output_signals = catcher(input_signals)
        loss = criterion(source_output_signals, target_output_signals)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        catcher.restrict()

        record["motifs"].append(deepcopy(catcher))
        record["losses"].append(float(loss))

    return record["motifs"], record["losses"]
