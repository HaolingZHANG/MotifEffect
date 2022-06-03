from copy import deepcopy
from multiprocessing.pool import Pool
from os.path import exists

from numpy import ptp
from pickle import dump, load
from torch import manual_seed, optim, nn
from warnings import filterwarnings

from hypothesis import Monitor
from hypothesis.operations import prepare_data

filterwarnings(action="ignore", category=UserWarning)


def calculate_similarity(value_range, points, source_motif_group, target_motif_group,
                         loss_threshold, check_threshold, iteration_threshold, save_path, seed=2022, processes=1):
    data = prepare_data(value_range=value_range, points=points)
    learn_rate = (value_range[1] - value_range[0]) / (points - 1) * 1e-2

    if processes == 1:
        for source_index, source_motifs in enumerate(source_motif_group):
            for target_index, target_motifs in enumerate(target_motif_group):
                maximum_minimum_search(data, source_motifs, target_motifs, source_index, target_index, learn_rate,
                                       loss_threshold, check_threshold, iteration_threshold, seed, save_path, True)
    else:
        pool = Pool(processes=processes)
        for source_index, source_motifs in enumerate(source_motif_group):
            for target_index, target_motifs in enumerate(target_motif_group):
                pool.apply_async(func=maximum_minimum_search,
                                 args=(data, source_motifs, target_motifs, source_index, target_index, learn_rate,
                                       loss_threshold, check_threshold, iteration_threshold, seed, save_path, False))
        pool.close()
        pool.join()


def maximum_minimum_search(input_data, source_motifs, target_motifs, source_index, target_index, learn_rate,
                           loss_threshold, check_threshold, iteration_threshold,
                           seed=None, save_path=None, verbose=False):
    if save_path is not None and exists(save_path + "max.s" + str(source_index) + "t" + str(target_index) + ".pkl"):
        with open(save_path + "max.s" + str(source_index) + "t" + str(target_index) + ".pkl") as file:
            return load(file=file)

    if seed is not None:
        manual_seed(seed=seed)

    data_records = []
    for source_motif in source_motifs:
        source_motif.reset()
        if verbose:
            print("initialized \"" + source_motif.t + " " + str(source_motif.i) + "\" motif is")
            print(str(source_motif) + "\n\n\n")

        optimizer, criterion = optim.Adam(source_motif.parameters(), lr=learn_rate), nn.HuberLoss(delta=learn_rate)
        data_record, loss_record, iteration = [], [], 1

        while True:
            if verbose:
                print("*" * 80)
                print("iteration " + str(iteration))
                print("-" * 80 + "\n")

            search_data = minimum_search(input_data=input_data, learn_rate=learn_rate, verbose=verbose,
                                         source_motif=source_motif, target_motifs=target_motifs,
                                         source_index=source_index, target_index=target_index,
                                         loss_threshold=loss_threshold, check_threshold=check_threshold,
                                         iteration_threshold=iteration_threshold, seed=seed, save_path=None)
            target_motif, target_loss, target_records = search_data
            data_record.append((deepcopy(source_motif), deepcopy(target_motif), target_records, float(target_loss)))

            source_data, target_data = source_motif(input_data), target_motif(input_data)
            loss = criterion(source_data, target_data)
            loss_record.append(float(loss))

            finished = False
            if iteration_threshold is not None and iteration == iteration_threshold:
                finished = True
            if len(loss_record) >= check_threshold and ptp(loss_record[-check_threshold:]) < loss_threshold:
                finished = True

            if finished:
                if verbose:
                    print("the maximum-minimum mean squared error is %.6e." % float(loss))
                    print("*" * 80)
                break

            optimizer.zero_grad()
            # noinspection PyArgumentList
            (-loss).backward(retain_graph=True)
            optimizer.step()
            source_motif.restrict()

            if verbose:
                name = source_motif.t.replace("-", " ")
                print("find maximum gap: training \"" + name + "\" motif by the gradient ascent.")
                print(str(source_motif) + "\n")
                print("*" * 80 + "\n\n")

            iteration += 1

        data_records.append(data_record)

    if save_path is not None:
        with open(save_path + "max.s" + str(source_index) + "t" + str(target_index) + ".pkl", "wb") as file:
            dump(obj=data_records, file=file)

    return data_records


def minimum_search(input_data, source_motif, target_motifs, source_index, target_index, learn_rate,
                   loss_threshold, check_threshold, iteration_threshold,
                   seed, save_path=None, verbose=False):
    if save_path is not None and exists(save_path + "min.s" + str(source_index) + "t" + str(target_index) + ".pkl"):
        with open(save_path + "min.s" + str(source_index) + "t" + str(target_index) + ".pkl", "rb") as file:
            return load(file=file)

    if seed is not None:
        manual_seed(seed=seed)

    source_data, best_motif, best_loss, best_record, monitor = source_motif(input_data), None, None, None, Monitor()
    if verbose:
        name = target_motifs[0].t.replace("-", " ")
        print("find minimum gap: training \"" + name + "\" motifs by the gradient descent.")

    for process, target_motif in enumerate(target_motifs):
        target_motif.reset()
        optimizer, criterion = optim.Adam(target_motif.parameters(), lr=learn_rate), nn.HuberLoss(delta=learn_rate)
        iteration, loss_record = 1, []

        while True:
            target_data = target_motif(input_data)
            loss = criterion(source_data, target_data)
            loss_record.append(float(loss))
            finished = False
            if iteration_threshold is not None and iteration == iteration_threshold:
                finished = True
            if len(loss_record) >= check_threshold and ptp(loss_record[-check_threshold:]) < loss_threshold:
                finished = True
            if finished:
                if best_loss is not None:
                    if best_loss > loss:
                        best_loss, best_motif, best_record = loss, deepcopy(target_motif), loss_record
                else:
                    best_loss, best_motif, best_record = loss, deepcopy(target_motif), loss_record
                break

            optimizer.zero_grad()
            # noinspection PyArgumentList
            loss.backward(retain_graph=True)
            optimizer.step()
            target_motif.restrict()
            iteration += 1

        if verbose:
            monitor.output(process + 1, len(target_motifs),
                           extra={"motif": target_motif.t + "-" + str(target_motif.i),
                                  "current loss": "%.2e" % float(loss_record[-1]),
                                  "best loss": "%.2e" % float(best_loss)})

    if verbose:
        print("the most similar \"" + best_motif.t + " " + str(best_motif.i) + "\" motif is")
        print(best_motif)
        print("with the minimum mean squared error %.6e." % float(best_loss))

    if save_path is not None:
        with open(save_path + "min.s" + str(source_index) + "t" + str(target_index) + ".pkl", "wb") as file:
            dump(obj=(best_motif, float(best_loss), best_record), file=file)

    return best_motif, float(best_loss), best_record
