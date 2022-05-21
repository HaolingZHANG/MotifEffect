from copy import deepcopy
from multiprocessing.pool import Pool
from numpy import sum
from pickle import dump
from torch import abs, manual_seed, optim, nn
from warnings import filterwarnings

from hypothesis import Monitor
from hypothesis.operations import prepare_data

filterwarnings(action="ignore", category=UserWarning)


def calculate_similarity(value_range, points, source_motif_group, target_motif_group,
                         threshold, save_path, seed=2022, processes=1):
    data = prepare_data(value_range=value_range, points=points)
    learning_rate = (value_range[1] - value_range[0]) / (points - 1)

    if processes == 1:
        for source_motifs in source_motif_group:
            maximum_minimum_search(data, source_motifs, target_motif_group, learning_rate,
                                   threshold, seed, save_path, True)
    else:
        pool = Pool(processes=processes)
        for source_motifs in source_motif_group:
            pool.apply_async(func=maximum_minimum_search,
                             args=(data, source_motifs, target_motif_group, learning_rate,
                                   threshold, seed, save_path, False))
        pool.close()
        pool.join()


def maximum_minimum_search(input_data, source_motifs, target_motif_groups, learning_rate,
                           threshold, seed=None, save_path=None, verbose=False):
    records = []
    if seed is not None:
        manual_seed(seed=seed)
    for source_motif in source_motifs:
        source_motif.reset()
        if verbose:
            print("initialized \"" + source_motif.t + " " + str(source_motif.i) + "\" motif is")
            print(str(source_motif) + "\n\n\n")

        optimizer = optim.Adam(source_motif.parameters(), lr=learning_rate * 1e-2)
        criterion = nn.HuberLoss(delta=learning_rate)
        record, iteration, former_loss = [], 1, None

        while True:
            if verbose:
                print("*" * 80)
                print("iteration " + str(iteration))
                print("-" * 80 + "\n")

            target_motif, target_loss = minimum_search(input_data=input_data, source_motif=source_motif,
                                                       target_motif_group=target_motif_groups,
                                                       learning_rate=learning_rate, threshold=threshold,
                                                       seed=seed, save_path=None, verbose=verbose)

            source_data, target_data = source_motif(input_data), target_motif(input_data)

            record.append((deepcopy(source_motif), deepcopy(target_motif), target_loss))

            latter_loss = criterion(source_data, target_data)
            if former_loss is not None:
                if abs(former_loss - latter_loss) < threshold:
                    if verbose:
                        # noinspection PyStringFormat
                        print("the maximum-minimum Huber error is %.6e\n" % former_loss)
                        print("*" * 80)
                    break

            optimizer.zero_grad()
            # noinspection PyArgumentList
            (-latter_loss).backward(retain_graph=True)
            optimizer.step()
            former_loss = latter_loss
            if verbose:
                print("find maximum gap: training \"" + source_motif.t + "\" motif by the gradient ascent.")
                print(str(source_motif) + "\n")
                print("*" * 80 + "\n\n")

            iteration += 1

        records.append(record)

    if save_path is None:
        return records
    else:
        info = source_motifs[0].t + " " + str(source_motifs[0].i) + " "
        info += str(source_motifs[0].a).replace("\'", "").replace(",", "") + " "
        info += str(source_motifs[0].g).replace("\'", "").replace(",", "")

        with open(save_path + info + ".pkl", "wb") as file:
            dump(obj=records, file=file)


def minimum_search(input_data, source_motif, target_motif_group, learning_rate, threshold, seed,
                   save_path=None, verbose=False):
    if seed is not None:
        manual_seed(seed=seed)
    source_data, best_motif, best_loss = source_motif(input_data), None, None
    total, monitor = sum([len(target_motifs) for target_motifs in target_motif_group]), Monitor()
    if verbose:
        print("find minimum gap: training \"" + target_motif_group[0][0].t + "\" motifs by the gradient descent.")

    process = 0
    for target_motifs in target_motif_group:
        for target_motif in target_motifs:
            target_motif.reset()
            optimizer = optim.Adam(target_motif.parameters(), lr=learning_rate * 1e-2)
            criterion = nn.HuberLoss(delta=learning_rate)
            former_loss = None

            while True:
                target_data = target_motif(input_data)
                latter_loss = criterion(source_data, target_data)
                if former_loss is not None and abs(former_loss - latter_loss) < threshold:
                    if best_loss is not None:
                        if best_loss > former_loss:
                            best_loss = former_loss
                            best_motif = deepcopy(target_motif)
                    else:
                        best_loss = former_loss
                        best_motif = deepcopy(target_motif)
                    break

                optimizer.zero_grad()
                # noinspection PyArgumentList
                latter_loss.backward(retain_graph=True)
                optimizer.step()
                former_loss = latter_loss

            if verbose:
                monitor.output(process + 1, total,  extra={"motif": target_motif.t + "-" + str(target_motif.i),
                                                           "current loss": "%.2e" % float(former_loss),
                                                           "best loss": "%.2e" % float(best_loss)})
                process += 1

    if verbose:
        print("the most similar \"" + best_motif.t + " " + str(best_motif.i) + "\" motif is")
        print(best_motif)
        # noinspection PyStringFormat
        print("with the minimum Huber error %.6e\n" % best_loss)

    if save_path is None:
        return best_motif, best_loss
    else:
        info = best_motif.t + " " + str(best_motif.i) + " "
        info += str(best_motif.a).replace("\'", "").replace(",", "") + " "
        info += str(best_motif.g).replace("\'", "").replace(",", "")

        with open(save_path + info + ".pkl", "wb") as file:
            dump(obj=(best_motif, best_loss), file=file)
