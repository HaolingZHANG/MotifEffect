from copy import deepcopy
from itertools import product
from pickle import dump
from torch import cat, linspace, meshgrid, unsqueeze, manual_seed
from torch.nn import MSELoss
from torch.optim import Adam

from hypothesis import NeuralMotif
from monitor import Monitor


def prepare_data(minimum_value=-1, maximum_value=+1, times=101):
    x, y = linspace(minimum_value, maximum_value, times), linspace(minimum_value, maximum_value, times)
    x, y = meshgrid([x, y])
    x, y = unsqueeze(x.reshape(-1), dim=1), unsqueeze(y.reshape(-1), dim=1)
    return cat((x, y), dim=1)


def prepare_motif(motif_type, motif_indices, sample, activation_group, aggregation_group, weights=None, biases=None):
    for motif_index in motif_indices:
        for activations in activation_group:
            for aggregations in aggregation_group:
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
                            except AssertionError:
                                pass
                        yield motif, sample_index
                    else:
                        yield NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                          activations=activations, aggregations=aggregations,
                                          weights=weights, biases=biases), sample_index


# noinspection PyArgumentList,PyStringFormat,PyUnresolvedReferences
def maximum_minimum_search(value_range, times, seed, sample, epochs, save_path, motif_types, motif_indices,
                           activations, aggregations, repeats):
    assert motif_types["s"] in ["coherent-loop", "incoherent-loop"]
    assert motif_types["t"] == "collider"

    manual_seed(seed=seed)
    considered_data = prepare_data(minimum_value=value_range[0], maximum_value=value_range[1], times=times)
    source_activation_group = list(product(activations, repeat=repeats["s"][0]))
    source_aggregation_group = list(product(aggregations, repeat=repeats["s"][1]))
    records = []
    for source_motif, index_1 in prepare_motif(motif_type=motif_types["s"], motif_indices=motif_indices["s"],
                                               activation_group=source_activation_group,
                                               aggregation_group=source_aggregation_group,
                                               sample=sample):
        print("initialized \"" + motif_types["s"] + "\" motif is")
        print(str(source_motif) + "\n\n\n")

        source_optimizer, source_criterion = Adam(source_motif.parameters()), MSELoss()
        source_results = source_motif(considered_data)

        # initialize target motifs with the known weights and biases for faster fitness.
        target_weights = [source_motif.w[1].get_info(), source_motif.w[2].get_info()]
        target_biases = [source_motif.b[1].get_info()]

        record = []
        for source_epoch in range(epochs + 1):
            saved_target_motif, saved_target_results, saved_loss, monitor = None, None, None, Monitor()
            target_motif_number, target_process = len(motif_indices["t"]), 1
            target_motif_number *= (len(activations) ** repeats["t"][0]) * (len(aggregations) ** repeats["t"][1])
            print("*" * 80)
            print("iteration " + str(source_epoch))
            print("-" * 80 + "\n")

            print("find minimum gap: training \"" + motif_types["t"] + "\" motifs by the gradient descent.")
            target_activation_group = list(product(activations, repeat=repeats["t"][0]))
            target_aggregation_group = list(product(aggregations, repeat=repeats["t"][1]))
            for target_motif, index_2 in prepare_motif(motif_type=motif_types["t"], motif_indices=motif_indices["t"],
                                                       activation_group=target_activation_group,
                                                       aggregation_group=target_aggregation_group,
                                                       sample=1, weights=target_weights, biases=target_biases):
                target_optimizer, target_criterion = Adam(target_motif.parameters()), MSELoss()
                for target_epoch in range(epochs + 1):
                    target_results = target_motif(considered_data)
                    target_loss = target_criterion(source_results, target_results)
                    if saved_loss is not None:
                        if target_loss < saved_loss:
                            saved_target_results, saved_loss = target_results, float(target_loss)
                            saved_target_motif = deepcopy(target_motif)
                    else:
                        saved_target_results, saved_loss = target_results, float(target_loss)
                        saved_target_motif = deepcopy(target_motif)

                    if target_epoch < epochs:
                        target_optimizer.zero_grad()
                        target_loss.backward(retain_graph=True)
                        target_optimizer.step()

                monitor.output(target_process, target_motif_number)
                target_process += 1

            print("the most similar \"" + motif_types["t"] + "\" motif is")
            print(saved_target_motif)
            print("with the mean squared error %.5e\n" % saved_loss)

            if source_epoch < epochs:
                source_loss = source_criterion(source_results, saved_target_results)
                source_optimizer.zero_grad()
                (-source_loss).backward(retain_graph=True)
                source_optimizer.step()
                print("find maximum gap: training \"" + motif_types["s"] + "\" motif by the gradient ascent.")
                print(str(source_motif) + "\n")
                print("*" * 80 + "\n\n")
            else:
                print("the maximum-minimum mean squared error is %.5e\n" % saved_loss)
                print("*" * 80)

            record.append((deepcopy(source_motif), deepcopy(saved_target_motif), saved_loss))
        records.append(record)

        if len(records) == sample:
            info = source_motif.t + " " + str(source_motif.i) + " "
            info += str(source_motif.a).replace("\'", "").replace(",", "") + " "
            info += str(source_motif.g).replace("\'", "").replace(",", "") + ".pkl"

            with open(save_path + info, "wb") as file:
                dump(obj=records, file=file)
            records = []
