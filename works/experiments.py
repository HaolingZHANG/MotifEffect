from copy import deepcopy
from itertools import product
from numpy import linspace, array, zeros, mean, sum, max, abs, argmax, cumsum
from sklearn.manifold import TSNE
from os.path import exists

from effect import NeuralMotif, prepare_motifs
from effect import calculate_landscape, calculate_gradients
from effect import calculate_rugosity, evaluate_propagation, estimate_lipschitz
from effect import maximum_minimum_loss_search, minimum_loss_search

from practice import NEATCartPoleTask, NEATAgent, NormNoiseGenerator, count_motifs_from_matrices
from practice import create_agent_config, obtain_best, calculate_matrix_from_agent, GraphType

from works import Monitor, get_reference_motifs, save_data, load_data


def experiment1(raw_path, task_path):
    value_range, points, times, weight_range, bias_range, monitor = (-1, +1), 41, 5, (-1, +1), (-1, +1), Monitor()
    activation_selection, aggregation_selection = ["relu", "tanh", "sigmoid"], ["sum", "max"]

    if not exists(path=raw_path + "property.landscape.pkl"):
        print("Calculate the output landscape of the incoherent loop population and the collider population.")
        records, flag, current, total = [], 1, 0, 4 * (3 ** 2) * (2 ** 2) * (5 ** (3 + 2))
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=2):
                for aggregations in product(aggregation_selection, repeat=2):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=3):
                        for biases in product(linspace(bias_range[0], bias_range[1], times), repeat=2):
                            try:
                                motif = NeuralMotif(motif_type="incoherent-loop", motif_index=motif_index,
                                                    activations=activations, aggregations=aggregations,
                                                    weights=weights, biases=biases)
                                records.append(("incoherent loop", flag,
                                                calculate_landscape(value_range, points, motif)))
                                flag += 1
                            except ValueError:
                                pass

                            monitor.output(current_state=current + 1, total_state=total)
                            current += 1

        flag, current, total = 1, 0, 4 * 3 * 2 * (5 ** (2 + 1))
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=1):
                for aggregations in product(aggregation_selection, repeat=1):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=2):
                        for biases in product(linspace(bias_range[0], bias_range[1], times), repeat=1):
                            try:
                                motif = NeuralMotif(motif_type="collider", motif_index=motif_index,
                                                    activations=activations, aggregations=aggregations,
                                                    weights=weights, biases=biases)
                                records.append(("collider", flag,
                                                calculate_landscape(value_range, points, motif)))
                                flag += 1
                            except ValueError:
                                pass

                            monitor.output(current_state=current + 1, total_state=total)
                            current += 1

        save_data(save_path=raw_path + "property.landscape.pkl", information=records)

    if not exists(path=raw_path + "property.difference.npy"):
        print("Calculate the difference between motifs.")
        records = load_data(load_path=raw_path + "property.landscape.pkl")
        differences = zeros(shape=(len(records), len(records)))
        current, total = 0, (len(records) - 1) * len(records) // 2
        for index_1 in range(len(records)):
            for index_2 in range(index_1 + 1, len(records)):
                difference = mean(abs(records[index_1][-1] - records[index_2][-1])) / (value_range[1] - value_range[0])
                differences[index_1, index_2], differences[index_2, index_1] = difference, difference
                monitor.output(current_state=current + 1, total_state=total)
                current += 1
        save_data(save_path=raw_path + "property.difference.npy", information=differences)

    if not exists(path=raw_path + "property.lipschitz.pkl"):
        print("Calculate the Lipschitz constant of the incoherent loop population and the collider population.")
        records, flag, current, total = [], 1, 0, 4 * (3 ** 2) * (2 ** 2) * (5 ** (3 + 2))
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=2):
                for aggregations in product(aggregation_selection, repeat=2):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=3):
                        for biases in product(linspace(bias_range[0], bias_range[1], times), repeat=2):
                            try:
                                motif = NeuralMotif(motif_type="incoherent-loop", motif_index=motif_index,
                                                    activations=activations, aggregations=aggregations,
                                                    weights=weights, biases=biases)
                                records.append(("incoherent loop", flag,
                                                estimate_lipschitz(value_range, points, motif, "L-2")))
                                flag += 1
                            except ValueError:
                                pass

                            monitor.output(current_state=current + 1, total_state=total)
                            current += 1

        flag, current, total = 1, 0, 4 * 3 * 2 * (5 ** (2 + 1))
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=1):
                for aggregations in product(aggregation_selection, repeat=1):
                    for weights in product(linspace(weight_range[0], weight_range[1], 11), repeat=2):
                        for biases in product(linspace(bias_range[0], bias_range[1], times), repeat=1):
                            try:
                                motif = NeuralMotif(motif_type="collider", motif_index=motif_index,
                                                    activations=activations, aggregations=aggregations,
                                                    weights=weights, biases=biases)
                                records.append(("collider", flag,
                                                estimate_lipschitz(value_range, points, motif, "L-2")))
                                flag += 1
                            except ValueError:
                                pass

                            monitor.output(current_state=current + 1, total_state=total)
                            current += 1

        save_data(save_path=raw_path + "property.lipschitz.pkl", information=records)

    if not exists(path=raw_path + "property.rugosity.pkl"):
        print("Calculate the rugosity index of the incoherent loop population and the collider population.")
        records, flag, current, total = [], 1, 0, 4 * (3 ** 2) * (2 ** 2) * (5 ** (3 + 2))
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=2):
                for aggregations in product(aggregation_selection, repeat=2):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=3):
                        for biases in product(linspace(bias_range[0], bias_range[1], times), repeat=2):
                            try:
                                motif = NeuralMotif(motif_type="incoherent-loop", motif_index=motif_index,
                                                    activations=activations, aggregations=aggregations,
                                                    weights=weights, biases=biases)
                                records.append(("incoherent loop", flag,
                                                calculate_rugosity(value_range, points, motif)))
                                flag += 1
                            except ValueError:
                                pass

                            monitor.output(current_state=current + 1, total_state=total)
                            current += 1

        flag, current, total = 1, 0, 4 * 3 * 2 * (5 ** (2 + 1))
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=1):
                for aggregations in product(aggregation_selection, repeat=1):
                    for weights in product(linspace(weight_range[0], weight_range[1], 11), repeat=2):
                        for biases in product(linspace(bias_range[0], bias_range[1], times), repeat=1):
                            try:
                                motif = NeuralMotif(motif_type="collider", motif_index=motif_index,
                                                    activations=activations, aggregations=aggregations,
                                                    weights=weights, biases=biases)
                                records.append(("collider", flag,
                                                calculate_rugosity(value_range, points, motif)))
                                flag += 1
                            except ValueError:
                                pass

                            monitor.output(current_state=current + 1, total_state=total)
                            current += 1

        save_data(save_path=raw_path + "property.rugosity.pkl", information=records)

    if not exists(path=raw_path + "property.propagation.pkl"):
        print("Calculate the maximum error propagation of the incoherent loop population and the collider population.")
        records, flag, current, total = [], 1, 0, 4 * (3 ** 2) * (2 ** 2) * (5 ** (3 + 2))
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=2):
                for aggregations in product(aggregation_selection, repeat=2):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=3):
                        for biases in product(linspace(bias_range[0], bias_range[1], times), repeat=2):
                            try:
                                motif = NeuralMotif(motif_type="incoherent-loop", motif_index=motif_index,
                                                    activations=activations, aggregations=aggregations,
                                                    weights=weights, biases=biases)
                                records.append(("incoherent loop", flag,
                                                evaluate_propagation(value_range, points, motif, "max")))
                                flag += 1
                            except ValueError:
                                pass

                            monitor.output(current_state=current + 1, total_state=total)
                            current += 1

        flag, current, total = 1, 0, 4 * 3 * 2 * (5 ** (2 + 1))
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=1):
                for aggregations in product(aggregation_selection, repeat=1):
                    for weights in product(linspace(weight_range[0], weight_range[1], 11), repeat=2):
                        for biases in product(linspace(bias_range[0], bias_range[1], times), repeat=1):
                            try:
                                motif = NeuralMotif(motif_type="collider", motif_index=motif_index,
                                                    activations=activations, aggregations=aggregations,
                                                    weights=weights, biases=biases)
                                records.append(("collider", flag,
                                                evaluate_propagation(value_range, points, motif, "max")))
                                flag += 1
                            except ValueError:
                                pass

                            monitor.output(current_state=current + 1, total_state=total)
                            current += 1

        save_data(save_path=raw_path + "property.propagation.pkl", information=records)

    if not exists(task_path + "locations.npy") \
            or (not exists(task_path + "location loop.npy")) or (not exists(task_path + "location collider.npy")):
        difference_record = load_data(load_path=raw_path + "property.difference.npy")
        method = TSNE(n_components=2, metric="precomputed")
        locations = method.fit_transform(X=difference_record)
        save_data(save_path=raw_path + "locations.npy", information=locations)
        loop_count = 4 * (3 ** 2) * (2 ** 2) * (2 ** 3) * (5 ** 2)
        save_data(save_path=raw_path + "location loop.npy", information=locations[:loop_count])
        save_data(save_path=raw_path + "location collider.npy", information=locations[loop_count:])

    if (not exists(task_path + "rugosity indices.npy")) \
            or (not exists(task_path + "rugosity loop.npy")) or (not exists(task_path + "rugosity collider.npy")):
        rugosity_indices, rugosity_record = [[], []], load_data(load_path=raw_path + "property.rugosity.pkl")
        for index, (name, _, rugosity_index) in enumerate(rugosity_record):
            if name == "incoherent loop":
                rugosity_indices[0].append(rugosity_index)
            else:
                rugosity_indices[1].append(rugosity_index)
        rugosity_totally = rugosity_indices[0] + rugosity_indices[1]
        save_data(save_path=task_path + "rugosity indices.npy", information=array(rugosity_totally))
        save_data(save_path=task_path + "rugosity loop.npy", information=array(rugosity_indices[0]))
        save_data(save_path=task_path + "rugosity collider.npy", information=array(rugosity_indices[1]))

    if (not exists(task_path + "lipschitz constants.npy")) \
            or (not exists(task_path + "lipschitz loop.npy")) or (not exists(task_path + "lipschitz collider.npy")):
        lipschitz_values, lipschitz_record = [[], []], load_data(load_path=raw_path + "property.lipschitz.pkl")
        for index, (name, _, lipschitz_value) in enumerate(lipschitz_record):
            if name == "incoherent loop":
                lipschitz_values[0].append(lipschitz_value)
            else:
                lipschitz_values[1].append(lipschitz_value)
        lipschitz_totally = lipschitz_values[0] + lipschitz_values[1]
        save_data(save_path=task_path + "lipschitz constants.npy", information=array(lipschitz_totally))
        save_data(save_path=task_path + "lipschitz loop.npy", information=array(lipschitz_values[0]))
        save_data(save_path=task_path + "lipschitz collider.npy", information=array(lipschitz_values[1]))

    if (not exists(task_path + "propagation effect.npy")) \
            or (not exists(task_path + "propagation loop.npy")) or (not exists(task_path + "propagation collider.npy")):
        propagation_record = load_data(raw_path + "property.propagation.pkl")
        propagations = zeros(shape=(2, points, points))
        for index, (name, _, matrix) in enumerate(propagation_record):
            if name == "incoherent loop":
                propagations[0] = max([matrix, propagations[0]], axis=0)
            else:
                propagations[1] = max([matrix, propagations[1]], axis=0)
        propagation_totally = propagations[0] + propagations[1]
        save_data(save_path=task_path + "propagation effect.npy", information=array(propagation_totally))
        save_data(save_path=task_path + "propagation loop.npy", information=propagations[0])
        save_data(save_path=task_path + "propagation collider.npy", information=propagations[1])


def experiment2(raw_path, task_path):
    value_range, points, threshold, monitor = (-1, +1), 41, 2e-3, Monitor()
    learn_rate = (value_range[1] - value_range[0]) / (points - 1) * 1e-1
    activation_selections, aggregation_selections = ["relu", "tanh", "sigmoid"], ["sum", "max"]
    loss_threshold, check_threshold, iteration_threshold = 1e-5, 5, 1000

    if not exists(raw_path + "intuition.search.pkl"):
        print("Find the most similar collider of customized incoherent loops in the intuition task.")
        source_motifs, target_motifs = [], []
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selections, repeat=2):
                for aggregations in product(aggregation_selections, repeat=2):
                    source_motif = prepare_motifs(motif_type="incoherent-loop", motif_index=motif_index,
                                                  activations=activations, aggregations=aggregations,
                                                  sample=1, weights=(1e0, 1e-3, 1e-3), biases=(0.0, 0.0))[0]
                    source_motifs.append(source_motif)
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selections, repeat=1):
                for aggregations in product(aggregation_selections, repeat=1):
                    for weights in product([1e-3, 1e0], repeat=2):
                        for biases in product([1e-3, 1e0], repeat=1):
                            target_motif = prepare_motifs(motif_type="collider", motif_index=motif_index,
                                                          activations=activations, aggregations=aggregations,
                                                          sample=1, weights=weights, biases=biases)[0]
                            target_motifs.append(target_motif)

        records = []
        for source_index, source_motif in enumerate(source_motifs):
            record, minimum_loss = [], 2.0
            print("Calculate " + str(source_index + 1) + "-th motif.")
            for target_motif in target_motifs:
                result = minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                             source_motif=source_motif, target_motif=deepcopy(target_motif),
                                             loss_threshold=loss_threshold, check_threshold=check_threshold,
                                             iteration_threshold=iteration_threshold,
                                             need_lipschitz=False, need_rugosity=False, verbose=False)
                motifs, losses, _, _ = result
                if minimum_loss > losses[-1]:
                    record, minimum_loss = (motifs, losses), losses[-1]
                if losses[-1] <= threshold:
                    break

            records.append(record)

        save_data(save_path=raw_path + "intuition.search.pkl", information=(source_motifs, records))

    if not exists(path=raw_path + "intuition.landscapes.npy"):
        print("Calculate the output landscape of motifs in the intuition task.")
        landscapes, records = [], load_data(load_path=raw_path + "intuition.search.pkl")
        for index, (source_motif, (target_motifs, _)) in enumerate(zip(records[0], records[1])):
            target_motif = target_motifs[-1]
            landscapes += [calculate_landscape(value_range=value_range, points=points, motif=source_motif),
                           calculate_landscape(value_range=value_range, points=points, motif=target_motif)]
            monitor.output(index + 1, len(records[0]))
        save_data(save_path=raw_path + "intuition.landscapes.npy", information=array(landscapes))

    if not exists(path=raw_path + "intuition.difference.npy"):
        print("Calculate the difference between motifs in the intuition task.")
        landscapes = load_data(load_path=raw_path + "intuition.landscapes.npy")
        differences = zeros(shape=(len(landscapes), len(landscapes)))
        current, total = 0, (len(landscapes) - 1) * len(landscapes) // 2
        for index_1 in range(len(landscapes)):
            for index_2 in range(index_1 + 1, len(landscapes)):
                difference = mean(abs(landscapes[index_1] - landscapes[index_2])) / (value_range[1] - value_range[0])
                differences[index_1, index_2], differences[index_2, index_1] = difference, difference
                monitor.output(current_state=current + 1, total_state=total)
                current += 1
        save_data(save_path=raw_path + "intuition.difference.npy", information=differences)

    if not exists(path=task_path + "locations.npy"):
        difference_record = load_data(load_path=raw_path + "intuition.difference.npy")
        method = TSNE(n_components=2, metric="precomputed")
        locations = method.fit_transform(X=difference_record)
        save_data(save_path=task_path + "locations.npy", information=locations)

    if not exists(path=task_path + "minimum losses.npy"):
        source_motifs, search_records = load_data(load_path=raw_path + "intuition.search.pkl")
        minimum_losses = [search_record[1][-1] for search_record in search_records]
        save_data(save_path=task_path + "minimum losses.npy", information=array(minimum_losses))

    if not exists(path=task_path + "lipschitz constants.npy"):
        constants, records = [], load_data(load_path=raw_path + "intuition.search.pkl")
        for index, (source_motif, (target_motifs, losses)) in enumerate(zip(records[0], records[1])):
            target_motif = target_motifs[-1]
            source_constant = estimate_lipschitz(value_range=value_range, points=points, motif=source_motif)
            target_constant = estimate_lipschitz(value_range=value_range, points=points, motif=target_motif)
            constants.append([source_constant, target_constant])
        save_data(save_path=task_path + "lipschitz constants.npy", information=array(constants))

    if not exists(path=task_path + "rugosity indices.npy"):
        rugosity_indices, records = [], load_data(load_path=raw_path + "intuition.search.pkl")
        for index, (source_motif, (target_motifs, losses)) in enumerate(zip(records[0], records[1])):
            target_motif = target_motifs[-1]
            source_rugosity = calculate_rugosity(value_range=value_range, points=points, motif=source_motif)
            target_rugosity = calculate_rugosity(value_range=value_range, points=points, motif=target_motif)
            rugosity_indices.append([source_rugosity, target_rugosity])
        save_data(save_path=task_path + "rugosity indices.npy", information=array(rugosity_indices))

    if not exists(path=task_path + "intuition landscapes.npy"):
        landscapes, source_motifs = [], load_data(load_path=raw_path + "intuition.search.pkl")[0]
        for source_motif in source_motifs:
            landscapes.append(calculate_landscape(value_range=value_range, points=points, motif=source_motif))
        save_data(save_path=task_path + "intuition landscapes.npy", information=array(landscapes))

    if not exists(path=task_path + "terminal cases.npy"):
        case_pairs, records = [], load_data(load_path=raw_path + "intuition.search.pkl")
        minimum_loss, maximum_loss, minimum_index, maximum_index = 2.0, 0.0, 0, 0
        for index, (source_motif, (target_motifs, losses)) in enumerate(zip(records[0], records[1])):
            target_motif = target_motifs[-1]
            source_landscape = calculate_landscape(value_range=value_range, points=points, motif=source_motif)
            target_landscape = calculate_landscape(value_range=value_range, points=points, motif=target_motif)
            case_pairs.append((index, (source_motif, source_landscape), (target_motif, target_landscape), losses[-1]))
            if minimum_loss > losses[-1]:
                minimum_loss, minimum_index = losses[-1], index
            if maximum_loss < losses[-1]:
                maximum_loss, maximum_index = losses[-1], index
        terminal_cases = {"min": case_pairs[minimum_index], "max": case_pairs[maximum_index]}
        save_data(save_path=task_path + "terminal cases.pkl", information=terminal_cases)


def experiment3(raw_path, task_path):
    value_range, points, monitor = (-1, +1), 41, Monitor()
    learn_rate = (value_range[1] - value_range[0]) / (points - 1) * 1e-1
    activation_selections, aggregation_selections = ["relu", "tanh", "sigmoid"], ["sum", "max"]
    loss_threshold, check_threshold, iteration_thresholds = 1e-5, 5, (1000, 200)

    if not exists(raw_path + "max-min.search.pkl"):
        source_motifs, target_motifs = [], []
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selections, repeat=2):
                for aggregations in product(aggregation_selections, repeat=2):
                    source_motif = prepare_motifs(motif_type="incoherent-loop", motif_index=motif_index,
                                                  activations=activations, aggregations=aggregations,
                                                  sample=1, weights=(1e-3, 1e0, 1e0), biases=(0.0, 0.0))[0]
                    source_motifs.append(source_motif)
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selections, repeat=1):
                for aggregations in product(aggregation_selections, repeat=1):
                    target_motif = prepare_motifs(motif_type="collider", motif_index=motif_index,
                                                  activations=activations, aggregations=aggregations,
                                                  sample=1, weights=(1e0, 1e0), biases=(0.0,))[0]
                    target_motifs.append(target_motif)

        print("Find the maximum-minimum losses of the incoherent loop population and the collider population.")
        records = []
        for source_index, source_motif in enumerate(source_motifs):
            record = maximum_minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                                 source_motif=source_motif, target_motifs=target_motifs,
                                                 loss_threshold=loss_threshold, check_threshold=check_threshold,
                                                 iteration_thresholds=iteration_thresholds,
                                                 need_lipschitz=False, need_rugosity=False, verbose=True)
            records.append(record)
            monitor.output(source_index + 1, len(source_motifs))
        save_data(save_path=raw_path + "max-min.search.pkl", information=records)

    if not exists(path=raw_path + "max-min.propagation.pkl"):
        print("Calculate the error propagation of each final source motif.")
        propagation_data, records = [], load_data(load_path=raw_path + "max-min.search.pkl")
        for index, (motif_pairs, _, _, _) in enumerate(records):
            source_motif, target_motif = motif_pairs[-1]
            source_error = evaluate_propagation(value_range=value_range, points=points,
                                                motif=source_motif, compute_type="mean")
            target_error = evaluate_propagation(value_range=value_range, points=points,
                                                motif=target_motif, compute_type="mean")
            propagation_data.append(source_error + target_error)
            monitor.output(index + 1, len(records))
        save_data(save_path=raw_path + "max-min.propagation.pkl", information=propagation_data)

    if not exists(path=raw_path + "max-min.lipschitz.pkl"):
        print("Calculate the Lipschitz content of each final source motif.")
        lipschitz_data, records = [], load_data(load_path=raw_path + "max-min.search.pkl")
        for index, (motif_pairs, _, _, _) in enumerate(records):
            lipschitz_values = []
            for motif, _ in motif_pairs:
                lipschitz_values.append(estimate_lipschitz(value_range=value_range, points=points, motif=motif))
            lipschitz_data.append(array(lipschitz_values))
            monitor.output(index + 1, len(records))
        save_data(save_path=raw_path + "max-min.lipschitz.pkl", information=lipschitz_data)

    if not exists(path=raw_path + "max-min.rugosity.pkl"):
        print("Calculate the rugosity index of each final source motif.")
        rugosity_data, records = [], load_data(load_path=raw_path + "max-min.search.pkl")
        for index, (motif_pairs, _, _, _) in enumerate(records):
            rugosity_indices = []
            for motif, _ in motif_pairs:
                rugosity_indices.append(calculate_rugosity(value_range=value_range, points=points, motif=motif))
            rugosity_data.append(array(rugosity_indices))
            monitor.output(index + 1, len(records))
        save_data(save_path=raw_path + "max-min.rugosity.pkl", information=rugosity_data)

    if not exists(path=raw_path + "max-min.gradient.pkl"):
        print("Calculate the gradient matrix of each final source motif.")
        gradient_data, records = [], load_data(load_path=raw_path + "max-min.search.pkl")
        for index, (motif_pairs, _, _, _) in enumerate(records):
            gradient_path = []
            for motif, _ in motif_pairs:
                gradient_path.append(calculate_gradients(value_range=value_range, points=points, motif=motif))
            gradient_data.append(array(gradient_path))
            monitor.output(index + 1, len(records))
        save_data(save_path=raw_path + "max-min.gradient.pkl", information=gradient_data)

    if not exists(path=task_path + "landscapes.npy"):
        landscapes, records = [], load_data(load_path=raw_path + "max-min.search.pkl")
        for record in records:
            landscapes.append(calculate_landscape(value_range=value_range, points=points, motif=record[0][-1][0]))
        save_data(save_path=task_path + "landscapes.npy", information=array(landscapes))

    if not exists(path=task_path + "max-min losses.npy"):
        loss_data, records = [], load_data(load_path=raw_path + "max-min.search.pkl")
        for index, (_, losses, _, _) in enumerate(records):
            loss_data.append(losses[-1])
        save_data(save_path=task_path + "max-min losses.npy", information=array(loss_data))

    if not exists(path=task_path + "max-min params.npy"):
        param_data, records = [[], [], [], []], load_data(load_path=raw_path + "max-min.search.pkl")
        for index, (motif_pairs, _, _, _) in enumerate(records):
            motif = motif_pairs[-1][0]
            param_data[motif.i - 1].append([weight.value() for weight in motif.w])
        save_data(save_path=task_path + "max-min params.npy", information=array(param_data))

    if not exists(path=task_path + "replaceable.npy"):
        propagations = load_data("../data/results/raw/max-min.propagation.pkl")
        chuck_losses = load_data("../data/results/task03/max-min losses.npy")
        counts = zeros(shape=(41,), dtype=int)
        for index, (propagation, chuck_loss) in enumerate(zip(propagations, chuck_losses)):
            for i in range(1, len(propagation)):
                if max(propagation[:i, :i]) > chuck_loss:
                    counts[i] += 1
                    break
        save_data(save_path=task_path + "replaceable.npy", information=cumsum(array(counts)))

    if not exists(path=task_path + "lipschitz constants.npy"):
        lipschitz_data, records = [], load_data(load_path=raw_path + "max-min.lipschitz.pkl")
        for record in records:
            lipschitz_data.append([record[0], record[-1]])
        save_data(save_path=task_path + "lipschitz constants.npy", information=array(lipschitz_data))

    if not exists(path=task_path + "loss paths.pkl"):
        loss_paths, records = [], load_data(load_path=raw_path + "max-min.search.pkl")
        for index, (_, losses, _, _) in enumerate(records):
            loss_paths.append(losses)
        save_data(save_path=task_path + "loss paths.pkl", information=array(loss_paths))

    if not exists(path=task_path + "lipschitz paths.pkl"):
        records = load_data(load_path=raw_path + "max-min.lipschitz.pkl")
        save_data(save_path=task_path + "lipschitz paths.pkl", information=records)

    if not exists(path=task_path + "rugosity paths.pkl"):
        records = load_data(load_path=raw_path + "max-min.rugosity.pkl")
        save_data(save_path=task_path + "rugosity paths.pkl", information=records)

    if not exists(path=task_path + "maximum case.pkl"):
        maximum_index = argmax(load_data(load_path=task_path + "max-min losses.npy"))
        loss_change = load_data(load_path=task_path + "loss paths.pkl")[maximum_index]
        lipschitz_change = load_data(load_path=task_path + "lipschitz paths.pkl")[maximum_index]
        motif_record = load_data(load_path=raw_path + "max-min.search.pkl")[maximum_index][0]
        param_change = zeros(shape=(1000, 5))

        source_motifs, indices = [], (linspace(0, 1000, 11, dtype=int) - 1)[1:]
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selections, repeat=2):
                for aggregations in product(aggregation_selections, repeat=2):
                    source_motif = prepare_motifs(motif_type="incoherent-loop", motif_index=motif_index,
                                                  activations=activations, aggregations=aggregations,
                                                  sample=1, weights=(1e-3, 1e0, 1e0), biases=(0.0, 0.0))[0]
                    source_motifs.append(source_motif)
        start_motif = source_motifs[maximum_index]

        output_change = [calculate_landscape(value_range=value_range, points=points, motif=start_motif)]
        for index in range(len(motif_record)):
            motif = motif_record[index][0]
            param_change[index] = array([weight.value() for weight in motif.w] + [bias.value() for bias in motif.b])
            output_change.append(calculate_landscape(value_range=value_range, points=points, motif=motif))

        rugosity_change = load_data(load_path=task_path + "rugosity paths.pkl")[maximum_index]
        saved_data = (loss_change, lipschitz_change, rugosity_change, param_change, output_change)
        save_data(save_path=task_path + "maximum case.pkl", information=saved_data)


def experiment4(raw_path, task_path, config_path):
    def record_handle(a):
        return calculate_matrix_from_agent(a, need_mapping=True)

    agent_names, agent_count, monitor = ["baseline", "geometry", "novelty"], 100, Monitor()
    agent_configs = [create_agent_config(path=config_path + "cartpole-v0.default." + name) for name in agent_names]
    radios = [0.0, 0.1, 0.2, 0.3]

    noise_generator = NormNoiseGenerator(norm_type="L-2", noise_scale=0.0)
    task = NEATCartPoleTask(maximum_generation=20, noise_generator=noise_generator, verbose=False)

    if not exists(path=raw_path + "practice.train.pkl"):
        print("Train agents in the CartPole environment.")
        train_records = {}
        for agent_name, agent_config in zip(agent_names, agent_configs):
            for radio in radios:
                train_records[(agent_name[0], int(radio * 10))] = []
                task.noise_generator.noise_scale = radio
                count = 1
                while count < agent_count:
                    print("Train " + agent_name + " in the CartPole: " + str(count) + " / " + str(agent_count))
                    task.reset_experience(record_handle=record_handle)
                    best_genome = obtain_best(task, agent_config, need_stdout=True)
                    if best_genome is not None:
                        agent = NEATAgent(model_genome=best_genome, neat_config=agent_config, description=agent_name,
                                          action_handle=argmax)
                        experience = task.get_experience()
                        record = {"agent": agent, "experience": experience}
                        train_records[(agent_name[0], int(radio * 10))].append(record)
                        count += 1
                    print()
        save_data(save_path=raw_path + "practice.train.pkl", information=train_records)

    if not exists(path=raw_path + "practice.motif.pkl"):
        print("Calculate motifs in agents.")
        reference_motifs, current = get_reference_motifs(), 1
        motif_records, train_records = {"final": {}, "evolve": {}}, load_data(load_path=raw_path + "practice.train.pkl")
        for (name_index, noise_index), train_info in train_records.items():
            motif_records["final"][(name_index, noise_index)] = zeros(shape=(len(reference_motifs),))
            motif_records["evolve"][(name_index, noise_index)] = zeros(shape=(len(reference_motifs), 20))
            temp_counts = zeros(shape=(len(reference_motifs), 20))
            for agent_index, agent_info in enumerate(train_info):
                experience, matrices = agent_info["experience"], []
                for matrix, _ in experience:
                    matrices.append(matrix)
                motif_counts = count_motifs_from_matrices(matrices=matrices, search_size=3,
                                                          graph_type=GraphType.pn, pruning=False,
                                                          reference_motifs=reference_motifs)
                final_counts = motif_counts[-1]

                evolved_counts = zeros(shape=(len(reference_motifs), 20))
                for index, values in enumerate(motif_counts):
                    evolved_counts[index] = values
                if len(motif_counts) < 20:
                    for index in range(len(motif_counts), 20):
                        evolved_counts[index] = evolved_counts[len(motif_counts) - 1]

                motif_records["evolve"][(name_index, noise_index)] += evolved_counts
                motif_records["final"][(name_index, noise_index)] += final_counts
                temp_counts[evolved_counts > 0] += 1
                monitor.output(current, 1200)
                current += 1

            temp_counts[temp_counts == 0] += 1
            motif_records["evolve"][(name_index, noise_index)] /= temp_counts
            motif_records["final"][(name_index, noise_index)] /= 100.0

        save_data(save_path=raw_path + "practice.motif.pkl", information=motif_records)

    if not exists(path=raw_path + "practice.noise.pkl"):
        print("Evaluate the agents under different noise scales.")
        noise_records, train_records = {}, load_data(load_path=raw_path + "practice.train.pkl")
        for (name_index, noise_index), train_info in train_records.items():
            noise_record = []
            print("Evaluate " + {"b": "baseline", "g": "geometry", "n": "novelty"}[name_index] + ".")
            for agent_index, agent_info in enumerate(train_info):
                agent, record, current = agent_info["agent"], zeros(shape=(len(radios), agent_count)), 1
                for index_1, noise_scale in enumerate(radios):
                    task.noise_generator.noise_scale = noise_scale
                    reward_collector = task.run(agent=agent)["rewards"]
                    for index_2, rewards in enumerate(reward_collector):
                        record[index_1, index_2] = sum(rewards)
                        monitor.output(current, len(radios) * agent_count)
                        current += 1
                noise_record.append(record)
            noise_records[(name_index, noise_index)] = array(noise_record)
        save_data(save_path=raw_path + "practice.noise.pkl", information=noise_records)

    if not exists(path=raw_path + "practice.additions.pkl"):
        print("Train agents in the CartPole environment (from 20 generations to 100 generations).")
        train_1_records = {}
        for agent_name, agent_config in zip(agent_names, agent_configs):
            for radio in radios:
                train_1_records[(agent_name[0], int(radio * 10))] = []
                task.noise_generator.noise_scale = radio
                count = 1
                while count < agent_count:
                    print("Train " + agent_name + " in the CartPole: " + str(count) + " / " + str(agent_count))
                    task.reset_experience(record_handle=record_handle)
                    best_genome = obtain_best(task, agent_config, need_stdout=True)
                    if best_genome is not None:
                        agent = NEATAgent(model_genome=best_genome, neat_config=agent_config, description=agent_name,
                                          action_handle=argmax)
                        experience = task.get_experience()
                        record = {"agent": agent, "experience": experience}
                        train_1_records[(agent_name[0], int(radio * 10))].append(record)
                        count += 1
                    print()

        print("Train agents in the CartPole environment (allow "
              "\"relu\", \"tanh\", and \"sigmoid\" for activations and \"sum\" and \"max\" for aggregations).")
        train_2_records = {}
        agent_configs = [create_agent_config(path=config_path + "cartpole-v0.combine." + name) for name in agent_names]
        for agent_name, agent_config in zip(agent_names, agent_configs):
            for radio in radios:
                train_2_records[(agent_name[0], int(radio * 10))] = []
                task.noise_generator.noise_scale = radio
                count = 1
                while count < agent_count:
                    print("Train " + agent_name + " in the CartPole: " + str(count) + " / " + str(agent_count))
                    task.reset_experience(record_handle=record_handle)
                    best_genome = obtain_best(task, agent_config, need_stdout=True)
                    if best_genome is not None:
                        agent = NEATAgent(model_genome=best_genome, neat_config=agent_config, description=agent_name,
                                          action_handle=argmax)
                        experience = task.get_experience()
                        record = {"agent": agent, "experience": experience}
                        train_2_records[(agent_name[0], int(radio * 10))].append(record)
                        count += 1
                    print()

        print("Evaluate the agents under different noise scales.")
        test_1_records = {}
        for (name_index, noise_index), train_info in train_1_records.items():
            noise_record = []
            print("Evaluate " + {"b": "baseline", "g": "geometry", "n": "novelty"}[name_index] + ".")
            for agent_index, agent_info in enumerate(train_info):
                agent, record, current = agent_info["agent"], zeros(shape=(len(radios), agent_count)), 1
                for index_1, noise_scale in enumerate(radios):
                    task.noise_generator.noise_scale = noise_scale
                    reward_collector = task.run(agent=agent)["rewards"]
                    for index_2, rewards in enumerate(reward_collector):
                        record[index_1, index_2] = sum(rewards)
                        monitor.output(current, len(radios) * agent_count)
                        current += 1
                noise_record.append(record)
            test_1_records[(name_index, noise_index)] = array(noise_record)

        test_2_records = {}
        for (name_index, noise_index), train_info in train_2_records.items():
            noise_record = []
            print("Evaluate " + {"b": "baseline", "g": "geometry", "n": "novelty"}[name_index] + ".")
            for agent_index, agent_info in enumerate(train_info):
                agent, record, current = agent_info["agent"], zeros(shape=(len(radios), agent_count)), 1
                for index_1, noise_scale in enumerate(radios):
                    task.noise_generator.noise_scale = noise_scale
                    reward_collector = task.run(agent=agent)["rewards"]
                    for index_2, rewards in enumerate(reward_collector):
                        record[index_1, index_2] = sum(rewards)
                        monitor.output(current, len(radios) * agent_count)
                        current += 1
                noise_record.append(record)
            test_2_records[(name_index, noise_index)] = array(noise_record)

        save_data(save_path="practice.additions.pkl", information={"20 to 100": (train_1_records, test_1_records),
                                                                   "combine": (train_2_records, test_2_records)})

    if not exists(path=task_path + "train results.npy"):
        train_records = load_data(load_path=raw_path + "practice.train.pkl")
        results = zeros(shape=(3, 4, 100))
        for (name_index, noise_index), train_info in train_records.items():
            for agent_index, agent_info in enumerate(train_info):
                agent = agent_info["agent"]
                results[{"g": 0, "b": 1, "n": 2}[name_index], noise_index, agent_index] = agent.get_fitness()
        save_data(save_path=task_path + "train results.npy", information=results)

    if not exists(path=task_path + "generations.npy"):
        train_records = load_data(load_path=raw_path + "practice.train.pkl")
        results = zeros(shape=(3, 4, 20))
        for (name_index, noise_index), train_info in train_records.items():
            for agent_index, agent_info in enumerate(train_info):
                experience = agent_info["experience"]
                results[{"g": 0, "b": 1, "n": 2}[name_index], noise_index, len(experience) - 1] += 1
        save_data(save_path=task_path + "generations.npy", information=results)

    if not exists(path=task_path + "performances.npy"):
        noise_records = load_data(load_path=raw_path + "practice.noise.pkl")
        results = zeros(shape=(3, 4, 4))
        for (name_index, noise_index), record in noise_records.items():
            results[{"g": 0, "b": 1, "n": 2}[name_index], noise_index] = mean(mean(record, axis=0), axis=1)
        save_data(save_path=task_path + "performances.npy", information=results)

    if not exists(path=task_path + "final motifs.npy"):
        motif_records = load_data(load_path=raw_path + "practice.motif.pkl")["final"]
        results = zeros(shape=(3, 4, 2))
        for (name_index, noise_index), record in motif_records.items():
            results[{"g": 0, "b": 1, "n": 2}[name_index], noise_index] = [sum(record[-4:]), sum(record[:4])]
        save_data(save_path=task_path + "final motifs.npy", information=results)

    if not exists(path=task_path + "evolved motifs.npy"):
        motif_records = load_data(load_path=raw_path + "practice.motif.pkl")["evolve"]
        results = zeros(shape=(3, 4, 2, 20))
        for (name_index, noise_index), record in motif_records.items():
            evolved_counts = array([sum(record[:, -4:], axis=1), sum(record[:, :4], axis=1)])
            results[{"g": 0, "b": 1, "n": 2}[name_index], noise_index] = evolved_counts
        save_data(save_path=task_path + "evolved motifs.npy", information=results)

    if not exists(path=task_path + "changed performances.pkl"):
        records = load_data(load_path=raw_path + "practice.additions.pkl")
        train_1_results, train_2_results = zeros(shape=(3, 4, 100)), zeros(shape=(3, 4, 100))
        for (name_index, noise_index), train_info in records["20 to 100"][0].items():
            for agent_index, agent_info in enumerate(train_info):
                fitness = agent_info["agent"].get_fitness()
                train_1_results[{"g": 0, "b": 1, "n": 2}[name_index], noise_index, agent_index] = fitness
        for (name_index, noise_index), train_info in records["combine"][0].items():
            for agent_index, agent_info in enumerate(train_info):
                fitness = agent_info["agent"].get_fitness()
                train_2_results[{"g": 0, "b": 1, "n": 2}[name_index], noise_index, agent_index] = fitness
        test_1_results, test_2_results = zeros(shape=(3, 4, 4)), zeros(shape=(3, 4, 4))
        for (name_index, noise_index), test_info in records["20 to 100"][1].items():
            test_1_results[{"g": 0, "b": 1, "n": 2}[name_index], noise_index] = mean(mean(test_info, axis=0), axis=1)
        for (name_index, noise_index), test_info in records["combine"][1].items():
            test_2_results[{"g": 0, "b": 1, "n": 2}[name_index], noise_index] = mean(mean(test_info, axis=0), axis=1)

        results = {"20 to 100": (train_1_results, test_1_results), "combine": (train_2_results, test_2_results)}
        save_data(save_path=task_path + "changed performances.pkl", information=results)

    if not exists(path=task_path + "changed motifs.pkl"):
        records = load_data(load_path=raw_path + "practice.additions.pkl")
        motif_records, reference_motifs = {"20 to 100": {}, "combine": {}}, get_reference_motifs()
        for (name_index, noise_index), train_info in records["20 to 100"][0].items():
            matrices = []
            for agent_index, agent_info in enumerate(train_info):
                matrices.append(agent_info["experience"][-1][0])
            motif_counts = count_motifs_from_matrices(matrices=matrices, search_size=3,
                                                      graph_type=GraphType.pn, pruning=False,
                                                      reference_motifs=reference_motifs)
            motif_records["20 to 100"][(name_index, noise_index)] = mean(motif_counts, axis=0)
        for (name_index, noise_index), train_info in records["combine"][0].items():
            matrices = []
            for agent_index, agent_info in enumerate(train_info):
                matrices.append(agent_info["experience"][-1][0])
            motif_counts = count_motifs_from_matrices(matrices=matrices, search_size=3,
                                                      graph_type=GraphType.pn, pruning=False,
                                                      reference_motifs=reference_motifs)
            motif_records["combine"][(name_index, noise_index)] = mean(motif_counts, axis=0)

        save_data(save_path=task_path + "changed motifs.pkl", information=motif_records)


def experiment5(raw_path, task_path, config_path):
    def record_handle(a):
        return calculate_matrix_from_agent(a, need_mapping=True)

    agent_names, agent_count, monitor = ["baseline", "adjusted"], 100, Monitor()
    agent_configs = [create_agent_config(path=config_path + "cartpole-v0.default." + name) for name in agent_names]
    radios = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    noise_generator = NormNoiseGenerator(norm_type="L-2", noise_scale=0.0)
    task = NEATCartPoleTask(maximum_generation=100, noise_generator=noise_generator, verbose=False)

    if not exists(path=raw_path + "intervene.train.pkl"):
        print("Train agents in the CartPole environment.")
        train_records = {}
        for agent_name, agent_config in zip(agent_names, agent_configs):
            for radio in radios:
                train_records[(agent_name[0], int(radio * 10))] = []
                task.noise_generator.noise_scale = radio
                count = 1
                while count < agent_count:
                    print("Train " + agent_name + " in the CartPole: " + str(count) + " / " + str(agent_count))
                    task.reset_experience(record_handle=record_handle)
                    best_genome = obtain_best(task, agent_config, need_stdout=True)
                    if best_genome is not None:
                        agent = NEATAgent(model_genome=best_genome, neat_config=agent_config, description=agent_name,
                                          action_handle=argmax)
                        experience = task.get_experience()
                        record = {"agent": agent, "experience": experience}
                        train_records[(agent_name[0], int(radio * 100))].append(record)
                        count += 1
                    print()
        save_data(save_path=raw_path + "intervene.train.pkl", information=train_records)

    if not exists(path=raw_path + "intervene.test.pkl"):
        print("Evaluate the agents under different noise scales.")
        noise_records, train_records = {}, load_data(load_path=raw_path + "intervene.train.pkl")
        for (name_index, noise_index), train_info in train_records.items():
            noise_record = []
            print("Evaluate " + {"b": "baseline", "a": "adjusted"}[name_index] + ".")
            for agent_index, agent_info in enumerate(train_info):
                agent, record, current = agent_info["agent"], zeros(shape=(len(radios), agent_count)), 1
                for index_1, noise_scale in enumerate(radios):
                    task.noise_generator.noise_scale = noise_scale
                    reward_collector = task.run(agent=agent)["rewards"]
                    for index_2, rewards in enumerate(reward_collector):
                        record[index_1, index_2] = sum(rewards)
                        monitor.output(current, len(radios) * agent_count)
                        current += 1
                noise_record.append(record)
            noise_records[(name_index, noise_index)] = array(noise_record)
        save_data(save_path=raw_path + "intervene.test.pkl", information=noise_records)

    if not exists(path=task_path + "generations.npy"):
        generation_data, mapping = zeros(shape=(2, len(radios), agent_count)), {"b": 0, "a": 1}
        for (name_index, noise_index), train_data in load_data(load_path=raw_path + "intervene.train.pkl").items():
            for agent_index, agent_info in enumerate(train_data):
                generation_data[mapping[name_index], noise_index // 5, agent_index] = len(agent_info["experience"])
        save_data(save_path=task_path + "generations.npy", information=generation_data)

    if not exists(path=task_path + "final loops.npy"):
        reference_motifs, incoherent_loops = get_reference_motifs(), zeros(shape=(len(radios), agent_count))
        for (name_index, noise_index), train_info in load_data(load_path=raw_path + "intervene.train.pkl").items():
            if name_index != "b":
                continue
            for agent_index, agent_info in enumerate(train_info):
                motif_count = count_motifs_from_matrices(matrices=[calculate_matrix_from_agent(agent_info["agent"])],
                                                         search_size=3, graph_type=GraphType.pn, pruning=False,
                                                         reference_motifs=reference_motifs)[0]
                incoherent_loops[noise_index // 5, agent_index] = sum(motif_count[-4:])
        save_data(save_path=task_path + "final loops.npy", information=incoherent_loops)

    if not exists(path=task_path + "accesses.npy"):
        access_matrix, mapping = zeros(shape=(2, len(radios), len(radios)), dtype=bool), {"b": 0, "a": 1}
        for (name_index, noise_index), test_data in load_data(load_path=raw_path + "intervene.test.pkl").items():
            access_matrix[mapping[name_index], noise_index // 5] = mean(mean(test_data, axis=0), axis=1) >= 195
        save_data(save_path=task_path + "accesses.npy", information=access_matrix)


if __name__ == "__main__":
    experiment1(raw_path="../data/results/raw/", task_path="../data/results/task01/")
    experiment2(raw_path="../data/results/raw/", task_path="../data/results/task02/")
    experiment3(raw_path="../data/results/raw/", task_path="../data/results/task03/")
    experiment4(raw_path="../data/results/raw/", task_path="../data/results/task04/", config_path="../data/configs/")
    experiment5(raw_path="../data/results/raw/", task_path="../data/results/task05/", config_path="../data/configs/")
