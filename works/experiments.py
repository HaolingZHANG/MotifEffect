from itertools import product
from numpy import linspace, array, zeros, ones, random, expand_dims, vstack, where, uint8
from numpy import min, mean, sum, abs, median, argmax, all
from os import path, mkdir, listdir

from effect import NeuralMotif, calculate_landscape, estimate_lipschitz_by_signals, maximum_minimum_loss_search
from practice import NEATCartPoleTask, NEATAgent, NormNoiseGenerator, count_motifs_from_matrices, GraphType
from practice import acyclic_motifs, create_agent_config, obtain_best, calculate_matrix_from_agent

from works import Monitor, load_data, save_data

sample_number = 100
activation_selection, aggregation_selection = ["tanh", "sigmoid", "relu"], ["sum", "max"]
motif_types, motif_indices = ["incoherent-loop", "coherent-loop", "collider"], [1, 2, 3, 4]
weight_values, bias_values = linspace(+0.1, +1.0, 10), linspace(-1.0, +1.0, 21)
value_range, points, monitor = (-1, +1), 41, Monitor()
learn_rate = (value_range[1] - value_range[0]) / (points - 1) * 1e-1
loss_threshold, check_threshold, iteration_thresholds = 1e-5, 5, (100, 100)


def experiment1(raw_path, sort_path):
    if not path.exists(path=raw_path + "population/") or not path.exists(path=raw_path + "motif/"):
        print("Generate motifs of the loop populations and the collider population with interval.")
        mkdir(raw_path + "population/")
        mkdir(raw_path + "motif/")
        index, total = 0, len(motif_types) * len(motif_indices) * len(activation_selection) * len(aggregation_selection)
        for motif_type in motif_types:
            for motif_index in motif_indices:
                weight_flags, motif_structure = [], acyclic_motifs[motif_type][motif_index - 1]
                for former, latter in motif_structure.edges:
                    weight_flags.append(motif_structure.get_edge_data(former, latter)["weight"])
                weight_groups = [weight_flag * weight_values for weight_flag in weight_flags]
                if len(motif_structure.edges) == 3:
                    bias_groups = [bias_values, bias_values]
                else:
                    bias_groups = [bias_values]
                for act in activation_selection:
                    for agg in aggregation_selection:
                        if len(motif_structure.edges) == 3:
                            activations, aggregations = [act, act], [agg, agg]
                        else:
                            activations, aggregations = [act], [agg]
                        feature = motif_type + "." + str(motif_index) + "." + activations[0] + "." + aggregations[0]
                        signal_groups, saved_motifs = None, []
                        for weights in product(*weight_groups):
                            for biases in product(*bias_groups):
                                motif = NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                                    activations=activations, aggregations=aggregations,
                                                    weights=weights, biases=biases)
                                signals = calculate_landscape(value_range=value_range, points=points, motif=motif)
                                signals = expand_dims(signals.reshape(-1), axis=0)
                                if signal_groups is not None:
                                    if min(mean(abs(signal_groups - signals), axis=1)) > 0.01:
                                        signal_groups = vstack((signal_groups, signals))
                                        saved_motifs.append(motif)
                                else:
                                    signal_groups = signals
                                    saved_motifs.append(motif)

                        save_data(save_path=raw_path + "population/" + feature + ".npy", information=signal_groups)
                        save_data(save_path=raw_path + "motif/" + feature + ".pkl", information=saved_motifs)
                        monitor(index + 1, total)
                        index += 1

    if not path.exists(path=raw_path + "robustness/"):
        print("Estimate Lipschitz constant for each motif.")
        mkdir(raw_path + "robustness/")
        child_paths = listdir(raw_path + "population/")
        for path_index, child_path in enumerate(child_paths):
            data, lipschitz_values = load_data(load_path=raw_path + "population/" + child_path), []
            for index, signals in enumerate(data):
                output = signals.reshape(points, points)
                lipschitz_values.append(estimate_lipschitz_by_signals(value_range, points, output, norm_type="L-2"))
            save_data(save_path=raw_path + "robustness/" + child_path, information=array(lipschitz_values))
            monitor(path_index + 1, len(child_paths))

    if not path.exists(path=raw_path + "difference/"):
        print("Calculate population difference.")
        mkdir(raw_path + "difference/")
        input_parent_path, output_parent_path = raw_path + "population/", raw_path + "difference/"
        index, total = 0, len(activation_selection) * len(aggregation_selection)
        total *= (len(motif_types) * len(motif_indices)) * (len(motif_types) * len(motif_indices) - 1)
        for act in activation_selection:
            for agg in aggregation_selection:
                for mt1 in motif_types:
                    for mt2 in motif_types:
                        for mi1 in [str(i) for i in motif_indices]:
                            for mi2 in [str(i) for i in motif_indices]:
                                if mt1 != mt2 or mi1 != mi2:
                                    input_path_1 = input_parent_path + mt1 + "." + mi1 + "." + act + "." + agg + ".npy"
                                    input_path_2 = input_parent_path + mt2 + "." + mi2 + "." + act + "." + agg + ".npy"
                                    info = "[" + act + "." + agg + "]"
                                    info_1 = mt1 + "." + mi1
                                    info_2 = mt2 + "." + mi2
                                    output_path = output_parent_path + info + " " + info_1 + " for " + info_2 + ".npy"
                                    if not path.exists(output_path):
                                        sources, targets, results = load_data(input_path_1), load_data(input_path_2), []
                                        sources = expand_dims(sources, axis=1)
                                        for source in sources:
                                            matrix = abs(targets - source)
                                            distances = mean(matrix, axis=1)
                                            results.append(min(distances))
                                        save_data(save_path=output_path, information=array(results))
                                    output_path = output_parent_path + info + " " + info_2 + " for " + info_1 + ".npy"
                                    if not path.exists(output_path):
                                        sources, targets, results = load_data(input_path_2), load_data(input_path_1), []
                                        sources = expand_dims(sources, axis=1)
                                        for source in sources:
                                            matrix = abs(targets - source)
                                            distances = mean(matrix, axis=1)
                                            results.append(min(distances))
                                        save_data(save_path=output_path, information=array(results))

                                    monitor(index + 1, total)
                                    index += 1

    if not path.exists(path=sort_path + "supp01.pkl"):
        task_data = {}
        attrs = []
        for mit in motif_types:
            for number in [1, 2, 3, 4]:
                for act in activation_selection:
                    for agg in aggregation_selection:
                        attrs.append((mit, str(number), act, agg))
        matrix = -ones(shape=(len(attrs), len(attrs)))
        for index_1 in range(len(attrs)):
            attr_1 = attrs[index_1]
            for index_2 in range(len(attrs)):
                attr_2 = attrs[index_2]
                if attr_1[2] == attr_2[2] and attr_1[3] == attr_2[3]:
                    used_path = raw_path + "difference/" + "[" + attr_1[2] + "." + attr_2[3] + "] "
                    used_path += attr_1[0] + "." + attr_1[1] + " for " + attr_2[0] + "." + attr_2[1]
                    used_path += ".npy"
                    if path.exists(used_path):
                        data = load_data(load_path=used_path)
                        matrix[index_1, index_2] = len(data[data <= 0.1]) / len(data)
        task_data["a"] = matrix
        save_data(save_path=sort_path + "supp01.pkl", information=task_data)

    if not path.exists(path=sort_path + "supp02.pkl"):
        task_data = {}
        matrix_1, matrix_2 = -ones(shape=(18, 4)), -ones(shape=(18, 4))
        sizes = [(len(weight_values) ** 3) * (len(bias_values) ** 2),
                 (len(weight_values) ** 3) * (len(bias_values) ** 2),
                 (len(weight_values) ** 2) * (len(bias_values) ** 1)]
        for child_path in listdir(raw_path + "population/"):
            data = load_data(load_path=raw_path + "population/" + child_path)
            info = child_path.split(".")
            # noinspection PyTypeChecker
            location_value = motif_types.index(info[0]) * 6
            # noinspection PyTypeChecker
            location_value += activation_selection.index(info[2]) * 2
            # noinspection PyTypeChecker
            location_value += aggregation_selection.index(info[3])
            matrix_1[location_value, int(info[1]) - 1] = len(data)
            # noinspection PyTypeChecker
            matrix_2[location_value, int(info[1]) - 1] = len(data) / float(sizes[motif_types.index(info[0])])
        task_data["a"] = matrix_1
        task_data["b"] = matrix_2
        save_data(save_path=sort_path + "supp02.pkl", information=task_data)

    if not path.exists(path=sort_path + "supp03.pkl"):
        task_data = {}
        matrix = -ones(shape=(18, 4))
        for child_path in listdir(raw_path + "robustness/"):
            data = load_data(load_path=raw_path + "robustness/" + child_path)
            info = child_path.split(".")
            # noinspection PyTypeChecker
            location_value = motif_types.index(info[0]) * 6
            # noinspection PyTypeChecker
            location_value += activation_selection.index(info[2]) * 2
            # noinspection PyTypeChecker
            location_value += aggregation_selection.index(info[3])
            matrix[location_value, int(info[1]) - 1] = median(data)
        task_data["a"] = matrix
        save_data(save_path=sort_path + "supp03.pkl", information=task_data)

    if not path.exists(path=sort_path + "main01.pkl"):
        task_data = {}
        counts = load_data(load_path=sort_path + "supp02.pkl")["a"]
        references = {}
        for activation in activation_selection:
            for aggregation in aggregation_selection:
                references[(activation, aggregation)] = {}
        index = 0
        for source_motif_type in motif_types:
            for activation in activation_selection:
                for aggregation in aggregation_selection:
                    references[(activation, aggregation)][source_motif_type] = []
                    for motif_index in motif_indices:
                        values = zeros(shape=(counts[index, motif_index - 1], 3), dtype=uint8)
                        references[(activation, aggregation)][source_motif_type].append(values)
                    index += 1
        attrs = []
        for mit in motif_types:
            for number in motif_indices:
                for act in activation_selection:
                    for agg in aggregation_selection:
                        attrs.append((mit, str(number), act, agg))
        for index_1 in range(len(attrs)):
            attr_1 = attrs[index_1]
            for index_2 in range(len(attrs)):
                attr_2 = attrs[index_2]
                if attr_1[2] == attr_2[2] and attr_1[3] == attr_2[3]:
                    used_path = raw_path + "difference/" + "[" + attr_1[2] + "." + attr_2[3] + "] "
                    used_path += attr_1[0] + "." + attr_1[1] + " for " + attr_2[0] + "." + attr_2[1]
                    used_path += ".npy"
                    if path.exists(used_path):
                        locations = where(load_data(load_path=used_path) <= 0.1)[0]
                        target_index = motif_types.index(attr_2[0])
                        references[(attr_1[2], attr_1[3])][attr_1[0]][int(attr_1[1]) - 1][locations, target_index] = 1
        occupation, records = {}, {}
        for (activation, aggregation), reference in references.items():
            occupation[(activation, aggregation)] = {}
            for source_motif_type, value_groups in reference.items():
                ignore_index = motif_types.index(source_motif_type)
                used_indices = []
                for index in range(3):
                    if index != ignore_index:
                        used_indices.append(index)
                individual, for_1, for_2, for_both = 0, 0, 0, 0
                for values in value_groups:
                    used_intersections = []
                    for index, intersections in enumerate(values.T):
                        if ignore_index == index:
                            continue
                        used_intersections.append(intersections)
                    intersection_identities = used_intersections[0] * 2 + used_intersections[1]
                    individual += len(where(intersection_identities == 0)[0])
                    for_2 += len(where(intersection_identities == 1)[0])
                    for_1 += len(where(intersection_identities == 2)[0])
                    for_both += len(where(intersection_identities == 3)[0])
                occupation[(activation, aggregation)][source_motif_type] = {}
                occupation[(activation, aggregation)][source_motif_type]["self"] = individual
                occupation[(activation, aggregation)][source_motif_type][motif_types[used_indices[0]]] = for_1
                occupation[(activation, aggregation)][source_motif_type][motif_types[used_indices[1]]] = for_2
                occupation[(activation, aggregation)][source_motif_type]["both"] = for_both
        intersections = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        locations = {("incoherent-loop", "self"): (2, 0),
                     ("incoherent-loop", "coherent-loop"): (5, 0),
                     ("incoherent-loop", "collider"): (4, 0),
                     ("incoherent-loop", "both"): (6, 0),
                     ("coherent-loop", "self"): (1, 1),
                     ("coherent-loop", "incoherent-loop"): (5, 1),
                     ("coherent-loop", "collider"): (3, 1),
                     ("coherent-loop", "both"): (6, 1),
                     ("collider", "self"): (0, 2),
                     ("collider", "incoherent-loop"): (4, 2),
                     ("collider", "coherent-loop"): (3, 2),
                     ("collider", "both"): (6, 2)}
        for (activation, aggregation), record in occupation.items():
            records[(activation, aggregation)] = zeros(shape=(len(intersections), 3))
            for motif_type, information in record.items():
                total_value = sum(list(information.values()))
                for target, value in information.items():
                    records[(activation, aggregation)][locations[(motif_type, target)]] = value / total_value
        task_data["b"] = records
        save_data(save_path=sort_path + "main01.pkl", information=task_data)

    if not path.exists(path=sort_path + "main02.pkl") or not path.exists(path=sort_path + "supp04.pkl"):
        task_data = {}
        counts = load_data(sort_path + "supp02.pkl")["a"].astype(int)
        size_data, location = {}, 0
        for _ in motif_types:
            for activation in activation_selection:
                for aggregation in aggregation_selection:
                    if (activation, aggregation) in size_data:
                        size_data[(activation, aggregation)].append(counts[location, :].tolist())
                    else:
                        size_data[(activation, aggregation)] = [counts[location, :].tolist()]
                    location += 1
        lipschitz_values = load_data(sort_path + "supp03.pkl")["a"]
        robust_data, location = {}, 0
        for _ in motif_types:
            for activation in activation_selection:
                for aggregation in aggregation_selection:
                    if (activation, aggregation) in robust_data:
                        robust_data[(activation, aggregation)].append(lipschitz_values[location, :].tolist())
                    else:
                        robust_data[(activation, aggregation)] = [lipschitz_values[location, :].tolist()]
                    location += 1
        task_data["a"] = (size_data, robust_data)
        references = {}
        for activation in activation_selection:
            for aggregation in aggregation_selection:
                references[(activation, aggregation)] = {}
        index = 0
        for source_motif_type in motif_types:
            for activation in activation_selection:
                for aggregation in aggregation_selection:
                    references[(activation, aggregation)][source_motif_type] = []
                    for motif_index in motif_indices:
                        reference = zeros(shape=(counts[index, motif_index - 1], 3), dtype=uint8)
                        references[(activation, aggregation)][source_motif_type].append(reference)
                    index += 1
        attrs = []
        for mit in motif_types:
            for number in motif_indices:
                for act in activation_selection:
                    for agg in aggregation_selection:
                        attrs.append((mit, str(number), act, agg))
        for index_1 in range(len(attrs)):
            attr_1 = attrs[index_1]
            for index_2 in range(len(attrs)):
                attr_2 = attrs[index_2]
                if attr_1[2] == attr_2[2] and attr_1[3] == attr_2[3]:
                    used_path = raw_path + "diff/" + "[" + attr_1[2] + "." + attr_2[3] + "] "
                    used_path += attr_1[0] + "." + attr_1[1] + " for " + attr_2[0] + "." + attr_2[1]
                    used_path += ".npy"
                    if path.exists(used_path):
                        locations = where(load_data(load_path=used_path) <= 0.1)[0]
                        target_index = motif_types.index(attr_2[0])
                        references[(attr_1[2], attr_1[3])][attr_1[0]][int(attr_1[1]) - 1][locations, target_index] = 1
        robust_distributions = {}
        locations = {("incoherent-loop", "self"): (2, 0),
                     ("incoherent-loop", "coherent-loop"): (5, 0),
                     ("incoherent-loop", "collider"): (4, 0),
                     ("incoherent-loop", "both"): (6, 0),
                     ("coherent-loop", "self"): (1, 1),
                     ("coherent-loop", "incoherent-loop"): (5, 1),
                     ("coherent-loop", "collider"): (3, 1),
                     ("coherent-loop", "both"): (6, 1),
                     ("collider", "self"): (0, 2),
                     ("collider", "incoherent-loop"): (4, 2),
                     ("collider", "coherent-loop"): (3, 2),
                     ("collider", "both"): (6, 2)}
        for (activation, aggregation), reference in references.items():
            robust_distributions[(activation, aggregation)] = {}
            for source_motif_type, value_groups in reference.items():
                robust_groups = []
                for motif_index in motif_indices:
                    used_path = raw_path + "robustness/" + source_motif_type + "." + str(motif_index) + "."
                    used_path += activation + "." + aggregation + ".npy"
                    robust_data = load_data(load_path=used_path)
                    robust_groups.append(robust_data)
                ignore_index = motif_types.index(source_motif_type)
                used_indices = []
                for index in range(3):
                    if index != ignore_index:
                        used_indices.append(index)
                iid, for_1, for_2, both = [], [], [], []
                for intersected_values, robust_values in zip(value_groups, robust_groups):
                    used_intersections = []
                    for index, intersections in enumerate(intersected_values.T):
                        if ignore_index == index:
                            continue
                        used_intersections.append(intersections)
                    intersection_identities = used_intersections[0] * 2 + used_intersections[1]
                    iid += robust_values[where(intersection_identities == 0)[0]].tolist()
                    for_2 += robust_values[where(intersection_identities == 1)[0]].tolist()
                    for_1 += robust_values[where(intersection_identities == 2)[0]].tolist()
                    both += robust_values[where(intersection_identities == 3)[0]].tolist()
                key = locations[(source_motif_type, "self")]
                robust_distributions[(activation, aggregation)][key] = array(iid)
                key = locations[(source_motif_type, motif_types[used_indices[0]])]
                robust_distributions[(activation, aggregation)][key] = array(for_1)
                key = locations[(source_motif_type, motif_types[used_indices[1]])]
                robust_distributions[(activation, aggregation)][key] = array(for_2)
                key = locations[(source_motif_type, "both")]
                robust_distributions[(activation, aggregation)][key] = array(both)
        task_data["b"] = robust_distributions[("tanh", "sum")]
        save_data(save_path=sort_path + "main02.pkl", information=task_data)
        task_data, panel_indices, index = {}, ["a", "b", "c", "d", "e", "f"], 0
        for activation in activation_selection:
            for aggregation in aggregation_selection:
                task_data[panel_indices[index]] = robust_distributions[(activation, aggregation)]
        save_data(save_path=sort_path + "supp04.pkl", information=task_data)


def experiment2(raw_path, sort_path):
    if not path.exists(path=raw_path + "max-min.search.pkl"):
        attrs = []
        for mit in motif_types:
            for number in motif_indices:
                attrs.append((mit, str(number), "tanh", "sum"))
        references, index, counts = {}, 0, load_data(sort_path + "supp02.pkl")["a"].astype(int)
        for source_motif_type in motif_types:
            for activation in activation_selection:
                for aggregation in aggregation_selection:
                    if activation == "tanh" and aggregation == "sum":
                        references[source_motif_type] = []
                        for motif_index in motif_indices:
                            reference = zeros(shape=(counts[index, motif_index - 1], 3), dtype=uint8)
                            references[source_motif_type].append(reference)
                    index += 1
        for index_1 in range(len(attrs)):
            attr_1 = attrs[index_1]
            for index_2 in range(len(attrs)):
                attr_2 = attrs[index_2]
                if attr_1[2] == attr_2[2] and attr_1[3] == attr_2[3]:
                    used_path = raw_path + "difference/" + "[" + attr_1[2] + "." + attr_2[3] + "] "
                    used_path += attr_1[0] + "." + attr_1[1] + " for " + attr_2[0] + "." + attr_2[1] + ".npy"
                    if path.exists(used_path):
                        locations = where(load_data(load_path=used_path) <= 0.1)[0]
                        target_index = motif_types.index(attr_2[0])
                        references[attr_1[0]][int(attr_1[1]) - 1][locations, target_index] = 1
        # noinspection PyTypeChecker
        random.seed(2023)
        motif_collection = {}
        for source_motif_type, value_groups in references.items():
            if source_motif_type in ["incoherent-loop", "coherent-loop"]:
                ignore_index = motif_types.index(source_motif_type)
                used_indices = []
                for index in range(3):
                    if index != ignore_index:
                        used_indices.append(index)
                saved_motifs = []
                for source_motif_index, intersected_values in enumerate(value_groups):
                    used_intersections = []
                    for index, intersections in enumerate(intersected_values.T):
                        if ignore_index == index:
                            continue
                        used_intersections.append(intersections)
                    intersection_identities = used_intersections[0] * 2 + used_intersections[1]
                    satisfied_indices = where(intersection_identities == 0)[0]
                    data_path = raw_path + "motif/" + source_motif_type + "." + str(source_motif_index + 1)
                    data_path += ".tanh.sum.pkl"
                    motifs, satisfied_motifs = load_data(load_path=data_path), []
                    # noinspection PyArgumentList
                    random.shuffle(satisfied_indices)
                    satisfied_motifs = []
                    for satisfied_index in satisfied_indices[:sample_number]:
                        satisfied_motifs.append(motifs[satisfied_index])
                    saved_motifs.append(satisfied_motifs)
                motif_collection[source_motif_type] = saved_motifs
        # noinspection PyTypeChecker
        random.seed(None)
        record = {"incoherent-loop": {}, "coherent-loop": {}}
        for index, source_motifs in enumerate(motif_collection["incoherent-loop"]):
            record["incoherent-loop"][index + 1] = []
            for source_index, source_motif in enumerate(source_motifs):
                target_motifs = []
                for motif_index in motif_indices:
                    weight_flags, motif_structure = [], acyclic_motifs["collider"][motif_index - 1]
                    for former, latter in motif_structure.edges:
                        weight_flags.append(motif_structure.get_edge_data(former, latter)["weight"])
                    for weights in product(*[weight_flag * array([0.5, 1.0]) for weight_flag in weight_flags]):
                        for biases in product(*[array([-0.5, 0.0, 0.5])]):
                            motif = NeuralMotif(motif_type="collider", motif_index=motif_index,
                                                activations=["tanh"], aggregations=["sum"],
                                                weights=weights, biases=biases)
                            target_motifs.append(motif)
                result = maximum_minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                                     source_motif=source_motif, target_motifs=target_motifs,
                                                     loss_threshold=loss_threshold, check_threshold=check_threshold,
                                                     iteration_thresholds=iteration_thresholds, verbose=True)
                motifs, losses = result
                robust = []
                for motif_1, motif_2 in motifs:
                    landscape = calculate_landscape(value_range, points, motif_1)
                    value_1 = estimate_lipschitz_by_signals(value_range, points, landscape)
                    landscape = calculate_landscape(value_range, points, motif_2)
                    value_2 = estimate_lipschitz_by_signals(value_range, points, landscape)
                    robust.append([value_1, value_2])
                record["incoherent-loop"][index + 1].append((motifs, array(losses), array(robust)))
        for index, source_motifs in enumerate(motif_collection["coherent-loop"]):
            record["coherent-loop"][index + 1] = []
            for source_index, source_motif in enumerate(source_motifs):
                target_motifs = []
                for motif_index in motif_indices:
                    weight_flags, motif_structure = [], acyclic_motifs["collider"][motif_index - 1]
                    for former, latter in motif_structure.edges:
                        weight_flags.append(motif_structure.get_edge_data(former, latter)["weight"])
                    for weights in product(*[weight_flag * array([0.5, 1.0]) for weight_flag in weight_flags]):
                        for biases in product(*[array([-0.5, 0.0, 0.5])]):
                            motif = NeuralMotif(motif_type="collider", motif_index=motif_index,
                                                activations=["tanh"], aggregations=["sum"],
                                                weights=weights, biases=biases)
                            target_motifs.append(motif)
                result = maximum_minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                                     source_motif=source_motif, target_motifs=target_motifs,
                                                     loss_threshold=loss_threshold, check_threshold=check_threshold,
                                                     iteration_thresholds=iteration_thresholds, verbose=True)
                motifs, losses = result
                robust = []
                for motif_1, motif_2 in motifs:
                    landscape = calculate_landscape(value_range, points, motif_1)
                    value_1 = estimate_lipschitz_by_signals(value_range, points, landscape)
                    landscape = calculate_landscape(value_range, points, motif_2)
                    value_2 = estimate_lipschitz_by_signals(value_range, points, landscape)
                    robust.append([value_1, value_2])
                record["coherent-loop"][index + 1].append((motifs, array(losses), array(robust)))
        save_data(save_path=raw_path + "max-min.search.pkl", information=record)

    if not path.exists(path=sort_path + "main03.pkl"):
        task_data = {}
        robust, losses = [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []]
        record, index = load_data(load_path=raw_path + "max-min.search.pkl"), 0
        for motif_type in ["incoherent-loop", "coherent-loop"]:
            for motif_index in [1, 2, 3, 4]:
                for sample in record[motif_type][motif_index - 1]:
                    losses[index].append(sample[1][-1] - sample[1][0])
                    robust[index].append(sample[2][-1, 0] - sample[2][0, 0])
                index += 1
        task_data["a"] = (robust, losses)
        # terminal cases after visualization.
        sample_1 = record["incoherent-loop"][3][99]
        case_1 = [(calculate_landscape(value_range, points, sample_1[0][0][0]), sample_1[2][0, 0]),
                  (calculate_landscape(value_range, points, sample_1[0][0][1]), sample_1[2][0, 1]),
                  (calculate_landscape(value_range, points, sample_1[0][-1][0]), sample_1[2][-1, 0]),
                  (calculate_landscape(value_range, points, sample_1[0][-1][1]), sample_1[2][-1, 1])]
        sample_2 = record["incoherent-loop"][2][89]
        case_2 = [(calculate_landscape(value_range, points, sample_2[0][0][0]), sample_2[2][0, 0]),
                  (calculate_landscape(value_range, points, sample_2[0][0][1]), sample_2[2][0, 1]),
                  (calculate_landscape(value_range, points, sample_2[0][-1][0]), sample_2[2][-1, 0]),
                  (calculate_landscape(value_range, points, sample_2[0][-1][1]), sample_2[2][-1, 1])]
        sample_3 = record["coherent-loop"][3][42]
        case_3 = [(calculate_landscape(value_range, points, sample_3[0][0][0]), sample_3[2][0, 0]),
                  (calculate_landscape(value_range, points, sample_3[0][0][1]), sample_3[2][0, 1]),
                  (calculate_landscape(value_range, points, sample_3[0][-1][0]), sample_3[2][-1, 0]),
                  (calculate_landscape(value_range, points, sample_3[0][-1][1]), sample_3[2][-1, 1])]
        sample_4 = record["coherent-loop"][1][17]
        case_4 = [(calculate_landscape(value_range, points, sample_4[0][0][0]), sample_4[2][0, 0]),
                  (calculate_landscape(value_range, points, sample_4[0][0][1]), sample_4[2][0, 1]),
                  (calculate_landscape(value_range, points, sample_4[0][-1][0]), sample_4[2][-1, 0]),
                  (calculate_landscape(value_range, points, sample_4[0][-1][1]), sample_4[2][-1, 1])]
        task_data["b"] = {"i": (case_1, case_2), "c": (case_3, case_4)}
        results = {"i": [], "c": []}
        sample_1 = record["incoherent-loop"][0][69]
        former = calculate_landscape(value_range, points, sample_1[0][0][0])
        latter = calculate_landscape(value_range, points, sample_1[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
        results["i"].append([former, latter - former, latter, value_1, value_2])
        sample_2 = record["incoherent-loop"][0][25]
        former = calculate_landscape(value_range, points, sample_2[0][0][0])
        latter = calculate_landscape(value_range, points, sample_2[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
        results["i"].append([former, latter - former, latter, value_1, value_2])
        sample_3 = record["incoherent-loop"][0][48]
        former = calculate_landscape(value_range, points, sample_3[0][0][0])
        latter = calculate_landscape(value_range, points, sample_3[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
        results["i"].append([former, latter - former, latter, value_1, value_2])
        sample_4 = record["incoherent-loop"][0][34]
        former = calculate_landscape(value_range, points, sample_4[0][0][0])
        latter = calculate_landscape(value_range, points, sample_4[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
        results["i"].append([former, latter - former, latter, value_1, value_2])
        sample_5 = record["coherent-loop"][0][84]
        former = calculate_landscape(value_range, points, sample_5[0][0][0])
        latter = calculate_landscape(value_range, points, sample_5[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
        results["c"].append([former, latter - former, latter, value_1, value_2])
        sample_6 = record["coherent-loop"][0][90]
        former = calculate_landscape(value_range, points, sample_6[0][0][0])
        latter = calculate_landscape(value_range, points, sample_6[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
        results["c"].append([former, latter - former, latter, value_1, value_2])
        sample_7 = record["coherent-loop"][0][25]
        former = calculate_landscape(value_range, points, sample_7[0][0][0])
        latter = calculate_landscape(value_range, points, sample_7[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
        results["c"].append([former, latter - former, latter, value_1, value_2])
        sample_8 = record["coherent-loop"][0][17]
        former = calculate_landscape(value_range, points, sample_8[0][0][0])
        latter = calculate_landscape(value_range, points, sample_8[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
        results["c"].append([former, latter - former, latter, value_1, value_2])
        task_data["c"] = results
        save_data(save_path=sort_path + "main03.pkl", information=task_data)


def experiment3(raw_path, sort_path, config_path):
    agent_names = ["baseline", "adjusted[i]", "adjusted[c]", "adjusted[i+c]"]
    agent_configs = [create_agent_config(path=config_path + "cartpole-v0.default." + name) for name in agent_names]
    radios, saved_names = [0.0, 0.1, 0.2, 0.3, 0.4], ["b", "i", "c", "a"]

    def record_handle(a):
        motif_counts = count_motifs_from_matrices(matrices=[calculate_matrix_from_agent(a)], search_size=3,
                                                  graph_type=GraphType.pn, pruning=False)[0]
        return a, sum(motif_counts.reshape(3, 4), axis=1)

    if not path.exists(path=raw_path + "practice.adjustments.1.pkl"):
        record, maximum_generation = {}, 20
        for agent_name, agent_config, saved_name in zip(agent_names, agent_configs, saved_names):
            record[saved_name] = {}
            for train_radio in radios:
                record[saved_name][train_radio] = []
                for repeat in range(sample_number):
                    noise_generator = NormNoiseGenerator(norm_type="L-2", noise_scale=train_radio)
                    task = NEATCartPoleTask(maximum_generation=maximum_generation, noise_generator=noise_generator)
                    task.reset_experience(record_handle=record_handle)
                    best_genome = obtain_best(task, agent_config, need_stdout=False)
                    if best_genome is not None:
                        agent = NEATAgent(model_genome=best_genome, neat_config=agent_config, description=agent_name,
                                          action_handle=argmax)
                        experience = task.get_experience()
                        record_1, record_2 = {"agent": agent, "experience": experience}, {}
                        for index, noise_scale in enumerate(radios):
                            task.noise_generator.noise_scale, fitnesses = noise_scale, []
                            reward_collector = task.run(agent=agent)["rewards"]
                            for rewards in reward_collector:
                                fitnesses.append(sum(rewards))
                            record_2[noise_scale] = fitnesses
                        record[saved_name][train_radio].append((record_1, record_2))
        save_data(save_path=raw_path + "practice.adjustments.1.pkl", information=record)

    if not path.exists(path=raw_path + "practice.adjustments.2.pkl"):
        record, maximum_generation, train_radio = {}, 100, 0.3
        for agent_name, agent_config, saved_name in zip(agent_names, agent_configs, saved_names):
            record[saved_name] = []
            for repeat in range(sample_number):
                noise_generator = NormNoiseGenerator(norm_type="L-2", noise_scale=train_radio)
                task = NEATCartPoleTask(maximum_generation=maximum_generation, noise_generator=noise_generator)
                task.reset_experience(record_handle=record_handle)
                best_genome = obtain_best(task, agent_config, need_stdout=False)
                if best_genome is not None:
                    agent = NEATAgent(model_genome=best_genome, neat_config=agent_config, description=agent_name,
                                      action_handle=argmax)
                    experience = task.get_experience()
                    record_1, record_2 = {"agent": agent, "experience": experience}, {}
                    for index, noise_scale in enumerate(radios):
                        task.noise_generator.noise_scale, fitnesses = noise_scale, []
                        reward_collector = task.run(agent=agent)["rewards"]
                        for rewards in reward_collector:
                            fitnesses.append(sum(rewards))
                        record_2[noise_scale] = fitnesses
                    record[saved_name].append((record_1, record_2))
        save_data(save_path=raw_path + "practice.adjustments.2.pkl", information=record)

    if not path.exists(path=sort_path + "main04.pkl"):
        task_data = {}
        record_1 = load_data(raw_path + "practice.adjustments.1.pkl")
        matrix = zeros(shape=(4, 5))
        for strategy_index, strategy in enumerate(["b", "i", "c", "a"]):
            for noise_index, noise in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
                average_performance = mean([a[0]["agent"].get_fitness() for a in record_1[strategy][noise]])
                matrix[strategy_index, noise_index] = average_performance
        task_data["b"] = matrix
        matrix = zeros(shape=(4, 5))
        for strategy_index, strategy in enumerate(["b", "i", "c", "a"]):
            for noise_index, noise in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
                median_generation = median([len(a[0]["experience"]) for a in record_1[strategy][noise]])
                matrix[strategy_index, noise_index] = median_generation
        task_data["c"] = matrix
        matrix = zeros(shape=(4, 5, 5))
        for strategy_index, strategy in enumerate(["b", "i", "c", "a"]):
            values = zeros(shape=(5, 5))
            for train_noise_index, noise_1 in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
                for a in record_1[strategy][noise_1]:
                    sub_values = []
                    for evaluate_noise_index, noise_2 in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
                        sub_values.append(mean(a[1][noise_2]))
                    values[train_noise_index] += sub_values
            matrix[strategy_index] = values / 100.0
        task_data["d"] = matrix
        record_2 = load_data(raw_path + "practice.adjustments.2.pkl")
        matrix = zeros(shape=(4, 2))
        cases = {}
        for strategy_index, strategy in enumerate(["b", "i", "c", "a"]):
            count_1 = 0
            for a in record_1[strategy][0.3]:
                sub_values = []
                for evaluate_noise_index, noise_2 in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
                    sub_values.append(mean(a[1][noise_2]))
                if min(sub_values[:4]) >= 195:
                    count_1 += 1
            count_2, cases[strategy] = 0, [[], [], []]
            for a in record_2[strategy]:
                sub_values = []
                for evaluate_noise_index, noise_2 in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
                    sub_values.append(mean(a[1][noise_2]))
                if min(sub_values[:4]) >= 195:
                    count_2 += 1
                else:
                    sub_values = array(sub_values)
                    if all(sub_values < 195) and sub_values[0] > sub_values[-1] and sub_values[0] > sub_values[2]:
                        cases[strategy][0].append(sub_values.tolist())
                    elif all(sub_values < 195):
                        cases[strategy][1].append(sub_values.tolist())
                    else:
                        cases[strategy][2].append(sub_values.tolist())
            matrix[strategy_index] = [count_1 / 100.0, count_2 / 100.0]
            a = (mean(array(cases[strategy][2]), axis=0), len(cases[strategy][2]), (100 - count_2))
            if len(cases[strategy][0]) > 0:
                b = (mean(array(cases[strategy][0]), axis=0), len(cases[strategy][0]), (100 - count_2))
            else:
                b = (None, 0, (100 - count_2))
            if len(cases[strategy][1]) > 0:
                c = (mean(array(cases[strategy][1]), axis=0), len(cases[strategy][1]), (100 - count_2))
            else:
                c = (None, 0, (100 - count_2))
            cases[strategy] = [a, b, c]
        task_data["e"] = matrix
        task_data["f"] = cases
        save_data(save_path=sort_path + "main04.pkl", information=task_data)


if __name__ == "__main__":
    # experiment1(raw_path="./raw/", sort_path="./data/")
    # experiment2(raw_path="./raw/", sort_path="./data/")
    # experiment3(raw_path="./raw/", sort_path="./data/", config_path="./confs/")
    task_data = {}
    robust, losses = [[], []], [[], []]
    record = load_data(load_path="./raw/max-min.search.pkl")
    for index, motif_type in enumerate(["incoherent-loop", "coherent-loop"]):
        for motif_index in [1, 2, 3, 4]:
            for sample in record[motif_type][motif_index]:
                losses[index].append(sample[1][-1] - sample[1][0])
                robust[index].append(sample[2][-1, 0] - sample[2][0, 0])
    task_data["a"] = (losses, robust)
    # terminal cases after visualization.
    sample_1 = record["incoherent-loop"][4][99]
    case_1 = [(calculate_landscape(value_range, points, sample_1[0][0][0]), sample_1[2][0, 0]),
              (calculate_landscape(value_range, points, sample_1[0][0][1]), sample_1[2][0, 1]),
              (calculate_landscape(value_range, points, sample_1[0][-1][0]), sample_1[2][-1, 0]),
              (calculate_landscape(value_range, points, sample_1[0][-1][1]), sample_1[2][-1, 1])]
    sample_2 = record["incoherent-loop"][3][89]
    case_2 = [(calculate_landscape(value_range, points, sample_2[0][0][0]), sample_2[2][0, 0]),
              (calculate_landscape(value_range, points, sample_2[0][0][1]), sample_2[2][0, 1]),
              (calculate_landscape(value_range, points, sample_2[0][-1][0]), sample_2[2][-1, 0]),
              (calculate_landscape(value_range, points, sample_2[0][-1][1]), sample_2[2][-1, 1])]
    sample_3 = record["coherent-loop"][4][42]
    case_3 = [(calculate_landscape(value_range, points, sample_3[0][0][0]), sample_3[2][0, 0]),
              (calculate_landscape(value_range, points, sample_3[0][0][1]), sample_3[2][0, 1]),
              (calculate_landscape(value_range, points, sample_3[0][-1][0]), sample_3[2][-1, 0]),
              (calculate_landscape(value_range, points, sample_3[0][-1][1]), sample_3[2][-1, 1])]
    sample_4 = record["coherent-loop"][2][17]
    case_4 = [(calculate_landscape(value_range, points, sample_4[0][0][0]), sample_4[2][0, 0]),
              (calculate_landscape(value_range, points, sample_4[0][0][1]), sample_4[2][0, 1]),
              (calculate_landscape(value_range, points, sample_4[0][-1][0]), sample_4[2][-1, 0]),
              (calculate_landscape(value_range, points, sample_4[0][-1][1]), sample_4[2][-1, 1])]
    task_data["b"] = {"i": (case_1, case_2), "c": (case_3, case_4)}
    results = {"i": [], "c": []}
    sample_1 = record["incoherent-loop"][1][69]
    former = calculate_landscape(value_range, points, sample_1[0][0][0])
    latter = calculate_landscape(value_range, points, sample_1[0][-1][0])
    value_1 = estimate_lipschitz_by_signals(value_range, points, former)
    value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
    results["i"].append([former, latter - former, latter, value_1, value_2])
    sample_2 = record["incoherent-loop"][1][25]
    former = calculate_landscape(value_range, points, sample_2[0][0][0])
    latter = calculate_landscape(value_range, points, sample_2[0][-1][0])
    value_1 = estimate_lipschitz_by_signals(value_range, points, former)
    value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
    results["i"].append([former, latter - former, latter, value_1, value_2])
    sample_3 = record["incoherent-loop"][1][48]
    former = calculate_landscape(value_range, points, sample_3[0][0][0])
    latter = calculate_landscape(value_range, points, sample_3[0][-1][0])
    value_1 = estimate_lipschitz_by_signals(value_range, points, former)
    value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
    results["i"].append([former, latter - former, latter, value_1, value_2])
    sample_4 = record["incoherent-loop"][1][34]
    former = calculate_landscape(value_range, points, sample_4[0][0][0])
    latter = calculate_landscape(value_range, points, sample_4[0][-1][0])
    value_1 = estimate_lipschitz_by_signals(value_range, points, former)
    value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
    results["i"].append([former, latter - former, latter, value_1, value_2])
    sample_5 = record["coherent-loop"][1][84]
    former = calculate_landscape(value_range, points, sample_5[0][0][0])
    latter = calculate_landscape(value_range, points, sample_5[0][-1][0])
    value_1 = estimate_lipschitz_by_signals(value_range, points, former)
    value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
    results["c"].append([former, latter - former, latter, value_1, value_2])
    sample_6 = record["coherent-loop"][1][90]
    former = calculate_landscape(value_range, points, sample_6[0][0][0])
    latter = calculate_landscape(value_range, points, sample_6[0][-1][0])
    value_1 = estimate_lipschitz_by_signals(value_range, points, former)
    value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
    results["c"].append([former, latter - former, latter, value_1, value_2])
    sample_7 = record["coherent-loop"][1][25]
    former = calculate_landscape(value_range, points, sample_7[0][0][0])
    latter = calculate_landscape(value_range, points, sample_7[0][-1][0])
    value_1 = estimate_lipschitz_by_signals(value_range, points, former)
    value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
    results["c"].append([former, latter - former, latter, value_1, value_2])
    sample_8 = record["coherent-loop"][1][17]
    former = calculate_landscape(value_range, points, sample_8[0][0][0])
    latter = calculate_landscape(value_range, points, sample_8[0][-1][0])
    value_1 = estimate_lipschitz_by_signals(value_range, points, former)
    value_2 = estimate_lipschitz_by_signals(value_range, points, latter)
    results["c"].append([former, latter - former, latter, value_1, value_2])
    task_data["c"] = results
    save_data(save_path="./data/main03.pkl", information=task_data)
