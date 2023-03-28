from hashlib import md5
from numpy import linspace, array, zeros, ones, random, min, mean, sum, median, ceil, all, where, uint8
from os import path, mkdir, listdir
from umap import UMAP

from effect import calculate_landscape, calculate_gradients, estimate_lipschitz_by_signals, generate_qualified_motifs
from effect import calculate_motif_differences, calculate_population_differences, maximum_minimum_loss_search

from practice import acyclic_motifs, NEATCartPoleTask, NormNoiseGenerator, create_agent_config, train_and_test

from works import load_data, save_data

sample_number = 100
activation_selection, aggregation_selection = ["tanh", "sigmoid", "relu"], ["sum", "max"]
motif_types, motif_indices = ["incoherent-loop", "coherent-loop", "collider"], [1, 2, 3, 4]
weight_values, bias_values = linspace(+0.1, +1.0, 10), linspace(-1.0, +1.0, 21)
value_range, points, threshold = (-1, +1), 41, 1e-2
learn_rate = (value_range[1] - value_range[0]) / (points - 1) * 1e-1
loss_threshold, check_threshold, iteration_thresholds = 1e-5, 5, (100, 100)


def experiment1(raw_path, sort_path):
    if not path.exists(path=raw_path + "motif/") or not path.exists(path=raw_path + "population/"):
        mkdir(raw_path + "motif/")
        mkdir(raw_path + "population/")
        for motif_type in motif_types:
            for motif_index in motif_indices:
                for activation in activation_selection:
                    for aggregation in aggregation_selection:
                        weight_flags, motif_structure = [], acyclic_motifs[motif_type][motif_index - 1]
                        for former, latter in motif_structure.edges:
                            weight_flags.append(motif_structure.get_edge_data(former, latter)["weight"])

                        weight_groups = [weight_flag * weight_values for weight_flag in weight_flags]
                        if len(motif_structure.edges) == 3:
                            bias_groups = [bias_values, bias_values]
                        else:
                            bias_groups = [bias_values]
                        if len(motif_structure.edges) == 3:
                            activations, aggregations = [activation, activation], [aggregation, aggregation]
                        else:
                            activations, aggregations = [activation], [aggregation]
                        feature = motif_type + "." + str(motif_index) + "." + activation + "." + aggregation
                        result = generate_qualified_motifs(motif_type, motif_index, activations, aggregations,
                                                           weight_groups, bias_groups, value_range, points, threshold)
                        motifs, signals = result
                        save_data(save_path=raw_path + "motif/" + feature + ".pkl", information=motifs)
                        save_data(save_path=raw_path + "population/" + feature + ".npy", information=signals)

    if not path.exists(path=raw_path + "robustness/"):
        mkdir(raw_path + "robustness/")
        child_paths = listdir(raw_path + "population/")
        for path_index, child_path in enumerate(child_paths):
            data, lipschitz_values = load_data(load_path=raw_path + "population/" + child_path), []
            for index, signals in enumerate(data):
                output = signals.reshape(points, points)
                lipschitz_values.append(estimate_lipschitz_by_signals(value_range, points, output, norm_type="L-2"))
            save_data(save_path=raw_path + "robustness/" + child_path, information=array(lipschitz_values))

    if not path.exists(path=raw_path + "difference/"):
        mkdir(raw_path + "difference/")
        input_parent_path, output_parent_path = raw_path + "population/", raw_path + "difference/"
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
                                        sources, targets = load_data(input_path_1), load_data(input_path_2)
                                        results = calculate_population_differences(sources, targets)
                                        save_data(save_path=output_path, information=results)
                                    output_path = output_parent_path + info + " " + info_2 + " for " + info_1 + ".npy"
                                    if not path.exists(output_path):
                                        sources, targets = load_data(input_path_2), load_data(input_path_1)
                                        results = calculate_population_differences(sources, targets)
                                        save_data(save_path=output_path, information=results)

    if not path.exists(path=sort_path + "supp01.pkl"):
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
        save_data(save_path=sort_path + "supp01.pkl", information=task_data)

    if not path.exists(path=sort_path + "supp02.pkl"):
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

    if not path.exists(path=sort_path + "main02.pkl") or \
            not path.exists(path=sort_path + "supp04.pkl") or \
            not path.exists(path=sort_path + "supp05.pkl"):
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
        save_data(save_path=sort_path + "supp04.pkl", information={"a": task_data["b"]})
        task_data, panel_indices, index = {}, ["a", "b", "c", "d", "e", "f"], 0
        for activation in activation_selection:
            for aggregation in aggregation_selection:
                task_data[panel_indices[index]] = robust_distributions[(activation, aggregation)]
        save_data(save_path=sort_path + "supp05.pkl", information=task_data)


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
                    activations, aggregations = ["tanh"], ["sum"]
                    weight_flags, motif_structure = [], acyclic_motifs["collider"][motif_index - 1]
                    for former_landscape, latter_landscape in motif_structure.edges:
                        weight_flags.append(motif_structure.get_edge_data(former_landscape, latter_landscape)["weight"])
                    weight_groups = [weight_flag * weight_values for weight_flag in weight_flags]
                    bias_groups = [bias_values]
                    target_motifs += generate_qualified_motifs("collider", motif_index, activations, aggregations,
                                                               weight_groups, bias_groups, value_range, points)[0]
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
                    activations, aggregations = ["tanh"], ["sum"]
                    weight_flags, motif_structure = [], acyclic_motifs["collider"][motif_index - 1]
                    for former_landscape, latter_landscape in motif_structure.edges:
                        weight_flags.append(motif_structure.get_edge_data(former_landscape, latter_landscape)["weight"])
                    weight_groups = [weight_flag * weight_values for weight_flag in weight_flags]
                    bias_groups = [bias_values]
                    target_motifs += generate_qualified_motifs("collider", motif_index, activations, aggregations,
                                                               weight_groups, bias_groups, value_range, points)[0]
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
        robust_change, loss_change = [[], []], [[], []]
        record = load_data(load_path=raw_path + "max-min.search.pkl")
        for index, motif_type in enumerate(["incoherent-loop", "coherent-loop"]):
            for motif_index in [1, 2, 3, 4]:
                for sample in record[motif_type][motif_index]:
                    robust_change[index].append(sample[2][-1, 0] - sample[2][0, 0])
                    loss_change[index].append(sample[1][-1] - sample[1][0])
        task_data["a"] = (robust_change, loss_change)
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
        sample_1 = record["incoherent-loop"][1][69]
        former_landscape = calculate_landscape(value_range, points, sample_1[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_1[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former_landscape)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter_landscape)
        results["i"].append([former_landscape, latter_landscape - former_landscape, latter_landscape, value_1, value_2])
        sample_2 = record["incoherent-loop"][1][25]
        former_landscape = calculate_landscape(value_range, points, sample_2[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_2[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former_landscape)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter_landscape)
        results["i"].append([former_landscape, latter_landscape - former_landscape, latter_landscape, value_1, value_2])
        sample_3 = record["incoherent-loop"][1][48]
        former_landscape = calculate_landscape(value_range, points, sample_3[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_3[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former_landscape)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter_landscape)
        results["i"].append([former_landscape, latter_landscape - former_landscape, latter_landscape, value_1, value_2])
        sample_4 = record["incoherent-loop"][1][34]
        former_landscape = calculate_landscape(value_range, points, sample_4[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_4[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former_landscape)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter_landscape)
        results["i"].append([former_landscape, latter_landscape - former_landscape, latter_landscape, value_1, value_2])
        sample_5 = record["coherent-loop"][1][84]
        former_landscape = calculate_landscape(value_range, points, sample_5[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_5[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former_landscape)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter_landscape)
        results["c"].append([former_landscape, latter_landscape - former_landscape, latter_landscape, value_1, value_2])
        sample_6 = record["coherent-loop"][1][90]
        former_landscape = calculate_landscape(value_range, points, sample_6[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_6[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former_landscape)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter_landscape)
        results["c"].append([former_landscape, latter_landscape - former_landscape, latter_landscape, value_1, value_2])
        sample_7 = record["coherent-loop"][1][25]
        former_landscape = calculate_landscape(value_range, points, sample_7[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_7[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former_landscape)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter_landscape)
        results["c"].append([former_landscape, latter_landscape - former_landscape, latter_landscape, value_1, value_2])
        sample_8 = record["coherent-loop"][1][17]
        former_landscape = calculate_landscape(value_range, points, sample_8[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_8[0][-1][0])
        value_1 = estimate_lipschitz_by_signals(value_range, points, former_landscape)
        value_2 = estimate_lipschitz_by_signals(value_range, points, latter_landscape)
        results["c"].append([former_landscape, latter_landscape - former_landscape, latter_landscape, value_1, value_2])
        task_data["c"] = results
        save_data(save_path=sort_path + "main03.pkl", information=task_data)

    if not path.exists(path=sort_path + "supp06.pkl"):
        task_data = {}
        robust_change, loss_change = [[] for _ in range(8)], [[] for _ in range(8)]
        record, index = load_data(load_path=raw_path + "max-min.search.pkl"), 0
        for motif_type in ["incoherent-loop", "coherent-loop"]:
            for motif_index in [1, 2, 3, 4]:
                for sample in record[motif_type][motif_index]:
                    robust_change[index].append(sample[2][-1, 0] - sample[2][0, 0])
                    loss_change[index].append(sample[1][-1] - sample[1][0])
                index += 1
        for index in range(8):
            task_data[chr(ord("a") + index)] = (robust_change[index], loss_change[index])
        save_data(save_path=sort_path + "supp06.pkl", information=task_data)

    if not path.exists(path=sort_path + "supp07.pkl"):
        task_data = {}
        motif_params = [[] for _ in range(8)]
        record, index = load_data(load_path=raw_path + "max-min.search.pkl"), 0
        for motif_type in ["incoherent-loop", "coherent-loop"]:
            for motif_index in [1, 2, 3, 4]:
                for sample in record[motif_type][motif_index]:
                    trained_motif = sample[0][-1][0]
                    weights = [weight.value() for weight in trained_motif.w]
                    biases = [bias.value() for bias in trained_motif.b]
                    motif_params[index].append(weights + biases)
                index += 1
        for index in range(8):
            task_data[chr(ord("a") + index)] = motif_params[index]
        save_data(save_path=sort_path + "supp07.pkl", information=task_data)

    if not path.exists(path=sort_path + "supp08.pkl"):
        task_data = {}
        record, index = load_data(load_path=raw_path + "max-min.search.pkl"), 0
        for motif_type in ["incoherent-loop", "coherent-loop"]:
            for motif_index in [1, 2, 3, 4]:
                landscapes = []
                for sample in record[motif_type][motif_index]:
                    landscapes.append(calculate_landscape(value_range, points, sample[0][-1][0]).reshape(-1).tolist())
                differences = calculate_motif_differences(array(landscapes))
                locations = UMAP(n_components=2, metric="precomputed").fit_transform(X=differences)
                task_data[chr(ord("a") + index)] = (differences, locations)
                index += 1
        save_data(save_path=sort_path + "supp08.pkl", information=task_data)

    if not path.exists(path=sort_path + "supp09.pkl"):
        task_data = {}
        record = load_data(load_path=raw_path + "max-min.search.pkl")
        sample_1 = record["incoherent-loop"][1][69]
        former_landscape = calculate_landscape(value_range, points, sample_1[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_1[0][-1][0])
        former_gradients = calculate_gradients(value_range, points, sample_1[0][0][0])
        latter_gradients = calculate_gradients(value_range, points, sample_1[0][-1][0])
        task_data["a"] = (former_landscape, latter_landscape, former_gradients, latter_gradients)
        sample_2 = record["incoherent-loop"][1][25]
        former_landscape = calculate_landscape(value_range, points, sample_2[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_2[0][-1][0])
        former_gradients = calculate_gradients(value_range, points, sample_2[0][0][0])
        latter_gradients = calculate_gradients(value_range, points, sample_2[0][-1][0])
        task_data["b"] = (former_landscape, latter_landscape, former_gradients, latter_gradients)
        sample_3 = record["incoherent-loop"][1][48]
        former_landscape = calculate_landscape(value_range, points, sample_3[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_3[0][-1][0])
        former_gradients = calculate_gradients(value_range, points, sample_3[0][0][0])
        latter_gradients = calculate_gradients(value_range, points, sample_3[0][-1][0])
        task_data["c"] = (former_landscape, latter_landscape, former_gradients, latter_gradients)
        sample_4 = record["incoherent-loop"][1][34]
        former_landscape = calculate_landscape(value_range, points, sample_4[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_4[0][-1][0])
        former_gradients = calculate_gradients(value_range, points, sample_4[0][0][0])
        latter_gradients = calculate_gradients(value_range, points, sample_4[0][-1][0])
        task_data["d"] = (former_landscape, latter_landscape, former_gradients, latter_gradients)
        sample_5 = record["coherent-loop"][1][84]
        former_landscape = calculate_landscape(value_range, points, sample_5[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_5[0][-1][0])
        former_gradients = calculate_gradients(value_range, points, sample_5[0][0][0])
        latter_gradients = calculate_gradients(value_range, points, sample_5[0][-1][0])
        task_data["e"] = (former_landscape, latter_landscape, former_gradients, latter_gradients)
        sample_6 = record["coherent-loop"][1][90]
        former_landscape = calculate_landscape(value_range, points, sample_6[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_6[0][-1][0])
        former_gradients = calculate_gradients(value_range, points, sample_6[0][0][0])
        latter_gradients = calculate_gradients(value_range, points, sample_6[0][-1][0])
        task_data["f"] = (former_landscape, latter_landscape, former_gradients, latter_gradients)
        sample_7 = record["coherent-loop"][1][25]
        former_landscape = calculate_landscape(value_range, points, sample_7[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_7[0][-1][0])
        former_gradients = calculate_gradients(value_range, points, sample_7[0][0][0])
        latter_gradients = calculate_gradients(value_range, points, sample_7[0][-1][0])
        task_data["g"] = (former_landscape, latter_landscape, former_gradients, latter_gradients)
        sample_8 = record["coherent-loop"][1][17]
        former_landscape = calculate_landscape(value_range, points, sample_8[0][0][0])
        latter_landscape = calculate_landscape(value_range, points, sample_8[0][-1][0])
        former_gradients = calculate_gradients(value_range, points, sample_8[0][0][0])
        latter_gradients = calculate_gradients(value_range, points, sample_8[0][-1][0])
        task_data["h"] = (former_landscape, latter_landscape, former_gradients, latter_gradients)
        save_data(save_path=sort_path + "supp09.pkl", information=task_data)


def experiment3(raw_path, sort_path, config_path):
    agent_names, radios = ["b", "i", "c", "a"], [0.0, 0.1, 0.2, 0.3, 0.4]

    if not path.exists(path=raw_path + "practice.adjustments.1.pkl"):
        config_names = ["baseline.config", "adjusted[i].config", "adjusted[c].config", "adjusted[a].config"]
        agent_configs = [create_agent_config(config_path + config_name) for config_name in config_names]
        noise_generators, norm_type = {}, "L-2"
        for radio in radios:
            noise_generators[radio] = NormNoiseGenerator(norm_type=norm_type, noise_scale=radio)
        record, maximum_generation = {}, 20
        for agent_name, agent_config in zip(agent_names, agent_configs):
            record[agent_name] = {}
            for train_radio in radios:
                result = train_and_test(task=NEATCartPoleTask(maximum_generation=maximum_generation),
                                        agent_name=agent_name, agent_config=agent_config, repeats=sample_number,
                                        train_noise_generator=noise_generators[train_radio],
                                        test_noise_generators=noise_generators)
                record[agent_name][train_radio] = result
        save_data(save_path=raw_path + "practice.adjustments.1.pkl", information=record)

    if not path.exists(path=raw_path + "practice.adjustments.2.pkl"):
        config_names = ["baseline.config", "adjusted[i].config", "adjusted[c].config", "adjusted[a].config"]
        agent_configs = [create_agent_config(config_path + config_name) for config_name in config_names]
        noise_generators, norm_type = {}, "L-2"
        for radio in radios:
            noise_generators[radio] = NormNoiseGenerator(norm_type=norm_type, noise_scale=radio)
        record, maximum_generation, train_radio = {}, 100, 0.3
        for agent_name, agent_config in zip(agent_names, agent_configs):
            result = train_and_test(task=NEATCartPoleTask(maximum_generation=maximum_generation),
                                    agent_name=agent_name, agent_config=agent_config, repeats=sample_number,
                                    train_noise_generator=noise_generators[train_radio],
                                    test_noise_generators=noise_generators)
            record[agent_name] = result
        save_data(save_path=raw_path + "practice.adjustments.2.pkl", information=record)

    if not path.exists(path=sort_path + "main04.pkl"):
        task_data = {}
        record_1 = load_data(raw_path + "practice.adjustments.1.pkl")
        matrix = zeros(shape=(4, 5))
        for strategy_index, strategy in enumerate(agent_names):
            for noise_index, noise in enumerate(radios):
                average_performance = mean([sample[0].get_fitness() for sample in record_1[strategy][noise]])
                matrix[strategy_index, noise_index] = average_performance
        task_data["b"] = matrix
        matrix = zeros(shape=(4, 5))
        for strategy_index, strategy in enumerate(agent_names):
            for noise_index, noise in enumerate(radios):
                median_generation = median([len(sample[1]) for sample in record_1[strategy][noise]])
                matrix[strategy_index, noise_index] = median_generation
        task_data["c"] = matrix
        matrix = zeros(shape=(4, 5, 5))
        for strategy_index, strategy in enumerate(agent_names):
            values = zeros(shape=(5, 5))
            for train_noise_index, noise_1 in enumerate(radios):
                for sample in record_1[strategy][noise_1]:
                    values[train_noise_index] += [sample[2][noise] for noise in radios]
            matrix[strategy_index] = values / float(sample_number)
        task_data["d"] = matrix
        record_2 = load_data(raw_path + "practice.adjustments.2.pkl")
        matrix = zeros(shape=(4, 2))
        cases = {}
        for strategy_index, strategy in enumerate(agent_names):
            count_1 = 0
            for sample in record_1[strategy][0.3]:
                evaluate_result = [sample[2][noise] for noise in radios]
                if min(evaluate_result[:4]) >= 195:
                    count_1 += 1
            count_2, cases[strategy] = 0, [[], [], []]
            for sample in record_2[strategy]:
                evaluate_result = [sample[2][noise] for noise in radios]
                if min(evaluate_result[:4]) >= 195:
                    count_2 += 1
                else:
                    evaluate_result = array(evaluate_result)
                    if all(evaluate_result < 195) and \
                            evaluate_result[0] > evaluate_result[-1] and \
                            evaluate_result[0] > evaluate_result[2]:
                        cases[strategy][0].append(evaluate_result.tolist())
                    elif all(evaluate_result < 195):
                        cases[strategy][1].append(evaluate_result.tolist())
                    else:
                        cases[strategy][2].append(evaluate_result.tolist())
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

    if not path.exists(path=sort_path + "supp10.pkl"):
        task_data = {}
        record_1, result_1 = load_data(raw_path + "practice.adjustments.1.pkl"), [[], [], [], [], []]
        for noise_index, noise in enumerate(radios):
            for strategy_index, strategy in enumerate(agent_names):
                motif_collections = []
                for sample in record_1[strategy][noise]:
                    motif_collections.append(sum(sample[0].get_motif_counts().reshape(3, 4), axis=1).tolist())
                motif_collections = array(motif_collections)
                result_1[noise_index].append(mean(motif_collections, axis=0).tolist())
        task_data["a"] = result_1
        record_2, result_2 = load_data(raw_path + "practice.adjustments.2.pkl"), []
        for strategy_index, strategy in enumerate(agent_names):
            motif_collections = []
            for sample in record_2[strategy]:
                motif_collections.append(sum(sample[0].get_motif_counts().reshape(3, 4), axis=1).tolist())
            motif_collections = array(motif_collections)
            result_2.append(mean(motif_collections, axis=0).tolist())
        task_data["b"] = result_2
        save_data(save_path=sort_path + "supp10.pkl", information=task_data)

    if not path.exists(path=sort_path + "supp11.pkl"):
        task_data = {}
        record = load_data(raw_path + "practice.adjustments.2.pkl")
        result_1, result_2, result_3 = [], [], zeros(shape=(4, 5))
        for strategy_index, strategy in enumerate(agent_names):
            result_1.append(mean([sample[0].get_fitness() for sample in record[strategy]]))
            result_2.append(median([len(sample[1]) for sample in record[strategy]]))
            collection = zeros(shape=(5,))
            for sample in record[strategy]:
                collection += [sample[2][noise] for noise in radios]
            collection /= float(sample_number)
            result_3[strategy_index] = collection
        task_data["a"] = result_1
        task_data["b"] = result_2
        task_data["c"] = result_3
        save_data(save_path=sort_path + "supp11.pkl", information=task_data)

    if not path.exists(path=sort_path + "supp12.pkl"):
        task_data = {}
        record = load_data(raw_path + "practice.adjustments.2.pkl")
        for strategy_index, strategy in enumerate(agent_names):
            cases = [[], [], [], []]
            for sample in record[strategy]:
                sub_values = [sample[2][noise] for noise in radios]
                collection = sum(sample[0].get_motif_counts().reshape(3, 4), axis=1).tolist()
                if min(sub_values[:4]) < 195:
                    sub_values = array(sub_values)
                    if all(sub_values < 195) and sub_values[0] > sub_values[-1] and sub_values[0] > sub_values[2]:
                        cases[1].append(collection)
                    elif all(sub_values < 195):
                        cases[2].append(collection)
                    else:
                        cases[3].append(collection)
                else:
                    cases[0].append(collection)
            a = mean(array(cases[0]), axis=0).tolist()
            b = mean(array(cases[3]), axis=0).tolist()
            if len(cases[1]) > 0:
                c = mean(array(cases[1]), axis=0).tolist()
            else:
                c = None
            if len(cases[2]) > 0:
                d = mean(array(cases[2]), axis=0).tolist()
            else:
                d = None
            task_data[chr(ord("a") + strategy_index)] = [a, b, c, d]
        save_data(save_path=sort_path + "supp12.pkl", information=task_data)


def data_summary():
    for parent_path in ["motif", "population", "robustness", "difference"]:
        for child_path in listdir("./raw/" + parent_path + "/"):
            md5_hash = md5()
            with open("./raw/" + parent_path + "/" + child_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    md5_hash.update(byte_block)
            file_size = path.getsize("./raw/" + parent_path + "/" + child_path)
            info = [parent_path, child_path, md5_hash.hexdigest().upper(), ceil(file_size / 1024).astype(int)]
            print("|", info[0], "|", info[1], "|", info[2], "|", info[3], "|")

    md5_hash = md5()
    with open("./raw/max-min.search.pkl", "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    file_size = path.getsize("./raw/max-min.search.pkl")
    info = ["N.A", "max-min.search.pkl", md5_hash.hexdigest().upper(), ceil(file_size / 1024).astype(int)]
    print("|", info[0], "|", info[1], "|", info[2], "|", info[3], "|")

    md5_hash = md5()
    with open("./raw/practice.adjustments.1.pkl", "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    file_size = path.getsize("./raw/practice.adjustments.1.pkl")
    info = ["N.A", "practice.adjustments.1.pkl", md5_hash.hexdigest().upper(), ceil(file_size / 1024).astype(int)]
    print("|", info[0], "|", info[1], "|", info[2], "|", info[3], "|")

    md5_hash = md5()
    with open("./raw/practice.adjustments.2.pkl", "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    file_size = path.getsize("./raw/practice.adjustments.2.pkl")
    info = ["N.A", "practice.adjustments.2.pkl", md5_hash.hexdigest().upper(), ceil(file_size / 1024).astype(int)]
    print("|", info[0], "|", info[1], "|", info[2], "|", info[3], "|")


if __name__ == "__main__":
    experiment1(raw_path="./raw/", sort_path="./data/")
    experiment2(raw_path="./raw/", sort_path="./data/")
    experiment3(raw_path="./raw/", sort_path="./data/", config_path="./confs/")
    data_summary()
