"""
@Author      : Haoling Zhang
@Description : Package all the presented data from the experimental results.
"""
from collections import Counter
from copy import deepcopy
from itertools import product, combinations_with_replacement
from numpy import array, zeros, arange, linspace, expand_dims, vstack, mgrid, all, sort, where
from numpy import abs, mean, min, max, sum, power, argmax, argmin, log10, isnan
from os import path, mkdir, environ
environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from scipy.stats import gaussian_kde, spearmanr  # noqa
from umap import UMAP  # noqa
from warnings import filterwarnings # noqa

from effect import NeuralMotif, calculate_landscape, calculate_gradients, detect_curvature_feature, minimum_loss_search  # noqa
from effect import get_landscape  # noqa
from practice import acyclic_motifs  # noqa
from works import load_data, save_data  # noqa

filterwarnings("ignore")

motif_types, motif_indices = ["incoherent-loop", "coherent-loop", "collider"], [1, 2, 3, 4]
activation, aggregation = "tanh", "sum"
weight_values, bias_values = linspace(+0.1, +1.0, 10), linspace(-1.0, +1.0, 21)
value_range, points, sample_number = (-1, +1), 41, 100
norm_type = "L-2"
learn_rate, iteration_thresholds = 1e-3, (100, 100)

landscape_names = ["Quadratic Saddle", "Monkey Saddle", "Branin", "three-hump Camel", "six-hump Camel"]

agent_names, radios = ["b", "i", "c", "a"], [0.0, 0.1, 0.2, 0.3, 0.4]

raw_path, sort_path = "./raw/", "./data/"


def main_02():
    """
    Collect plot data from Figure 2 in main text.
    """
    if not path.exists(sort_path + "main02.pkl"):
        landscapes = []
        fixed_loop = NeuralMotif(motif_type="incoherent-loop", motif_index=1,
                                 activations=["tanh", "tanh"], aggregations=["sum", "sum"],
                                 weights=[-1.00e+00, +6.52e-01, +1.00e+00], biases=[+1.00e+00, +3.37e-01])
        landscapes.append(calculate_landscape(value_range, points, fixed_loop).reshape(-1).tolist())
        for motif_index in motif_indices:
            weight_flags, motif_structure = [], acyclic_motifs[motif_types[-1]][motif_index - 1]
            for former, latter in motif_structure.edges:
                weight_flags.append(motif_structure.get_edge_data(former, latter)["weight"])

            weight_groups = [weight_flag * array([0.5]) for weight_flag in weight_flags]
            bias_groups = [array([0.0])]
            activations, aggregations = [activation], [aggregation]
            for weights in product(*weight_groups):
                for biases in product(*bias_groups):
                    target_motif = NeuralMotif(motif_type=motif_types[-1], motif_index=motif_index,
                                               activations=activations, aggregations=aggregations,
                                               weights=weights, biases=biases)
                    landscapes.append(calculate_landscape(value_range, points, target_motif).reshape(-1).tolist())
                    recorded_motifs, recorded_losses = minimum_loss_search(value_range=value_range, points=points,
                                                                           escaper=fixed_loop,
                                                                           catcher=deepcopy(target_motif),
                                                                           learn_rate=learn_rate,
                                                                           threshold=iteration_thresholds[0])
                    for motif in recorded_motifs:
                        landscapes.append(calculate_landscape(value_range, points, motif).reshape(-1).tolist())
        landscapes, differences = array(landscapes), zeros(shape=(len(landscapes), len(landscapes)))
        for index, landscape in enumerate(landscapes):
            differences[index] = mean(power(landscapes - expand_dims(landscape, axis=0), 2), axis=1)
        reducer = UMAP(metric='precomputed')
        locations = reducer.fit_transform(differences)
        locations[:, 0] -= min(locations[:, 0])
        locations[:, 0] /= max(locations[:, 0])
        locations[:, 1] -= min(locations[:, 1])
        locations[:, 1] /= max(locations[:, 1])

        difference_distributions = {}
        for motif_type in motif_types[:-1]:
            difference_data = []
            for motif_index in [1, 2, 3, 4]:
                structure = motif_type + "." + str(motif_index)
                difference_data += load_data("./raw/trade-offs/" + structure + ".npy")[:, 1].tolist()
            difference_data = array(difference_data)
            difference_data = difference_data[difference_data <= 0.03]
            x = linspace(0.00, 0.03, 100)
            y = gaussian_kde(difference_data)(x)
            difference_distributions[motif_type] = (x, y)

        robustness_distributions = {}
        for motif_type in motif_types[:-1]:
            robustness_data = []
            for motif_index in [1, 2, 3, 4]:
                structure = motif_type + "." + str(motif_index)
                robustness_data += load_data("./raw/robustness/" + structure + ".npy").tolist()
            robustness_data = array(robustness_data)
            x = linspace(0.60, 2.40, 100)
            y = gaussian_kde(robustness_data)(x)
            robustness_distributions[motif_type] = (x, y)

        density_map, maximum_value, counts = {}, 0, []
        for motif_type in motif_types[:-1]:
            trade_off, count = [], 0
            for motif_index in [1, 2, 3, 4]:
                structure = motif_type + "." + str(motif_index)
                data_1 = load_data("./raw/robustness/" + structure + ".npy")
                data_2 = load_data("./raw/trade-offs/" + structure + ".npy")
                for index, (source_robust, values) in enumerate(zip(data_1, data_2)):
                    if values[1] <= 0.03 and 0.6 <= source_robust <= 2.4:
                        trade_off.append([values[1], source_robust])
                        count += 1
            trade_off, count = array(trade_off), count / (441000.0 * 4.0)
            x_data, y_data = trade_off[:, 0], trade_off[:, 1]
            x_bins, y_bins = linspace(0.00, 0.03, 100), linspace(0.60, 2.40, 100)
            matrix = zeros(shape=(100, 100))
            for x, y in zip(x_data, y_data):
                matrix[max(where(x - x_bins > 0)[0]), max(where(y - y_bins > 0)[0])] += 1

            maximum_value = max([maximum_value, max(matrix)])
            counts.append(count)
            density_map[motif_type] = matrix

        for motif_type in motif_types[:-1]:
            density_map[motif_type] = density_map[motif_type] / maximum_value

        task_data = {"a": locations, "b": robustness_distributions, "c": difference_distributions,
                     "d": density_map["incoherent-loop"], "e": density_map["coherent-loop"]}

        save_data(save_path=sort_path + "main02.pkl", information=task_data)


def main_03():
    """
    Collect plot data from Figure 3 in main text.
    """
    if not path.exists(sort_path + "main03.pkl"):
        task_data = {}

        pairs_1, pairs_2 = [], []
        for motif_index in motif_indices:
            feature = motif_types[0] + "." + str(motif_index)
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                point_1, point_2 = argmin(losses), argmax(losses)
                pairs_1.append([abs(motifs[point_1][0].w[0].value()), abs(motifs[point_2][0].w[0].value())])
                pairs_2.append([abs(motifs[point_1][0].b[0].value()), abs(motifs[point_2][0].b[0].value())])
        task_data["b"], task_data["c"] = array(pairs_1).T, array(pairs_2).T

        pairs_1, pairs_2 = [], []
        for motif_index in motif_indices:
            feature = motif_types[1] + "." + str(motif_index)
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                point_1, point_2 = argmin(losses), argmax(losses)
                pairs_1.append([abs(motifs[point_1][0].w[0].value()), abs(motifs[point_2][0].w[0].value())])
                pairs_2.append([abs(motifs[point_1][0].b[0].value()), abs(motifs[point_2][0].b[0].value())])
        task_data["d"], task_data["e"] = array(pairs_1).T, array(pairs_2).T

        save_data(save_path=sort_path + "main03.pkl", information=task_data)


def main_04():
    """
    Collect plot data from Figure 4 in main text.
    """
    if not path.exists(sort_path + "main04.pkl"):
        task_data = {}

        escape_data = load_data(load_path=raw_path + "particular/" + motif_types[0] + ".1.escape-process.pkl")
        motifs, losses = escape_data[55]
        source_motif, target_motif = motifs[argmin(losses)][0], motifs[argmax(losses)][0]
        source_landscape = calculate_landscape(value_range, points, source_motif)
        target_landscape = calculate_landscape(value_range, points, target_motif)
        source_feature = detect_curvature_feature(source_landscape, 1.0 / (points - 1))
        target_feature = detect_curvature_feature(target_landscape, 1.0 / (points - 1))
        task_data["a"] = tuple([source_landscape, target_landscape, source_feature, target_feature])

        escape_data = load_data(load_path=raw_path + "particular/" + motif_types[1] + ".1.escape-process.pkl")
        motifs, losses = escape_data[42]
        source_motif, target_motif = motifs[argmin(losses)][0], motifs[argmax(losses)][0]
        source_landscape = calculate_landscape(value_range, points, source_motif)
        target_landscape = calculate_landscape(value_range, points, target_motif)
        source_feature = detect_curvature_feature(source_landscape, 1.0 / (points - 1))
        target_feature = detect_curvature_feature(target_landscape, 1.0 / (points - 1))
        task_data["b"] = tuple([source_landscape, target_landscape, source_feature, target_feature])

        records = []
        for motif_index in motif_indices:
            feature = motif_types[0] + "." + str(motif_index)
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                source, target = motifs[argmin(losses)][0], motifs[argmax(losses)][0]
                source_landscape = calculate_landscape(value_range, sample_number + 1, source)
                target_landscape = calculate_landscape(value_range, sample_number + 1, target)
                source_concavity = detect_curvature_feature(source_landscape, 1.0 / sample_number)
                target_concavity = detect_curvature_feature(target_landscape, 1.0 / sample_number)
                counter_1, counter_2 = Counter(source_concavity.reshape(-1)), Counter(target_concavity.reshape(-1))
                used_value_1, used_value_2 = max([counter_1[1], counter_1[-1]]), max([counter_2[1], counter_2[-1]])
                used_ratio_1 = used_value_1 / ((sample_number + 1) ** 2)
                used_ratio_2 = used_value_2 / ((sample_number + 1) ** 2)
                source, target = motifs[argmin(losses)][1], motifs[argmax(losses)][1]
                source_landscape = calculate_landscape(value_range, sample_number + 1, source)
                target_landscape = calculate_landscape(value_range, sample_number + 1, target)
                source_concavity = detect_curvature_feature(source_landscape, 1.0 / sample_number)
                target_concavity = detect_curvature_feature(target_landscape, 1.0 / sample_number)
                counter_3, counter_4 = Counter(source_concavity.reshape(-1)), Counter(target_concavity.reshape(-1))
                used_value_3, used_value_4 = max([counter_3[1], counter_3[-1]]), max([counter_4[1], counter_4[-1]])
                used_ratio_3 = used_value_3 / ((sample_number + 1) ** 2)
                used_ratio_4 = used_value_4 / ((sample_number + 1) ** 2)
                records.append([used_ratio_1, used_ratio_2, used_ratio_3, used_ratio_4])
        task_data["c"] = array(records)

        records = []
        for motif_index in motif_indices:
            feature = motif_types[1] + "." + str(motif_index)
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                source, target = motifs[argmin(losses)][0], motifs[argmax(losses)][0]
                source_landscape = calculate_landscape(value_range, sample_number + 1, source)
                target_landscape = calculate_landscape(value_range, sample_number + 1, target)
                source_concavity = detect_curvature_feature(source_landscape, 1.0 / sample_number)
                target_concavity = detect_curvature_feature(target_landscape, 1.0 / sample_number)
                counter_1, counter_2 = Counter(source_concavity.reshape(-1)), Counter(target_concavity.reshape(-1))
                used_value_1, used_value_2 = max([counter_1[1], counter_1[-1]]), max([counter_2[1], counter_2[-1]])
                used_ratio_1 = used_value_1 / ((sample_number + 1) ** 2)
                used_ratio_2 = used_value_2 / ((sample_number + 1) ** 2)
                source, target = motifs[argmin(losses)][1], motifs[argmax(losses)][1]
                source_landscape = calculate_landscape(value_range, sample_number + 1, source)
                target_landscape = calculate_landscape(value_range, sample_number + 1, target)
                source_concavity = detect_curvature_feature(source_landscape, 1.0 / sample_number)
                target_concavity = detect_curvature_feature(target_landscape, 1.0 / sample_number)
                counter_3, counter_4 = Counter(source_concavity.reshape(-1)), Counter(target_concavity.reshape(-1))
                used_value_3, used_value_4 = max([counter_3[1], counter_3[-1]]), max([counter_4[1], counter_4[-1]])
                used_ratio_3 = used_value_3 / ((sample_number + 1) ** 2)
                used_ratio_4 = used_value_4 / ((sample_number + 1) ** 2)
                records.append([used_ratio_1, used_ratio_2, used_ratio_3, used_ratio_4])
        task_data["d"] = array(records)

        save_data(save_path=sort_path + "main04.pkl", information=task_data)


def main_05():
    """
    Collect plot data from Figure 5 in main text.
    """
    if not path.exists(sort_path + "main05.pkl"):
        task_data, record = {}, load_data(raw_path + "real-world/adjustments.1.pkl")

        matrix = zeros(shape=(4, 5))
        for strategy_index, strategy in enumerate(agent_names):
            for noise_index, noise in enumerate(radios):
                average_performance = mean([sample[0].get_fitness() for sample in record[strategy][noise]])
                matrix[strategy_index, noise_index] = average_performance
        task_data["b"] = matrix

        for strategy_index, (label, strategy) in enumerate(zip(["c", "d", "e", "f"], agent_names)):
            values = zeros(shape=(5, 5))
            for train_noise_index, noise_1 in enumerate(radios):
                for sample in record[strategy][noise_1]:
                    values[train_noise_index] += [sample[2][noise] for noise in radios]
            task_data[label] = values / float(sample_number)

        save_data(save_path=sort_path + "main05.pkl", information=task_data)


def main_06():
    """
    Collect plot data from Figure 6 in main text.
    """
    if not path.exists(sort_path + "main06.pkl"):
        task_data = {"a": load_data(raw_path + "real-world/iterations.pkl")}

        record = load_data(raw_path + "real-world/adjustments.2.pkl")
        for strategy_index, (panel_index, strategy) in enumerate(zip(["c", "d", "e", "f"], agent_names)):
            count, samples = 0, [[], [], []]
            for sample in record[strategy]:
                evaluation = [sample[2][noise] for noise in radios]
                if min(evaluation[:4]) >= 195:
                    count += 1
                else:
                    evaluation = array(evaluation)
                    if all(evaluation < 195) and evaluation[0] > evaluation[-1] and evaluation[0] > evaluation[2]:
                        samples[0].append(evaluation.tolist())
                    elif all(evaluation < 195):
                        samples[1].append(evaluation.tolist())
                    else:
                        samples[2].append(evaluation.tolist())

            task_data[panel_index] = [strategy, count, len(samples[2]), len(samples[0]), len(samples[1])]

        save_data(save_path=sort_path + "main06.pkl", information=task_data)


def supp_01():
    """
    Collect plot data from Figure S1 in supplementary file.
    """
    if not path.exists(sort_path + "supp01.pkl"):
        task_data = {}
        index, labels = 0, ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        for motif_type in motif_types[:-1]:
            total_data = []
            for motif_index in [1, 2, 3, 4]:
                structure = motif_type + "." + str(motif_index)
                total_data += load_data("./raw/robustness/" + structure + ".npy").tolist()
            total_data = array(total_data)
            x = linspace(0.60, 2.40, 100)
            y = gaussian_kde(total_data)(x)
            task_data[labels[index]] = (motif_type, 0, (x, y))
            index += 1
            for motif_index in [1, 2, 3, 4]:
                structure = motif_type + "." + str(motif_index)
                sub_data = load_data("./raw/robustness/" + structure + ".npy")
                x = linspace(0.60, 2.40, 100)
                y = gaussian_kde(sub_data)(x)
                task_data[labels[index]] = (motif_type, motif_index, (x, y))
                index += 1
        save_data(save_path=sort_path + "supp01.pkl", information=task_data)


def supp_02():
    """
    Collect plot data from Figure S2 in supplementary file.
    """
    if not path.exists(sort_path + "supp02.pkl"):
        task_data = {}
        index, labels = 0, ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        for motif_type in motif_types[:-1]:
            total_data = []
            for motif_index in [1, 2, 3, 4]:
                structure = motif_type + "." + str(motif_index)
                total_data += load_data("./raw/trade-offs/" + structure + ".npy")[:, 1].tolist()
            total_data = array(total_data)
            x = linspace(0.00, 0.03, 100)
            y = gaussian_kde(total_data)(x)
            task_data[labels[index]] = (motif_type, 0, (x, y))
            index += 1
            for motif_index in [1, 2, 3, 4]:
                structure = motif_type + "." + str(motif_index)
                sub_data = load_data("./raw/trade-offs/" + structure + ".npy")[:, 1]
                y = gaussian_kde(sub_data)(x)
                task_data[labels[index]] = (motif_type, motif_index, (x, y))
                index += 1
        save_data(save_path=sort_path + "supp02.pkl", information=task_data)


def supp_04():
    """
    Collect plot data from Figure S4 in supplementary file.
    """
    if not path.exists(sort_path + "supp04.pkl"):
        task_data = {}

        origin_data = load_data(load_path=raw_path + "particular/coherent-loop.1.initialization.pkl")
        escape_data = load_data(load_path=raw_path + "particular/coherent-loop.1.escape-process.pkl")
        # case 43
        case_o_1 = origin_data[42][0]
        case_e_1 = escape_data[42][0]
        task_data["a"] = [
            calculate_landscape(value_range, points, case_o_1),
            calculate_landscape(value_range, points, case_e_1[9][0]),
            calculate_landscape(value_range, points, case_e_1[19][0]),
            calculate_landscape(value_range, points, case_e_1[29][0]),
            calculate_landscape(value_range, points, case_e_1[39][0]),
            calculate_landscape(value_range, points, case_e_1[49][0]),
            calculate_landscape(value_range, points, case_e_1[59][0]),
            calculate_landscape(value_range, points, case_e_1[69][0]),
            calculate_landscape(value_range, points, case_e_1[79][0]),
            calculate_landscape(value_range, points, case_e_1[89][0]),
            calculate_landscape(value_range, points, case_e_1[99][0]),
        ]

        # case 61
        case_o_2 = origin_data[60][0]
        case_e_2 = escape_data[60][0]
        task_data["b"] = [
            calculate_landscape(value_range, points, case_o_2),
            calculate_landscape(value_range, points, case_e_2[9][0]),
            calculate_landscape(value_range, points, case_e_2[19][0]),
            calculate_landscape(value_range, points, case_e_2[29][0]),
            calculate_landscape(value_range, points, case_e_2[39][0]),
            calculate_landscape(value_range, points, case_e_2[49][0]),
            calculate_landscape(value_range, points, case_e_2[59][0]),
            calculate_landscape(value_range, points, case_e_2[69][0]),
            calculate_landscape(value_range, points, case_e_2[79][0]),
            calculate_landscape(value_range, points, case_e_2[89][0]),
            calculate_landscape(value_range, points, case_e_2[99][0]),
        ]
        save_data(save_path=sort_path + "supp04.pkl", information=task_data)


def supp_05():
    """
    Collect plot data from Figure S5 in supplementary file.
    """
    if not path.exists(sort_path + "supp05.pkl"):
        task_data = {}
        for motif_index, panel_index in zip(motif_indices, ["a", "b", "c", "d"]):
            feature, records = motif_types[1] + "." + str(motif_index), []
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                start, stop, correlations = argmin(losses), argmax(losses), []
                for location in range(start, stop - 1):
                    source_landscape = calculate_landscape(value_range, points, motifs[location + 0][0])
                    target_landscape = calculate_landscape(value_range, points, motifs[location + 1][0])
                    values_x = calculate_gradients(value_range, points, motifs[location + 0][0]).reshape(-1)
                    values_y = abs(target_landscape - source_landscape).reshape(-1)
                    # noinspection PyTypeChecker
                    correlation, _ = spearmanr(values_x, values_y)
                    correlations.append(correlation)
                x = linspace(-1, +1, 100)
                y = gaussian_kde(correlations)(x)
                y /= sum(y)
                records.append(y.tolist())
            task_data[panel_index] = array(records)
        save_data(save_path=sort_path + "supp05.pkl", information=task_data)


def supp_06():
    """
    Collect plot data from Figure S6 in supplementary file.
    """
    if not path.exists(sort_path + "supp06.pkl"):
        task_data = {}
        for panel_label, motif_index in zip(["a", "b", "c", "d"], motif_indices):
            feature, records = motif_types[0] + "." + str(motif_index), []
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                source, target = motifs[argmin(losses)][0], motifs[argmax(losses)][0]
                source_concavity = detect_curvature_feature(calculate_landscape(value_range, 101, source), 0.01)
                target_concavity = detect_curvature_feature(calculate_landscape(value_range, 101, target), 0.01)
                counter_1, counter_2 = Counter(source_concavity.reshape(-1)), Counter(target_concavity.reshape(-1))
                used_value_1, used_value_2 = max([counter_1[1], counter_1[-1]]), max([counter_2[1], counter_2[-1]])
                records.append([used_value_1 / (101 ** 2), used_value_2 / (101 ** 2)])
            task_data[panel_label] = array(records)
        save_data(save_path=sort_path + "supp06.pkl", information=task_data)


def supp_07():
    """
    Collect plot data from Figure S7 in supplementary file.
    """
    if not path.exists(sort_path + "supp07.pkl"):
        task_data, flag = {"b": []}, True
        for motif_index in motif_indices:
            feature = motif_types[0] + "." + str(motif_index)
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                source, target = motifs[argmin(losses)][0], motifs[argmax(losses)][0]
                source_landscape = calculate_landscape(value_range, 101, source)
                target_landscape = calculate_landscape(value_range, 101, target)
                source_concavity = detect_curvature_feature(source_landscape, 0.01)
                target_concavity = detect_curvature_feature(target_landscape, 0.01)
                counter_1, counter_2 = Counter(source_concavity.reshape(-1)), Counter(target_concavity.reshape(-1))
                used_value_1, used_value_2 = max([counter_1[1], counter_1[-1]]), max([counter_2[1], counter_2[-1]])
                use_rate_1, use_rate_2 = used_value_1 / (101 ** 2), used_value_2 / (101 ** 2)
                if use_rate_2 < use_rate_1:
                    if 1 not in counter_1:
                        source_region = where(source_landscape > 0, 1, 0)
                        target_region = where(target_landscape > 0, 1, 0)
                    else:
                        source_region = where(source_landscape < 0, 1, 0)
                        target_region = where(target_landscape < 0, 1, 0)
                    if flag:
                        task_data["a"] = (source_landscape, target_landscape, source_concavity, target_concavity,
                                          source_region, target_region)
                        flag = False
                    # noinspection PyUnresolvedReferences
                    task_data["b"].append([sum(source_region.reshape(-1)) / (101 ** 2),
                                           sum(target_region.reshape(-1)) / (101 ** 2)])
        # noinspection PyUnresolvedReferences
        task_data["b"] = array(task_data["b"])

        save_data(save_path=sort_path + "supp07.pkl", information=task_data)


def supp_08():
    """
    Collect plot data from Figure S8 in supplementary file.
    """
    if not path.exists(sort_path + "supp08.pkl"):
        task_data = {}
        for panel_label, motif_index in zip(["a", "b", "c", "d"], motif_indices):
            feature, records = motif_types[1] + "." + str(motif_index), []
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                source, target = motifs[argmin(losses)][0], motifs[argmax(losses)][0]
                source_concavity = detect_curvature_feature(calculate_landscape(value_range, 101, source), 0.01)
                target_concavity = detect_curvature_feature(calculate_landscape(value_range, 101, target), 0.01)
                counter_1, counter_2 = Counter(source_concavity.reshape(-1)), Counter(target_concavity.reshape(-1))
                used_value_1, used_value_2 = max([counter_1[1], counter_1[-1]]), max([counter_2[1], counter_2[-1]])
                records.append([used_value_1 / (101 ** 2), used_value_2 / (101 ** 2)])
            task_data[panel_label] = array(records)
        save_data(save_path=sort_path + "supp08.pkl", information=task_data)


def supp_09():
    """
    Collect plot data from Figure S9 in supplementary file.
    """
    if not path.exists(sort_path + "supp09.pkl"):
        task_data = {}

        for panel_index, landscape_name in enumerate(landscape_names):
            landscape = get_landscape(name=landscape_name).detach().numpy()
            task_data[chr(ord("b") + panel_index)] = (landscape_name, landscape)

        save_data(save_path=sort_path + "supp09.pkl", information=task_data)


def supp_10():
    """
    Collect plot data from Figure S10 in supplementary file.
    """
    if not path.exists(sort_path + "supp10.pkl"):
        task_data = {}

        records = load_data(raw_path + "network-scale/loop.vs.collider.Quadratic Saddle.pkl")

        panel_a, panel_b, panel_c = [], [], []
        for motif_number in arange(1, 11):
            panel_c.append(["", "", "", ""])
            values = array([[sample[-1], len(sample)] for sample in records[("collider", motif_number)]])
            panel_a.append([mean(values[:, 0]), mean(values[:, 1])])
            temp_a = ("%.2E" % mean(values[:, 0])).replace("-0", "-")
            temp_b = ("%.2E" % mean(values[:, 1])).replace("+0", "+")
            panel_c[-1][0], panel_c[-1][1] = temp_a, temp_b

            values = array([[sample[-1], len(sample)] for sample in records[("loop", motif_number)]])
            panel_b.append([mean(values[:, 0]), mean(values[:, 1])])
            temp_a = ("%.2E" % mean(values[:, 0])).replace("-0", "-")
            temp_b = ("%.2E" % mean(values[:, 1])).replace("+0", "+")
            panel_c[-1][2], panel_c[-1][3] = temp_a, temp_b

        task_data["a"], task_data["b"], task_data["c"] = array(panel_a), array(panel_b), panel_c

        save_data(save_path=sort_path + "supp10.pkl", information=task_data)


def supp_11():
    """
    Collect plot data from Figure S11 in supplementary file.
    """
    if not path.exists(sort_path + "supp11.pkl"):
        task_data = {}

        records = load_data(raw_path + "network-scale/loop.vs.collider.Monkey Saddle.pkl")

        panel_a, panel_b, panel_c = [], [], []
        for motif_number in arange(1, 11):
            panel_c.append(["", "", "", ""])
            values = array([[sample[-1], len(sample)] for sample in records[("collider", motif_number)]])
            panel_a.append([mean(values[:, 0]), mean(values[:, 1])])
            temp_a = ("%.2E" % mean(values[:, 0])).replace("-0", "-")
            temp_b = ("%.2E" % mean(values[:, 1])).replace("+0", "+")
            panel_c[-1][0], panel_c[-1][1] = temp_a, temp_b

            values = array([[sample[-1], len(sample)] for sample in records[("loop", motif_number)]])
            panel_b.append([mean(values[:, 0]), mean(values[:, 1])])
            temp_a = ("%.2E" % mean(values[:, 0])).replace("-0", "-")
            temp_b = ("%.2E" % mean(values[:, 1])).replace("+0", "+")
            panel_c[-1][2], panel_c[-1][3] = temp_a, temp_b

        task_data["a"], task_data["b"], task_data["c"] = array(panel_a), array(panel_b), panel_c

        save_data(save_path=sort_path + "supp11.pkl", information=task_data)


def supp_12():
    """
    Collect plot data from Figure S12 in supplementary file.
    """
    if not path.exists(sort_path + "supp12.pkl"):
        task_data = {}

        records = load_data(raw_path + "network-scale/loop.vs.collider.Branin.pkl")

        panel_a, panel_b, panel_c = [], [], []
        for motif_number in arange(1, 11):
            panel_c.append(["", "", "", ""])
            values = array([[sample[-1], len(sample)] for sample in records[("collider", motif_number)]])
            panel_a.append([mean(values[:, 0]), mean(values[:, 1])])
            temp_a = ("%.2E" % mean(values[:, 0])).replace("-0", "-")
            temp_b = ("%.2E" % mean(values[:, 1])).replace("+0", "+")
            panel_c[-1][0], panel_c[-1][1] = temp_a, temp_b

            values = array([[sample[-1], len(sample)] for sample in records[("loop", motif_number)]])
            panel_b.append([mean(values[:, 0]), mean(values[:, 1])])
            temp_a = ("%.2E" % mean(values[:, 0])).replace("-0", "-")
            temp_b = ("%.2E" % mean(values[:, 1])).replace("+0", "+")
            panel_c[-1][2], panel_c[-1][3] = temp_a, temp_b

        task_data["a"], task_data["b"], task_data["c"] = array(panel_a), array(panel_b), panel_c

        save_data(save_path=sort_path + "supp12.pkl", information=task_data)


def supp_13():
    """
    Collect plot data from Figure S13 in supplementary file.
    """
    if not path.exists(sort_path + "supp13.pkl"):
        task_data = {}

        records = load_data(raw_path + "network-scale/loop.vs.collider.three-hump Camel.pkl")

        panel_a, panel_b, panel_c = [], [], []
        for motif_number in arange(1, 11):
            panel_c.append(["", "", "", ""])
            values = array([[sample[-1], len(sample)] for sample in records[("collider", motif_number)]])
            panel_a.append([mean(values[:, 0]), mean(values[:, 1])])
            temp_a = ("%.2E" % mean(values[:, 0])).replace("-0", "-")
            temp_b = ("%.2E" % mean(values[:, 1])).replace("+0", "+")
            panel_c[-1][0], panel_c[-1][1] = temp_a, temp_b

            values = array([[sample[-1], len(sample)] for sample in records[("loop", motif_number)]])
            panel_b.append([mean(values[:, 0]), mean(values[:, 1])])
            temp_a = ("%.2E" % mean(values[:, 0])).replace("-0", "-")
            temp_b = ("%.2E" % mean(values[:, 1])).replace("+0", "+")
            panel_c[-1][2], panel_c[-1][3] = temp_a, temp_b

        task_data["a"], task_data["b"], task_data["c"] = array(panel_a), array(panel_b), panel_c

        save_data(save_path=sort_path + "supp13.pkl", information=task_data)


def supp_14():
    """
    Collect plot data from Figure S14 in supplementary file.
    """
    if not path.exists(sort_path + "supp14.pkl"):
        task_data = {}

        records = load_data(raw_path + "network-scale/loop.vs.collider.six-hump Camel.pkl")

        panel_a, panel_b, panel_c = [], [], []
        for motif_number in arange(1, 11):
            panel_c.append(["", "", "", ""])
            values = array([[sample[-1], len(sample)] for sample in records[("collider", motif_number)]])
            panel_a.append([mean(values[:, 0]), mean(values[:, 1])])
            temp_a = ("%.2E" % mean(values[:, 0])).replace("-0", "-")
            temp_b = ("%.2E" % mean(values[:, 1])).replace("+0", "+")
            panel_c[-1][0], panel_c[-1][1] = temp_a, temp_b

            values = array([[sample[-1], len(sample)] for sample in records[("loop", motif_number)]])
            panel_b.append([mean(values[:, 0]), mean(values[:, 1])])
            temp_a = ("%.2E" % mean(values[:, 0])).replace("-0", "-")
            temp_b = ("%.2E" % mean(values[:, 1])).replace("+0", "+")
            panel_c[-1][2], panel_c[-1][3] = temp_a, temp_b

        task_data["a"], task_data["b"], task_data["c"] = array(panel_a), array(panel_b), panel_c

        save_data(save_path=sort_path + "supp14.pkl", information=task_data)


def supp_15():
    """
    Collect plot data from Figure S15 in supplementary file.
    """
    if not path.exists(sort_path + "supp15.pkl"):
        task_data = {}

        motif_number, totals = 10, zeros(shape=(11,), dtype=int)
        for coherent_number in range(0, 11):
            for _ in combinations_with_replacement([1, 2, 3, 4], coherent_number):
                for _ in combinations_with_replacement([1, 2, 3, 4], motif_number - coherent_number):
                    totals[coherent_number] += 1

        for panel_index, landscape_name in enumerate(["Quadratic Saddle", "Monkey Saddle", "Branin",
                                                      "three-hump Camel", "six-hump Camel"]):
            records = load_data(raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl")
            counts = zeros(shape=(11,), dtype=int)
            for key, record in records.items():
                incoherent_number = key.split("-")[0].count("0")
                if record is not None and record["training loss"][-1] <= 1e-3:
                    counts[incoherent_number] += 1
            task_data[chr(ord("a") + panel_index)] = (landscape_name, counts, totals)

        save_data(save_path=sort_path + "supp15.pkl", information=task_data)


def supp_16():
    """
    Collect plot data from Figure S16 in supplementary file.
    """
    if not path.exists(sort_path + "supp16.pkl"):
        task_data, landscape_name = {}, "Quadratic Saddle"
        milestones = [1e+0, 5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]
        labels = ["1E+1", "5E-1", "2E-1", "1E-1", "5E-2", "2E-2", "1E-2", "5E-3", "2E-3", "1E-3"]

        records_1 = load_data(raw_path + "network-scale/loop.vs.collider." + landscape_name + ".pkl")

        panel_a = {label: [] for label in labels}
        for index, sample in enumerate(records_1[("loop", 10)]):
            for milestone, label in zip(milestones, labels):
                iterations = where(sample <= milestone)[0]
                if len(iterations) > 0:
                    panel_a[label].append(iterations[0])
                else:
                    panel_a[label].append(0)
        task_data["a"] = panel_a

        thresholds = {}
        for label in labels:
            thresholds[label] = max(panel_a[label])

        records_2 = load_data(raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl")

        matrix = zeros(shape=(2, 11, 10))
        for key, record in records_2.items():
            incoherent_number, losses = key.split("-")[0].count("0"), record["training loss"]
            for index, (label, milestone) in enumerate(zip(labels, milestones)):
                if losses[0] < milestone:
                    continue
                elif losses[-1] > milestone:
                    matrix[0, incoherent_number, index] += 1
                else:
                    iterations = where(losses <= milestone)[0]
                    if len(iterations) > 0 and iterations[0] > thresholds[label]:
                        matrix[0, incoherent_number, index] += 1
                matrix[1, incoherent_number, index] += 1

        matrix = matrix[0] / matrix[1]
        task_data["b"] = matrix

        save_data(save_path=sort_path + "supp16.pkl", information=task_data)


def supp_17():
    """
    Collect plot data from Figure S17 in supplementary file.
    """
    if not path.exists(sort_path + "supp17.pkl"):
        task_data, landscape_name = {}, "Quadratic Saddle"

        records = load_data(raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl")
        counts, loss_data = zeros(shape=(2, 11)), [[] for _ in range(11)]
        for key, record in records.items():
            incoherent_number, final_loss = key.split("-")[0].count("0"), record["training loss"][-1]
            if final_loss > 1e-3:
                loss_data[incoherent_number].append(final_loss)
                counts[0, incoherent_number] += 1
            counts[1, incoherent_number] += 1

        task_data["a"] = counts[0] / counts[1]

        task_data["b"] = []
        for index in range(11):
            task_data["b"].append(sort(loss_data[index]))

        save_data(save_path=sort_path + "supp17.pkl", information=task_data)


def supp_18():
    """
    Collect plot data from Figure S18 in supplementary file.
    """
    if not path.exists(sort_path + "supp18.pkl"):
        task_data, landscape_name = {}, "Quadratic Saddle"

        records = load_data(raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl")

        divided_data = [[[], [], []] for _ in range(11)]
        for key, record in records.items():
            incoherent_number = key.split("-")[0].count("0")
            losses, utilization_rates = record["training loss"], record["sparsity"]
            divided_data[incoherent_number][0].append(losses[-1])
            if incoherent_number < 10:
                divided_data[incoherent_number][1].append(utilization_rates[-1, 1])
            else:
                divided_data[incoherent_number][1].append(0)
            if incoherent_number > 0:
                divided_data[incoherent_number][2].append(utilization_rates[-1, 2])
            else:
                divided_data[incoherent_number][2].append(0)

        for incoherent_number, sub_divided_data in enumerate(divided_data):
            sub_divided_data = array(sub_divided_data)
            plot_indices = where(sub_divided_data[0] > 1e-3)[0]
            if incoherent_number < 10:
                values_1 = (log10(sub_divided_data[0, plot_indices]), sub_divided_data[1, plot_indices])
                correlation_1, _ = spearmanr(sub_divided_data[0], sub_divided_data[1])
            else:
                values_1, correlation_1 = None, None
            task_data[chr(ord("a") + incoherent_number * 2 + 0)] = (values_1, correlation_1)

            if incoherent_number > 0:
                values_2 = (log10(sub_divided_data[0, plot_indices]), sub_divided_data[2, plot_indices])
                correlation_2, _ = spearmanr(sub_divided_data[0], sub_divided_data[2])
            else:
                values_2, correlation_2 = None, None
            task_data[chr(ord("a") + incoherent_number * 2 + 1)] = (values_2, correlation_2)

        save_data(save_path=sort_path + "supp18.pkl", information=task_data)


def supp_19():
    """
    Collect plot data from Figure S19 in supplementary file.
    """
    if not path.exists(sort_path + "supp19.pkl"):
        task_data, landscape_name = {}, "Quadratic Saddle"

        records = load_data(raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl")

        divided_data = [[[], []] for _ in range(11)]
        for key, record in records.items():
            incoherent_number = key.split("-")[0].count("0")
            values_1, values_2 = record["training loss"], record["sparsity"]
            delta_losses = values_1[:-1] - values_1[1:]
            if values_2[-1, 3] < 1.0:
                end = max(where(values_2[:, -1] == 1)[0]) - 1
                used_delta_losses = delta_losses[:end].tolist()
            else:
                used_delta_losses = delta_losses.tolist()
            divided_data[incoherent_number][0] += used_delta_losses
            divided_data[incoherent_number][1] += arange(len(used_delta_losses)).tolist()
        panel_data = []
        for incoherent_number, sub_divided_data in enumerate(divided_data):
            loss_data, iteration_data = sub_divided_data[0], sub_divided_data[1]
            correlation, _ = spearmanr(loss_data, iteration_data)
            panel_data.append(correlation)
        task_data["a"] = array(panel_data)

        divided_data = [[[], []] for _ in range(11)]
        for key, record in records.items():
            incoherent_number = key.split("-")[0].count("0")
            values_1, values_2, values_3 = record["training loss"], record["lipschitz constant"], record["sparsity"]
            deltas_1, deltas_2 = values_1[:-1] - values_1[1:], values_2[:-1] - values_2[1:]
            if values_3[-1, 3] < 1.0:
                end = max(where(values_3[:, -1] == 1.0)[0]) - 1
                if end > 0:
                    divided_data[incoherent_number][0] += deltas_1[:end].tolist()
                    divided_data[incoherent_number][1] += deltas_2[:end].tolist()
            else:
                divided_data[incoherent_number][0] += deltas_1.tolist()
                divided_data[incoherent_number][1] += deltas_2.tolist()
        for incoherent_number in range(11):
            pairs_0, pairs_1 = divided_data[incoherent_number]
            x_values, y_values = linspace(0, 6e-2, 100), linspace(-4e-3, +8e-3, 100)
            x_indices, y_indices = mgrid[x_values[0]:x_values[-1]:100j, y_values[0]:y_values[-1]:100j]
            positions = vstack([x_indices.ravel(), y_indices.ravel()])
            z_values = gaussian_kde(vstack([pairs_0, pairs_1]))(positions)
            z_values = z_values.reshape(len(x_values), len(y_values))
            correlation, _ = spearmanr(pairs_0, pairs_1)
            task_data[chr(ord("a") + 1 + incoherent_number)] = (x_values, y_values, z_values, correlation)

        save_data(save_path=sort_path + "supp19.pkl", information=task_data)


def supp_20():
    """
    Collect plot data from Figure S20 in supplementary file.
    """
    if not path.exists(sort_path + "supp20.pkl"):
        task_data, landscape_name = {}, "Quadratic Saddle"

        records = load_data(raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl")

        divided_data = [[[], []] for _ in range(11)]
        for key, record in records.items():
            incoherent_number = key.split("-")[0].count("0")
            values_1, values_2 = record["spectral norm"], record["sparsity"]
            if values_2[-1, 3] < 1.0:
                end = max(where(values_2[:, -1] == 1.0)[0]) - 1
                if end > 0:
                    divided_data[incoherent_number][0].append(values_1[:end, 1])
                    divided_data[incoherent_number][1].append(values_1[:end, 2])
            else:
                divided_data[incoherent_number][0].append(values_1[:, 1])
                divided_data[incoherent_number][1].append(values_1[:, 2])

        for incoherent_number in range(11):
            pairs_0, pairs_1 = divided_data[incoherent_number]
            task_data[chr(ord("a") + incoherent_number * 2 + 0)] = pairs_0 if incoherent_number < 10 else None
            task_data[chr(ord("a") + incoherent_number * 2 + 1)] = pairs_1 if incoherent_number > 0 else None

        save_data(save_path=sort_path + "supp20.pkl", information=task_data)


def supp_21():
    """
    Collect plot data from Figure S21 in supplementary file.
    """
    if not path.exists(sort_path + "supp21.pkl"):
        task_data, landscape_name = {}, "Quadratic Saddle"

        records = load_data(raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl")

        divided_data = [[[], []] for _ in range(11)]
        for key, record in records.items():
            incoherent_number = key.split("-")[0].count("0")
            values_1, values_2 = record["spectral norm"], record["sparsity"]
            if incoherent_number < 10:  # have coherent loops
                collection = [[values_1[0, 1], values_2[0, 1]]]
                for value_1, value_2 in zip(values_1[1:], values_2[1:]):
                    if collection[-1][1] - value_2[1] > 1e-4:
                        collection.append([value_1[1], value_2[1]])
                curve = array(collection).T
                correlation, _ = spearmanr(curve[0], curve[1])
                if not isnan(correlation):
                    divided_data[incoherent_number][0].append(correlation)
            if incoherent_number > 0:  # have incoherent loops
                collection = [[values_1[0, 2], values_2[0, 2]]]
                for value_1, value_2 in zip(values_1[1:], values_2[1:]):
                    if collection[-1][1] - value_2[2] > 1e-4:
                        collection.append([value_1[2], value_2[2]])
                curve = array(collection).T
                correlation, _ = spearmanr(curve[0], curve[1])
                if not isnan(correlation):
                    divided_data[incoherent_number][1].append(correlation)

        for incoherent_number in range(11):
            pairs_0, pairs_1 = divided_data[incoherent_number]
            if len(pairs_0) > 0:
                x_values_0 = linspace(min(pairs_0), max(pairs_0), 100)
                y_values_0 = gaussian_kde(pairs_0)(x_values_0)
                y_values_0 /= sum(y_values_0)
            else:
                x_values_0, y_values_0 = None, None
            if len(pairs_1) > 0:
                x_values_1 = linspace(min(pairs_1), max(pairs_1), 100)
                y_values_1 = gaussian_kde(pairs_1)(x_values_1)
                y_values_1 /= sum(y_values_1)
            else:
                x_values_1, y_values_1 = None, None
            task_data[chr(ord("a") + incoherent_number)] = (x_values_0, y_values_0, x_values_1, y_values_1)

        save_data(save_path=sort_path + "supp21.pkl", information=task_data)


def supp_22():
    """
    Collect plot data from Figure S22 in supplementary file.
    """
    if not path.exists(sort_path + "supp22.pkl"):
        task_data, landscape_name = {}, "Quadratic Saddle"

        records = load_data(raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl")

        divided_data = [[[], []] for _ in range(11)]
        for key, record in records.items():
            incoherent_number = key.split("-")[0].count("0")
            values_1, values_2, values_3 = record["gradient variance"], record["lipschitz constant"], record["sparsity"]
            values_1, values_2 = values_1[1:, 3] - values_1[:-1, 3], values_2[1:] - values_2[:-1]
            if values_3[-1, 3] < 1.0:
                end = max(where(values_3[:, -1] == 1)[0]) - 1
                divided_data[incoherent_number][0] += values_1[:end].tolist()
                divided_data[incoherent_number][1] += values_2[:end].tolist()
            else:
                divided_data[incoherent_number][0] += values_1.tolist()
                divided_data[incoherent_number][1] += values_2.tolist()

        x_values, y_values = linspace(-0.012, +0.002, 40), linspace(-0.008, +0.006, 40)
        x_indices, y_indices = mgrid[x_values[0]:x_values[-1]:40j, y_values[0]:y_values[-1]:40j]
        positions = vstack([x_indices.ravel(), y_indices.ravel()])
        for incoherent_number in range(11):
            pairs_0, pairs_1 = divided_data[incoherent_number]
            z_values = gaussian_kde(vstack([pairs_0, pairs_1]))(positions).reshape(40, 40)
            z_values /= max(z_values)
            correlation, _ = spearmanr(pairs_0, pairs_1)
            task_data[chr(ord("a") + incoherent_number)] = (z_values, correlation)

        save_data(save_path=sort_path + "supp22.pkl", information=task_data)


def supp_23():
    """
    Collect plot data from Figure S23 in supplementary file.
    """
    if not path.exists(sort_path + "supp23.pkl"):
        task_data, landscape_name = {}, "Quadratic Saddle"

        records = load_data(raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl")

        divided_data = [zeros(shape=(3, 3)) for _ in range(11)]
        for key, record in records.items():
            incoherent_number = key.split("-")[0].count("0")
            values_1, values_2 = record["hessian eigenvalue"], record["lipschitz constant"]
            values_1, values_2 = values_1[1:] - values_1[:-1], values_2[1:] - values_2[:-1]
            for value_1, value_2 in zip(values_1[:, 1], values_2):
                if value_1 > 0:
                    index_1 = 2
                elif value_1 < 0:
                    index_1 = 0
                else:
                    index_1 = 1

                if value_2 > 0:
                    index_2 = 2
                elif value_2 < 0:
                    index_2 = 0
                else:
                    index_2 = 1
                divided_data[incoherent_number][index_1, index_2] += 1

        for incoherent_number in range(11):
            task_data[chr(ord("a") + incoherent_number)] = divided_data[incoherent_number]

        save_data(save_path=sort_path + "supp23.pkl", information=task_data)


def supp_24():
    """
    Collect plot data from Figure S24 in supplementary file.
    """
    if not path.exists(sort_path + "supp24.pkl"):
        task_data = {}
        record = load_data(raw_path + "real-world/adjustments.2.pkl")
        for strategy_index, (panel_index, strategy) in enumerate(zip(["b", "i", "c", "a"], agent_names)):
            count, cases = 0, [[], [], []]
            for sample in record[strategy]:
                evaluation = [sample[2][noise] for noise in radios]
                if min(evaluation[:4]) >= 195:
                    count += 1
                else:
                    evaluation = array(evaluation)
                    if all(evaluation < 195) and evaluation[0] > evaluation[-1] and evaluation[0] > evaluation[2]:
                        cases[0].append(evaluation.tolist())
                    elif all(evaluation < 195):
                        cases[1].append(evaluation.tolist())
                    else:
                        cases[2].append(evaluation.tolist())
            a = array(cases[2])
            if len(cases[0]) > 0:
                b = array(cases[0])
            else:
                b = None
            if len(cases[1]) > 0:
                c = array(cases[1])
            else:
                c = None

            task_data[panel_index] = [a, b, c]

        save_data(save_path=sort_path + "supp24.pkl", information=task_data)


def supp_25():
    """
    Collect plot data from Figure S25 in supplementary file.
    """
    if not path.exists(sort_path + "supp25.pkl"):
        task_data = {}
        record = load_data(raw_path + "real-world/adjustments.2.pkl")
        for strategy_index, strategy in enumerate(agent_names):
            cases = []
            for sample in record[strategy]:
                evaluation = array([sample[2][noise] for noise in radios])
                if all(evaluation < 195) and evaluation[0] > evaluation[-1] and evaluation[0] > evaluation[2]:
                    pass
                elif all(evaluation < 195):
                    cases.append(evaluation)
            task_data[chr(ord("a") + strategy_index)] = cases
        save_data(save_path=sort_path + "supp25.pkl", information=task_data)


def supp_26():
    """
    Collect plot data from Figure S26 in supplementary file.
    """
    if not path.exists(sort_path + "supp26.pkl"):
        task_data = {}
        record = load_data(raw_path + "real-world/uci.datasets.pkl")
        for label, (info, (_, trained_labels, _, tested_labels, _, _, _, _)) in zip(["a", "b", "c"], record.items()):
            task_data[label] = (info[len("case study in "):].replace(" and ", "-"),
                                Counter(trained_labels), Counter(tested_labels))
        save_data(save_path=sort_path + "supp26.pkl", information=task_data)


def supp_27():
    """
    Collect plot data from Figure S27 in supplementary file.
    """
    if not path.exists(sort_path + "supp27.pkl"):
        task_data = {}
        record = load_data(raw_path + "real-world/uci.datasets.pkl")
        for label, (info, (trained_data, _, tested_data, _, _, _, _, _)) in zip(["a", "b", "c"], record.items()):
            task_data[label] = (info[len("case study in "):].replace(" and ", "-"), trained_data, tested_data)
        save_data(save_path=sort_path + "supp27.pkl", information=task_data)


def supp_28():
    """
    Collect plot data from Figure S28 in supplementary file.
    """
    if not path.exists(sort_path + "supp28.pkl"):
        task_data = {}

        records_1, matrix_1 = load_data(raw_path + "real-world/biology.pkl"), zeros(shape=(4, 5))
        for agent_index, agent_name in enumerate(agent_names):
            for radio_index in range(5):
                values = []
                for best_agent, _, _ in records_1[agent_name][radio_index]:
                    values.append(best_agent.get_fitness())
                matrix_1[agent_index, radio_index] = mean(values)
        task_data["a"] = matrix_1

        records_2, matrix_2 = load_data(raw_path + "real-world/physics-chemistry.pkl"), zeros(shape=(4, 5))
        for agent_index, agent_name in enumerate(agent_names):
            for radio_index in range(5):
                values = []
                for best_agent, _, _ in records_2[agent_name][radio_index]:
                    values.append(best_agent.get_fitness())
                matrix_2[agent_index, radio_index] = mean(values)
        task_data["b"] = matrix_2

        records_3, matrix_3 = load_data(raw_path + "real-world/health-medicine.pkl"), zeros(shape=(4, 5))
        for agent_index, agent_name in enumerate(agent_names):
            for radio_index in range(5):
                values = []
                for best_agent, _, _ in records_3[agent_name][radio_index]:
                    values.append(best_agent.get_fitness())
                matrix_3[agent_index, radio_index] = mean(values)
        task_data["c"] = matrix_3

        save_data(save_path=sort_path + "supp28.pkl", information=task_data)


def supp_29():
    """
    Collect plot data from Figure S29 in supplementary file.
    """
    if not path.exists(sort_path + "supp29.pkl"):
        task_data = {}

        records_1, matrix_1 = load_data(raw_path + "real-world/biology.pkl"), zeros(shape=(4, 5, 5))
        for agent_index, agent_name in enumerate(agent_names):
            for radio_index_1 in range(5):
                values = [[] for _ in range(5)]
                for _, _, test_record in records_1[agent_name][radio_index_1]:
                    for radio_value, value in test_record.items():
                        radio_index_2 = int(radio_value * 10)
                        values[radio_index_2].append(value)
                values = array(values)
                matrix_1[agent_index, radio_index_1] = mean(values, axis=1)
        task_data["a"] = matrix_1[0]
        task_data["b"] = matrix_1[1]
        task_data["c"] = matrix_1[2]
        task_data["d"] = matrix_1[3]

        records_2, matrix_2 = load_data(raw_path + "real-world/physics-chemistry.pkl"), zeros(shape=(4, 5, 5))
        for agent_index, agent_name in enumerate(agent_names):
            for radio_index_1 in range(5):
                values = [[] for _ in range(5)]
                for _, _, test_record in records_2[agent_name][radio_index_1]:
                    for radio_value, value in test_record.items():
                        radio_index_2 = int(radio_value * 10)
                        values[radio_index_2].append(value)
                values = array(values)
                matrix_2[agent_index, radio_index_1] = mean(values, axis=1)
        task_data["e"] = matrix_2[0]
        task_data["f"] = matrix_2[1]
        task_data["g"] = matrix_2[2]
        task_data["h"] = matrix_2[3]

        records_3, matrix_3 = load_data(raw_path + "real-world/health-medicine.pkl"), zeros(shape=(4, 5, 5))
        for agent_index, agent_name in enumerate(agent_names):
            for radio_index_1 in range(5):
                values = [[] for _ in range(5)]
                for _, _, test_record in records_3[agent_name][radio_index_1]:
                    for radio_value, value in test_record.items():
                        radio_index_2 = int(radio_value * 10)
                        values[radio_index_2].append(value)
                values = array(values)
                matrix_3[agent_index, radio_index_1] = mean(values, axis=1)
        task_data["i"] = matrix_3[0]
        task_data["j"] = matrix_3[1]
        task_data["k"] = matrix_3[2]
        task_data["l"] = matrix_3[3]

        save_data(save_path=sort_path + "supp29.pkl", information=task_data)


def supp_30():
    """
    Collect plot data from Figure S30 in supplementary file.
    """
    if not path.exists(sort_path + "supp30.pkl"):
        task_data = {}

        records_1, counts_1 = load_data(raw_path + "real-world/biology.pkl"), zeros(shape=(4, 5), dtype=int)
        for agent_index, agent_name in enumerate(agent_names):
            for radio_index_1 in range(5):
                for sample_index, (_, _, test_record) in enumerate(records_1[agent_name][radio_index_1]):
                    values = [0 for _ in range(5)]
                    for radio_value, value in test_record.items():
                        radio_index_2 = int(radio_value * 10)
                        values[radio_index_2] = value
                    if max(values) == values[radio_index_1]:
                        counts_1[agent_index, radio_index_1] += 1
        task_data["a"] = counts_1

        records_2, counts_2 = load_data(raw_path + "real-world/physics-chemistry.pkl"), zeros(shape=(4, 5), dtype=int)
        for agent_index, agent_name in enumerate(agent_names):
            for radio_index_1 in range(5):
                for sample_index, (_, _, test_record) in enumerate(records_2[agent_name][radio_index_1]):
                    values = [0 for _ in range(5)]
                    for radio_value, value in test_record.items():
                        radio_index_2 = int(radio_value * 10)
                        values[radio_index_2] = value
                    if max(values) == values[radio_index_1]:
                        counts_2[agent_index, radio_index_1] += 1
        task_data["b"] = counts_2

        records_3, counts_3 = load_data(raw_path + "real-world/health-medicine.pkl"), zeros(shape=(4, 5), dtype=int)
        for agent_index, agent_name in enumerate(agent_names):
            for radio_index_1 in range(5):
                for sample_index, (_, _, test_record) in enumerate(records_3[agent_name][radio_index_1]):
                    values = [0 for _ in range(5)]
                    for radio_value, value in test_record.items():
                        radio_index_2 = int(radio_value * 10)
                        values[radio_index_2] = value
                    if max(values) == values[radio_index_1]:
                        counts_3[agent_index, radio_index_1] += 1
        task_data["c"] = counts_3

        save_data(save_path=sort_path + "supp30.pkl", information=task_data)


if __name__ == "__main__":
    if not path.exists(sort_path):
        mkdir(sort_path)

    if not path.exists(raw_path):
        raise ValueError("Please run the tasks (run_1_tasks.py) first!")

    main_02()
    main_03()
    main_04()
    main_05()
    main_06()
    supp_01()
    supp_02()
    supp_04()
    supp_05()
    supp_06()
    supp_07()
    supp_08()
    supp_09()
    supp_10()
    supp_11()
    supp_12()
    supp_13()
    supp_14()
    supp_15()
    supp_16()
    supp_17()
    supp_18()
    supp_19()
    supp_20()
    supp_21()
    supp_22()
    supp_23()
    supp_24()
    supp_25()
    supp_26()
    supp_27()
    supp_28()
    supp_29()
    supp_30()
