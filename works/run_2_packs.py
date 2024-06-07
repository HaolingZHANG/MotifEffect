"""
@Author      : Haoling Zhang
@Description : Package all the presented data from the experimental results.
"""
from collections import Counter
from copy import deepcopy
from itertools import product
from numpy import array, zeros, linspace, expand_dims, abs, mean, min, max, sum, power, argmax, argmin, all, where
from os import path, mkdir, environ
environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from scipy.stats import gaussian_kde  # noqa
from umap import UMAP  # noqa
from warnings import filterwarnings # noqa

from effect import NeuralMotif, calculate_landscape, detect_curvature_feature, minimum_loss_search  # noqa
from practice import acyclic_motifs  # noqa
from works import load_data, save_data  # noqa

filterwarnings("ignore")

motif_types, motif_indices = ["incoherent-loop", "coherent-loop", "collider"], [1, 2, 3, 4]
activation, aggregation = "tanh", "sum"
weight_values, bias_values = linspace(+0.1, +1.0, 10), linspace(-1.0, +1.0, 21)
value_range, points, sample_number = (-1, +1), 41, 100
norm_type = "L-2"
learn_rate, iteration_thresholds = 1e-3, (100, 100)

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


def supp_03():
    """
    Collect plot data from Figure S3 in supplementary file.
    """
    if not path.exists(sort_path + "supp03.pkl"):
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
        save_data(save_path=sort_path + "supp03.pkl", information=task_data)


def supp_04():
    """
    Collect plot data from Figure S4 in supplementary file.
    """
    if not path.exists(sort_path + "supp04.pkl"):
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
                if use_rate_2 <= 0.4:
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
                    task_data["b"].append([sum(source_region.reshape(-1)) / (101 ** 2),
                                           sum(target_region.reshape(-1)) / (101 ** 2)])
        # noinspection PyUnresolvedReferences
        task_data["b"] = array(task_data["b"])

        save_data(save_path=sort_path + "supp04.pkl", information=task_data)


def supp_05():
    """
    Collect plot data from Figure S5 in supplementary file.
    """
    if not path.exists(sort_path + "supp05.pkl"):
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

        save_data(save_path=sort_path + "supp05.pkl", information=task_data)


def supp_06():
    """
    Collect plot data from Figure S6 in supplementary file.
    """
    if not path.exists(sort_path + "supp06.pkl"):
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
        save_data(save_path=sort_path + "supp06.pkl", information=task_data)


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
    supp_03()
    supp_04()
    supp_05()
    supp_06()
