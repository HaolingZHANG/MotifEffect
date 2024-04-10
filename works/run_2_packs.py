"""
@Author      : Haoling Zhang
@Description : Package all the presented data from the experimental results.
"""
from collections import Counter
from numpy import array, zeros, linspace, abs, mean, min, max, sum, argmax, argmin, all, where
from os import path, mkdir
from scipy.stats import spearmanr, gaussian_kde

from effect import calculate_landscape, calculate_gradients, detect_concavity
from works import load_data, save_data

motif_types, motif_indices = ["incoherent-loop", "coherent-loop", "collider"], [1, 2, 3, 4]
activation, aggregation = "tanh", "sum"
weight_values, bias_values = linspace(+0.1, +1.0, 10), linspace(-1.0, +1.0, 21)
value_range, points, sample_number = (-1, +1), 41, 100

agent_names, radios = ["b", "i", "c", "a"], [0.0, 0.1, 0.2, 0.3, 0.4]

raw_path, sort_path = "./raw/", "./data/"


def main_01():
    """
    Collect plot data from Figure 1 in main text.
    """
    if not path.exists(sort_path + "main01.pkl"):
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

        task_data = {"b": robustness_distributions, "c": difference_distributions,
                     "d": density_map["incoherent-loop"], "e": density_map["coherent-loop"]}

        save_data(save_path=sort_path + "main01.pkl", information=task_data)


def main_02():
    """
    Collect plot data from Figure 2 in main text.
    """
    if not path.exists(sort_path + "main02.pkl"):
        task_data = {}

        escape_data = load_data(load_path=raw_path + "particular/" + motif_types[0] + ".1.escape-process.pkl")
        maximum_index, maximum_loss = 0, -1
        for index, (motifs, losses) in enumerate(escape_data):
            start, stop = argmin(losses), argmax(losses)
            if stop <= start:
                continue
            if losses[stop] - losses[start] > maximum_loss:
                maximum_loss, maximum_index = losses[stop] - losses[start], index
        motifs, losses = escape_data[maximum_index]
        source_motif, target_motif = motifs[argmin(losses)][0], motifs[argmax(losses)][0]
        source_landscape = calculate_landscape(value_range, points, source_motif)
        target_landscape = calculate_landscape(value_range, points, target_motif)

        task_data["b"] = tuple([source_landscape, target_landscape,
                                detect_concavity(calculate_landscape(value_range, 101, source_motif), 0.01),
                                detect_concavity(calculate_landscape(value_range, 101, target_motif), 0.01)])

        escape_data = load_data(load_path=raw_path + "particular/" + motif_types[1] + ".1.escape-process.pkl")
        maximum_index, maximum_loss = 0, -1
        for index, (motifs, losses) in enumerate(escape_data):
            start, stop = argmin(losses), argmax(losses)
            if stop <= start:
                continue
            if losses[stop] - losses[start] > maximum_loss:
                maximum_loss, maximum_index = losses[stop] - losses[start], index
        motifs, losses = escape_data[maximum_index]
        source, target = motifs[argmin(losses)][0], motifs[argmax(losses)][0]
        task_data["c"] = tuple([calculate_landscape(value_range, points, source),
                                calculate_landscape(value_range, points, target),
                                calculate_gradients(value_range, points, source)])

        records = []
        for motif_index in motif_indices:
            feature = motif_types[0] + "." + str(motif_index)
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                source, target = motifs[argmin(losses)][0], motifs[argmax(losses)][0]
                source_concavity = detect_concavity(calculate_landscape(value_range, 101, source), 0.01)
                target_concavity = detect_concavity(calculate_landscape(value_range, 101, target), 0.01)
                counter_1, counter_2 = Counter(source_concavity.reshape(-1)), Counter(target_concavity.reshape(-1))
                used_value_1, used_value_2 = max([counter_1[1], counter_1[-1]]), max([counter_2[1], counter_2[-1]])
                records.append([used_value_1 / (101 ** 2), used_value_2 / (101 ** 2)])
        task_data["d"] = array(records)

        records = []
        for motif_index in motif_indices:
            feature = motif_types[1] + "." + str(motif_index)
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                start, stop = argmin(losses), argmax(losses)
                source_landscape = calculate_landscape(value_range, points, motifs[start][0])
                target_landscape = calculate_landscape(value_range, points, motifs[stop][0])
                values_x = calculate_gradients(value_range, points, motifs[start][0]).reshape(-1)
                values_y = abs(target_landscape - source_landscape).reshape(-1)
                # noinspection PyTypeChecker
                correlation, p_value = spearmanr(values_x, values_y)
                records.append([correlation, p_value])
        task_data["e"] = array(records)

        save_data(save_path=sort_path + "main02.pkl", information=task_data)


def main_03():
    """
    Collect plot data from Figure 3 in main text.
    """
    if not path.exists(sort_path + "main03.pkl"):
        record = load_data(raw_path + "real-world/adjustments.1.pkl")

        task_data = {}

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

        save_data(save_path=sort_path + "main03.pkl", information=task_data)


def main_04():
    """
    Collect plot data from Figure 4 in main text.
    """
    if not path.exists(sort_path + "main04.pkl"):
        task_data = {}

        record = load_data(raw_path + "real-world/iterations.pkl")
        task_data["a"] = record

        record = load_data(raw_path + "real-world/adjustments.2.pkl")
        for strategy_index, (panel_index, strategy) in enumerate(zip(["b", "c", "d", "e"], agent_names)):
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
            a = (mean(array(cases[2]), axis=0), len(cases[2]), (100 - count))
            if len(cases[0]) > 0:
                b = (mean(array(cases[0]), axis=0), len(cases[0]), (100 - count))
            else:
                b = (None, 0, (100 - count))
            if len(cases[1]) > 0:
                c = (mean(array(cases[1]), axis=0), len(cases[1]), (100 - count))
            else:
                c = (None, 0, (100 - count))

            task_data[panel_index] = [a, b, c]

        save_data(save_path=sort_path + "main04.pkl", information=task_data)


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
                source_concavity = detect_concavity(calculate_landscape(value_range, 101, source), 0.01)
                target_concavity = detect_concavity(calculate_landscape(value_range, 101, target), 0.01)
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
                source_concavity = detect_concavity(source_landscape, 0.01)
                target_concavity = detect_concavity(target_landscape, 0.01)
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
        task_data["b"] = array(task_data["b"])

        save_data(save_path=sort_path + "supp04.pkl", information=task_data)


def supp_05():
    """
    Collect plot data from Figure S5 in supplementary file.
    """
    if not path.exists(sort_path + "supp05.pkl"):
        task_data = {}
        for panel_label, motif_index in zip(["a", "b", "c", "d"], motif_indices):
            feature, records = motif_types[1] + "." + str(motif_index), []
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                start, stop = argmin(losses), argmax(losses)
                source_landscape = calculate_landscape(value_range, points, motifs[start][0])
                target_landscape = calculate_landscape(value_range, points, motifs[stop][0])
                values_x = calculate_gradients(value_range, points, motifs[start][0]).reshape(-1)
                values_y = abs(target_landscape - source_landscape).reshape(-1)
                # noinspection PyTypeChecker
                correlation, p_value = spearmanr(values_x, values_y)
                records.append(correlation)
            task_data[panel_label] = array(records)
        save_data(save_path=sort_path + "supp05.pkl", information=task_data)


def supp_06():
    """
    Collect plot data from Figure S6 in supplementary file.
    """
    if not path.exists(sort_path + "supp06.pkl"):
        task_data = {}
        for panel_label, motif_index in zip(["a", "b", "c", "d"], motif_indices):
            feature, records = motif_types[1] + "." + str(motif_index), []
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                start, stop = argmin(losses), argmax(losses)
                source_landscape = calculate_landscape(value_range, points, motifs[start][0])
                target_landscape = calculate_landscape(value_range, points, motifs[stop][0])
                values_x = calculate_gradients(value_range, points, motifs[start][0]).reshape(-1)
                values_y = abs(target_landscape - source_landscape).reshape(-1)
                # noinspection PyTypeChecker
                correlation, p_value = spearmanr(values_x, values_y)
                if correlation < 0.2 and losses[stop] < 0.013:
                    records.append([losses[start], losses[stop]])
            task_data[panel_label] = array(records)
        save_data(save_path=sort_path + "supp06.pkl", information=task_data)


def supp_07():
    """
    Collect plot data from Figure S7 in supplementary file.
    """
    if not path.exists(sort_path + "supp07.pkl"):
        task_data, flag, records = {}, True, []
        for motif_index in motif_indices:
            feature = motif_types[1] + "." + str(motif_index)
            escape_data = load_data(load_path=raw_path + "particular/" + feature + ".escape-process.pkl")
            for index, (motifs, losses) in enumerate(escape_data):
                start, stop = argmin(losses), argmax(losses)
                source_landscape = calculate_landscape(value_range, points, motifs[start][0])
                target_landscape = calculate_landscape(value_range, points, motifs[stop][0])
                values_x = calculate_gradients(value_range, points, motifs[start][0]).reshape(-1)
                values_y = abs(target_landscape - source_landscape).reshape(-1)
                # noinspection PyTypeChecker
                correlation, _ = spearmanr(values_x, values_y)
                if correlation < 0.2 and losses[stop] > 0.013:
                    correlations = []
                    for location in range(start, stop - 1):
                        new_source_landscape = calculate_landscape(value_range, points, motifs[location][0])
                        new_target_landscape = calculate_landscape(value_range, points, motifs[location + 1][0])
                        new_values_x = calculate_gradients(value_range, points, motifs[location][0]).reshape(-1)
                        new_values_y = abs(new_target_landscape - new_source_landscape).reshape(-1)
                        # noinspection PyTypeChecker
                        new_correlation, _ = spearmanr(new_values_x, new_values_y)
                        correlations.append(new_correlation)
                    if flag:
                        task_data["a"] = (source_landscape, target_landscape, correlation, correlations)
                        flag = False
                    records.append([correlation, mean(correlations)])
            task_data["b"] = array(records)
        save_data(save_path=sort_path + "supp07.pkl", information=task_data)


def supp_08():
    """
    Collect plot data from Figure S8 in supplementary file.
    """
    if not path.exists(sort_path + "supp08.pkl"):
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
        save_data(save_path=sort_path + "supp08.pkl", information=task_data)


if __name__ == "__main__":
    if not path.exists(sort_path):
        mkdir(sort_path)

    if not path.exists(raw_path):
        raise ValueError("Please run the tasks (run_1_tasks.py) first!")

    main_01()
    main_02()
    main_03()
    main_04()
    supp_01()
    supp_02()
    supp_03()
    supp_04()
    supp_05()
    supp_06()
    supp_07()
    supp_08()
