"""
@Author      : Haoling Zhang
@Description : Run all experiments for this work.
"""
from itertools import product
from numpy import array, linspace, arange, zeros, min, argsort, where
from os import path, mkdir

from effect import NeuralMotif, generate_outputs, estimate_lipschitz, estimate_lipschitz_by_motif
from effect import calculate_differences, execute_catch_processes, execute_escape_processes

from practice import acyclic_motifs, NEATCartPoleTask, NormNoiseGenerator
from practice import create_agent_config, train_and_evaluate

from works import load_data, save_data

motif_types, motif_indices = ["incoherent-loop", "coherent-loop", "collider"], [1, 2, 3, 4]
activation, aggregation = "tanh", "sum"
weight_values, bias_values = linspace(+0.1, +1.0, 10), linspace(-1.0, +1.0, 21)
selected_weight_values, selected_bias_values = array([+0.1, +0.5, +1.0]), array([-1.0, 0.0, +1.0])
value_range, points, sample_number = (-1, +1), 41, 100
norm_type = "L-2"

learn_rate, iteration_thresholds = 1e-3, (100, 100)

agent_names, radios = ["b", "i", "c", "a"], [0.0, 0.1, 0.2, 0.3, 0.4]
config_names = ["baseline.config", "adjusted[i].config", "adjusted[c].config", "adjusted[a].config"]

raw_path, config_path = "./raw/", "./confs/"


def task_1():
    """
    By averaging sampling, the population for each motif structure is established for estimating
    the trade-off between representational capacity and numerical stability.
    """
    if not path.exists(raw_path + "parameters/"):
        mkdir(raw_path + "parameters/")
    if not path.exists(raw_path + "landscapes/"):
        mkdir(raw_path + "landscapes/")
    if not path.exists(raw_path + "robustness/"):
        mkdir(raw_path + "robustness/")
    if not path.exists(raw_path + "trade-offs/"):
        mkdir(raw_path + "trade-offs/")

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
            if len(motif_structure.edges) == 3:
                activations, aggregations = [activation, activation], [aggregation, aggregation]
            else:
                activations, aggregations = [activation], [aggregation]

            structure = motif_type + "." + str(motif_index)

            completed = True
            if not path.exists(raw_path + "parameters/" + structure + ".npy"):
                completed = False
            if not path.exists(raw_path + "landscapes/" + structure + ".npy"):
                completed = False
            if not completed:
                result = generate_outputs(motif_type=motif_type, motif_index=motif_index,
                                          activations=activations, aggregations=aggregations,
                                          weight_groups=weight_groups, bias_groups=bias_groups,
                                          value_range=value_range, points=points)
                collection = []
                for landscape in result[1]:
                    landscape = landscape.reshape(points, points)
                    values = estimate_lipschitz(value_range=value_range, points=points,
                                                output=landscape, norm_type=norm_type)
                    collection.append(values)
                save_data(save_path=raw_path + "parameters/" + structure + ".npy", information=result[0])
                save_data(save_path=raw_path + "landscapes/" + structure + ".npy", information=result[1])
                save_data(save_path=raw_path + "robustness/" + structure + ".npy", information=array(collection))

    if not path.exists(raw_path + "difference/"):
        mkdir(raw_path + "difference/")

    for motif_type_1 in motif_types:
        for motif_index_1 in motif_indices:
            for motif_type_2 in motif_types:
                for motif_index_2 in motif_indices:
                    source = motif_type_1 + "." + str(motif_index_1)
                    target = motif_type_2 + "." + str(motif_index_2)
                    save_feature = source + " for " + target
                    if not path.exists(raw_path + "difference/" + save_feature + ".npy"):
                        load_feature_1 = motif_type_1 + "." + str(motif_index_1)
                        load_feature_2 = motif_type_2 + "." + str(motif_index_2)

                        if load_feature_1 == load_feature_2:
                            landscapes = load_data(load_path=raw_path + "landscapes/" + load_feature_1 + ".npy")
                            result = calculate_differences(landscapes_1=landscapes, norm_type=norm_type)
                            save_data(save_path=raw_path + "difference/" + save_feature + ".npy", information=result)
                        else:
                            landscapes_1 = load_data(load_path=raw_path + "landscapes/" + load_feature_1 + ".npy")
                            landscapes_2 = load_data(load_path=raw_path + "landscapes/" + load_feature_2 + ".npy")
                            result = calculate_differences(landscapes_1=landscapes_1, landscapes_2=landscapes_2,
                                                           norm_type=norm_type)
                            save_data(save_path=raw_path + "difference/" + save_feature + ".npy", information=result)

    target_motifs = []
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
                target_motifs.append(target_motif)

    for motif_type in motif_types[:-1]:
        for motif_index in motif_indices:
            feature = motif_type + "." + str(motif_index)
            if not path.exists(raw_path + "trade-offs/" + feature + ".npy"):
                parameters = load_data(load_path=raw_path + "parameters/" + feature + ".npy")
                references = []
                for parameter in parameters:
                    references.append(NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                                  activations=[activation, activation],
                                                  aggregations=[aggregation, aggregation],
                                                  weights=parameter[:3], biases=parameter[3:]))

                record = execute_catch_processes(references=references, catchers=target_motifs,
                                                 value_range=value_range, points=points,
                                                 learn_rate=learn_rate, threshold=iteration_thresholds[1])
                results = []
                for target, loss in record:
                    robust_target = estimate_lipschitz_by_motif(value_range=value_range, points=points, motif=target)
                    results.append([robust_target, loss])
                save_data(save_path=raw_path + "trade-offs/" + feature + ".npy", information=array(results))


def task_2():
    """
    Use motif escape process to investigate how incoherent loops or coherent loops
    achieve their specificity compared with colliders.
    """
    if not path.exists(raw_path + "particular/"):
        mkdir(raw_path + "particular/")

    target_motifs = []
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
                target_motifs.append(target_motif)

    for motif_type in motif_types[:-1]:
        for motif_index in motif_indices:
            source_feature = motif_type + "." + str(motif_index)
            if not path.exists(raw_path + "particular/" + source_feature + ".initialization.pkl"):
                differences = []
                for another_index in motif_indices:
                    mixed_feature = source_feature + " for " + motif_types[-1] + "." + str(another_index)
                    differences.append(load_data(load_path=raw_path + "difference/" + mixed_feature + ".npy").tolist())
                available_indices = where(min(array(differences), axis=0) <= 0.03)[0]

                self_feature = source_feature + " for " + source_feature
                self_difference = load_data(load_path=raw_path + "difference/" + self_feature + ".npy")
                self_difference = self_difference[available_indices]

                source_parameters, motifs = load_data(load_path=raw_path + "parameters/" + source_feature + ".npy"), []
                for index in available_indices[argsort(self_difference)[::-1]][:sample_number]:
                    source_motif = NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                               activations=[activation, activation],
                                               aggregations=[aggregation, aggregation],
                                               weights=source_parameters[index, :3],
                                               biases=source_parameters[index, 3:])
                    motifs.append((source_motif, target_motifs))

                save_data(save_path=raw_path + "particular/" + source_feature + ".initialization.pkl",
                          information=motifs)

    for motif_type in motif_types[:-1]:
        for motif_index in motif_indices:
            source_feature = motif_type + "." + str(motif_index)
            if not path.exists(raw_path + "particular/" + source_feature + ".escape-process.pkl"):
                motif_data = load_data(load_path=raw_path + "sacrifices/" + source_feature + ".initialization.pkl")
                records = execute_escape_processes(motif_pairs=motif_data, value_range=value_range, points=points,
                                                   learn_rate=learn_rate, thresholds=iteration_thresholds)

                save_data(save_path=raw_path + "particular/" + source_feature + ".escape-process.pkl",
                          information=records)


def task_3():
    """
    Use classical neuroevolution method (NEAT) and its variations to learn CartPole environment,
    for verifying the influence of the robustness of motif usages on entire neural networks.
    """
    if not path.exists(raw_path + "real-world/"):
        mkdir(raw_path + "real-world/")

    agent_configs = [create_agent_config(config_path + config_name) for config_name in config_names]

    noise_generators = {}
    for radio in radios:
        noise_generators[radio] = NormNoiseGenerator(norm_type=norm_type, noise_scale=radio)

    if not path.exists(path=raw_path + "real-world/adjustments.1.pkl"):
        record, maximum_generation = {}, 20
        for agent_name, agent_config in zip(agent_names, agent_configs):
            record[agent_name] = {}
            for train_radio in radios:
                result = train_and_evaluate(task=NEATCartPoleTask(maximum_generation=maximum_generation),
                                            agent_name=agent_name, agent_config=agent_config, repeats=sample_number,
                                            train_noise_generator=noise_generators[train_radio],
                                            test_noise_generators=noise_generators)
                record[agent_name][train_radio] = result
        save_data(save_path=raw_path + "real-world/adjustments.1.pkl", information=record)

    if not path.exists(path=raw_path + "real-world/iterations.pkl"):
        record, train_radio, generations = {}, 0.3, arange(30, 151, 10)
        for agent_name, agent_config in zip(agent_names, agent_configs):
            record[agent_name] = []
            for generation in generations:
                result = train_and_evaluate(task=NEATCartPoleTask(maximum_generation=generation),
                                            agent_name=agent_name, agent_config=agent_config, repeats=sample_number,
                                            train_noise_generator=noise_generators[train_radio],
                                            test_noise_generators=noise_generators)
                values = []
                for _, _, test_record in result:
                    values.append(list(test_record.values()))
                values, matrix = array(values), zeros(shape=(sample_number,), dtype=int)
                for index in range(len(radios)):
                    matrix[where(values[:, index] >= 195)] += 1
                record[agent_name].append(len(where(matrix >= 4)[0]))
            # noinspection PyUnresolvedReferences
            record[agent_name] = array(record[agent_name])
        save_data(save_path=raw_path + "real-world/iterations.pkl", information=record)

    if not path.exists(path=raw_path + "real-world/adjustments.2.pkl"):
        record, maximum_generation, train_radio = {}, 100, 0.3
        for agent_name, agent_config in zip(agent_names, agent_configs):
            result = train_and_evaluate(task=NEATCartPoleTask(maximum_generation=maximum_generation),
                                        agent_name=agent_name, agent_config=agent_config, repeats=sample_number,
                                        train_noise_generator=noise_generators[train_radio],
                                        test_noise_generators=noise_generators)
            record[agent_name] = result
        save_data(save_path=raw_path + "real-world/adjustments.2.pkl", information=record)


if __name__ == "__main__":
    if not path.exists(raw_path):
        mkdir(raw_path)

    if not path.exists(config_path):
        raise ValueError("Configures have not been declared!")

    task_1()
    task_2()
    task_3()
