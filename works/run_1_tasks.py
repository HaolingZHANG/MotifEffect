"""
@Author      : Haoling Zhang
@Description : Run all experiments for this work.
"""
from hashlib import md5
from itertools import product, combinations_with_replacement
from numpy import array, linspace, arange, zeros, ones, random, concatenate, hstack, argsort, min, max, ceil, where
from os import path, mkdir, listdir
from ucimlrepo import fetch_ucirepo

from effect import NeuralMotif, ColliderNetwork, LoopNetwork, RestrictedLoopNetwork
from effect import generate_outputs, estimate_lipschitz, estimate_lipschitz_by_motif, fit
from effect import calculate_differences, execute_catch_processes, execute_escape_processes

from practice import acyclic_motifs, NEATCartPoleTask, SupervisionTask, NormNoiseGenerator
from practice import create_agent_config, train_and_evaluate

from works import load_data, save_data

motif_types, motif_indices = ["incoherent-loop", "coherent-loop", "collider"], [1, 2, 3, 4]
activation, aggregation = "tanh", "sum"
weight_values, bias_values = linspace(+0.1, +1.0, 10), linspace(-1.0, +1.0, 21)
selected_weight_values, selected_bias_values = array([+0.1, +0.5, +1.0]), array([-1.0, 0.0, +1.0])
value_range, points, sample_number = (-1, +1), 41, 100
norm_type = "L-2"

learn_rate, iteration_thresholds = 1e-3, (100, 100)
patience, fitted_threshold = 1000, 1e-3

agent_names, radios = ["b", "i", "c", "a"], [0.0, 0.1, 0.2, 0.3, 0.4]
config_names = ["baseline.config", "adjusted[i].config", "adjusted[c].config", "adjusted[a].config"]

landscape_names = ["Quadratic Saddle", "Monkey Saddle", "Branin", "three-hump Camel", "six-hump Camel"]

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
    Use classical neuroevolution method (NEAT) and its variations to learn reinforcement learning task (CartPole),
    for verifying the influence of the robustness of motif usages on entire neural networks.
    """
    if not path.exists(raw_path + "real-world/"):
        mkdir(raw_path + "real-world/")

    noise_generators = {}
    for radio in radios:
        noise_generators[radio] = NormNoiseGenerator(norm_type=norm_type, noise_scale=radio)

    if not path.exists(path=raw_path + "real-world/adjustments.1.pkl"):
        agent_configs = [create_agent_config(config_path + "main/" + config_name) for config_name in config_names]
        record, maximum_generation = {}, 20
        for agent_name, agent_config in zip(agent_names, agent_configs):
            record[agent_name] = {}
            for train_radio in radios:
                result = train_and_evaluate(task=NEATCartPoleTask(maximum_generation=maximum_generation),
                                            agent_name=agent_name, agent_config=agent_config, repeats=sample_number,
                                            train_noise_generator=noise_generators[train_radio],
                                            test_noise_generators=noise_generators,
                                            evaluation_type="reinforcement")
                record[agent_name][train_radio] = result
        save_data(save_path=raw_path + "real-world/adjustments.1.pkl", information=record)

    if not path.exists(path=raw_path + "real-world/iterations.pkl"):
        agent_configs = [create_agent_config(config_path + "main/" + config_name) for config_name in config_names]
        record, train_radio, generations = {}, 0.3, arange(30, 151, 10)
        for agent_name, agent_config in zip(agent_names, agent_configs):
            record[agent_name] = []
            for generation in generations:
                result = train_and_evaluate(task=NEATCartPoleTask(maximum_generation=generation),
                                            agent_name=agent_name, agent_config=agent_config, repeats=sample_number,
                                            train_noise_generator=noise_generators[train_radio],
                                            test_noise_generators=noise_generators,
                                            evaluation_type="reinforcement")
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
        agent_configs = [create_agent_config(config_path + "main/" + config_name) for config_name in config_names]
        record, maximum_generation, train_radio = {}, 100, 0.3
        for agent_name, agent_config in zip(agent_names, agent_configs):
            result = train_and_evaluate(task=NEATCartPoleTask(maximum_generation=maximum_generation),
                                        agent_name=agent_name, agent_config=agent_config, repeats=sample_number,
                                        train_noise_generator=noise_generators[train_radio],
                                        test_noise_generators=noise_generators,
                                        evaluation_type="reinforcement")
            record[agent_name] = result
        save_data(save_path=raw_path + "real-world/adjustments.2.pkl", information=record)


def task_4():
    """
    Use classical neuroevolution method (NEAT) and its variations to learn 3 supervision learning tasks in real world,
    for verifying the influence of the robustness of motif usages on entire neural networks.
    """
    noise_generators = {}
    for radio in radios:
        noise_generators[radio] = NormNoiseGenerator(norm_type=norm_type, noise_scale=radio)

    if not path.exists(path=raw_path + "real-world/uci.datasets.pkl"):
        # (1) biology https://archive.ics.uci.edu/dataset/39/ecoli
        # (2) physics and chemistry https://archive.ics.uci.edu/dataset/42/glass+identification
        # (3) health and medicine https://archive.ics.uci.edu/dataset/212/vertebral+column
        records = {}
        for uci_index, domain in zip([39, 42, 212], ["biology", "physics and chemistry", "health and medicine"]):
            dataset = fetch_ucirepo(id=uci_index)
            inputs, outputs, output_collector = dataset.data.features.to_numpy(), dataset.data.targets.to_numpy(), {}
            for index, output in enumerate(outputs):
                if output[0] in output_collector:
                    # noinspection PyUnresolvedReferences
                    output_collector[output[0]].append(index)
                else:
                    output_collector[output[0]] = [index]
            trained_inputs, trained_outputs, tested_inputs, tested_outputs = [], [], [], []
            for index, (key, values) in enumerate(output_collector.items()):
                trained_number = int(len(values) / 2.0 + 0.5)
                tested_number = len(values) - trained_number
                flags = hstack((ones(shape=(trained_number,), dtype=bool), zeros(shape=(tested_number,), dtype=bool)))
                random.shuffle(flags)
                for flag, value in zip(flags, values):
                    if flag:
                        trained_inputs.append(inputs[value])
                        trained_outputs.append(index)
                    else:
                        tested_inputs.append(inputs[value])
                        tested_outputs.append(index)
            trained_inputs, trained_outputs = array(trained_inputs), array(trained_outputs)
            tested_inputs, tested_outputs = array(tested_inputs), array(tested_outputs)
            data_ranges = array([[min([min(trained_inputs[:, index]), min(tested_inputs[:, index])]),
                                  max([max(trained_inputs[:, index]), max(tested_inputs[:, index])])]
                                 for index in range(len(trained_inputs[0]))])
            data_types = [data_type.lower() for data_type in list(dataset.variables.to_numpy()[:, 2])[1:-1]]
            data_names = list(dataset.variables.to_numpy()[:, 0])[1:-1]
            label_number = len(output_collector)
            records["case study in " + domain] = (trained_inputs, trained_outputs, tested_inputs, tested_outputs,
                                                  label_number, data_names, data_types, data_ranges)
        save_data(save_path=raw_path + "real-world/uci.datasets.pkl", information=records)

    if (not path.exists(path=raw_path + "real-world/biology.pkl")) or \
            (not path.exists(path=raw_path + "real-world/physics-chemistry.pkl")) or \
            (not path.exists(path=raw_path + "real-world/health-medicine.pkl")):
        maximum_generation = 100
        for domain, (trained_inputs, trained_outputs, tested_inputs, tested_outputs, label_number, data_names,
                     data_types, data_ranges) in load_data(raw_path + "real-world/uci.datasets.pkl").items():
            label, agent_configs, record = domain[len("case study in "):].replace(" and ", "-"), [], {}
            for config_name in config_names:
                agent_configs.append(create_agent_config(config_path + "supp/" + label + "." + config_name))
            task = SupervisionTask(trained_inputs=trained_inputs, trained_outputs=trained_outputs,
                                   tested_inputs=tested_inputs, tested_outputs=tested_outputs,
                                   description=domain, label_number=label_number, data_types=data_types,
                                   data_ranges=data_ranges, maximum_generation=maximum_generation)
            for agent_name, agent_config in zip(agent_names, agent_configs):
                record[agent_name] = {}
                for radio_index, train_radio in enumerate(radios):
                    result = train_and_evaluate(task=task, agent_name=agent_name, agent_config=agent_config,
                                                repeats=sample_number,
                                                train_noise_generator=noise_generators[train_radio],
                                                test_noise_generators=noise_generators,
                                                evaluation_type="supervision")
                    record[agent_name][radio_index] = result
            save_data(save_path=raw_path + "real-world/" + label + ".pkl", information=record)


def task_5():
    """
    Analyze the differences in representational capacity between collider and loop motifs at the network level,
    and further examine the multi-parameter distinctions between coherent and incoherent loops.
    """
    if not path.exists(raw_path + "network-scale/"):
        mkdir(raw_path + "network-scale/")

    for landscape_name in landscape_names:
        if not path.exists(path=raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl"):
            records = {}
            for motif_number in range(1, 11):
                sub_records = []
                for _ in range(sample_number):
                    network = ColliderNetwork(motif_number=motif_number)
                    record = fit(network=network, name=landscape_name,
                                 patience=patience, learn_rate=learn_rate, threshold=fitted_threshold)
                    sub_records.append(record["training loss"])
                records[("collider", motif_number)] = sub_records

                for _ in range(sample_number):
                    network = LoopNetwork(motif_number=motif_number)
                    record = fit(network=network, name=landscape_name,
                                 patience=patience, learn_rate=learn_rate, threshold=fitted_threshold)
                    sub_records.append(record["training loss"])
                records[("loop", motif_number)] = sub_records

            save_data(save_path=raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl",
                      information=records)

    motif_number, small_sampling_number = 10, 10
    for landscape_name in landscape_names:
        if not path.exists(path=raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl"):
            previous_records = load_data(raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl")
            counts = []
            for losses in previous_records[("loop", motif_number)]:
                counts.append(len(losses))
            maximum_iteration = max(counts)

            records = {}
            for coherent_number in range(0, 11):
                flags_1 = concatenate((zeros(shape=(coherent_number,), dtype=int),
                                       ones(shape=(motif_number - coherent_number,), dtype=int)))
                for flags_2_1 in combinations_with_replacement([1, 2, 3, 4], coherent_number):
                    for flags_2_2 in combinations_with_replacement([1, 2, 3, 4], motif_number - coherent_number):
                        flags_2 = concatenate([flags_2_1, flags_2_2]).astype(int)
                        network_info = str(flags_1)[1:-1].replace(" ", "") + "-" + str(flags_2)[1:-1].replace(" ", "")
                        motif_combination, sub_records = (tuple(flags_1), tuple(flags_2)), []
                        for _ in range(small_sampling_number):
                            network = RestrictedLoopNetwork(flags_1=flags_1, flags_2=flags_2, motif_number=motif_number)
                            record = fit(network=network, name=landscape_name, patience=1000, learn_rate=1e-3,
                                         threshold=1e-3, maximum_iteration=maximum_iteration)
                            sub_records.append(record)
                            if record["training loss"][-1] <= 1e-3:
                                sub_records.append(record)
                            else:
                                sub_records = []
                                break
                        records[network_info] = sub_records

            save_data(save_path=raw_path + "network-scale/incoherent.vs.coherent." + landscape_name + ".pkl",
                      information=records)


if __name__ == "__main__":
    if not path.exists(raw_path):
        mkdir(raw_path)

    if not path.exists(config_path):
        raise ValueError("Configures have not been declared!")

    task_1()
    task_2()
    task_3()
    task_4()
    task_5()

    print("| parent path in the /raw/ folder | file name | MD5 | file size (KB) |")
    print("| --- | --- | --- | --- |")
    for fold_name in ["difference", "landscapes", "parameters", "particular", "network-scale",
                      "real-world", "robustness", "trade-offs", "videos"]:
        for child_path in listdir(raw_path + fold_name + "/"):
            md5_hash = md5()
            with open(raw_path + fold_name + "/" + child_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    md5_hash.update(byte_block)
            md5_value = (md5_hash.hexdigest()).upper()
            file_size = ceil(path.getsize(raw_path + fold_name + "/" + child_path) / 1024).astype(int)
            print("| " + fold_name + " | " + child_path + " | " + md5_value + " | " + str(file_size) + " |")
