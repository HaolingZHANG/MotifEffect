from copy import deepcopy
from itertools import product
from numpy import linspace, array, zeros, mean, min, max, abs, argmin, argmax
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from os.path import exists

from effect import NeuralMotif, prepare_motifs
from effect import calculate_landscape, calculate_rugosity, calculate_gradients, evaluate_propagation
from effect import maximum_minimum_loss_search, minimum_loss_search

from grace import NEATCartPoleTask, NEATAgent, InputNormNoiseGenerator, NormType
from grace import create_agent_config, obtain_best

from works import Monitor, save_data, load_data


def experiment_1(raw_path, task_path):
    value_range, points, times, weight_range, monitor = (-1, +1), 41, 11, (-1, +1), Monitor()
    activation_selection, aggregation_selection = ["relu", "tanh", "sigmoid"], ["sum", "max"]

    if not exists(path=raw_path + "property.landscape.pkl"):
        print("Calculate the output landscape of the incoherent loop population and the collider population.")
        records = []
        current, total = 0, 4 * (len(activation_selection) ** 2) * (len(aggregation_selection) ** 2) * (times ** 3)
        flag = 1
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=2):
                for aggregations in product(aggregation_selection, repeat=2):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=3):
                        try:
                            motif = NeuralMotif(motif_type="incoherent-loop", motif_index=motif_index,
                                                activations=activations, aggregations=aggregations,
                                                weights=weights, biases=(0.0, 0.0))
                            records.append(("incoherent loop", flag,
                                            calculate_landscape(value_range, points, motif)))
                            flag += 1
                        except ValueError:
                            pass

                        monitor.output(current_state=current + 1, total_state=total)
                        current += 1

        current, total = 0, 4 * len(activation_selection) * len(aggregation_selection) * (times ** 2)
        flag = 1
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=1):
                for aggregations in product(aggregation_selection, repeat=1):
                    for weights in product(linspace(-1.0, 1.0, 11), repeat=2):
                        try:
                            motif = NeuralMotif(motif_type="collider", motif_index=motif_index,
                                                activations=activations, aggregations=aggregations,
                                                weights=weights, biases=(0.0,))
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

    if not exists(path=raw_path + "property.rugosity.pkl"):
        print("Calculate the rugosity index of the incoherent loop population and the collider population.")
        records = []
        current, total = 0, 4 * (len(activation_selection) ** 2) * (len(aggregation_selection) ** 2) * (times ** 3)
        flag = 1
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=2):
                for aggregations in product(aggregation_selection, repeat=2):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=3):
                        try:
                            motif = NeuralMotif(motif_type="incoherent-loop", motif_index=motif_index,
                                                activations=activations, aggregations=aggregations,
                                                weights=weights, biases=(0.0, 0.0))
                            records.append(("incoherent loop", flag,
                                            calculate_rugosity(value_range, points, motif)))
                            flag += 1
                        except ValueError:
                            pass

                        monitor.output(current_state=current + 1, total_state=total)
                        current += 1

        current, total = 0, 4 * len(activation_selection) * len(aggregation_selection) * (times ** 2)
        flag = 1
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=1):
                for aggregations in product(aggregation_selection, repeat=1):
                    for weights in product(linspace(-1.0, 1.0, 11), repeat=2):
                        try:
                            motif = NeuralMotif(motif_type="collider", motif_index=motif_index,
                                                activations=activations, aggregations=aggregations,
                                                weights=weights, biases=(0.0,))
                            records.append(("collider", flag,
                                            calculate_rugosity(value_range, points, motif)))
                            flag += 1
                        except ValueError:
                            pass

                        monitor.output(current_state=current + 1, total_state=total)
                        current += 1

        save_data(save_path=raw_path + "property.rugosity.pkl", information=records)

    if not exists(path=raw_path + "property.propagation.extreme.pkl"):
        print("Calculate the maximum error propagation of the incoherent loop population and the collider population.")
        records = []
        current, total = 0, 4 * (len(activation_selection) ** 2) * (len(aggregation_selection) ** 2) * (times ** 3)
        flag = 1
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=2):
                for aggregations in product(aggregation_selection, repeat=2):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=3):
                        try:
                            motif = NeuralMotif(motif_type="incoherent-loop", motif_index=motif_index,
                                                activations=activations, aggregations=aggregations,
                                                weights=weights, biases=(0.0, 0.0))
                            records.append(("incoherent loop", flag,
                                            evaluate_propagation(value_range, points, motif, "max")))
                            flag += 1
                        except ValueError:
                            pass

                        monitor.output(current_state=current + 1, total_state=total)
                        current += 1

        current, total = 0, 4 * len(activation_selection) * len(aggregation_selection) * (times ** 2)
        flag = 1
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=1):
                for aggregations in product(aggregation_selection, repeat=1):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=2):
                        try:
                            motif = NeuralMotif(motif_type="collider", motif_index=motif_index,
                                                activations=activations, aggregations=aggregations,
                                                weights=weights, biases=(0.0,))
                            records.append(("collider", flag,
                                            evaluate_propagation(value_range, points, motif, "max")))
                            flag += 1
                        except ValueError:
                            pass

                        monitor.output(current_state=current + 1, total_state=total)
                        current += 1

        save_data(save_path=raw_path + "property.propagation.extreme.pkl", information=records)

    if not exists(path=raw_path + "property.propagation.average.pkl"):
        print("Calculate the average error propagation of the incoherent loop population and the collider population.")
        records = []
        current, total = 0, 4 * (len(activation_selection) ** 2) * (len(aggregation_selection) ** 2) * (times ** 3)
        flag = 1
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=2):
                for aggregations in product(aggregation_selection, repeat=2):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=3):
                        try:
                            motif = NeuralMotif(motif_type="incoherent-loop", motif_index=motif_index,
                                                activations=activations, aggregations=aggregations,
                                                weights=weights, biases=(0.0, 0.0))
                            records.append(("incoherent loop", flag,
                                            evaluate_propagation(value_range, points, motif, "mean")))
                            flag += 1
                        except ValueError:
                            pass

                        monitor.output(current_state=current + 1, total_state=total)
                        current += 1

        current, total = 0, 4 * len(activation_selection) * len(aggregation_selection) * (times ** 2)
        flag = 1
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selection, repeat=1):
                for aggregations in product(aggregation_selection, repeat=1):
                    for weights in product(linspace(weight_range[0], weight_range[1], times), repeat=2):
                        try:
                            motif = NeuralMotif(motif_type="collider", motif_index=motif_index,
                                                activations=activations, aggregations=aggregations,
                                                weights=weights, biases=(0.0,))
                            records.append(("collider", flag,
                                            evaluate_propagation(value_range, points, motif, "mean")))
                            flag += 1
                        except ValueError:
                            pass

                        monitor.output(current_state=current + 1, total_state=total)
                        current += 1

        save_data(save_path=raw_path + "property.propagation.average.pkl", information=records)

    if (not exists(task_path + "rugosity loop.npy")) or (not exists(task_path + "rugosity collider.npy")):
        rugosity_indices, rugosity_record = [[], []], load_data(load_path=raw_path + "property.rugosity.pkl")
        for index, (name, _, rugosity_index) in enumerate(rugosity_record):
            if name == "incoherent loop":
                rugosity_indices[0].append(rugosity_index)
            else:
                rugosity_indices[1].append(rugosity_index)
        save_data(save_path=task_path + "rugosity loop.npy", information=array(rugosity_indices[0]))
        save_data(save_path=task_path + "rugosity collider.npy", information=array(rugosity_indices[1]))

    if not exists(task_path + "locations.npy"):
        difference_record = load_data(load_path=raw_path + "property.difference.npy")
        method = TSNE(n_components=2, metric="precomputed")
        locations = method.fit_transform(X=difference_record)
        save_data(save_path=task_path + "locations.npy", information=locations)

    if (not exists(task_path + "propagation extreme loop.npy")) \
            or (not exists(task_path + "propagation extreme collider.npy")):
        propagation_record = load_data(raw_path + "property.propagation.extreme.pkl")
        propagations = zeros(shape=(2, points, points))
        for index, (name, _, matrix) in enumerate(propagation_record):
            if name == "incoherent loop":
                propagations[0] = max([matrix, propagations[0]], axis=0)
            else:
                propagations[1] = max([matrix, propagations[1]], axis=0)
        save_data(save_path=task_path + "propagation extreme loop.npy", information=propagations[0])
        save_data(save_path=task_path + "propagation extreme collider.npy", information=propagations[1])

    if (not exists(task_path + "propagation average loop.npy")) \
            or (not exists(task_path + "propagation average collider.npy")):
        print("in")
        propagation_record = load_data(raw_path + "property.propagation.average.pkl")
        propagations, counts = zeros(shape=(2, points, points)), [0, 0]
        for index, (name, _, matrix) in enumerate(propagation_record):
            if name == "incoherent loop":
                propagations[0] += matrix
                counts[0] += 1
            else:
                propagations[1] += matrix
                counts[1] += 1
        save_data(save_path=task_path + "propagation average loop.npy", information=propagations[0] / counts[0])
        save_data(save_path=task_path + "propagation average collider.npy", information=propagations[1] / counts[1])


def experiment_2(raw_path, task_path):
    value_range, points, monitor = (-1, +1), 41, Monitor()
    learn_rate = (value_range[1] - value_range[0]) / (points - 1) * 1e-1
    activation_selections, aggregation_selections = ["relu", "tanh", "sigmoid"], ["sum", "max"]
    loss_threshold, check_threshold, iteration_threshold = 1e-5, 5, 1000

    if not exists(raw_path + "minimum.thresholds.pkl"):
        source_motifs, target_motifs, records = [], [], []
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selections, repeat=2):
                for aggregations in product(aggregation_selections, repeat=2):
                    source_motif = prepare_motifs(motif_type="incoherent-loop", motif_index=motif_index,
                                                  activations=activations, aggregations=aggregations,
                                                  sample=1, weights=(1e0, 1e-3, 1e0), biases=(0.0, 0.0))[0]
                    source_motifs.append(source_motif)
        for motif_index in [1, 2, 3, 4]:
            for activations in product(activation_selections, repeat=1):
                for aggregations in product(aggregation_selections, repeat=1):
                    target_motif = prepare_motifs(motif_type="collider", motif_index=motif_index,
                                                  activations=activations, aggregations=aggregations,
                                                  sample=1, weights=(1e-3, 1e0), biases=(0.0,))[0]
                    target_motifs.append(target_motif)

        records = []
        for source_index, source_motif in enumerate(source_motifs):
            record, minimum_loss, t = [], 2.0, None
            print("Calculate " + str(source_index + 1) + "-th motif.")
            for target_index in range(len(target_motifs)):
                target_motif = deepcopy(target_motifs[target_index])
                result = minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                             source_motif=source_motif, target_motif=target_motif,
                                             loss_threshold=loss_threshold, check_threshold=check_threshold,
                                             iteration_threshold=iteration_threshold, verbose=False)
                target_motif, target_loss = result
                minimum_loss = min([minimum_loss, target_loss])
                record.append((deepcopy(source_motif), deepcopy(target_motif), target_loss))
                monitor.output(target_index + 1, len(target_motifs),
                               extra={"current": target_loss, "best": minimum_loss})

            records.append(record)

        save_data(save_path=raw_path + "minimum.thresholds.pkl", information=records)

    if not exists(path=raw_path + "minimum.landscapes.npy"):
        print("Calculate the output landscape of motifs in minimum threshold task.")
        landscapes, records = [], load_data(load_path=raw_path + "minimum.thresholds.pkl")
        for record_index, record in enumerate(records):
            minimum_loss, similar_motif = None, None
            for source_motif, target_motif, loss in record:
                if minimum_loss is not None and minimum_loss > loss:
                    minimum_loss, similar_motif = loss, target_motif
                elif minimum_loss is None:
                    minimum_loss, similar_motif = loss, target_motif
            landscapes += [calculate_landscape(value_range=value_range, points=points, motif=record[0][0]),
                           calculate_landscape(value_range=value_range, points=points, motif=similar_motif)]
            monitor.output(record_index + 1, len(records))
        save_data(save_path=raw_path + "minimum.landscapes.npy", information=array(landscapes))

    if not exists(path=raw_path + "minimum.difference.npy"):
        print("Calculate the difference between motifs in minimum threshold task.")
        landscapes = load_data(load_path=raw_path + "minimum.landscapes.npy")
        differences = zeros(shape=(len(landscapes), len(landscapes)))
        current, total = 0, (len(landscapes) - 1) * len(landscapes) // 2
        for index_1 in range(len(landscapes)):
            for index_2 in range(index_1 + 1, len(landscapes)):
                difference = mean(abs(landscapes[index_1] - landscapes[index_2])) / (value_range[1] - value_range[0])
                differences[index_1, index_2], differences[index_2, index_1] = difference, difference
                monitor.output(current_state=current + 1, total_state=total)
                current += 1
        save_data(save_path=raw_path + "minimum.difference.npy", information=differences)

    if (not exists(task_path + "minimum losses.npy")) or (not exists(task_path + "rugosity indices.npy")) \
            or (not exists(task_path + "locations.npy")) or (not exists(task_path + "case.pkl")):
        minimum_losses, rugosity_indices, case_pairs = [], [], []
        training_records = load_data(raw_path + "minimum.thresholds.pkl")
        for record in training_records:
            minimum_loss, similar_motif = None, None
            for source_motif, target_motif, loss in record:
                if minimum_loss is not None and minimum_loss > loss:
                    minimum_loss, similar_motif = loss, target_motif
                elif minimum_loss is None:
                    minimum_loss, similar_motif = loss, target_motif
            minimum_losses.append(minimum_loss)
            rugosity_indices.append(calculate_rugosity(value_range=value_range, points=points, motif=record[0][0]))
            case_pairs.append((record[0][0], similar_motif,
                               calculate_landscape(value_range=value_range, points=points, motif=record[0][0]),
                               calculate_landscape(value_range=value_range, points=points, motif=similar_motif)))
        difference_record = load_data(load_path=raw_path + "minimum.difference.npy")
        method = TSNE(n_components=2, metric="precomputed")
        locations = method.fit_transform(X=difference_record)
        save_data(save_path=task_path + "minimum losses.npy", information=array(minimum_losses))
        save_data(save_path=task_path + "rugosity indices.npy", information=array(rugosity_indices))
        save_data(save_path=task_path + "locations.npy", information=locations)
        minimum_index, maximum_index = argmin(minimum_losses), argmax(minimum_losses)
        cases = {"min": (minimum_index, case_pairs[minimum_index]), "max": (maximum_index, case_pairs[maximum_index])}
        save_data(save_path=task_path + "terminal cases.pkl", information=cases)


def experiment_3(raw_path, task_path):
    value_range, points, monitor = (-1, +1), 41, Monitor()
    learn_rate = (value_range[1] - value_range[0]) / (points - 1) * 1e-1
    activation_selections, aggregation_selections = ["relu", "tanh", "sigmoid"], ["sum", "max"]
    loss_threshold, check_threshold, iteration_thresholds = 1e-5, 5, (1000, 200)

    if not exists(raw_path + "maximum-minimum.search.pkl"):
        source_motifs, target_motifs, records = [], [], {}
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

        for source_index, source_motif in enumerate(source_motifs):
            record = maximum_minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                                 source_motif=source_motif, target_motifs=target_motifs,
                                                 loss_threshold=loss_threshold, check_threshold=check_threshold,
                                                 iteration_thresholds=iteration_thresholds, verbose=True)
            # noinspection PyUnresolvedReferences
            records[str(source_index + 1).zfill(3)] = record
            monitor.output(source_index + 1, len(source_motifs))

        save_data(save_path=raw_path + "maximum-minimum.search.pkl", information=records)

    if not exists(raw_path + "maximum-minimum.propagation.pkl"):
        records, matrices = load_data(load_path=raw_path + "maximum-minimum.search.pkl"), []
        for process in linspace(1, len(records), len(records), dtype=int):
            record = records[str(process).zfill(3)]
            saved_info = (record[-1][0], record[-1][1], record[-1][-1])
            source_motif = saved_info[0]
            matrix = evaluate_propagation(value_range=value_range, points=points, motif=source_motif,
                                          compute_type="mean")
            matrices.append(matrix)
        save_data(save_path=raw_path + "maximum-minimum.propagation.npy", information=array(matrices))

    if (not exists(task_path + "max-min losses.npy")) or (not exists(task_path + "max-min params.npy")) \
            or (not exists(task_path + "rugosity indices.npy")) or (not exists(task_path + "rugosity changes.npy")):
        records = load_data(load_path=raw_path + "maximum-minimum.search.pkl")
        losses, params, rugosity_indices, rugosity_changes, total = [], [[], [], [], []], [], [[], []], len(records)
        for index in linspace(1, total, total, dtype=int):
            record = records[str(index).zfill(3)]
            saved_info = (record[-1][0], record[-1][2], record[-1][3], record[-1][4])
            losses.append(saved_info[-1])
            params[saved_info[0].i - 1].append([weight.value() for weight in saved_info[0].w])
            rugosity_indices.append(saved_info[1])
            rugosity_changes[0].append([record[0][2], record[0][3]])
            rugosity_changes[1].append([saved_info[1], saved_info[2]])
            monitor.output(index, total)
        save_data(save_path=task_path + "max-min losses.npy", information=array(losses))
        save_data(save_path=task_path + "max-min params.npy", information=array(params))
        save_data(save_path=task_path + "rugosity indices.npy", information=array(rugosity_indices))
        save_data(save_path=task_path + "rugosity changes.npy", information=array(rugosity_changes))

    # example = load(raw_path + "s007t09.pkl", allow_pickle=True)[1]
    # record_change, gradient_matrix = zeros(shape=(101, 101)), zeros(shape=(101, 101))
    # rugosity_change, loss_change = [], []
    # motif_set = {"first": (example[0][0], example[0][1]), "last": (example[-1][0], example[-1][1])}
    # for iteration, (source_motif, target_motif, records, target_loss) in enumerate(example):
    #     record_change[iteration, :len(records)] = log10(records)
    #     gradients = calculate_gradients(value_range=value_range, points=points, motif=source_motif)
    #     gradient_distribution = gaussian_kde(dataset=gradients.reshape(-1)).evaluate(linspace(0, 1.6, 101))
    #
    #     gradient_matrix[iteration + 1] = gradient_distribution / max(gradient_distribution)
    #     rugosity_change.append(calculate_rugosity(value_range=value_range, points=points, motif=source_motif))
    #     loss_change.append(target_loss)
    #
    #     monitor.output(iteration + 1, len(example), extra={"max gradient": max(gradients)})
    #
    # with open(task_path + "case.pkl", "wb") as file:
    #     dump(obj=(motif_set, record_change, gradient_matrix, array(loss_change), array(rugosity_change)), file=file)


def experiment_4(raw_path, config_path, task_path):
    agent_names, agent_count = ["baseline", "geometry", "novelty"], 100
    agent_configs = [create_agent_config(path="/hy-tmp/configs/" + "cartpole-v0." + name) for name in agent_names]
    gym_task = NEATCartPoleTask(maximum_generation=20, verbose=False)
    bounds, radios = gym_task.get_state_range(), [0.0, 0.1, 0.2, 0.3]

    if not exists(raw_path + "practice.agents.pkl"):
        agent_records = {}
        for agent_name, agent_config in zip(agent_names, agent_configs):
            for radio in radios:
                agent_records[(agent_name, str(int(radio * 100)) + "%")] = []
                task = deepcopy(gym_task)
                if radio > 0:
                    noise_generator = InputNormNoiseGenerator(norm_type=NormType.l2, noise_size=1,
                                                              minimum_bound=-bounds * radio,
                                                              maximum_bound=+bounds * radio)
                    task.set_noises(noise_generator=noise_generator)
                while len(agent_records[(agent_name, str(int(radio * 100)) + "%")]) < agent_count:
                    current = str(len(agent_records[(agent_name, str(int(radio * 100)) + "%")]) + 1)
                    print("Train " + agent_name + " in the CartPole environment: " + current + "/" + str(agent_count))
                    best_genome = obtain_best(task, agent_config, need_stdout=True)
                    if best_genome is not None:
                        agent = NEATAgent(model_genome=best_genome, neat_config=agent_config, description=agent_name)
                        experience = task.get_experience()
                        record = {"agent": agent, "experience": experience}
                        agent_records[(agent_name, str(int(radio * 100)) + "%")].append(record)
                    task.reset_experience()
                    print()
        save_data(save_path=raw_path + "practice.agents.pkl", information=agent_records)

    # if not exists(raw_path + "practice.robustness.pkl"):
    #     data, differences = load_data(path=raw_path + "practice.agents.pkl"), bound
    #     noise_generator = GradientNormNoiseGenerator(norm_type=NormType.l2, attack_step_size=20,
    #                                                  minimum_bound=-differences, maximum_bound=differences)
    #     trained_models = {}
    #     for agent_name in agent_names:
    #         for radio in radios:
    #             agents = []
    #             for agent_index in range(40):
    #                 agents.append(data[(agent_name, str(int(radio * 100)) + "%", agent_index + 1)]["agent"])
    #             trained_models[agent_name + "-" + str(int(radio * 100)) + "%"] = agents
    #
    #     results = {}
    #     for agent_name, agents in trained_models.items():
    #         for agent_index, agent in enumerate(agents):  # load an agent and collect the states in the environment.
    #             statistics = []
    #             while len(statistics) < repeats:
    #                 print("replay " + str(agent_index + 1) + "-th agent (" + agent_name + ") " +
    #                       "in environment " + task.description + " for " + str(len(statistics) + 1) + "-th time")
    #
    #                 records = task.run_1_iteration(agent=agent, total_steps=task.fit_total_steps, perturbations=None)
    #
    #                 clever = DefaultCleverScore(task=task, agent=agent, noise_generator=noise_generator,
    #                                             saved_states=records["states"], fit_type=None)
    #                 clever.launch(verbose=True)
    #
    #                 if sum(clever.scores) < 1e-16:
    #                     print("Invalid: the total amount of CLEVER scores is not compliant.")
    #                     continue
    #
    #                 if min(clever.scores) < score_bounds[0] or max(clever.scores) > score_bounds[1]:
    #                     print("Invalid: at least one CLEVER scores doesn't satisfy the range of assumptions.")
    #                     continue
    #
    #                 statistics.append(clever.scores)
    #
    #             data = -ones(shape=(len(statistics), task.fit_total_steps))
    #             for value_index, value in enumerate(statistics):
    #                 data[value_index, :len(value)] = value
    #
    #             results[(agent_name, agent_index + 1)] = data
    #
    #     save_data(save_path=raw_path + "practice.robustness.pkl", information=results)


if __name__ == "__main__":
    # experiment_1(raw_path="../data/results/raw/", task_path="../data/results/task01/")
    # experiment_2(raw_path="../data/results/raw/", task_path="../data/results/task02/")
    # experiment_3(raw_path="../data/results/raw/", task_path="../data/results/task03/")
    experiment_4(raw_path="../data/results/raw/", task_path="../data/results/task04/")
