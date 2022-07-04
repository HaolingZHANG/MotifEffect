from numpy import zeros, array, linspace, argmin, ceil, ptp, min, max, log10

from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from torch import optim, nn
from warnings import filterwarnings

from effect.operations import Monitor, prepare_data, calculate_gradients
from effect.robustness import calculate_rugosity

filterwarnings(action="ignore", category=UserWarning)


def maximum_minimum_loss_search(value_range, points, source_motif, target_motifs,
                                learn_rate, loss_threshold, check_threshold, iteration_thresholds, verbose=True):
    record, source_loss_record, iteration, monitor = [], [], 1, Monitor()
    input_signals = prepare_data(value_range=value_range, points=points)
    optimizer, criterion = optim.Adam(source_motif.parameters(), lr=learn_rate), nn.L1Loss()

    for iteration in range(iteration_thresholds[0]):
        if verbose:
            print("*" * 80)
            print("ITERATION " + str(iteration + 1))
            print("-" * 80)
            print("We train the target motifs (to approach the source motif) using the gradient descent.")

        target_loss_record = []
        for target_index in range(len(target_motifs)):
            result = minimum_loss_search(value_range=value_range, points=points, learn_rate=learn_rate,
                                         source_motif=source_motif, target_motif=target_motifs[target_index],
                                         loss_threshold=loss_threshold, check_threshold=check_threshold,
                                         iteration_threshold=iteration_thresholds[1], verbose=False)
            target_motif, target_loss = result
            target_motifs[target_index] = target_motif
            target_loss_record.append(target_loss)

            if verbose:
                monitor.output(target_index + 1, len(target_motifs))

        choice = argmin(target_loss_record)  # choose the most similar target motif.
        target_motif, target_loss = target_motifs[choice], min(target_loss_record)

        if verbose:
            print("The loss of each trained target motif is")
            print(array([list(range(1, len(target_loss_record) + 1)), target_loss_record]))
            print("We find the most similar \"" + target_motif.t + " " + str(target_motif.i) + "\" motif is")
            print(target_motif)
            print(str(choice + 1) + "-th motif in the target motifs " +
                  "with the minimum error %.6f.\n" % float(target_loss))

        source_output_signals, target_output_signals = source_motif(input_signals), target_motif(input_signals)
        source_loss = criterion(source_output_signals, target_output_signals)
        optimizer.zero_grad()
        (-source_loss).backward(retain_graph=True)
        optimizer.step()
        source_motif.restrict()

        maximum_minimum_loss = float(criterion(source_motif(input_signals), target_motif(input_signals)))
        source_loss_record.append(maximum_minimum_loss)

        source_rugosity = calculate_rugosity(value_range=value_range, points=points, motif=source_motif)
        target_rugosity = calculate_rugosity(value_range=value_range, points=points, motif=target_motif)

        if verbose:
            print("Reversely, we train the source motif (to away from the targets motifs) using the gradient ascent.")
            print(source_motif)
            print("\nNow, the maximum-minimum loss is %.6f, and the rugosity indices of the two final motifs "
                  "are %.6f and %.6f respectively.\n" % (maximum_minimum_loss, source_rugosity, target_rugosity))

        record.append((source_motif, target_motif, source_rugosity, target_rugosity, maximum_minimum_loss))

        if iteration > check_threshold and ptp(source_loss_record[-check_threshold:]) < loss_threshold:
            break

    return record


def minimum_loss_search(value_range, points, source_motif, target_motif,
                        learn_rate, loss_threshold, check_threshold, iteration_threshold, verbose=True):
    optimizer, criterion, record = optim.Adam(target_motif.parameters(), lr=learn_rate), nn.L1Loss(), []
    input_signals, monitor = prepare_data(value_range=value_range, points=points), Monitor()
    source_output_signals = source_motif(input_signals)

    for iteration in range(iteration_threshold):
        target_output_signals = target_motif(input_signals)
        loss = criterion(source_output_signals, target_output_signals)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        target_motif.restrict()
        record.append(float(loss))

        if verbose:
            monitor.output(iteration + 1, iteration_threshold, extra={"loss": record[-1]})

        if len(record) > check_threshold and ptp(record[-check_threshold:]) < loss_threshold:
            if verbose:
                monitor.output(iteration_threshold, iteration_threshold, extra={"loss": record[-1]})
            break

    return target_motif, record[-1]


# noinspection PyTypeChecker
def assemble_example(examples, value_range, points, verbose=False):
    monitor, evolution_paths, parameter_samples, counts, maximum_index, maximum_loss = Monitor(), {}, [], [], 0, 0
    if verbose:
        print("Load the example data in a sample of maximum-minimum loss search.")
    for example_index, example in enumerate(examples):
        if example[-1][-1] > maximum_loss:
            maximum_index, maximum_loss = example_index, example[-1][-1]
        counts.append(len(example))
        evolution_paths[example_index] = {"parameters": [], "locations": None}
        for motif, _, _, _ in example:
            parameters = [weight.value() for weight in motif.w] + [bias.value() for bias in motif.b]
            evolution_paths[example_index]["parameters"].append(parameters)
            parameter_samples.append(parameters)
        evolution_paths[example_index]["parameters"] = array(evolution_paths[example_index]["parameters"])
        if verbose:
            monitor.output(example_index + 1, len(examples))

    method = TSNE(n_components=2, random_state=2022, method="barnes_hut")
    locations, count = method.fit_transform(array(parameter_samples)), 0
    for count_index in range(len(counts)):
        evolution_paths[count_index]["locations"] = locations[count: count + counts[count_index]]
        count += counts[count_index]

    example, gradient_set, gradient_matrix, maximum_gradient = examples[maximum_index], [], zeros(shape=(101, 101)), 0.0
    record_change, rugosity_change, loss_change = zeros(shape=(101, 101)), [], []
    motif_set = {"first": (example[0][0], example[0][1]), "last": (example[-1][0], example[-1][1])}
    if verbose:
        print("Calculate the gradient adjustment during the evolution process.")
    for iteration, (source_motif, target_motif, records, target_loss) in enumerate(example):
        record_change[iteration, :len(records)] = log10(records)
        gradients = calculate_gradients(value_range=value_range, points=points, motif=source_motif).reshape(-1)
        maximum_gradient = max([max(gradients), maximum_gradient])
        gradient_set.append(gradients)
        rugosity_change.append(calculate_rugosity(value_range=value_range, points=points, motif=source_motif))
        loss_change.append(target_loss)
        if verbose:
            monitor.output(iteration + 1, len(example), extra={"maximum gradient": maximum_gradient})

    for iteration, gradients in enumerate(gradient_set):
        gradient_distribution = gaussian_kde(dataset=gradients).evaluate(linspace(0, ceil(maximum_gradient), 101))
        gradient_matrix[iteration + 1] = gradient_distribution / max(gradient_distribution)

    loss_change, rugosity_change = array(loss_change), array(rugosity_change)

    return evolution_paths, motif_set, record_change, gradient_matrix, loss_change, rugosity_change
