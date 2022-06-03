from numpy import sum, zeros, abs, max, sqrt
from torch import squeeze

from hypothesis.operations import prepare_data
from hypothesis import Monitor


def intervene_entrances(value_range, times, scales, motif, verbose=False):
    data, value_size = prepare_data(value_range=value_range, points=times), value_range[1] - value_range[0]
    clean_output = motif(data).reshape(times, times).detach().numpy()
    differences = zeros(shape=(len(scales), times, times))
    for scale_index, scale in enumerate(scales):
        expanded_range = (value_range[0] - scale * value_size, value_range[1] + scale * value_size)
        scale_size = int((times - 1) * scale)
        expanded_times = times + 2 * scale_size
        expanded_data = prepare_data(value_range=expanded_range, points=expanded_times)
        expanded_output = motif(expanded_data).reshape(expanded_times, expanded_times).detach().numpy()
        intervened_output, count, monitor = zeros(shape=((2 * scale_size + 1) ** 2, times, times)), 0, Monitor()
        for index_1 in range(2 * scale_size + 1):
            for index_2 in range(2 * scale_size + 1):
                values = abs(expanded_output[index_1: index_1 + times, index_2: index_2 + times] - clean_output)
                intervened_output[count] = values

                if verbose:
                    monitor.output(count + 1, (2 * scale_size + 1) ** 2)

                count += 1

        differences[scale_index] = max(intervened_output, axis=0)

    return differences[0]


def calculate_gradients(value_range, points, motif, verbose=False):
    sources = prepare_data(value_range=value_range, points=points)
    sources.requires_grad = True
    targets = motif(sources)
    gradients, monitor = zeros(shape=(points ** 2,)), Monitor()

    if verbose:
        print("estimate the gradient in the current resolution of the landscape.")

    for index in range(points ** 2):
        # noinspection PyArgumentList
        squeeze(targets[index]).backward(retain_graph=True)
        values = sources.grad[index].detach().numpy()
        gradients[index] = sqrt(sum(values ** 2))
        if verbose:
            monitor.output(index + 1, points ** 2, extra={"source": gradients[index]})

    return gradients.reshape(points, points)


def calculate_clever_score():
    pass
