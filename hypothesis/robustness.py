# from numpy import std, inf
from os import listdir, path
from pickle import load

from matplotlib import pyplot
from numpy import sum, sqrt, zeros, arange
from torch import squeeze

from hypothesis.operations import prepare_data
from hypothesis import Monitor


def intervene_equivalents(value_range, times, load_parent_path, save_parent_path):
    considered_data_1 = prepare_data(value_range=value_range, points=times)
    considered_data_2 = prepare_data(value_range=value_range, points=times)
    for sub_path in listdir(load_parent_path):
        if path.exists(save_parent_path + sub_path):
            continue

        with open(load_parent_path + sub_path, "rb") as file:
            motif_data = load(file)

        print("calculate " + sub_path)
        minimum_loss, motif_1, motif_2 = None, None, None
        for source_motif, target_motif, loss in motif_data:
            if minimum_loss is None or loss < minimum_loss:
                minimum_loss = loss
                motif_1, motif_2 = source_motif, target_motif

        print(minimum_loss)
        considered_data_1.requires_grad, considered_data_2.requires_grad = True, True
        sources = [considered_data_1, considered_data_2]
        targets = motif_1(sources[0]), motif_2(sources[1])
        pyplot.figure(figsize=(10, 5), tight_layout=True)
        pyplot.subplot(1, 2, 1)
        pyplot.pcolormesh(arange(times), arange(times), targets[0].reshape(times, times).detach().numpy(),
                          cmap="rainbow")
        pyplot.subplot(1, 2, 2)
        pyplot.pcolormesh(arange(times), arange(times), targets[1].reshape(times, times).detach().numpy(),
                          cmap="rainbow")
        pyplot.show()
        pyplot.close()


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
