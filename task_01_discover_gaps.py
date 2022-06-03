from itertools import product

from hypothesis import prepare_motifs, calculate_similarity


if __name__ == "__main__":
    seed, samples, value_range, points, loss_threshold, check_threshold = 2022, 10, (-1, +1), 41, 1e-6, 5
    parent_path = "./results/data/task01/"

    target_group = []
    for motif_index in [1, 2, 3, 4]:
        for activations in product(["relu", "tanh", "sigmoid"], repeat=1):
            for aggregations in product(["sum", "avg", "max"], repeat=1):
                target_motifs = prepare_motifs(motif_type="collider", motif_index=motif_index,
                                               activations=activations, aggregations=aggregations,
                                               sample=samples, weights=None, biases=None)
                target_group.append(target_motifs)

    source_group = []
    for motif_index in [1, 2, 3, 4]:
        for activations in product(["relu", "tanh", "sigmoid"], repeat=2):
            for aggregations in product(["sum", "avg", "max"], repeat=2):
                source_motifs = prepare_motifs(motif_type="incoherent-loop", motif_index=motif_index,
                                               activations=activations, aggregations=aggregations,
                                               sample=samples, weights=None, biases=None)
                source_group.append(source_motifs)

    calculate_similarity(value_range=value_range, points=points,
                         source_motif_group=source_group, target_motif_group=target_group,
                         loss_threshold=loss_threshold, check_threshold=check_threshold, iteration_threshold=None,
                         save_path=parent_path, seed=seed, processes=1)
