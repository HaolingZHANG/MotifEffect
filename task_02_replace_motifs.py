from hypothesis.similarity import maximum_minimum_search


if __name__ == "__main__":
    save_path = "./results/data/mm_replace/"
    maximum_minimum_search(value_range=(-1, +1), times=101, seed=2022, sample=10, epochs=20, save_path=save_path,
                           motif_types={"s": "incoherent-loop", "t": "collider"},
                           motif_indices={"s": [1, 2, 3, 4], "t": [1, 2, 3, 4]},
                           activations=["relu", "tanh", "sigmoid"], aggregations=["sum", "avg", "max"],
                           repeats={"s": (2, 2), "t": (1, 1)})
