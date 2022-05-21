from datetime import datetime
from itertools import product
from torch import cat, linspace, meshgrid, unsqueeze

from hypothesis.networks import NeuralMotif


class Monitor(object):

    def __init__(self):
        self.last_time = None

    def output(self, current_state, total_state, extra=None):
        if self.last_time is None:
            self.last_time = datetime.now()

        if current_state == 0:
            return

        position = int(current_state / total_state * 100)

        string = "|"

        for index in range(0, 100, 5):
            if position >= index:
                string += "█"
            else:
                string += " "

        string += "|"

        pass_time = (datetime.now() - self.last_time).total_seconds()
        wait_time = int(pass_time * (total_state - current_state) / current_state)

        string += " " * (3 - len(str(position))) + str(position) + "% ("

        string += " " * (len(str(total_state)) - len(str(current_state))) + str(current_state) + "/" + str(total_state)

        if current_state < total_state:
            minute, second = divmod(wait_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") wait " + "%04d:%02d:%02d" % (hour, minute, second)
        else:
            minute, second = divmod(pass_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") used " + "%04d:%02d:%02d" % (hour, minute, second)

        if extra is not None:
            string += " " + str(extra).replace("\'", "").replace("{", "(").replace("}", ")") + "."
        else:
            string += "."

        print("\r" + string, end="", flush=True)

        if current_state >= total_state:
            self.last_time = None
            print()


def prepare_data(value_range, points=101):
    x, y = linspace(value_range[0], value_range[1], points), linspace(value_range[0], value_range[1], points)
    x, y = meshgrid((x, y), indexing="ij")
    x, y = unsqueeze(x.reshape(-1), dim=1), unsqueeze(y.reshape(-1), dim=1)

    data = cat((x, y), dim=1)

    return data


def prepare_motifs(motif_type, motif_index, activations, aggregations, sample, weights=None, biases=None):
    motifs = []

    for sample_index in range(sample):
        if weights is not None and biases is not None:
            motif = None
            for flags in product([1.0, -1.0], repeat=len(weights)):
                try:
                    adjusted_weights = [flag * weight for flag, weight in zip(flags, weights)]
                    motif = NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                        activations=activations, aggregations=aggregations,
                                        weights=adjusted_weights, biases=biases)
                    break
                except ValueError:
                    pass

        else:
            motif = NeuralMotif(motif_type=motif_type, motif_index=motif_index,
                                activations=activations, aggregations=aggregations,
                                weights=weights, biases=biases)

        motifs.append(motif)

    return motifs
