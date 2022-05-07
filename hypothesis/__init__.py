from networkx import DiGraph
from torch import tensor, unsqueeze, sum, max, mean, relu, tanh, sigmoid, cat, rand
from torch.nn import Module, ModuleList, Parameter


acyclic_motifs = {
    "collider": [DiGraph([(1, 3, {"weight": +1}), (2, 3, {"weight": +1})]),
                 DiGraph([(1, 3, {"weight": +1}), (2, 3, {"weight": -1})]),
                 DiGraph([(1, 3, {"weight": -1}), (2, 3, {"weight": -1})])],
    "fork": [DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": +1})]),
             DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": -1})]),
             DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": -1})])],
    "chain": [DiGraph([(1, 2, {"weight": +1}), (2, 3, {"weight": +1})]),
              DiGraph([(1, 2, {"weight": +1}), (2, 3, {"weight": -1})]),
              DiGraph([(1, 2, {"weight": -1}), (2, 3, {"weight": +1})]),
              DiGraph([(1, 2, {"weight": -1}), (2, 3, {"weight": -1})])],
    "coherent-loop": [DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": +1}), (2, 3, {"weight": +1})]),
                      DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": +1}), (2, 3, {"weight": -1})]),
                      DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": -1}), (2, 3, {"weight": +1})]),
                      DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": -1}), (2, 3, {"weight": -1})])],
    "incoherent-loop": [DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": +1}), (2, 3, {"weight": +1})]),
                        DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": +1}), (2, 3, {"weight": -1})]),
                        DiGraph([(1, 2, {"weight": +1}), (1, 3, {"weight": -1}), (2, 3, {"weight": +1})]),
                        DiGraph([(1, 2, {"weight": -1}), (1, 3, {"weight": -1}), (2, 3, {"weight": -1})])]
}


class NeuralMotif(Module):

    def __init__(self, motif_type, motif_index, activations, aggregations, weights=None, biases=None):
        super(NeuralMotif, self).__init__()

        if aggregations is None:
            aggregations = []

        assert motif_index in [1, 2, 3, 4]
        if motif_type == "collider":
            assert len(activations) == 1 and len(aggregations) == 1
        elif motif_type == "fork":
            assert len(activations) == 2 and len(aggregations) == 0
        elif motif_type == "chain":
            assert len(activations) == 2 and len(aggregations) == 0
        elif motif_type in ["coherent-loop", "incoherent-loop"]:
            assert len(activations) == 2 and len(aggregations) == 2
        else:
            assert motif_type in ["collider", "fork", "chain", "coherent-loop", "incoherent-loop"]

        for activation in activations:
            assert activation in ["relu", "tanh", "sigmoid"]
        for aggregation in aggregations:
            assert aggregation in ["sum", "avg", "max"]

        self.t, self.i, self.a, self.g, self.w, self.b = motif_type, motif_index, activations, aggregations, [], []

        if self.t == "collider":
            w_flags, b_size = [self.i <= 2, self.i in [1, 3]], 1
        elif self.t in ["fork", "chain"]:
            w_flags, b_size = [self.i <= 2, self.i in [1, 3]], 2
        elif self.t == "coherent-loop":
            w_flags, b_size = [self.i in [1, 4], self.i <= 2, self.i in [1, 3]], 2
        else:  # self.t == "incoherent-loop"
            w_flags, b_size = [self.i in [2, 3], self.i <= 2, self.i in [1, 3]], 2
        
        if weights is not None:
            assert len(weights) == len(w_flags)
            for weight_value, weight_flag in zip(weights, w_flags):
                assert weight_flag * weight_value > 0
            self.w = ModuleList([NeuralMotif.RestrictedWeight(flag, value) for flag, value in zip(w_flags, weights)])
        else:
            self.w = ModuleList([NeuralMotif.RestrictedWeight(flag) for flag in w_flags])
            
        if biases is not None:
            assert len(biases) == b_size
            self.b = ModuleList([NeuralMotif.Bias(bias_value) for bias_value in biases])
        else:
            self.b = ModuleList([NeuralMotif.Bias() for _ in range(b_size)])

    def forward(self, values):
        if self.t in "collider":
            assert values.size()[1] == 2
            return self.activate(self.add_bias(self.aggregate(self.add_weight(values, [0, 1]), 0), 0), 0)
        elif self.t == "fork":
            assert values.size()[1] == 1
            values = self.add_weight(values, [0, 1])
            return cat(tensors=(self.activate(self.add_bias(unsqueeze(values[:, 0], dim=1), 0), 0),
                                self.activate(self.add_bias(unsqueeze(values[:, 1], dim=1), 1), 1)),
                       dim=1)
        elif self.t == "chain":
            assert values.size()[1] == 1
            values = self.activate(self.add_bias(self.add_weight(values, [0]), 0), 0)
            return self.activate(self.add_bias(self.add_weight(values, [1]), 1), 1)
        else:  # self.t in ["coherent-loop", "incoherent-loop"]:
            assert values.size()[1] == 2
            values_for_2 = cat(tensors=(self.add_weight(unsqueeze(values[:, 0], dim=1), [0]),
                                        unsqueeze(values[:, 1], dim=1)),
                               dim=1)
            values_for_3 = cat(tensors=(unsqueeze(values[:, 0], dim=1),
                                        self.activate(self.add_bias(self.aggregate(values_for_2, 0), 0), 0)),
                               dim=1)
            return self.activate(self.add_bias(self.aggregate(self.add_weight(values_for_3, [1, 2]), 1), 1), 1)

    def activate(self, values, activate_index):
        if self.a[activate_index] == "relu":
            return relu(values)
        elif self.a[activate_index] == "tanh":
            return tanh(values)
        else:
            return sigmoid(values)

    def aggregate(self, values, aggregate_index):
        if self.g[aggregate_index] == "sum":
            return unsqueeze(sum(values, dim=1), dim=1)
        elif self.g[aggregate_index] == "max":
            return unsqueeze(max(values, dim=1)[0], dim=1)
        else:
            return unsqueeze(mean(values, dim=1), dim=1)

    # noinspection PyUnresolvedReferences
    def add_weight(self, values, weight_indices):
        if len(weight_indices) == 1 and values.size()[1] == 1:
            return self.w[weight_indices[0]](values)
        if len(weight_indices) == 2 and values.size()[1] == 2:
            return cat(tuple([self.w[weight_indices[index]](unsqueeze(values[:, index], dim=1))
                              for index in range(len(weight_indices))]), dim=1)
        if len(weight_indices) == 2 and values.size()[1] == 1:
            return cat(tuple([self.w[weight_indices[index]](values)
                              for index in range(len(weight_indices))]), dim=1)

    # noinspection PyUnresolvedReferences
    def add_bias(self, values, bias_index):
        return self.b[bias_index](values)

    # noinspection PyTypeChecker
    def __str__(self):
        ws = [("+" if weight.get_info() >= 0 else "") + "%.1e" % weight.get_info() for weight in self.w]
        bs = [("+" if bias.get_info() >= 0 else "") + "%.1e" % bias.get_info() for bias in self.b]
        if self.t == "collider":
            info = "<NeuralMotif: " + self.t + " with type " + str((self.i - 1) // 2 + (self.i - 1) % 2 + 1) + "\n"
            info += "\t" + "activation  | (1),(2) >> " + self.a[0].rjust(8) + " >>     (3)" + "\n"
            info += "\t" + "aggregation | (1),(2) >> " + self.g[0].rjust(8) + " >>     (3)" + "\n"
            info += "\t" + "weight      |     (1) >> " + ws[0] + " >>     (3)" + "\n"
            info += "\t" + "weight      |     (2) >> " + ws[1] + " >>     (3)" + "\n"
            info += "\t" + "bias        | (1),(2) >> " + bs[0] + " >>     (3)" + ">"
        elif self.t == "fork":
            info = "<NeuralMotif: " + self.t + " with type " + str((self.i - 1) // 2 + (self.i - 1) % 2 + 1) + "\n"
            info += "\t" + "activation  |     (1) >> " + self.a[0].rjust(8) + " >>     (2)" + "\n"
            info += "\t" + "activation  |     (1) >> " + self.a[1].rjust(8) + " >>     (3)" + "\n"
            info += "\t" + "weight      |     (1) >> " + ws[0] + " >>     (2)" + "\n"
            info += "\t" + "weight      |     (1) >> " + ws[1] + " >>     (3)" + "\n"
            info += "\t" + "bias        |     (1) >> " + bs[0] + " >>     (2)" + "\n"
            info += "\t" + "bias        |     (1) >> " + bs[1] + " >>     (3)" + ">"
        elif self.t == "chain":
            info = "<NeuralMotif: " + self.t + " with type " + str(self.i) + "\n"
            info += "\t" + "activation  |     (1) >> " + self.a[0].rjust(8) + " >>     (2)" + "\n"
            info += "\t" + "activation  |     (2) >> " + self.a[0].rjust(8) + " >>     (3)" + "\n"
            info += "\t" + "weight      |     (1) >> " + ws[0] + " >>     (2)" + "\n"
            info += "\t" + "weight      |     (2) >> " + ws[1] + " >>     (3)" + "\n"
            info += "\t" + "bias        |     (1) >> " + bs[0] + " >>     (2)" + "\n"
            info += "\t" + "bias        |     (2) >> " + bs[1] + " >>     (3)" + ">"
        else:
            info = "<NeuralMotif: " + self.t + " with type " + str(self.i) + "\n"
            info += "\t" + "activation  |     (1) >> " + self.a[0].rjust(8) + " >>     (2)" + "\n"
            info += "\t" + "activation  | (1),(2) >> " + self.a[0].rjust(8) + " >>     (3)" + "\n"
            info += "\t" + "activation  |     (1) >> " + self.g[0].rjust(8) + " >>     (2)" + "\n"
            info += "\t" + "activation  | (1),(2) >> " + self.g[0].rjust(8) + " >>     (3)" + "\n"
            info += "\t" + "weight      |     (1) >> " + ws[0] + " >>     (2)" + "\n"
            info += "\t" + "weight      |     (1) >> " + ws[1] + " >>     (3)" + "\n"
            info += "\t" + "weight      |     (2) >> " + ws[2] + " >>     (3)" + "\n"
            info += "\t" + "bias        |     (1) >> " + bs[0] + " >>     (2)" + "\n"
            info += "\t" + "bias        | (1),(2) >> " + bs[1] + " >>     (3)" + ">"
        return info

    class RestrictedWeight(Module):

        def __init__(self, is_positive=True, value=None, threshold=1e-3):
            super(NeuralMotif.RestrictedWeight, self).__init__()
            assert threshold < 1.0
            if value is not None:
                assert not (value > 0) ^ is_positive
                self.weight = Parameter(data=tensor(value), requires_grad=True)
            else:
                if is_positive:
                    self.weight = Parameter(data=(threshold - 1.0) * rand(1) + 1.0, requires_grad=True)
                else:
                    self.weight = Parameter(data=(-1.0 + threshold) * rand(1) - threshold, requires_grad=True)

            self.is_positive, self.threshold = is_positive, threshold

        def forward(self, data):
            if self.is_positive and self.weight < +self.threshold:
                self.weight = Parameter(data=tensor(+self.threshold), requires_grad=True)
            elif not self.is_positive and self.weight > -self.threshold:
                self.weight = Parameter(data=tensor(-self.threshold), requires_grad=True)

            return self.weight * data

        def get_info(self):
            return float(self.weight)

    class Bias(Module):

        def __init__(self, value=None):
            super(NeuralMotif.Bias, self).__init__()
            if value is not None:
                self.bias = Parameter(data=tensor(value), requires_grad=True)
            else:
                self.bias = Parameter(data=-2.0 * rand(1) + 1.0, requires_grad=True)

        def forward(self, data):
            return self.bias + data

        def get_info(self):
            return float(self.bias)
