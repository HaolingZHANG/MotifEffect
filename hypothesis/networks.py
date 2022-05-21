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


class RestrictedWeight(Module):

    def __init__(self, is_positive=True, value=None, bound=None):
        super(RestrictedWeight, self).__init__()

        if bound is None:
            bound = (1e-3, 1e0)

        self.weight, self.is_positive, self.bound = None, is_positive, bound
        self.reset(value=value)

    def forward(self, data):
        self.restrict()

        return self.weight * data

    def value(self):
        return float(self.weight)

    def restrict(self):
        value = None
        if self.is_positive:
            if self.weight < +self.bound[0]:
                value = tensor(+self.bound[0])
            elif self.weight > +self.bound[1]:
                value = tensor(+self.bound[1])
        else:
            if self.weight > -self.bound[0]:
                value = tensor(-self.bound[0])
            elif self.weight < -self.bound[1]:
                value = tensor(-self.bound[1])

        if value is not None:
            self.weight = Parameter(data=value, requires_grad=True)

    def reset(self, value=None):
        if value is not None:
            if self.is_positive and self.bound[0] <= value <= self.bound[1]:
                value = tensor(value)
            elif not self.is_positive and -self.bound[1] <= value <= -self.bound[0]:
                value = tensor(self.value)
            else:
                raise ValueError("the inputted value is wrong, it needs to meet the established constraints!")

        elif self.is_positive:
            value = +self.bound[0] + (self.bound[1] - self.bound[0]) * rand(1)

        else:
            value = -self.bound[1] + (self.bound[1] - self.bound[0]) * rand(1)

        self.weight = Parameter(data=value, requires_grad=True)


class RestrictedBias(Module):

    def __init__(self, value=None, bound=None):
        super(RestrictedBias, self).__init__()
        if bound is None:
            bound = (-1e0, +1e0)

        self.bias, self.bound = None, bound
        self.reset(value=value)

    def forward(self, data):
        self.restrict()

        return self.bias + data

    def restrict(self):
        value = None
        if self.bias < self.bound[0]:
            value = tensor(self.bound[0])
        elif self.bias > self.bound[1]:
            value = tensor(self.bound[1])

        if value is not None:
            self.bias = Parameter(data=value, requires_grad=True)

    def value(self):
        return float(self.bias)

    def reset(self, value=None):
        if value is not None:
            value = tensor(value)
        else:
            value = self.bound[0] + (self.bound[1] - self.bound[0]) * rand(1)

        self.bias = Parameter(data=value, requires_grad=True)


class NeuralMotif(Module):

    def __init__(self, motif_type, motif_index, activations, aggregations, weights=None, biases=None,
                 weight_bound=None, bias_bound=None):
        super(NeuralMotif, self).__init__()

        if aggregations is None:
            aggregations = []

        if motif_index not in [1, 2, 3, 4]:
            raise ValueError("index of motif needs to belong to [1, 2, 3, 4], got " + str(motif_index) + ".")

        if motif_type == "collider":
            request_a, request_g = 1, 1
        elif motif_type == "fork":
            request_a, request_g = 2, 0
        elif motif_type == "chain":
            request_a, request_g = 2, 0
        elif motif_type in ["coherent-loop", "incoherent-loop"]:
            request_a, request_g = 2, 2
        else:
            raise ValueError("no such motif type, expect one in "
                             "[\"collider\", \"fork\", \"chain\", \"coherent-loop\", \"incoherent-loop\"].")

        if len(activations) != request_a:
            raise ValueError("wrong number of activation functions, "
                             "expect " + str(request_a) + ", got " + str(len(activations)))
        if len(aggregations) != request_g:
            raise ValueError("wrong number of aggregation functions, "
                             "expect " + str(request_g) + ", got " + str(len(aggregations)))

        for activation in activations:
            if activation not in ["relu", "tanh", "sigmoid"]:
                raise ValueError("no such activation type, expect one in "
                                 "[\"relu\", \"tanh\", \"sigmoid\"].")
        for aggregation in aggregations:
            if aggregation not in ["sum", "avg", "max"]:
                raise ValueError("no such aggregation type, expect one in "
                                 "[\"sum\", \"avg\", \"max\"].")

        self.t, self.i, self.a, self.g, self.w, self.b = motif_type, motif_index, activations, aggregations, [], []
        self.weight_bound, self.bias_bound = weight_bound, bias_bound
        self.reset(weights, biases)

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

    def reset(self, weights=None, biases=None):
        if self.t == "collider":
            weight_flags, bias_size = [self.i <= 2, self.i in [1, 3]], 1
        elif self.t in ["fork", "chain"]:
            weight_flags, bias_size = [self.i <= 2, self.i in [1, 3]], 2
        elif self.t == "coherent-loop":
            weight_flags, bias_size = [self.i in [1, 4], self.i <= 2, self.i in [1, 3]], 2
        else:  # self.t == "incoherent-loop"
            weight_flags, bias_size = [self.i in [2, 3], self.i <= 2, self.i in [1, 3]], 2

        if weights is not None:
            if len(weights) != len(weight_flags):
                raise ValueError("the number of weights should be "
                                 + str(len(weight_flags)) + " got " + str(len(weights)) + ".")
            self.w = ModuleList([RestrictedWeight(flag, value, bound=self.weight_bound)
                                 for flag, value in zip(weight_flags, weights)])
        else:
            self.w = ModuleList([RestrictedWeight(flag, bound=self.weight_bound)
                                 for flag in weight_flags])

        if biases is not None:
            if len(biases) != bias_size:
                raise ValueError("the number of weights should be "
                                 + str(bias_size) + " got " + str(len(biases)) + ".")
            self.b = ModuleList([RestrictedBias(bias_value, bound=self.bias_bound)
                                 for bias_value in biases])
        else:
            self.b = ModuleList([RestrictedBias(bound=self.bias_bound)
                                 for _ in range(bias_size)])

    def __str__(self):
        ws = [("+" if weight.value() >= 0 else "") + "%.2e" % weight.value() for weight in self.w]
        bs = [("+" if bias.value() >= 0 else "") + "%.2e" % bias.value() for bias in self.b]
        info = "<NeuralMotif" + "\n"
        if self.t == "collider":
            info += "\t" + "motif type   |  " + self.t + " " + str((self.i - 1) // 2 + (self.i - 1) % 2 + 1) + "\n"
            info += "\t" + "activation   |  (1),(2) >> " + self.a[0].rjust(9) + " >> (3)" + "\n"
            info += "\t" + "aggregation  |  (1),(2) >> " + self.g[0].rjust(9) + " >> (3)" + "\n"
            info += "\t" + "weight       |      (1) >> " + ws[0] + " >> (3)" + "\n"
            info += "\t" + "weight       |      (2) >> " + ws[1] + " >> (3)" + "\n"
            info += "\t" + "bias         |  (1),(2) >> " + bs[0] + " >> (3)" + ">"
        elif self.t == "fork":
            info += "\t" + "motif type   |  " + self.t + " " + str((self.i - 1) // 2 + (self.i - 1) % 2 + 1) + "\n"
            info += "\t" + "activation   |      (1) >> " + self.a[0].rjust(8) + " >> (2)" + "\n"
            info += "\t" + "activation   |      (1) >> " + self.a[1].rjust(8) + " >> (3)" + "\n"
            info += "\t" + "weight       |      (1) >> " + ws[0] + " >> (2)" + "\n"
            info += "\t" + "weight       |      (1) >> " + ws[1] + " >> (3)" + "\n"
            info += "\t" + "bias         |      (1) >> " + bs[0] + " >> (2)" + "\n"
            info += "\t" + "bias         |      (1) >> " + bs[1] + " >> (3)" + ">"
        elif self.t == "chain":
            info += "\t" + "motif type   |  " + self.t + " " + str(self.i) + "\n"
            info += "\t" + "activation   |      (1) >> " + self.a[0].rjust(9) + " >> (2)" + "\n"
            info += "\t" + "activation   |      (2) >> " + self.a[1].rjust(9) + " >> (3)" + "\n"
            info += "\t" + "weight       |      (1) >> " + ws[0] + " >> (2)" + "\n"
            info += "\t" + "weight       |      (2) >> " + ws[1] + " >> (3)" + "\n"
            info += "\t" + "bias         |      (1) >> " + bs[0] + " >> (2)" + "\n"
            info += "\t" + "bias         |      (2) >> " + bs[1] + " >> (3)" + ">"
        else:
            info += "\t" + "motif type   |  " + self.t + " " + str(self.i) + "\n"
            info += "\t" + "activation   |      (1) >> " + self.a[0].rjust(9) + " >> (2)" + "\n"
            info += "\t" + "activation   |  (1),(2) >> " + self.a[1].rjust(9) + " >> (3)" + "\n"
            info += "\t" + "aggregation  |      (1) >> " + self.g[0].rjust(9) + " >> (2)" + "\n"
            info += "\t" + "aggregation  |  (1),(2) >> " + self.g[1].rjust(9) + " >> (3)" + "\n"
            info += "\t" + "weight       |      (1) >> " + ws[0] + " >> (2)" + "\n"
            info += "\t" + "weight       |      (1) >> " + ws[1] + " >> (3)" + "\n"
            info += "\t" + "weight       |      (2) >> " + ws[2] + " >> (3)" + "\n"
            info += "\t" + "bias         |      (1) >> " + bs[0] + " >> (2)" + "\n"
            info += "\t" + "bias         |  (1),(2) >> " + bs[1] + " >> (3)" + ">"
        return info


class NeuralNetwork(Module):

    def __init__(self, connections, motif_set):
        super(NeuralNetwork, self).__init__()
