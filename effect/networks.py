from torch import tensor, unsqueeze, sum, min, max, mean, relu, tanh, sigmoid, cat, rand
from torch.nn import Module, ModuleList, Parameter


class RestrictedWeight(Module):

    def __init__(self, is_positive=True, value=None, bound=None):
        """
        Initialize the restricted weight module.

        :param is_positive: is positive weight.
        :type is_positive: bool

        :param value: established value.
        :type value: float

        :param bound: bound of the weight.
        :type bound: tuple
        """
        super(RestrictedWeight, self).__init__()

        if bound is None:
            bound = (1e-2, 1e0)

        self.weight, self.is_positive, self.bound = None, is_positive, bound
        self.reset(value=value)

    def forward(self, data):
        """
        Forward propagate through the restricted weight * data.

        :param data: input data.
        :type data: torch.Tensor

        :return: forwarded data.
        :rtype: torch.Tensor
        """
        self.restrict()
        return self.weight * data

    def value(self):
        """
        Obtain the weight value.

        :return: weight value.
        :rtype float
        """
        return float(self.weight)

    def restrict(self):
        """
        Restrict the weight before the forward propagation.
        """
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
        """
        Reset the weight value.

        :param value: established weight value.
        :type value: float or None.
        """
        if value is not None:
            if self.is_positive and self.bound[0] <= value <= self.bound[1]:
                value = tensor(value)
            elif not self.is_positive and -self.bound[1] <= value <= -self.bound[0]:
                value = tensor(value)
            else:
                raise ValueError("the inputted value is wrong, it needs to meet the established constraints!")

        elif self.is_positive:
            value = +self.bound[0] + (self.bound[1] - self.bound[0]) * rand(1)

        else:
            value = -self.bound[1] + (self.bound[1] - self.bound[0]) * rand(1)

        self.weight = Parameter(data=value, requires_grad=True)


class RestrictedBias(Module):

    def __init__(self, value=None, bound=None):
        """
        Initialize the restricted bias module.

        :param value: established value.
        :type value: float

        :param bound: bound of the weight.
        :type bound: tuple
        """
        super(RestrictedBias, self).__init__()
        if bound is None:
            bound = (-1e0, +1e0)

        self.bias, self.bound = None, bound
        self.reset(value=value)

    def forward(self, data):
        """
        Forward propagate through the restricted bias + data.

        :param data: input data.
        :type data: torch.Tensor

        :return: forwarded data.
        :rtype: torch.Tensor
        """
        self.restrict()

        return self.bias + data

    def value(self):
        """
        Obtain the bias value.

        :return: bias value.
        :rtype float
        """
        return float(self.bias)

    def restrict(self):
        """
        Restrict the bias before the forward propagation.
        """
        value = None
        if self.bias < self.bound[0]:
            value = tensor(self.bound[0])
        elif self.bias > self.bound[1]:
            value = tensor(self.bound[1])

        if value is not None:
            self.bias = Parameter(data=value, requires_grad=True)

    def reset(self, value=None):
        """
        Reset the bias value.

        :param value: established bias value.
        :type value: float or None.
        """
        if value is not None:
            value = tensor(value)
        else:
            value = self.bound[0] + (self.bound[1] - self.bound[0]) * rand(1)

        self.bias = Parameter(data=value, requires_grad=True)


class NeuralMotif(Module):

    def __init__(self, motif_type, motif_index, activations, aggregations, weights=None, biases=None,
                 weight_bound=None, bias_bound=None):
        """
        Initialize the neural network motif.

        :param motif_type: motif type ("collider", "fork", "chain", "coherent-loop", or "incoherent-loop").
        :type motif_type: str

        :param motif_index: index of network motif (1 ~ 4).
        :type motif_index: int

        :param activations: activation function list.
        :type activations: tuple or list

        :param aggregations: aggregation function list.
        :type aggregations: tuple or list

        :param weights: established weights.
        :type weights: tuple, list, or None

        :param biases: established biases.
        :type biases: tuple, list, or None

        :param weight_bound: bound of weight.
        :type weight_bound: tuple, list, or None

        :param bias_bound: bound of bias.
        :type bias_bound: tuple, list, or None
        """
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

        if weight_bound is None:
            weight_bound = (1e-3, 1e0)
        if bias_bound is None:
            bias_bound = (-1e0, +1e0)

        if len(activations) != request_a:
            raise ValueError("wrong number of activation functions, "
                             "expect " + str(request_a) + ", got " + str(len(activations)))
        if len(aggregations) != request_g:
            raise ValueError("wrong number of aggregation functions, "
                             "expect " + str(request_g) + ", got " + str(len(aggregations)))

        for activation in activations:
            if activation not in ["tanh", "sigmoid", "relu"]:
                raise ValueError("no such activation type, expect one in [\"tanh\", \"sigmoid\", \"relu\"].")
        for aggregation in aggregations:
            if aggregation not in ["sum", "max"]:
                raise ValueError("no such aggregation type, expect one in [\"sum\", \"max\"].")

        self.t, self.i, self.a, self.g, self.w, self.b = motif_type, motif_index, activations, aggregations, [], []
        self.weight_bound, self.bias_bound = weight_bound, bias_bound
        self.reset(weights, biases)

    def forward(self, input_signals):
        """
        Forward propagate through the neural network motif.

        :param input_signals: input signals.
        :type input_signals: torch.Tensor

        :return: output normalized signals.
        :rtype: torch.Tensor
        """
        if self.t in "collider":
            assert input_signals.size()[1] == 2
            output_signals = self.activate(self.add_bias(self.aggregate(self.add_weight(input_signals, [0, 1]),
                                                                        0), 0), 0)
        elif self.t == "fork":
            assert input_signals.size()[1] == 1
            input_signals = self.add_weight(input_signals, [0, 1])
            output_signals = cat(tensors=(self.activate(self.add_bias(unsqueeze(input_signals[:, 0], dim=1), 0), 0),
                                          self.activate(self.add_bias(unsqueeze(input_signals[:, 1], dim=1), 1), 1)),
                                 dim=1)
        elif self.t == "chain":
            assert input_signals.size()[1] == 1
            signals = self.activate(self.add_bias(self.add_weight(input_signals, [0]), 0), 0)
            output_signals = self.activate(self.add_bias(self.add_weight(signals, [1]), 1), 1)
        else:  # self.t in ["coherent-loop", "incoherent-loop"]:
            assert input_signals.size()[1] == 2
            v_for_2 = cat(tensors=(self.add_weight(unsqueeze(input_signals[:, 0], dim=1), [0]),
                                   unsqueeze(input_signals[:, 1], dim=1)),
                          dim=1)
            v_for_3 = cat(tensors=(unsqueeze(input_signals[:, 0], dim=1),
                                   self.activate(self.add_bias(self.aggregate(v_for_2, 0), 0), 0)),
                          dim=1)
            output_signals = self.activate(self.add_bias(self.aggregate(self.add_weight(v_for_3, [1, 2]), 1), 1), 1)

        if self.t != "fork":
            if max(output_signals) - min(output_signals) < 1e-12:
                output_signals = output_signals - max(output_signals)
            else:
                output_signals = (output_signals - min(output_signals)) / (max(output_signals) - min(output_signals))
                output_signals = (output_signals - 0.5) * 2.0
        else:
            for index in [0, 1]:
                if max(output_signals[:, index]) - min(output_signals[:, index]) < 1e-10:
                    output_signals[:, index] = output_signals[:, index] - mean(output_signals[:, index])
                else:
                    output_signals[:, index] -= min(output_signals[:, index])
                    output_signals[:, index] /= max(output_signals[:, index]) - min(output_signals[:, index])
                    output_signals[:, index] = (output_signals[:, index] - 0.5) * 2.0

        return output_signals

    def activate(self, values, activate_index):
        """
        Forward propagate through activating.

        :param values: input intersected_values.
        :type values: torch.Tensor

        :param activate_index: index of activations.
        :type activate_index: int

        :return: output intersected_values.
        :rtype: torch.Tensor
        """
        if self.a[activate_index] == "tanh":
            return tanh(values)
        elif self.a[activate_index] == "sigmoid":
            return sigmoid(values)
        elif self.a[activate_index] == "relu":
            return relu(values)
        else:
            raise ValueError("No such activation function type!")

    def aggregate(self, values, aggregate_index):
        """
        Forward propagate through aggregating.

        :param values: input intersected_values.
        :type values: torch.Tensor

        :param aggregate_index: index of aggregations.
        :type aggregate_index: int

        :return: output intersected_values.
        :rtype: torch.Tensor
        """
        if self.g[aggregate_index] == "sum":
            return unsqueeze(sum(values, dim=1), dim=1)
        elif self.g[aggregate_index] == "max":
            return unsqueeze(max(values, dim=1)[0], dim=1)
        else:
            raise ValueError("No such aggregation function type!")

    def add_weight(self, values, weight_indices):
        """
        Add weight for the intersected_values.

        :param values: input intersected_values.
        :type values: torch.Tensor

        :param weight_indices: indices of weight parameter.
        :type weight_indices: torch.Tensor, numpy.ndarray, or list

        :return: output intersected_values.
        :rtype: torch.Tensor
        """
        if len(weight_indices) == 1 and values.size()[1] == 1:
            return self.w[weight_indices[0]](values)
        if len(weight_indices) == 2 and values.size()[1] == 2:
            return cat(tuple([self.w[weight_indices[index]](unsqueeze(values[:, index], dim=1))
                              for index in range(len(weight_indices))]), dim=1)
        if len(weight_indices) == 2 and values.size()[1] == 1:
            return cat(tuple([self.w[weight_indices[index]](values)
                              for index in range(len(weight_indices))]), dim=1)

    def add_bias(self, values, bias_index):
        """
        Add bias for the intersected_values.

        :param values: input intersected_values.
        :type values: torch.Tensor

        :param bias_index: index of bias parameter.
        :type bias_index: int

        :return: output intersected_values.
        :rtype: torch.Tensor
        """
        return self.b[bias_index](values)

    def restrict(self):
        """
        Restrict the weights and biases before the forward propagation.
        """
        for index in range(len(self.w)):
            self.w[index].restrict()
        for index in range(len(self.b)):
            self.b[index].restrict()

    def reset(self, weights=None, biases=None):
        """
        Reset the weight and bias intersected_values.

        :param weights: established weight intersected_values.
        :type weights: list or None

        :param biases: established bias intersected_values.
        :type biases: list or None
        """
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
        name = self.t.replace("-", " ")
        if self.t == "collider":
            info += "\t" + "motif type   |  " + name + " " + str((self.i - 1) // 2 + (self.i - 1) % 2 + 1) + "\n"
            info += "\t" + "activation   |  (1),(2) >> " + self.a[0].rjust(9) + " >> (3)" + "\n"
            info += "\t" + "aggregation  |  (1),(2) >> " + self.g[0].rjust(9) + " >> (3)" + "\n"
            info += "\t" + "weight       |      (1) >> " + ws[0] + " >> (3)" + "\n"
            info += "\t" + "weight       |      (2) >> " + ws[1] + " >> (3)" + "\n"
            info += "\t" + "bias         |  (1),(2) >> " + bs[0] + " >> (3)" + ">"
        elif self.t == "fork":
            info += "\t" + "motif type   |  " + name + " " + str((self.i - 1) // 2 + (self.i - 1) % 2 + 1) + "\n"
            info += "\t" + "activation   |      (1) >> " + self.a[0].rjust(8) + " >> (2)" + "\n"
            info += "\t" + "activation   |      (1) >> " + self.a[1].rjust(8) + " >> (3)" + "\n"
            info += "\t" + "weight       |      (1) >> " + ws[0] + " >> (2)" + "\n"
            info += "\t" + "weight       |      (1) >> " + ws[1] + " >> (3)" + "\n"
            info += "\t" + "bias         |      (1) >> " + bs[0] + " >> (2)" + "\n"
            info += "\t" + "bias         |      (1) >> " + bs[1] + " >> (3)" + ">"
        elif self.t == "chain":
            info += "\t" + "motif type   |  " + name + " " + str(self.i) + "\n"
            info += "\t" + "activation   |      (1) >> " + self.a[0].rjust(9) + " >> (2)" + "\n"
            info += "\t" + "activation   |      (2) >> " + self.a[1].rjust(9) + " >> (3)" + "\n"
            info += "\t" + "weight       |      (1) >> " + ws[0] + " >> (2)" + "\n"
            info += "\t" + "weight       |      (2) >> " + ws[1] + " >> (3)" + "\n"
            info += "\t" + "bias         |      (1) >> " + bs[0] + " >> (2)" + "\n"
            info += "\t" + "bias         |      (2) >> " + bs[1] + " >> (3)" + ">"
        else:
            info += "\t" + "motif type   |  " + name + " " + str(self.i) + "\n"
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
