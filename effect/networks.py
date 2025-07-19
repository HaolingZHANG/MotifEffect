"""
@Author      : Haoling Zhang
@Description : Definition of neural motif
"""
from numpy import ndarray, array, concatenate, linalg, abs, var
from torch import Tensor, tensor, nn, no_grad, unsqueeze, stack, cat, rand, min, max, sum, mean, relu, tanh, sigmoid
from typing import Union


class RestrictedWeight(nn.Module):

    def __init__(self,
                 is_positive: bool = True,
                 value: float = None,
                 bound: tuple = (1e-3, 1e0)):
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
        self.weight, self.is_positive, self.bound = None, is_positive, bound
        self.reset(value=value)

    def forward(self,
                data: Tensor) \
            -> Tensor:
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
        :rtype: float
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
            self.weight = nn.Parameter(data=value, requires_grad=True)

    def reset(self,
              value: Union[float, None] = None):
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
                print(value, self.is_positive, self.bound)
                raise ValueError("the inputted value is wrong, it needs to meet the established constraints!")

        elif self.is_positive:
            value = +self.bound[0] + (self.bound[1] - self.bound[0]) * rand(1)

        else:
            value = -self.bound[1] + (self.bound[1] - self.bound[0]) * rand(1)

        self.weight = nn.Parameter(data=value, requires_grad=True)


class RestrictedBias(nn.Module):

    def __init__(self, value: float = None,
                 bound: tuple = (-1e0, +1e0)):
        """
        Initialize the restricted bias module.

        :param value: established value.
        :type value: float

        :param bound: bound of the weight.
        :type bound: tuple
        """
        super(RestrictedBias, self).__init__()
        self.bias, self.bound = None, bound
        self.reset(value=value)

    def forward(self,
                data: Tensor) \
            -> Tensor:
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
        :rtype: float
        """
        return float(self.bias)

    def restrict(self):
        """
        Restrict the bias before the forward propagation.
        """
        value = None
        if self.bias <= self.bound[0]:
            value = tensor(self.bound[0])
        elif self.bias >= self.bound[1]:
            value = tensor(self.bound[1])

        if value is not None:
            self.bias = nn.Parameter(data=value, requires_grad=True)

    def reset(self,
              value: Union[float, None] = None):
        """
        Reset the bias value.

        :param value: established bias value.
        :type value: float or None.
        """
        if value is not None:
            value = tensor(value)
        else:
            value = self.bound[0] + (self.bound[1] - self.bound[0]) * rand(1)

        self.bias = nn.Parameter(data=value, requires_grad=True)


class NeuralMotif(nn.Module):

    def __init__(self,
                 motif_type: str,
                 motif_index: int,
                 activations: Union[tuple, list],
                 aggregations: Union[tuple, list],
                 weights: Union[tuple, list, None] = None,
                 biases: Union[tuple, list, None] = None,
                 weight_bound: tuple = (+1e-3, +1e0),
                 bias_bound: tuple = (-1e0, +1e0)):
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
        :type weight_bound: tuple

        :param bias_bound: bound of bias.
        :type bias_bound: tuple
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

    def forward(self,
                input_signals: Tensor) \
            -> Tensor:
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
                # noinspection PyAugmentAssignment
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

    def activate(self,
                 values: Tensor,
                 activate_index: int) \
            -> Tensor:
        """
        Forward propagate through activating.

        :param values: input values.
        :type values: torch.Tensor

        :param activate_index: index of activations.
        :type activate_index: int

        :return: output values.
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

    def aggregate(self,
                  values: Tensor,
                  aggregate_index: int) \
            -> Tensor:
        """
        Forward propagate through aggregating.

        :param values: input values.
        :type values: torch.Tensor

        :param aggregate_index: index of aggregations.
        :type aggregate_index: int

        :return: output values.
        :rtype: torch.Tensor
        """
        if self.g[aggregate_index] == "sum":
            return unsqueeze(sum(values, dim=1), dim=1)
        elif self.g[aggregate_index] == "max":
            return unsqueeze(max(values, dim=1)[0], dim=1)
        else:
            raise ValueError("No such aggregation function type!")

    def add_weight(self,
                   values: Tensor,
                   weight_indices: Union[Tensor, ndarray, list]) \
            -> Tensor:
        """
        Add weight for the values.

        :param values: input values.
        :type values: torch.Tensor

        :param weight_indices: indices of weight parameter.
        :type weight_indices: torch.Tensor, numpy.ndarray, or list

        :return: output values.
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

    def add_bias(self,
                 values: Tensor,
                 bias_index: int) \
            -> Tensor:
        """
        Add bias for the intersected values.

        :param values: input values.
        :type values: torch.Tensor

        :param bias_index: index of bias parameter.
        :type bias_index: int

        :return: output values.
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

    def reset(self,
              weights: Union[list, None] = None,
              biases: Union[list, None] = None):
        """
        Reset the weight and bias values.

        :param weights: established weight values.
        :type weights: list or None

        :param biases: established bias values.
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
            self.w = nn.ModuleList([RestrictedWeight(flag, value, bound=self.weight_bound)
                                    for flag, value in zip(weight_flags, weights)])
        else:
            self.w = nn.ModuleList([RestrictedWeight(flag, bound=self.weight_bound)
                                    for flag in weight_flags])

        if biases is not None:
            if len(biases) != bias_size:
                raise ValueError("the number of weights should be "
                                 + str(bias_size) + " got " + str(len(biases)) + ".")
            self.b = nn.ModuleList([RestrictedBias(bias_value, bound=self.bias_bound)
                                    for bias_value in biases])
        else:
            self.b = nn.ModuleList([RestrictedBias(bound=self.bias_bound)
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


class Collider(nn.Module):

    def __init__(self):
        """
        Initialize a collider.
        """
        super(Collider, self).__init__()
        self.collider = nn.Linear(2, 1)

    def forward(self,
                input_signals: Tensor) \
            -> Tensor:
        """
        Forward propagate through the collider.

        :param input_signals: input signals.
        :type input_signals: torch.Tensor

        :return: output signals.
        :rtype: torch.Tensor
        """
        return tanh(self.collider(input_signals))

    def get_weights(self) \
            -> dict:
        """
        Get the weights of the collider.

        :return: weights.
        :rtype: dict
        """
        weight_data = self.collider.weight
        return {
            "x->z": weight_data[0, 0].item(),
            "y->z": weight_data[0, 1].item()
        }

    def get_utilization(self,
                        epsilon: float = 1e-6) \
            -> float:
        """
        Get the utilization of weights.

        :param epsilon: minimum absolute value of weight.
        :type epsilon: float

        :return: utilization of weights.
        :rtype: float
        """
        weight_data = self.get_weights()

        count = 0
        for value in [weight_data["x->z"], weight_data["y->z"]]:
            if abs(value) > epsilon:
                count += 1

        return count / 2.0

    def get_gradients(self) \
            -> dict:
        """
        Get the gradients of the collider (if contained).

        :return: weight gradients and bias gradients.
        :rtype: dict
        """
        weight_gradients = self.collider.weight.grad
        bias_gradient = self.collider.bias.grad

        if weight_gradients is None or bias_gradient is None:
            raise ValueError("No gradients now!")

        return {
            "w": weight_gradients.detach().numpy().flatten(),
            "b": bias_gradient.detach().numpy().flatten()
        }

    def get_spectral_norm(self) \
            -> float:
        """
        Get the spectral norm of the linear weight matrix.

        :return: spectral norm.
        :rtype: float
        """
        return linalg.norm(self.collider.weight.data.detach().numpy(), ord=2)

    def get_motif_information(self) \
            -> tuple:
        """
        Get collider type and collider index.

        :return: collider type and collider index.
        :rtype: tuple
        """
        weights = self.get_weights()

        if weights["x->z"] > 0.0 and weights["y->z"] > 0.0:
            return "collider", 1
        # noinspection PyChainedComparisons
        if weights["x->z"] > 0.0 and weights["y->z"] < 0.0:
            return "collider", 2
        # noinspection PyChainedComparisons
        if weights["x->z"] < 0.0 and weights["y->z"] > 0.0:
            return "collider", 3
        if weights["x->z"] < 0.0 and weights["y->z"] < 0.0:
            return "collider", 4

        raise ValueError("No such situation (may one weight is 0)!")


class Loop(nn.Module):

    def __init__(self,
                 original_direction: bool = True):
        """
        Initialize a loop.

        :param original_direction: whether to apply the direction from node x to node y (or reversed, if it is False).
        :type original_direction: bool
        """
        super(Loop, self).__init__()
        self.original_direction = original_direction
        self.collider, self.addition = Collider(), nn.Linear(1, 1)

    def forward(self,
                input_signals: Tensor) \
            -> Tensor:
        """
        Forward propagate through the loop.

        :param input_signals: input signals.
        :type input_signals: torch.Tensor

        :return: output signals.
        :rtype: torch.Tensor
        """
        if self.original_direction:
            signals = (
                input_signals[:, 0:1],
                input_signals[:, 1:2] + tanh(self.addition(input_signals[:, 0:1]))
            )
        else:
            signals = (
                input_signals[:, 0:1] + tanh(self.addition(input_signals[:, 1:2])),
                input_signals[:, 1:2]
            )
        return tanh(self.collider(cat(signals, dim=1)))

    def get_weights(self) \
            -> dict:
        """
        Get the weights of the loop.

        :return: weights.
        :rtype: dict
        """
        weight_data_1, weight_data_2 = self.collider.collider.weight, self.addition.weight

        return {
            "x->z": weight_data_1[0, 0].item(),
            "y->z": weight_data_1[0, 1].item(),
            "x->y": weight_data_2[0, 0].item(),
        }

    def get_utilization(self,
                        epsilon: float = 1e-6) \
            -> float:
        """
        Get the utilization of weights.

        :param epsilon: minimum absolute value of weight.
        :type epsilon: float

        :return: utilization of weights.
        :rtype: float
        """
        weight_data = self.get_weights()

        count = 0
        for value in [weight_data["x->z"], weight_data["y->z"], weight_data["x->y"]]:
            if abs(value) > epsilon:
                count += 1

        return count / 3.0

    def get_gradients(self) \
            -> dict:
        """
        Get the gradients of the loop (if contained).

        :return: weight gradients and bias gradients.
        :rtype: dict
        """
        collider_gradients = self.collider.get_gradients()

        addition_weight_gradient = self.addition.weight.grad
        addition_bias_gradient = self.addition.bias.grad

        if addition_weight_gradient is None or addition_bias_gradient is None:
            raise ValueError("No gradients now!")

        total_weight_gradient = concatenate([collider_gradients["w"],
                                             addition_weight_gradient.detach().numpy().flatten()])
        total_bias_gradient = concatenate([collider_gradients["b"],
                                           addition_bias_gradient.detach().numpy().flatten()])

        return {
            "w": total_weight_gradient,
            "b": total_bias_gradient
        }

    def get_spectral_norm(self) \
            -> float:
        """
        Get the spectral norm of the linear weight matrix.

        :return: spectral norm.
        :rtype: float
        """
        collider_norm = self.collider.get_spectral_norm()
        addition_norm = linalg.norm(self.addition.weight.data.detach().numpy(), ord=2)

        # In conventional feedforward neural networks, the spectral norm of each linear layer is often used as a proxy
        # for the Lipschitz constant, and the overall Lipschitz constant is upper-bounded by the
        # product of spectral norms across layers.
        # However, this multiplicative formulation assumes a strictly sequential (chain-like) composition of layers,
        # where each transformation is linearly followed by the next.
        # In contrast, our loop motifs are structured around nonlinear compositions and parallel interactions —
        # for example, in the forward path x to y to z, the signal from x affects z through two nonlinear branches:
        # one directly and another via a tanh-modulated path through y.
        # Since these branches are merged in a non-sequential and nonlinear fashion,
        # the product-based bound becomes theoretically invalid.
        # Therefore, we adopt a simple additive aggregation of spectral norms from the subcomponents
        # (e.g., collider and addition modules) within each motif.
        # This additive estimate reflects the total contribution of each path to potential signal amplification
        # and serves as a more appropriate proxy for robustness-related capacity within nonlinear motifs.
        return collider_norm + addition_norm

    def is_coherent(self) \
            -> bool:
        """
        Check if the loop is coherent.

        :return: True if coherent, False if incoherent.
        :rtype: bool
        """
        weight_data = self.get_weights()

        if weight_data["x->y"] > 0 and weight_data["x->z"] > 0 and weight_data["y->z"] > 0:
            return True
        # noinspection PyChainedComparisons
        if weight_data["x->y"] < 0 and weight_data["x->z"] > 0 and weight_data["y->z"] < 0:
            return True
        # noinspection PyChainedComparisons
        if weight_data["x->y"] < 0 and weight_data["x->z"] < 0 and weight_data["y->z"] > 0:
            return True
        # noinspection PyChainedComparisons
        if weight_data["x->y"] > 0 and weight_data["x->z"] < 0 and weight_data["y->z"] < 0:
            return True

        return False

    def get_motif_information(self) \
            -> tuple:
        """
        Get loop type and loop index.

        :return: loop type and loop index.
        :rtype: tuple
        """
        weight_data = self.get_weights()

        if weight_data["x->y"] > 0.0 and weight_data["x->z"] > 0.0 and weight_data["y->z"] > 0.0:
            return "coherent-loop", 1
        # noinspection PyChainedComparisons
        if weight_data["x->y"] < 0.0 and weight_data["x->z"] > 0.0 and weight_data["y->z"] < 0.0:
            return "coherent-loop", 2
        # noinspection PyChainedComparisons
        if weight_data["x->y"] < 0.0 and weight_data["x->z"] < 0.0 and weight_data["y->z"] > 0.0:
            return "coherent-loop", 3
        # noinspection PyChainedComparisons
        if weight_data["x->y"] > 0.0 and weight_data["x->z"] < 0.0 and weight_data["y->z"] < 0.0:
            return "coherent-loop", 4

        # noinspection PyChainedComparisons
        if weight_data["x->y"] < 0.0 and weight_data["x->z"] > 0.0 and weight_data["y->z"] > 0.0:
            return "incoherent-loop", 1
        # noinspection PyChainedComparisons
        if weight_data["x->y"] > 0.0 and weight_data["x->z"] > 0.0 and weight_data["y->z"] < 0.0:
            return "incoherent-loop", 2
        # noinspection PyChainedComparisons
        if weight_data["x->y"] > 0.0 and weight_data["x->z"] < 0.0 and weight_data["y->z"] > 0.0:
            return "incoherent-loop", 3
        # noinspection PyChainedComparisons
        if weight_data["x->y"] < 0.0 and weight_data["x->z"] < 0.0 and weight_data["y->z"] < 0.0:
            return "incoherent-loop", 4

        raise ValueError("No such situation (may one weight is 0)!")


class RestrictedLoop(Loop):

    def __init__(self,
                 loop_type: str,
                 loop_index: int,
                 epsilon: float = 1e-6,
                 original_direction: bool = True):
        """
        Initialize a restricted loop.

        :param loop_type: loop type: "i" for incoherent loop and "c" for coherent loop.
        :type loop_type: str

        :param loop_index: index of sub-loop type.
        :type loop_index: int

        :param epsilon: minimum absolute value of weight.
        :type epsilon: float

        :param original_direction: whether to apply the direction from node x to node y (or reversed, if it is False).
        :type original_direction: bool
        """
        if epsilon <= 0:
            raise ValueError("Epsilon needs to larger than 0.")

        super(RestrictedLoop, self).__init__(original_direction=original_direction)

        if loop_type == "i":  # incoherent
            if loop_index == 1:
                self.sign_config = {"x->y": "-", "x->z": "+", "y->z": "+"}
            elif loop_index == 2:
                self.sign_config = {"x->y": "+", "x->z": "+", "y->z": "-"}
            elif loop_index == 3:
                self.sign_config = {"x->y": "+", "x->z": "-", "y->z": "+"}
            elif loop_index == 4:
                self.sign_config = {"x->y": "-", "x->z": "-", "y->z": "-"}
            else:
                raise ValueError("Loop index must be 1, 2, 3, or 4!")

        elif loop_type == "c":  # coherent
            if loop_index == 1:
                self.sign_config = {"x->y": "+", "x->z": "+", "y->z": "+"}
            elif loop_index == 2:
                self.sign_config = {"x->y": "-", "x->z": "+", "y->z": "-"}
            elif loop_index == 3:
                self.sign_config = {"x->y": "-", "x->z": "+", "y->z": "+"}
            elif loop_index == 4:
                self.sign_config = {"x->y": "+", "x->z": "-", "y->z": "-"}
            else:
                raise ValueError("Loop index must be 1, 2, 3, or 4!")

        else:
            raise ValueError("Loop type must be either \"incoherent\" or \"coherent\"!")

        self.loop_type, self.loop_index, self.epsilon = loop_type, loop_index, epsilon

        self.adjust_weights()

    def adjust_weights(self):
        """
        Adjust weights to ensure proper initialization.
        """
        with no_grad():
            # clamp x->y
            if 0.0 < self.addition.weight[0, 0] < self.epsilon:
                self.addition.weight[0, 0] = 2 * self.epsilon
            elif -self.epsilon < self.addition.weight[0, 0] < 0.0:
                self.addition.weight[0, 0] = -2 * self.epsilon

            # enforce sign x->y
            if self.sign_config["x->y"] == "-" and self.addition.weight[0, 0] > 0.0:
                self.addition.weight[0, 0] *= -1.0
            elif self.sign_config["x->y"] == "+" and self.addition.weight[0, 0] < 0.0:
                self.addition.weight[0, 0] *= -1.0

            # clamp x->z
            if 0.0 < self.collider.collider.weight[0, 0] < self.epsilon:
                self.collider.collider.weight[0, 0] = 2 * self.epsilon
            elif -self.epsilon < self.collider.collider.weight[0, 0] < 0.0:
                self.collider.collider.weight[0, 0] = -2 * self.epsilon

            # enforce sign x->z
            if self.sign_config["x->z"] == "-" and self.collider.collider.weight[0, 0] > 0.0:
                self.collider.collider.weight[0, 0] *= -1
            elif self.sign_config["x->z"] == "+" and self.collider.collider.weight[0, 0] < 0.0:
                self.collider.collider.weight[0, 0] *= -1

            # clamp y->z
            if 0.0 < self.collider.collider.weight[0, 1] < self.epsilon:
                self.collider.collider.weight[0, 1] = 2 * self.epsilon
            elif -self.epsilon < self.collider.collider.weight[0, 1] < 0.0:
                self.collider.collider.weight[0, 1] = -2 * self.epsilon

            # enforce sign y->z
            if self.sign_config["y->z"] == "-" and self.collider.collider.weight[0, 1] > 0.0:
                self.collider.collider.weight[0, 1] *= -1
            elif self.sign_config["y->z"] == "+" and self.collider.collider.weight[0, 1] < 0.0:
                self.collider.collider.weight[0, 1] *= -1

    def restrict(self):
        """
        Restrict weights to ensure an unchanged network structure.
        """
        with no_grad():
            if self.sign_config["x->y"] == "-" and self.addition.weight[0, 0] > -self.epsilon:
                self.addition.weight[0, 0] = -self.epsilon
            elif self.sign_config["x->y"] == "+" and self.addition.weight[0, 0] < self.epsilon:
                self.addition.weight[0, 0] = self.epsilon

            if self.sign_config["x->z"] == "-" and self.collider.collider.weight[0, 0] > -self.epsilon:
                self.collider.collider.weight[0, 0] = -self.epsilon
            elif self.sign_config["x->z"] == "+" and self.collider.collider.weight[0, 0] < self.epsilon:
                self.collider.collider.weight[0, 0] = self.epsilon

            if self.sign_config["y->z"] == "-" and self.collider.collider.weight[0, 1] > -self.epsilon:
                self.collider.collider.weight[0, 1] = -self.epsilon
            elif self.sign_config["y->z"] == "+" and self.collider.collider.weight[0, 1] < self.epsilon:
                self.collider.collider.weight[0, 1] = self.epsilon

    def get_motif_information(self) \
            -> tuple:
        """
        Get loop type and loop index.

        :return: loop type and loop index.
        :rtype: tuple
        """
        return "coherent-loop" if self.loop_type == "c" else "incoherent-loop", self.loop_index


class MotifNetwork(nn.Module):

    def __init__(self,
                 motif_number: int = 1):
        """
        Initialize a motif perceptron network.

        :param motif_number: number of motifs.
        :type motif_number: int
        """
        super(MotifNetwork, self).__init__()
        self.bank, self.motif_number = nn.ModuleList(), motif_number
        self.build_bank()

    def build_bank(self):
        """
        Build layers according to the layer number.
        """
        raise NotImplementedError

    def forward(self,
                input_signals: Tensor) \
            -> Tensor:
        """
        Forward propagate through the given motif perceptron network.

        :param input_signals: input signals.
        :type input_signals: torch.Tensor

        :return: output signals.
        :rtype: torch.Tensor
        """
        return sum(stack([tanh(motif(input_signals)) for motif in self.bank]), dim=0)

    def restrict(self):
        """
        Restrict the parameters.
        """
        raise NotImplementedError

    def get_motif_types(self) \
            -> list:
        """
        Get motif types in the network.

        :return: motif types.
        :rtype: list
        """
        return [motif.get_motif_information()[0] for motif in self.bank]

    def get_motif_utilization(self,
                              epsilon: float = 1e-6) \
            -> list:
        """
        Get weight utilization of each motif in the network.

        :param epsilon: minimum absolute value of weight.
        :type epsilon: float

        :return: weight utilization of each motif.
        :rtype: list
        """
        return [motif.get_utilization(epsilon=epsilon) for motif in self.bank]

    def get_sparsity(self,
                     epsilon: float = 1e-6) \
            -> dict:
        """
        Get sparsity information of the network.

        :param epsilon: minimum absolute value of weight.
        :type epsilon: float

        :return: average motif sparsity and the entire network sparsity.
        :rtype: dict
        """
        groups = {"collider": [], "coherent-loop": [], "incoherent-loop": []}
        count, total = 0.0, 0.0
        for motif_type, utilization in zip(self.get_motif_types(), self.get_motif_utilization(epsilon=epsilon)):
            groups[motif_type].append(utilization)
            if motif_type == "collider":
                count += 2 * utilization
                total += 2
            else:
                count += 3 * utilization
                total += 3

        record = {key: mean(Tensor(value)).item() if len(value) > 0 else None for key, value in groups.items()}
        record["entire network"] = count / total

        return record

    def get_gradients(self) \
            -> list:
        """
        Get the gradients of the motif network (if contained).

        :return: weight gradients and bias gradients of each motif.
        :rtype: list
        """
        return [motif.get_gradients() for motif in self.bank]

    def get_gradient_variance(self) \
            -> dict:
        """
        Get the weight gradient variances of the motif network (if contained).

        :return: weight gradient variances divided by the motif type.
        :rtype: dict
        """
        groups = {"collider": [], "coherent-loop": [], "incoherent-loop": [], "entire network": []}

        for motif_type, gradient in zip(self.get_motif_types(), self.get_gradients()):
            weight_gradient_data = gradient["w"].tolist()
            groups[motif_type] += weight_gradient_data
            groups["entire network"] += weight_gradient_data

        return {key: var(array(value)) if len(value) > 0 else None for key, value in groups.items()}

    def get_spectral_norms(self) \
            -> list:
        """
        Get the spectral norms of motifs in the neural network.

        :return: motif spectral norms.
        :rtype: list
        """
        return [motif.get_spectral_norm() for motif in self.bank]

    def get_spectral_norm_summary(self) \
            -> dict:
        """
        Get the summary of spectral norms.

        :return: average spectral norm of different motif type and the spectral norm of the entire network.
        :rtype: dict
        """
        groups = {"collider": [], "coherent-loop": [], "incoherent-loop": [], "entire network": []}

        for motif_type, spectral_norm in zip(self.get_motif_types(), self.get_spectral_norms()):
            groups[motif_type].append(spectral_norm)
            groups["entire network"].append(spectral_norm)

        value_1 = mean(Tensor(groups["collider"])).item() if len(groups["collider"]) > 0 else None
        value_2 = mean(Tensor(groups["coherent-loop"])).item() if len(groups["coherent-loop"]) > 0 else None
        value_3 = mean(Tensor(groups["incoherent-loop"])).item() if len(groups["incoherent-loop"]) > 0 else None
        value_4 = sum(Tensor(groups["entire network"])).item()

        return {"collider": value_1, "coherent-loop": value_2, "incoherent-loop": value_3, "entire network": value_4}

    def get_hessian_eigenvalues(self):
        pass

    def get_motif_number(self) \
            -> int:
        """
        Get the number of motifs.

        :return: number of motifs.
        :rtype: int
        """
        return self.motif_number

    def get_parameter_number(self) \
            -> int:
        """
        Get the number of parameters of the network.

        :return: number of parameters.
        :rtype: int
        """
        number = 0

        for motif in self.bank:
            for parameter in motif.parameters():
                number += parameter.numel()

        return number


class ColliderNetwork(MotifNetwork):

    def build_bank(self):
        """
        Build motifs according to the motif number.
        """
        for index in range(self.motif_number):
            self.bank.append(Collider())

    def restrict(self):
        """
        Restrict the parameters.
        """
        pass  # not applicable.


class LoopNetwork(MotifNetwork):

    def build_bank(self):
        """
        Build motifs according to the motif number.
        """
        for index in range(self.motif_number):
            self.bank.append(Loop(original_direction=index % 2 == 0))

    def restrict(self):
        """
        Restrict the parameters.
        """
        pass  # not applicable.


class RestrictedLoopNetwork(MotifNetwork):

    def __init__(self,
                 flags_1: ndarray,
                 flags_2: ndarray,
                 motif_number: int = 1,
                 epsilon: float = 1e-6):
        if len(flags_1) != motif_number or len(flags_2) != motif_number:
            raise ValueError("The length of coherent flags and index flags must be %d." % motif_number)

        self.flags_1, self.flags_2, self.epsilon = flags_1, flags_2, epsilon

        super(RestrictedLoopNetwork, self).__init__(motif_number=motif_number)

    def build_bank(self):
        for index in range(self.motif_number):
            loop_type = "c" if self.flags_1[index] else "i"
            self.bank.append(RestrictedLoop(loop_type=loop_type, loop_index=self.flags_2[index],
                                            epsilon=self.epsilon, original_direction=index % 2 == 0))

    def restrict(self):
        """
        Restrict the parameters.
        """
        for motif in self.bank:
            motif.restrict()
