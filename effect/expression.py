"""
@Author      : Haoling Zhang
@Description : Network-scale optimization tasks
"""
from copy import deepcopy
from numpy import array
from torch import Tensor, nn, optim, linspace, meshgrid, max, min, sum, cos, pi, nan
from typing import Union

from effect.operations import prepare_data, calculate_hessian_eigenvalues
from effect.robustness import estimate_lipschitz


def get_landscape(name: str,
                  points: int = 41) \
        -> Tensor:
    """
    Get a landscape fitting task.

    :param name: landscape name.
    :type name: str

    :param points: sampling points.
    :type points: int

    :return: landscape matrix.
    :rtype: torch.Tensor
    """
    if name == "Quadratic Saddle":  # https://ganguli-gang.stanford.edu/pdf/14.SaddlePoint.NIPS.pdf
        # noinspection PyTypeChecker
        x, y = meshgrid((linspace(-1, +1, points), linspace(-1, 1, points)), indexing="ij")
        landscape = 5 * (x ** 2) - (y ** 2)

    elif name == "Monkey Saddle":  # https://ganguli-gang.stanford.edu/pdf/14.SaddlePoint.NIPS.pdf
        # noinspection PyTypeChecker
        x, y = meshgrid((linspace(-1, +1, points), linspace(-1, 1, points)), indexing="ij")
        landscape = (x ** 3) - 3 * x * (y ** 2)

    elif name == "Branin":  # https://www.sfu.ca/~ssurjano/branin.html
        # noinspection PyTypeChecker
        x, y = meshgrid((linspace(-5, 15, points), linspace(0, 15, points)), indexing="ij")
        a, b, c, r, s, t = 1.0, 5.1 / (4 * pi * pi), 5.0 / pi, 6.0, 10.0, 1 / (8 * pi)
        landscape = a * (y - b * (x ** 2) + c * x - r) ** 2 + s * (1 - t) * cos(x) + s

    elif name == "three-hump Camel":  # https://www.sfu.ca/~ssurjano/camel3.html
        # noinspection PyTypeChecker
        x, y = meshgrid((linspace(-2, 2, points), linspace(-2, 2, points)), indexing="ij")
        landscape = 2.0 * (x ** 2) - 1.05 * (x ** 4) + (x ** 6) / 6.0 + x * y + (y ** 2)

    elif name == "six-hump Camel":  # https://www.sfu.ca/~ssurjano/camel6.html
        # noinspection PyTypeChecker
        x, y = meshgrid((linspace(-2, 2, points), linspace(-1, 1, points)), indexing="ij")
        landscape = (4.0 - 2.1 * (x ** 2) + (x ** 4) / 3.0) * (x ** 2) + x * y + (-4.0 + 4.0 * (y ** 2)) * (y ** 2)
    else:
        raise ValueError("Unrecognized landscape name: " + name + "!")

    landscape -= min(landscape)
    landscape /= max(landscape)
    landscape = landscape * 2 - 1

    return landscape


def fit(network: nn.Module,
        name: str,
        learn_rate: float,
        patience: int,
        various_metrics: Union[list, str] = "none",
        threshold: float = 1e-2,
        points: int = 41,
        maximum_iteration: Union[int, None] = None) \
        -> dict:
    """
    Fit a 2D landscape by a given neural network.

    :param network: neural network.
    :type network: torch.nn.Module

    :param name: landscape like "Quadratic Saddle", "Monkey Saddle", "Branin", "three-hump Camel" and "six-hump Camel".
    :type name: str

    :param learn_rate: learn rate.
    :type learn_rate: float

    :param patience: decrease the current loss by at least 1/"patience" within "patience" steps.
    :type patience: int

    :param various_metrics: additional metrics names, including "lipschitz", "variance", ..., or "all".
    :type various_metrics: list, str, or None

    :param threshold: success threshold.
    :type threshold: float

    :param points: sampling points.
    :type points: int

    :param maximum_iteration: given maximum iteration of the training stage.
    :type maximum_iteration: int or None

    :return: training records.
    :rtype: dict
    """
    input_data = prepare_data(value_range=(-1, +1), points=points)
    expected_landscape = get_landscape(name).flatten().unsqueeze(1)

    optimizer, criterion = optim.Adam(network.parameters(), lr=learn_rate), nn.MSELoss()

    motif_order = ["collider", "coherent-loop", "incoherent-loop", "entire network"]
    metric_list = ["sparsity", "lipschitz constant", "gradient variance", "hessian eigenvalue", "spectral norm"]

    records = {"training loss": []}

    if type(various_metrics) == str:
        if various_metrics == "all":
            for one_metrics in metric_list:
                records[one_metrics] = []
        elif various_metrics == "none":
            pass
        else:
            raise ValueError("Unrecognized metrics: " + various_metrics + ".")
    elif type(various_metrics) == list:
        for one_metrics in various_metrics:
            if one_metrics in metric_list:
                records[one_metrics] = []
            else:
                raise ValueError("Unrecognized metrics: " + one_metrics + ".")
    else:
        raise ValueError("Invalid metrics!")

    while True:
        obtained_landscape = network(input_data)

        if "hessian eigenvalue" in records:
            network_copy = deepcopy(network)
        else:
            network_copy = None

        loss = criterion(obtained_landscape, expected_landscape)

        records["training loss"].append(loss.item())

        # record the sparsity.
        if "sparsity" in records:
            obtained_data = network.get_sparsity()
            saved_values = [obtained_data[key] if obtained_data[key] is not None else nan for key in motif_order]
            records["sparsity"].append(saved_values)

        # record the lipschitz constant
        if "lipschitz constant" in records:
            obtained_data = estimate_lipschitz(value_range=(-1, +1), points=points,
                                               output=obtained_landscape.reshape(points, points).detach().numpy())
            records["lipschitz constant"].append(obtained_data)

        # record the spectral norm.
        if "spectral norm" in records:
            obtained_data = network.get_spectral_norm_summary()
            saved_values = [obtained_data[key] if obtained_data[key] is not None else nan for key in motif_order]
            records["spectral norm"].append(saved_values)

        # optimize the parameters.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # record the gradient variance.
        if "gradient variance" in records:
            obtained_data = network.get_gradient_variance()
            saved_values = [obtained_data[key] if obtained_data[key] is not None else nan for key in motif_order]
            records["gradient variance"].append(saved_values)

        # record the hessian eigenvalue.
        if "hessian eigenvalue" in records:
            assert network_copy is not None
            hessian_eigenvalue = calculate_hessian_eigenvalues(network=network_copy, criterion=criterion,
                                                               input_data=input_data,
                                                               expected_landscape=expected_landscape)

            records["hessian eigenvalue"].append([
                min(Tensor(hessian_eigenvalue)).item(),
                max(Tensor(hessian_eigenvalue)).item(),
                sum(Tensor(hessian_eigenvalue)).item()
            ])

        optimizer.step()

        # restrict the parameters if needed.
        network.restrict()

        if records["training loss"][-1] < threshold:
            break

        if len(records["training loss"]) > patience:
            obtained_loss = records["training loss"][-1]
            expected_loss = (1.0 - 1.0 / patience) * records["training loss"][-patience]

            if obtained_loss > expected_loss:
                break

        if maximum_iteration is not None and len(records["training loss"]) == maximum_iteration:
            break

    # noinspection PyUnresolvedReferences
    records["training loss"] = array(records["training loss"])

    if "sparsity" in records:
        # noinspection PyUnresolvedReferences
        records["sparsity"] = array(records["sparsity"])

    if "lipschitz constant" in records:
        # noinspection PyUnresolvedReferences
        records["lipschitz constant"] = array(records["lipschitz constant"])
        reference_lipschitz = estimate_lipschitz(value_range=(-1, +1), points=points,
                                                 output=expected_landscape.reshape(points, points).detach().numpy())
        records["lipschitz constant"] /= reference_lipschitz

    if "gradient variance" in records:
        # noinspection PyUnresolvedReferences
        records["gradient variance"] = array(records["gradient variance"])

    if "spectral norm" in records:
        # noinspection PyUnresolvedReferences
        records["spectral norm"] = array(records["spectral norm"])

    if "hessian eigenvalue" in records:
        # noinspection PyUnresolvedReferences
        records["hessian eigenvalue"] = array(records["hessian eigenvalue"])

    return records
