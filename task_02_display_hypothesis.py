from matplotlib import pyplot, patches, markers
from numpy import arange, min, max, sum, linspace, random, std
from scipy.stats import gaussian_kde

from hypothesis import NeuralMotif, prepare_data, calculate_gradients


if __name__ == "__main__":
    input_range, points = (-1, +1), 41
    source_motif = NeuralMotif(motif_type="incoherent-loop", motif_index=2,
                               activations=("relu", "relu"), aggregations=("max", "max"),
                               weights=[0.90827476978302, 0.5210987329483032, -0.8107407093048096],
                               biases=[0.9697856903076172, 0.9693939685821533])
    target_motif = NeuralMotif(motif_type="collider", motif_index=2,
                               activations=("relu",), aggregations=("sum",),
                               weights=[0.4955233335494995, -0.04083000123500824],
                               biases=[0.9734159708023071])
    data = prepare_data(value_range=input_range, points=points)
    source_gradients = calculate_gradients(value_range=input_range, points=points, motif=source_motif)
    target_gradients = calculate_gradients(value_range=input_range, points=points, motif=target_motif)
    source_data = source_motif(data).reshape(points, points).detach().numpy()
    target_data = target_motif(data).reshape(points, points).detach().numpy()
    output_range = (min([source_data, target_data]), max([source_data, target_data]))

    draw_info = {"collider": ([0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1, 2], [3], []),
                 "incoherent-loop": ([0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2])}
    figure = pyplot.figure(figsize=(10, 4.5), tight_layout=True)
    grid = pyplot.GridSpec(2, 4)

    pyplot.subplot(grid[0, 0])
    pyplot.title("incoherent-loop motif", fontsize=12)
    # noinspection PyUnresolvedReferences
    info = draw_info[source_motif.t]
    for index, (px, py) in enumerate(zip(info[0], info[1])):
        if index + 1 in info[2]:
            pyplot.scatter(px, py, color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
        elif index + 1 in info[3]:
            pyplot.scatter(px, py, color="black", edgecolor="black", lw=1.5, s=120, zorder=2)
        elif index + 1 in info[4]:
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                           color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                           color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
        else:
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
    x, y = info[0], info[1]
    # noinspection PyUnresolvedReferences
    for former, latter, weight in [(1, 2, source_motif.w[0].value()), (1, 3, source_motif.w[1].value()),
                                   (2, 3, source_motif.w[2].value())]:
        if weight > 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            shrinkA=6, shrinkB=6, lw=1.5), zorder=2)
        elif weight < 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                            shrinkA=6, shrinkB=6, lw=1.5), zorder=2)
    pyplot.text(x=info[0][0], y=info[1][0] - 0.06, s="x", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[0][1], y=info[1][1] - 0.06, s="y", fontsize=9, va="top", ha="center")
    # noinspection PyUnresolvedReferences
    pyplot.text(x=info[0][1], y=info[1][1] + 0.1, s=source_motif.a[0] + "/" + source_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    # noinspection PyUnresolvedReferences
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=source_motif.a[1] + "/" + source_motif.g[1], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    pyplot.axis("off")

    pyplot.subplot(grid[0, 1])
    pyplot.title("collider motif", fontsize=12)
    # noinspection PyUnresolvedReferences
    info = draw_info[target_motif.t]
    for index, (px, py) in enumerate(zip(info[0], info[1])):
        if index + 1 in info[2]:
            pyplot.scatter(px, py, color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
        elif index + 1 in info[3]:
            pyplot.scatter(px, py, color="black", edgecolor="black", lw=1.5, s=120, zorder=2)
        elif index + 1 in info[4]:
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                           color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                           color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
        else:
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
    x, y = info[0], info[1]
    # noinspection PyUnresolvedReferences
    data = [(1, 3, target_motif.w[0].value()),
            (2, 3, target_motif.w[1].value())]
    # noinspection PyUnresolvedReferences
    for former, latter, weight in [(1, 3, target_motif.w[0].value()), (2, 3, target_motif.w[1].value())]:
        if weight > 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            shrinkA=6, shrinkB=6, lw=1.5), zorder=2)
        elif weight < 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                            shrinkA=6, shrinkB=6, lw=1.5), zorder=2)
    pyplot.text(x=info[0][0], y=info[1][0] - 0.06, s="x", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[0][1], y=info[1][1] - 0.06, s="y", fontsize=9, va="top", ha="center")
    # noinspection PyUnresolvedReferences
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=target_motif.a[0] + "/" + target_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    pyplot.axis("off")

    pyplot.subplot(grid[1, 0])
    pyplot.pcolormesh(arange(points), arange(points), source_data, cmap="rainbow",
                      vmin=output_range[0], vmax=output_range[1])
    pyplot.xlabel("input x")
    pyplot.ylabel("input y")
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])

    pyplot.subplot(grid[1, 1])
    pyplot.pcolormesh(arange(points), arange(points), target_data, cmap="rainbow",
                      vmin=output_range[0], vmax=output_range[1])
    pyplot.xlabel("input x")
    pyplot.ylabel("input y")
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])

    pyplot.subplot(grid[:, 2:])
    source_gradients, target_gradients = source_gradients.reshape(-1), target_gradients.reshape(-1)
    pyplot.title("gradient distribution", fontsize=12)
    if std(source_gradients) < 0.01:
        source_gradients += random.normal(size=(len(source_gradients),)) * 0.01 - 0.005
    method_1 = gaussian_kde(source_gradients)
    if std(target_gradients) < 0.01:
        target_gradients += random.normal(size=(len(target_gradients),)) * 0.01 - 0.005
    method_2 = gaussian_kde(target_gradients)
    gradients = linspace(0, 1, 101)
    distribution_1 = method_1.evaluate(gradients)
    distribution_2 = method_2.evaluate(gradients)
    distribution_1, distribution_2 = distribution_1 / sum(distribution_1), distribution_2 / sum(distribution_2)
    normalized_value = max([sum(distribution_1), sum(distribution_2)])
    distribution_1, distribution_2 = distribution_1 / normalized_value, distribution_2 / normalized_value
    pyplot.plot(gradients, distribution_1, color="#31A354", linewidth=1.5)
    pyplot.fill_between(gradients, 0, distribution_1, color="#31A354", alpha=0.5)
    pyplot.plot(gradients, distribution_2, color="#E6550D", linewidth=1.5)
    pyplot.fill_between(gradients, 0, distribution_2, color="#E6550D", alpha=0.5)
    legends = [patches.Patch(facecolor="#98D2AA", edgecolor="#31A354", linewidth=1,
                             label="incoherent-loop motif"),
               patches.Patch(facecolor="#F3AB85", edgecolor="#E6550D", linewidth=1,
                             label="collider motif")]
    pyplot.legend(handles=legends, loc="upper right", fontsize=9)
    pyplot.xlabel("normalized gradient")
    pyplot.xlim(0.4, 0.9)
    pyplot.xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    pyplot.ylabel("frequency")
    pyplot.ylim(0, 0.5)
    pyplot.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0%", "10%", "20%", "30%", "40%", "50%"])

    figure.text(0.02, 0.99, "A", va="center", ha="center", fontsize=10)
    figure.text(0.27, 0.99, "B", va="center", ha="center", fontsize=10)
    figure.text(0.51, 0.99, "C", va="center", ha="center", fontsize=10)

    pyplot.savefig("./results/figures/[02] hypothesis.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()
