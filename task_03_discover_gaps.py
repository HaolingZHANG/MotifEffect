from itertools import product
from torch import manual_seed
from matplotlib import pyplot, markers
from numpy import load, min, max, linspace, ones, arange, log10

from hypothesis import prepare_data, prepare_motifs, calculate_similarity


if __name__ == "__main__":
    seed, samples, value_range, points, loss_threshold, iteration_threshold = 2022, 1, (-1, +1), 41, 1e-6, 40

    manual_seed(seed=seed)

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
                         loss_threshold=loss_threshold, iteration_threshold=iteration_threshold,
                         save_path="./results/data/cases/", seed=seed, processes=1)

    manual_seed(seed=seed)

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
                source_motifs = prepare_motifs(motif_type="coherent-loop", motif_index=motif_index,
                                               activations=activations, aggregations=aggregations,
                                               sample=samples, weights=None, biases=None)
                source_group.append(source_motifs)

    calculate_similarity(value_range=value_range, points=points,
                         source_motif_group=source_group, target_motif_group=target_group,
                         loss_threshold=loss_threshold, iteration_threshold=iteration_threshold,
                         save_path="./results/data/cases/", seed=seed, processes=1)

    draw_info = {"collider": ([0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1, 2], [3], []),
                 "incoherent-loop": ([0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2])}

    example = load("./results/data/cases/incoherent-loop 1 (relu tanh) (max sum).pkl", allow_pickle=True)[1]

    input_range, points = (-1, +1), 41
    database = prepare_data(value_range=input_range, points=points)

    figure = pyplot.figure(figsize=(10, 9), tight_layout=True)
    grid = pyplot.GridSpec(4, 4)

    pyplot.subplot(grid[0, 0])
    source_motif = example[0][0]
    source_i_data = source_motif(database).reshape(points, points).detach().numpy()
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
    pyplot.text(x=(info[0][0] + info[0][1]) / 2.0, y=info[1][0] - 0.06,
                s="%.1e" % source_motif.w[0].value(), fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[0][0] + info[0][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0,
                s="%.1e" % source_motif.w[1].value(), fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[0][1] + info[0][2]) / 2.0 + 0.03, y=(info[1][1] + info[1][2]) / 2.0,
                s="%.1e" % source_motif.w[2].value(), fontsize=9, va="bottom", ha="left")
    pyplot.text(x=info[0][1] + 0.1, y=info[1][1],
                s="%.1e" % source_motif.b[0].value(), fontsize=9, va="center", ha="left")
    pyplot.text(x=info[0][2] + 0.1, y=info[1][2],
                s="%.1e" % source_motif.b[1].value(), fontsize=9, va="center", ha="left")
    pyplot.text(x=info[0][1], y=info[1][1] + 0.1, s=source_motif.a[0] + "/" + source_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=source_motif.a[1] + "/" + source_motif.g[1], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.05, 0.95)
    pyplot.axis("off")

    pyplot.subplot(grid[1, 0])
    target_motif = example[0][1]
    target_i_data = target_motif(database).reshape(points, points).detach().numpy()
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
    data = [(1, 3, target_motif.w[0].value()),
            (2, 3, target_motif.w[1].value())]
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
    pyplot.text(x=(info[0][0] + info[0][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0,
                s="%.1e" % target_motif.w[0].value(), fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[0][1] + info[0][2]) / 2.0 + 0.03, y=(info[1][1] + info[1][2]) / 2.0,
                s="%.1e" % target_motif.w[1].value(), fontsize=9, va="bottom", ha="left")
    pyplot.text(x=info[0][2] + 0.1, y=info[1][2],
                s="%.1e" % target_motif.b[0].value(), fontsize=9, va="center", ha="left")
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=target_motif.a[0] + "/" + target_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.05, 0.95)
    pyplot.axis("off")

    pyplot.subplot(grid[2, 0])
    source_motif = example[-1][0]
    source_f_data = source_motif(database).reshape(points, points).detach().numpy()
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
    pyplot.text(x=(info[0][0] + info[0][1]) / 2.0, y=info[1][0] - 0.06,
                s="%.1e" % source_motif.w[0].value(), fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[0][0] + info[0][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0,
                s="%.1e" % source_motif.w[1].value(), fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[0][1] + info[0][2]) / 2.0 + 0.03, y=(info[1][1] + info[1][2]) / 2.0,
                s="%.1e" % source_motif.w[2].value(), fontsize=9, va="bottom", ha="left")
    pyplot.text(x=info[0][1] + 0.1, y=info[1][1],
                s="%.1e" % source_motif.b[0].value(), fontsize=9, va="center", ha="left")
    pyplot.text(x=info[0][2] + 0.1, y=info[1][2],
                s="%.1e" % source_motif.b[1].value(), fontsize=9, va="center", ha="left")
    pyplot.text(x=info[0][1], y=info[1][1] + 0.1, s=source_motif.a[0] + "/" + source_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=source_motif.a[1] + "/" + source_motif.g[1], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.05, 0.95)
    pyplot.axis("off")

    pyplot.subplot(grid[3, 0])
    target_motif = example[-1][1]
    target_f_data = target_motif(database).reshape(points, points).detach().numpy()
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
    data = [(1, 3, target_motif.w[0].value()), (2, 3, target_motif.w[1].value())]
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
    pyplot.text(x=(info[0][0] + info[0][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0,
                s="%.1e" % target_motif.w[0].value(), fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[0][1] + info[0][2]) / 2.0 + 0.03, y=(info[1][1] + info[1][2]) / 2.0,
                s="%.1e" % target_motif.w[1].value(), fontsize=9, va="bottom", ha="left")
    pyplot.text(x=info[0][2] + 0.1, y=info[1][2],
                s="%.1e" % target_motif.b[0].value(), fontsize=9, va="center", ha="left")
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=target_motif.a[0] + "/" + target_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.05, 0.95)
    pyplot.axis("off")

    output_range = (min([source_i_data, source_f_data, target_i_data, target_f_data]),
                    max([source_i_data, source_f_data, target_i_data, target_f_data]))

    pyplot.subplot(grid[0, 1])
    pyplot.pcolormesh(arange(points), arange(points), source_i_data, cmap="rainbow",
                      vmin=output_range[0], vmax=output_range[1])
    pyplot.xlabel("input x")
    pyplot.ylabel("input y")
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])

    pyplot.subplot(grid[1, 1])
    pyplot.pcolormesh(arange(points), arange(points), target_i_data, cmap="rainbow",
                      vmin=output_range[0], vmax=output_range[1])
    pyplot.xlabel("input x")
    pyplot.ylabel("input y")
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])

    pyplot.subplot(grid[2, 1])
    pyplot.pcolormesh(arange(points), arange(points), source_f_data, cmap="rainbow",
                      vmin=output_range[0], vmax=output_range[1])
    pyplot.xlabel("input x")
    pyplot.ylabel("input y")
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])

    pyplot.subplot(grid[3, 1])
    pyplot.pcolormesh(arange(points), arange(points), target_f_data, cmap="rainbow",
                      vmin=output_range[0], vmax=output_range[1])
    pyplot.xlabel("input x")
    pyplot.ylabel("input y")
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["-1", "0", "+1"])

    pyplot.subplot(grid[:2, 2:])
    counts, values = [], []
    matrix = ones(shape=(100, 100)) * -6
    for iteration, (_, _, records, _) in enumerate(example):
        values.append(records[-1])
        counts.append(len(records))
        if len(records) < 100:
            records += [records[-1] for _ in range(100 - len(records))]
        matrix[iteration, : len(records)] = log10(records)
    maximum_value = max(matrix)
    pyplot.pcolormesh(arange(101), arange(101), matrix.T, vmin=-4, vmax=-1, cmap="rainbow")
    for location, count in enumerate(counts):
        pyplot.fill_between([location, location + 1], count, 100, color="white")

    gradient_colors = pyplot.get_cmap(name="rainbow")(linspace(0, 1, 24))
    for location, gradient_color in enumerate(gradient_colors):
        pyplot.fill_between([38 + location, 38 + location + 1], 90, 95, color=gradient_color)
    pyplot.text(x=50, y=96, s="Huber loss", va="bottom", ha="center", fontsize=8)
    pyplot.vlines(38, 90, 95, color="black", linewidth=0.75)
    pyplot.vlines(62, 90, 95, color="black", linewidth=0.75)
    pyplot.hlines(90, 38, 62, color="black", linewidth=0.75)
    pyplot.hlines(95, 38, 62, color="black", linewidth=0.75)
    pyplot.text(x=38, y=89, s="1E-4", va="top", ha="center", fontsize=8)
    pyplot.text(x=50, y=89, s="1E-2", va="top", ha="center", fontsize=8)
    pyplot.text(x=62, y=89, s="1E+0", va="top", ha="center", fontsize=8)
    pyplot.annotate(s="A", xy=(0, counts[0]), xytext=(10, 95),
                    arrowprops=dict(arrowstyle="-|>", color="black", shrinkA=3, shrinkB=3, lw=1), zorder=2)
    pyplot.annotate(s="C", xy=(100, counts[-1]), xytext=(90, 95),
                    arrowprops=dict(arrowstyle="-|>", color="black", shrinkA=3, shrinkB=3, lw=1), zorder=2)

    pyplot.xlabel("iteration of maximum loss search")
    pyplot.xlim(0, 100)
    pyplot.xticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100])
    pyplot.ylabel("iteration of minimum loss search")
    pyplot.ylim(0, 100)
    pyplot.yticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", " 100"])

    pyplot.subplot(grid[2:, 2:])
    pyplot.plot(arange(101), [0] + values, color="black", linewidth=0.75)
    pyplot.fill_between(arange(101), 0, [0] + values, color="silver")
    pyplot.xlabel("iteration of maximum loss search")
    pyplot.xlim(0, 100)
    pyplot.xticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100])
    pyplot.ylabel("minimum loss value")
    pyplot.ylim(0, 0.008)
    pyplot.yticks([0, 0.002, 0.004, 0.006, 0.008], ["0E-3", "2E-3", "4E-3", "6E-3", "8E-3"])

    figure.text(0.01, 0.99, "A", va="center", ha="center", fontsize=10)
    figure.text(0.49, 0.99, "B", va="center", ha="center", fontsize=10)
    figure.text(0.01, 0.50, "C", va="center", ha="center", fontsize=10)
    figure.text(0.49, 0.50, "D", va="center", ha="center", fontsize=10)

    pyplot.savefig("./results/figures/[03] search pipeline.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()
