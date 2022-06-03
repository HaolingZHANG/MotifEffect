from scipy.stats import gaussian_kde
from matplotlib import pyplot, markers
from numpy import load, zeros, ones, arange, linspace, max, log10

from hypothesis import Monitor, NeuralMotif, prepare_data, calculate_gradients, intervene_entrances, acyclic_motifs


def main_01():
    input_range, points = (-1, +1), 41
    source_motif = NeuralMotif(motif_type="incoherent-loop", motif_index=2,
                               activations=("relu", "relu"), aggregations=("max", "max"),
                               weights=[0.90827476978302, 0.5210987329483032, -0.8107407093048096],
                               biases=[0.9697856903076172, 0.9693939685821533])
    target_motif = NeuralMotif(motif_type="collider", motif_index=2,
                               activations=("relu",), aggregations=("sum",),
                               weights=[0.4955233335494995, -0.04083000123500824],
                               biases=[0.9734159708023071])
    draw_info = {"collider": ([0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1, 2], [3], []),
                 "incoherent-loop": ([0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2])}
    figure = pyplot.figure(figsize=(10, 4), tight_layout=True)
    grid = pyplot.GridSpec(2, 4)

    pyplot.subplot(grid[0, 0])
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
    pyplot.text(x=info[0][1], y=info[1][1] + 0.1, s=source_motif.a[0] + "/" + source_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=source_motif.a[1] + "/" + source_motif.g[1], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0, 1)
    pyplot.ylim(0.1, 0.9)
    pyplot.axis("off")

    pyplot.subplot(grid[1, 0])
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
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=target_motif.a[0] + "/" + target_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0, 1)
    pyplot.ylim(0.1, 0.9)
    pyplot.axis("off")

    data = prepare_data(value_range=input_range, points=points)
    source_data = source_motif(data).reshape(points, points).detach().numpy()
    target_data = target_motif(data).reshape(points, points).detach().numpy()

    pyplot.subplot(grid[0, 1])
    mesh = pyplot.pcolormesh(arange(points), arange(points), source_data, shading="gouraud",
                             cmap="rainbow", vmin=0.4, vmax=1.6)
    bar = pyplot.colorbar(mesh, ticks=[0.4, 1.0, 1.6])
    bar.set_label("output value", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    pyplot.xlabel("input x", fontsize=9)
    pyplot.ylabel("input y", fontsize=9)
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)

    pyplot.subplot(grid[1, 1])
    mesh = pyplot.pcolormesh(arange(points), arange(points), target_data, shading="gouraud",
                             cmap="rainbow", vmin=0.4, vmax=1.6)
    bar = pyplot.colorbar(mesh, ticks=[0.4, 1.0, 1.6])
    bar.set_label("output value", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    pyplot.xlabel("input x", fontsize=9)
    pyplot.ylabel("input y", fontsize=9)
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)

    source_gradients = calculate_gradients(value_range=input_range, points=points, motif=source_motif)
    target_gradients = calculate_gradients(value_range=input_range, points=points, motif=target_motif)

    pyplot.subplot(grid[0, 2])
    mesh = pyplot.pcolormesh(arange(points), arange(points), source_gradients,
                             shading="gouraud", cmap="rainbow", vmin=0, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[0, 0.5, 1])
    bar.set_label("gradient length", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    pyplot.xlabel("input x", fontsize=9)
    pyplot.ylabel("input y", fontsize=9)
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)

    pyplot.subplot(grid[1, 2])
    mesh = pyplot.pcolormesh(arange(points), arange(points), target_gradients,
                             shading="gouraud", cmap="rainbow", vmin=0, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[0, 0.5, 1])
    bar.set_label("gradient length", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    pyplot.xlabel("input x", fontsize=9)
    pyplot.ylabel("input y", fontsize=9)
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)

    source_differences = intervene_entrances(value_range=(-1.0, +1.0), times=41, scales=[0.2], motif=source_motif)
    target_differences = intervene_entrances(value_range=(-1.0, +1.0), times=41, scales=[0.2], motif=target_motif)

    pyplot.subplot(grid[0, 3])
    mesh = pyplot.pcolormesh(arange(points), arange(points), source_differences,
                             shading="gouraud", cmap="hot", vmin=0, vmax=0.4)
    bar = pyplot.colorbar(mesh, ticks=[0, 0.2, 0.4])
    bar.set_label("maximum error", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    pyplot.xlabel("input x", fontsize=9)
    pyplot.ylabel("input y", fontsize=9)
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)

    pyplot.subplot(grid[1, 3])
    mesh = pyplot.pcolormesh(arange(points), arange(points), target_differences,
                             shading="gouraud", cmap="hot", vmin=0, vmax=0.4)
    bar = pyplot.colorbar(mesh, ticks=[0, 0.2, 0.4])
    bar.set_label("maximum error", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    pyplot.xlabel("input x", fontsize=9)
    pyplot.ylabel("input y", fontsize=9)
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=8)

    figure.text(0.021, 0.98, "A", va="center", ha="center", fontsize=10)
    figure.text(0.021, 0.50, "B", va="center", ha="center", fontsize=10)

    pyplot.savefig("./results/figures/[M01] hypothesis.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_02():
    pass


def main_03():
    value_range, points = (-1, +1), 41
    draw_info = {"collider": ([0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1, 2], [3], []),
                 "incoherent-loop": ([0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2])}
    example = load("./results/data/cases/incoherent-loop 1 (relu tanh) (max sum).pkl", allow_pickle=True)[1]
    database = prepare_data(value_range=value_range, points=points)
    gradient_matrix, monitor = zeros(shape=(101, 101)), Monitor()
    print("collect the gradient data.")
    for index, (source_motif, target_motif, _, _) in enumerate(example):
        gradients = calculate_gradients(value_range=value_range, points=points, motif=source_motif)
        gradient_distribution = gaussian_kde(dataset=gradients.reshape(-1)).evaluate(linspace(0, 1.6, 101))
        gradient_matrix[index + 1] = gradient_distribution / max(gradient_distribution)
        monitor.output(index + 1, 100)

    figure = pyplot.figure(figsize=(10, 10), tight_layout=True)
    grid = pyplot.GridSpec(5, 4)
    pyplot.subplot(grid[:2, :])
    counts, values = [], []
    matrix = ones(shape=(100, 100)) * -6
    for iteration, (_, _, records, _) in enumerate(example):
        values.append(records[-1])
        counts.append(len(records))
        if len(records) < 100:
            records += [records[-1] for _ in range(100 - len(records))]
        matrix[iteration, :len(records)] = log10(records)
    pyplot.pcolormesh(arange(101), arange(101), matrix.T, vmin=-4, vmax=-1, cmap="spring")
    for location, count in enumerate(counts):
        pyplot.fill_between([location, location + 1], count, 100, color="white")
    gradient_colors = pyplot.get_cmap(name="spring")(linspace(0, 1, 24))
    for location, gradient_color in enumerate(gradient_colors):
        pyplot.fill_between([40 + location, 40 + location + 1], 90, 94, color=gradient_color)
    pyplot.text(x=52, y=95, s="Huber loss (delta=0.05)", va="bottom", ha="center", fontsize=8)
    pyplot.vlines(40, 90, 94, color="black", linewidth=0.75)
    pyplot.vlines(64, 90, 94, color="black", linewidth=0.75)
    pyplot.hlines(90, 40, 64, color="black", linewidth=0.75)
    pyplot.hlines(94, 40, 64, color="black", linewidth=0.75)
    pyplot.vlines(40, 89, 90, color="black", linewidth=0.75)
    pyplot.vlines(48, 89, 90, color="black", linewidth=0.75)
    pyplot.vlines(56, 89, 90, color="black", linewidth=0.75)
    pyplot.vlines(64, 89, 90, color="black", linewidth=0.75)
    pyplot.text(x=40, y=88, s="1e-04", va="top", ha="center", fontsize=8)
    pyplot.text(x=48, y=88, s="1e-03", va="top", ha="center", fontsize=8)
    pyplot.text(x=56, y=88, s="1e-02", va="top", ha="center", fontsize=8)
    pyplot.text(x=64, y=88, s="1e-01", va="top", ha="center", fontsize=8)
    pyplot.annotate(s="D", xy=(0, counts[0]), xytext=(10, 95),
                    arrowprops=dict(arrowstyle="-|>", color="black", shrinkA=3, shrinkB=10, lw=1), zorder=2)
    pyplot.annotate(s="E", xy=(100, counts[-1]), xytext=(90, 95),
                    arrowprops=dict(arrowstyle="-|>", color="black", shrinkA=3, shrinkB=10, lw=1), zorder=2)
    pyplot.xlabel("iteration of maximum search\n")
    pyplot.xlim(0, 100)
    pyplot.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    pyplot.ylabel("iteration of minimum search")
    pyplot.ylim(0, 100)
    pyplot.yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "   100"])

    pyplot.subplot(grid[2, :2])
    pyplot.plot(arange(101), [0] + values, color="black", linewidth=0.75)
    pyplot.fill_between(arange(101), 0, [0] + values, color="silver")
    pyplot.xlabel("iteration of maximum search")
    pyplot.xlim(0, 100)
    pyplot.xticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100])
    pyplot.ylabel("minimum loss")
    pyplot.ylim(0, 0.008)
    pyplot.yticks([0, 0.004, 0.008], ["0.000", "0.004", "0.008"])

    pyplot.subplot(grid[2, 2:])
    pyplot.pcolormesh(arange(101), arange(101), gradient_matrix.T, shading="gouraud", cmap="binary", vmin=0, vmax=1)
    pyplot.xlabel("iteration of maximum search")
    pyplot.xlim(0, 100)
    pyplot.xticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100])
    pyplot.ylabel("gradient module")
    pyplot.ylim(0, 100)
    pyplot.yticks([0, 50, 100], ["0.000", "1.000", "2.000"])
    gradient_colors = pyplot.get_cmap(name="binary")(linspace(0, 1, 24))
    for location, gradient_color in enumerate(gradient_colors):
        pyplot.fill_between([38 + location, 38 + location + 1], 82, 88, color=gradient_color)
    pyplot.text(x=50, y=90, s="normalized aggregation level", va="bottom", ha="center", fontsize=8)
    pyplot.vlines(38, 81, 88, color="black", linewidth=0.75)
    pyplot.vlines(62, 81, 88, color="black", linewidth=0.75)
    pyplot.hlines(88, 38, 62, color="black", linewidth=0.75)
    pyplot.hlines(82, 38, 62, color="black", linewidth=0.75)
    pyplot.vlines(50, 81, 82, color="black", linewidth=0.75)
    pyplot.text(x=38, y=80, s="0%", va="top", ha="center", fontsize=8)
    pyplot.text(x=50, y=80, s="50%", va="top", ha="center", fontsize=8)
    pyplot.text(x=62, y=80, s="100%", va="top", ha="center", fontsize=8)

    pyplot.subplot(grid[3, 0])
    source_motif = example[0][0]
    source_i_data = source_motif(database).reshape(points, points).detach().numpy()
    info = draw_info[source_motif.t]
    for index, (px, py) in enumerate(zip(info[0], info[1])):
        if index + 1 in info[2]:
            pyplot.scatter(px, py, color="white", edgecolor="black", lw=1, s=60, zorder=2)
        elif index + 1 in info[3]:
            pyplot.scatter(px, py, color="black", edgecolor="black", lw=1, s=60, zorder=2)
        elif index + 1 in info[4]:
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                           color="white", edgecolor="black", lw=1, s=60, zorder=2)
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                           color="gray", edgecolor="black", lw=1, s=60, zorder=2)
        else:
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1, s=60, zorder=2)
    x, y = info[0], info[1]
    for former, latter, weight in [(1, 2, source_motif.w[0].value()), (1, 3, source_motif.w[1].value()),
                                   (2, 3, source_motif.w[2].value())]:
        if weight > 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
        elif weight < 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
    pyplot.text(x=info[0][0], y=info[1][0] - 0.06, s="x", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[0][1], y=info[1][1] - 0.06, s="y", fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[0][0] + info[0][1]) / 2.0, y=info[1][0] - 0.06,
                s="%.0e" % source_motif.w[0].value(), fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[0][0] + info[0][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0,
                s="%.0e" % source_motif.w[1].value(), fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[0][1] + info[0][2]) / 2.0 + 0.03, y=(info[1][1] + info[1][2]) / 2.0,
                s="%.0e" % source_motif.w[2].value(), fontsize=9, va="bottom", ha="left")
    pyplot.text(x=info[0][1], y=info[1][1] + 0.1, s=source_motif.a[0] + "/" + source_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=source_motif.a[1] + "/" + source_motif.g[1], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.1, 0.9)
    pyplot.axis("off")

    pyplot.subplot(grid[4, 0])
    target_motif = example[0][1]
    target_i_data = target_motif(database).reshape(points, points).detach().numpy()
    info = draw_info[target_motif.t]
    for index, (px, py) in enumerate(zip(info[0], info[1])):
        if index + 1 in info[2]:
            pyplot.scatter(px, py, color="white", edgecolor="black", lw=1, s=60, zorder=2)
        elif index + 1 in info[3]:
            pyplot.scatter(px, py, color="black", edgecolor="black", lw=1, s=60, zorder=2)
        elif index + 1 in info[4]:
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                           color="white", edgecolor="black", lw=1, s=60, zorder=2)
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                           color="gray", edgecolor="black", lw=1, s=60, zorder=2)
        else:
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1, s=60, zorder=2)
    x, y = info[0], info[1]
    for former, latter, weight in [(1, 3, target_motif.w[0].value()), (2, 3, target_motif.w[1].value())]:
        if weight > 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
        elif weight < 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
    pyplot.text(x=info[0][0], y=info[1][0] - 0.06, s="x", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[0][1], y=info[1][1] - 0.06, s="y", fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[0][0] + info[0][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0,
                s="%.0e" % target_motif.w[0].value(), fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[0][1] + info[0][2]) / 2.0 + 0.03, y=(info[1][1] + info[1][2]) / 2.0,
                s="%.0e" % target_motif.w[1].value(), fontsize=9, va="bottom", ha="left")
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=target_motif.a[0] + "/" + target_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.1, 0.9)
    pyplot.axis("off")

    pyplot.subplot(grid[3, 2])
    source_motif = example[-1][0]
    source_f_data = source_motif(database).reshape(points, points).detach().numpy()
    info = draw_info[source_motif.t]
    for index, (px, py) in enumerate(zip(info[0], info[1])):
        if index + 1 in info[2]:
            pyplot.scatter(px, py, color="white", edgecolor="black", lw=1, s=60, zorder=2)
        elif index + 1 in info[3]:
            pyplot.scatter(px, py, color="black", edgecolor="black", lw=1, s=60, zorder=2)
        elif index + 1 in info[4]:
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                           color="white", edgecolor="black", lw=1, s=60, zorder=2)
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                           color="gray", edgecolor="black", lw=1, s=60, zorder=2)
        else:
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1, s=60, zorder=2)
    x, y = info[0], info[1]
    for former, latter, weight in [(1, 2, source_motif.w[0].value()), (1, 3, source_motif.w[1].value()),
                                   (2, 3, source_motif.w[2].value())]:
        if weight > 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
        elif weight < 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
    pyplot.text(x=info[0][0], y=info[1][0] - 0.06, s="x", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[0][1], y=info[1][1] - 0.06, s="y", fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[0][0] + info[0][1]) / 2.0, y=info[1][0] - 0.06,
                s="%.0e" % source_motif.w[0].value(), fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[0][0] + info[0][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0,
                s="%.0e" % source_motif.w[1].value(), fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[0][1] + info[0][2]) / 2.0 + 0.03, y=(info[1][1] + info[1][2]) / 2.0,
                s="%.0e" % source_motif.w[2].value(), fontsize=9, va="bottom", ha="left")
    pyplot.text(x=info[0][1], y=info[1][1] + 0.1, s=source_motif.a[0] + "/" + source_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=source_motif.a[1] + "/" + source_motif.g[1], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.1, 0.9)
    pyplot.axis("off")

    pyplot.subplot(grid[4, 2])
    target_motif = example[-1][1]
    target_f_data = target_motif(database).reshape(points, points).detach().numpy()
    info = draw_info[target_motif.t]
    for index, (px, py) in enumerate(zip(info[0], info[1])):
        if index + 1 in info[2]:
            pyplot.scatter(px, py, color="white", edgecolor="black", lw=1, s=60, zorder=2)
        elif index + 1 in info[3]:
            pyplot.scatter(px, py, color="black", edgecolor="black", lw=1, s=60, zorder=2)
        elif index + 1 in info[4]:
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                           color="white", edgecolor="black", lw=1, s=60, zorder=2)
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                           color="gray", edgecolor="black", lw=1, s=60, zorder=2)
        else:
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1, s=60, zorder=2)
    x, y = info[0], info[1]
    for former, latter, weight in [(1, 3, target_motif.w[0].value()), (2, 3, target_motif.w[1].value())]:
        if weight > 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
        elif weight < 0:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
    pyplot.text(x=info[0][0], y=info[1][0] - 0.06, s="x", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[0][1], y=info[1][1] - 0.06, s="y", fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[0][0] + info[0][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0,
                s="%.0e" % target_motif.w[0].value(), fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[0][1] + info[0][2]) / 2.0 + 0.03, y=(info[1][1] + info[1][2]) / 2.0,
                s="%.0e" % target_motif.w[1].value(), fontsize=9, va="bottom", ha="left")
    pyplot.text(x=info[0][2], y=info[1][2] + 0.1, s=target_motif.a[0] + "/" + target_motif.g[0], fontsize=9,
                va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.1, 0.9)
    pyplot.axis("off")

    pyplot.subplot(grid[3, 1])
    mesh = pyplot.pcolormesh(arange(points), arange(points), source_i_data, cmap="rainbow", vmin=-1, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[-1, 0, 1])
    bar.ax.set_yticklabels(["\N{MINUS SIGN}1", " 0", "+1"])
    pyplot.xlabel("input x")
    pyplot.ylabel("input y")
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"])
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"])

    pyplot.subplot(grid[4, 1])
    mesh = pyplot.pcolormesh(arange(points), arange(points), target_i_data, cmap="rainbow", vmin=-1, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[-1, 0, 1])
    bar.ax.set_yticklabels(["\N{MINUS SIGN}1", " 0", "+1"])
    pyplot.xlabel("input x")
    pyplot.ylabel("input y")
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"])
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"])

    pyplot.subplot(grid[3, 3])
    mesh = pyplot.pcolormesh(arange(points), arange(points), source_f_data, cmap="rainbow", vmin=-1, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[-1, 0, 1])
    bar.ax.set_yticklabels(["\N{MINUS SIGN}1", " 0", "+1"])
    pyplot.xlabel("input x")
    pyplot.ylabel("input y")
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"])
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"])

    pyplot.subplot(grid[4, 3])
    mesh = pyplot.pcolormesh(arange(points), arange(points), target_f_data, cmap="rainbow", vmin=-1, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[-1, 0, 1])
    bar.ax.set_yticklabels(["\N{MINUS SIGN}1", " 0", "+1"])
    pyplot.xlabel("input x")
    pyplot.ylabel("input y")
    pyplot.xlim(0, points - 1)
    pyplot.ylim(0, points - 1)
    pyplot.xticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"])
    pyplot.yticks([0, (points - 1) // 2, points - 1], ["\N{MINUS SIGN}1", "0", "+1"])

    figure.text(0.021, 0.99, "A", va="center", ha="center", fontsize=10)
    figure.text(0.021, 0.60, "B", va="center", ha="center", fontsize=10)
    figure.text(0.514, 0.60, "C", va="center", ha="center", fontsize=10)
    figure.text(0.021, 0.40, "D", va="center", ha="center", fontsize=10)
    figure.text(0.514, 0.40, "E", va="center", ha="center", fontsize=10)

    pyplot.savefig("./results/figures/[M03] search pipeline.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_01():
    draw_info = {
        "collider":
            ("#86E3CE", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1, 2], [3], []),
        "fork":
            ("#D0E6A5", [0.5, 0.2, 0.8], [0.20, 0.70, 0.70], [1], [2, 3], []),
        "chain":
            ("#FFDD94", [0.2, 0.5, 0.8], [0.20, 0.45, 0.70], [1], [3], []),
        "coherent-loop":
            ("#CCABD8", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2]),
        "incoherent-loop":
            ("#FA897B", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2])
    }

    pyplot.figure(figsize=(10, 8))
    pyplot.subplots_adjust(wspace=0, hspace=0)
    grid = pyplot.GridSpec(4, 5)
    for type_index, (motif_type, motifs) in enumerate(acyclic_motifs.items()):
        info = draw_info[motif_type]
        for motif_index, motif in enumerate(motifs):
            pyplot.subplot(grid[motif_index, type_index])
            if motif_index == 0:
                pyplot.title(motif_type.replace("-", " "))
            for index, (px, py) in enumerate(zip(info[1], info[2])):
                if index + 1 in info[3]:
                    pyplot.scatter(px, py, color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
                elif index + 1 in info[4]:
                    pyplot.scatter(px, py, color="black", edgecolor="black", lw=1.5, s=120, zorder=2)
                elif index + 1 in info[5]:
                    pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                                   color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
                    pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                                   color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
                else:
                    pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)

            x, y = info[1], info[2]
            for former, latter in motif.edges:
                if motif.get_edge_data(former, latter)["weight"] == 1:
                    pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                                    arrowprops=dict(arrowstyle="-|>", color="black",
                                                    shrinkA=6, shrinkB=6, lw=1.5), zorder=2)
                elif motif.get_edge_data(former, latter)["weight"] == -1:
                    pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                                    arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                                    shrinkA=6, shrinkB=6, lw=1.5), zorder=2)

            pyplot.fill_between([0, 1], [0], [1], color=info[0], zorder=1)
            pyplot.xlim(0, 1)
            pyplot.ylim(0, 1)
            pyplot.xticks([])
            pyplot.yticks([])

    pyplot.subplot(grid[3, 0: 2])
    pyplot.scatter([0.2], [0.2], color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.4], marker=markers.MarkerStyle("o", fillstyle="right"),
                   color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.4], marker=markers.MarkerStyle("o", fillstyle="left"),
                   color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.6], color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.8], color="black", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.annotate(s="", xy=(1.8, 0.7), xytext=(1.3, 0.7),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    shrinkA=0, shrinkB=0, lw=1.5), zorder=2)
    pyplot.annotate(s="", xy=(1.8, 0.3), xytext=(1.3, 0.3),
                    arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                    shrinkA=0, shrinkB=0, lw=1.5), zorder=2)
    pyplot.text(0.3, 0.8, "output node", ha="left", va="center")
    pyplot.text(0.3, 0.6, "hidden node", ha="left", va="center")
    pyplot.text(0.3, 0.4, "hidden / input node", ha="left", va="center")
    pyplot.text(0.3, 0.2, "input node", ha="left", va="center")
    pyplot.text(1.52, 0.7, "positive\n\neffect", ha="center", va="center")
    pyplot.text(1.52, 0.3, "negative\n\neffect", ha="center", va="center")
    pyplot.xlim(0, 2)
    pyplot.ylim(0, 1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.axis("off")

    pyplot.savefig("./results/figures/[S01] acyclic neural motifs.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    # main_02()
    supp_01()
