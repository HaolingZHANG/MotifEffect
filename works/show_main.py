from matplotlib import pyplot, rcParams, markers, patches, lines, colors
from numpy import array, arange, zeros, ones, linspace
from numpy import hstack, min, max, mean, argsort, sum, sin, clip, percentile, log10, where, inf, pi
from scipy.stats import gaussian_kde, spearmanr

from effect import NeuralMotif, calculate_landscape, estimate_lipschitz

from works import load_data, acyclic_motifs, draw_info, adjust_format


def main01():
    pyplot.figure(figsize=(10, 4))
    grid = pyplot.GridSpec(4, 10)
    pyplot.subplots_adjust(wspace=0, hspace=0)
    rcParams["font.family"] = "Times New Roman"

    pyplot.subplot(grid[:2, :2])
    motif, info = acyclic_motifs["incoherent-loop"][1], draw_info["incoherent-loop"]
    for index, (px, py) in enumerate(zip(info[1], info[2])):
        if index + 1 in info[3]:
            pyplot.scatter(px, py, color="white", edgecolor="black", lw=1, s=80, zorder=2)
        elif index + 1 in info[4]:
            pyplot.scatter(px, py, color="black", edgecolor="black", lw=1, s=80, zorder=2)
        elif index + 1 in info[5]:
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                           color="white", edgecolor="black", lw=1, s=80, zorder=2)
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                           color="gray", edgecolor="black", lw=1, s=80, zorder=2)
        else:
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1, s=80, zorder=2)
    x, y = info[1], info[2]
    for former, latter in [(1, 2), (1, 3), (2, 3)]:
        if motif.get_edge_data(former, latter)["weight"] == 1:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
        elif motif.get_edge_data(former, latter)["weight"] == -1:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
    pyplot.text(x=info[1][0], y=info[2][0] - 0.06, s="x", fontsize=10, va="top", ha="center")
    pyplot.text(x=info[1][1], y=info[2][1] - 0.06, s="y", fontsize=10, va="top", ha="center")
    pyplot.text(x=info[1][2], y=info[2][2] + 0.06, s="z", fontsize=10, va="bottom", ha="center")
    pyplot.text(-0.05, 0.45, "incoherent loop", va="center", ha="center", fontsize=10, rotation=90)
    pyplot.xlim(-0.10, 1.0)
    pyplot.ylim(-0.05, 1.0)
    pyplot.axis("off")

    pyplot.subplot(grid[2:4, :2])
    motif, info = acyclic_motifs["collider"][1], draw_info["collider"]
    for index, (px, py) in enumerate(zip(info[1], info[2])):
        if index + 1 in info[3]:
            pyplot.scatter(px, py, color="white", edgecolor="black", lw=1, s=80, zorder=2)
        elif index + 1 in info[4]:
            pyplot.scatter(px, py, color="black", edgecolor="black", lw=1, s=80, zorder=2)
        elif index + 1 in info[5]:
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                           color="white", edgecolor="black", lw=1, s=80, zorder=2)
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                           color="gray", edgecolor="black", lw=1, s=80, zorder=2)
        else:
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1, s=80, zorder=2)
    x, y = info[1], info[2]
    for former, latter in [(1, 3), (2, 3)]:
        if motif.get_edge_data(former, latter)["weight"] == 1:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
        elif motif.get_edge_data(former, latter)["weight"] == -1:
            pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
    pyplot.text(x=info[1][0], y=info[2][0] - 0.06, s="x", fontsize=10, va="top", ha="center")
    pyplot.text(x=info[1][1], y=info[2][1] - 0.06, s="y", fontsize=10, va="top", ha="center")
    pyplot.text(x=info[1][2], y=info[2][2] + 0.06, s="z", fontsize=10, va="bottom", ha="center")
    pyplot.text(-0.05, 0.45, "collider", va="center", ha="center", fontsize=10, rotation=90)
    pyplot.xlim(-0.10, 1.0)
    pyplot.ylim(-0.05, 1.0)
    pyplot.axis("off")

    def get_x():
        v1 = ones(shape=(10,)) * 0.5
        v2 = (sin(linspace(-0.5 * pi, 1.5 * pi, 40)) + 1.0) * 0.25 + 0.5
        v3 = ones(shape=(50,)) * 0.5
        return array(v1.tolist() + v2.tolist() + v3.tolist())

    def get_yi():
        v1 = ones(shape=(20,)) * 0.5
        v2 = (sin(linspace(-0.5 * pi, 1.5 * pi, 40)) + 1.0) * 0.25 + 0.5
        v3 = ones(shape=(40,)) * 0.5
        return array(v1.tolist() + v2.tolist() + v3.tolist())

    def get_yc():
        return ones(shape=(len(inputs_x),)) * 0.5

    def get_zi():
        v1 = ones(shape=(20,)) * 0.5
        v2 = (sin(linspace(-0.5 * pi, 1.5 * pi, 60)) + 1.0) * 0.15 + 0.5
        v3 = ones(shape=(20,)) * 0.5
        return array(v1.tolist() + v2.tolist() + v3.tolist())

    def get_zc():
        v1 = ones(shape=(20,)) * 0.5
        v2 = (sin(linspace(-0.5 * pi, 0.5 * pi, 30)) + 1.0) * 0.25 + 0.5
        v3 = ones(shape=(50,))
        return array(v1.tolist() + v2.tolist() + v3.tolist())

    inputs_x = get_x()
    inputs_yi = get_yi()
    inputs_yc = get_yc()
    outputs_zi = get_zi()
    outputs_zc = get_zc()

    pyplot.subplot(grid[:2, 2:6])
    pyplot.title("biological network", fontsize=10)
    pyplot.text(0.09, 0.675, "x molecule", fontsize=7, va="center", ha="right", rotation=90)
    pyplot.text(0.09, 0.325, "y molecule", fontsize=7, va="center", ha="right", rotation=90)
    pyplot.text(0.20, 0.18, "time", fontsize=7, va="top", ha="center")
    pyplot.text(0.20, 0.53, "time", fontsize=7, va="top", ha="center")
    pyplot.plot([0.1, 0.1, 0.3], [0.45, 0.20, 0.20], color="black", linewidth=0.75)
    pyplot.plot([0.1, 0.1, 0.3], [0.80, 0.55, 0.55], color="black", linewidth=0.75)
    values = 0.55 + inputs_x * 0.15
    pyplot.plot(linspace(0.1, 0.3, len(values)), values, color="k", linewidth=0.75)
    values = 0.20 + inputs_yc * 0.15
    pyplot.plot(linspace(0.1, 0.3, len(values))[20: 60], values[20: 60], color="k", linewidth=0.75, linestyle=":")
    values = 0.20 + inputs_yi * 0.15
    pyplot.plot(linspace(0.1, 0.3, len(values)), values, color="k", linewidth=0.75)
    pyplot.text(0.4, 0.53, "reaction", fontsize=7, va="bottom", ha="center")
    pyplot.annotate(s="", xy=(0.45, 0.5), xytext=(0.35, 0.5),
                    arrowprops=dict(arrowstyle="simple", edgecolor="black", facecolor="white",
                                    shrinkA=0, shrinkB=0, lw=0.75), zorder=2)
    pyplot.plot([0.55, 0.55, 0.9], [0.8, 0.2, 0.2], color="black", linewidth=0.75)
    pyplot.text(0.54, 0.50, "z molecule", fontsize=7, va="center", ha="right", rotation=90)
    values = 0.15 + outputs_zc * 0.5
    pyplot.plot(linspace(0.55, 0.90, len(values))[20:], values[20:], color="k", linewidth=0.75, linestyle=":")
    values = 0.15 + outputs_zi * 0.5
    pyplot.plot(linspace(0.55, 0.90, len(values)), values, color="k", linewidth=0.75)
    pyplot.fill_between(linspace(0.55, 0.90, len(values)), 0.4, values, color="silver")
    pyplot.text(0.725, 0.18, "time", fontsize=7, va="top", ha="center")
    pyplot.text(0.725, 0.81, "more robust", fontsize=10, color="red", va="bottom", ha="center")
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    pyplot.subplot(grid[2:4, 2:6])
    pyplot.text(0.09, 0.675, "x molecule", fontsize=7, va="center", ha="right", rotation=90)
    pyplot.text(0.09, 0.325, "y molecule", fontsize=7, va="center", ha="right", rotation=90)
    pyplot.text(0.20, 0.18, "time", fontsize=7, va="top", ha="center")
    pyplot.text(0.20, 0.53, "time", fontsize=7, va="top", ha="center")
    pyplot.plot([0.1, 0.1, 0.3], [0.45, 0.20, 0.20], color="black", linewidth=0.75)
    pyplot.plot([0.1, 0.1, 0.3], [0.80, 0.55, 0.55], color="black", linewidth=0.75)
    values = 0.55 + inputs_x * 0.15
    pyplot.plot(linspace(0.1, 0.3, len(values)), values, color="k", linewidth=0.75)
    values = 0.20 + inputs_yi * 0.15
    pyplot.plot(linspace(0.1, 0.3, len(values))[20: 60], values[20: 60], color="k", linewidth=0.75, linestyle=":")
    values = 0.20 + inputs_yc * 0.15
    pyplot.plot(linspace(0.1, 0.3, len(values)), values, color="k", linewidth=0.75)
    pyplot.text(0.4, 0.53, "reaction", fontsize=7, va="bottom", ha="center")
    pyplot.annotate(s="", xy=(0.45, 0.5), xytext=(0.35, 0.5),
                    arrowprops=dict(arrowstyle="simple", edgecolor="black", facecolor="white",
                                    shrinkA=0, shrinkB=0, lw=0.75), zorder=2)
    pyplot.plot([0.55, 0.55, 0.9], [0.8, 0.2, 0.2], color="black", linewidth=0.75)
    pyplot.text(0.54, 0.50, "z molecule", fontsize=7, va="center", ha="right", rotation=90)
    values = 0.15 + outputs_zi * 0.5
    pyplot.plot(linspace(0.55, 0.90, len(values))[20:], values[20:], color="k", linewidth=0.75, linestyle=":")
    values = 0.15 + outputs_zc * 0.5
    pyplot.plot(linspace(0.55, 0.90, len(values)), values, color="k", linewidth=0.75)
    pyplot.fill_between(linspace(0.55, 0.90, len(values)), 0.4, values, color="silver")
    pyplot.text(0.725, 0.18, "time", fontsize=7, va="top", ha="center")

    pyplot.text(0.725, 0.81, "less robust", fontsize=10, color="red", va="bottom", ha="center")
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    gradient_colors = pyplot.get_cmap(name="rainbow")(linspace(0, 1, 41))
    motif1 = NeuralMotif(motif_type="incoherent-loop", motif_index=2,
                         activations=("relu", "relu"), aggregations=("max", "max"),
                         weights=[0.90827476978302, 0.5210987329483032, -0.8107407093048096],
                         biases=[0.9697856903076172, 0.9693939685821533])
    motif2 = NeuralMotif(motif_type="collider", motif_index=2,
                         activations=("relu",), aggregations=("sum",),
                         weights=[0.4955233335494995, -0.04083000123500824],
                         biases=[0.9734159708023071])
    matrix1 = calculate_landscape(value_range=(-1, +1), points=41, motif=motif1).T
    matrix2 = calculate_landscape(value_range=(-1, +1), points=41, motif=motif2).T
    lipschitz1 = estimate_lipschitz(value_range=(-1, +1), points=41, motif=motif1, norm_type="L-2", verbose=False)
    lipschitz2 = estimate_lipschitz(value_range=(-1, +1), points=41, motif=motif2, norm_type="L-2", verbose=False)

    pyplot.subplot(grid[:2, 6:10])
    pyplot.title("artificial neural network", fontsize=10)
    c_interval = 0.6 / 41
    r_interval = c_interval * 0.5
    for r in range(41):
        for c in range(41):
            pyplot.fill_between([r * r_interval + 0.1, (r + 1) * r_interval + 0.1],
                                c * c_interval + 0.2, (c + 1) * c_interval + 0.2,
                                color=gradient_colors[int(matrix1[r, c] * 20 + 20.5)])
    pyplot.text(0.25, 0.18, "x signal", fontsize=7, va="top", ha="center")
    pyplot.text(0.09, 0.50, "y signal", fontsize=7, va="center", ha="right", rotation=90)
    pyplot.text(0.25, 0.81, "z signal", fontsize=7, va="bottom", ha="center")
    pyplot.plot([0.4, 0.1, 0.1], [0.2, 0.2, 0.8], color="black", linewidth=0.75)
    pyplot.annotate(s="", xy=(0.55, 0.5), xytext=(0.45, 0.5),
                    arrowprops=dict(arrowstyle="simple", edgecolor="black", facecolor="white",
                                    shrinkA=0, shrinkB=0, lw=0.75), zorder=2)
    pyplot.plot([0.65, 0.95, 0.65, 0.65], [0.2, 0.2, 0.8, 0.2], color="black", lw=0.75)
    pyplot.text(0.5, 0.53, "Lipschitz\nconstant", fontsize=7, va="bottom", ha="center")
    pyplot.plot([0.65, 0.95], [0.2 + 0.6 * (lipschitz2 / lipschitz1), 0.2],
                color="black", linestyle=":", linewidth=0.75, zorder=2)
    pyplot.fill_between([0.65, 0.95], [0.2, 0.2], [0.8, 0.2], color="silver")
    pyplot.text(0.8, 0.18, "input (x,y) change", fontsize=7, va="top", ha="center")
    pyplot.text(0.64, 0.5, "output (z) change", fontsize=7, va="center", ha="right", rotation=90)
    pyplot.text(0.8, 0.81, "less robust", fontsize=10, color="red", va="bottom", ha="center")
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    pyplot.subplot(grid[2:4, 6:10])
    c_interval = 0.6 / 41
    r_interval = c_interval * 0.5
    for r in range(41):
        for c in range(41):
            pyplot.fill_between([r * r_interval + 0.1, (r + 1) * r_interval + 0.1],
                                c * c_interval + 0.2, (c + 1) * c_interval + 0.2,
                                color=gradient_colors[int(matrix2[r, c] * 20 + 20.5)])
    pyplot.text(0.25, 0.18, "x signal", fontsize=7, va="top", ha="center")
    pyplot.text(0.09, 0.50, "y signal", fontsize=7, va="center", ha="right", rotation=90)
    pyplot.text(0.25, 0.81, "z signal", fontsize=7, va="bottom", ha="center")
    pyplot.plot([0.4, 0.1, 0.1], [0.2, 0.2, 0.8], color="black", linewidth=0.75)
    pyplot.annotate(s="", xy=(0.55, 0.5), xytext=(0.45, 0.5),
                    arrowprops=dict(arrowstyle="simple", edgecolor="black", facecolor="white",
                                    shrinkA=0, shrinkB=0, lw=0.75), zorder=2)
    pyplot.plot([0.65, 0.95, 0.65, 0.65], [0.2, 0.2, 0.2 + 0.6 * (lipschitz2 / lipschitz1), 0.2],
                color="black", lw=0.75)
    pyplot.text(0.5, 0.53, "Lipschitz\nconstant", fontsize=7, va="bottom", ha="center")
    pyplot.text(0.8, 0.18, "input (x,y) change", fontsize=7, va="top", ha="center")
    pyplot.text(0.64, 0.5, "output (z) change", fontsize=7, va="center", ha="right", rotation=90)
    pyplot.text(0.8, 0.81, "more robust", fontsize=10, color="red", va="bottom", ha="center")
    pyplot.plot([0.65, 0.65, 0.95], [0.2 + 0.6 * (lipschitz2 / lipschitz1), 0.8, 0.2],
                color="black", linestyle=":", linewidth=0.75)
    pyplot.fill_between([0.65, 0.95], [0.2, 0.2], [0.2 + 0.6 * (lipschitz2 / lipschitz1), 0.2], color="silver")
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    pyplot.savefig("../data/figures/main01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


# noinspection PyArgumentList,PyTypeChecker
def main02():
    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    grid = pyplot.GridSpec(2, 3)
    rcParams["font.family"] = "Times New Roman"

    pyplot.subplot(grid[:, 0])
    pyplot.bar([0], [log10(28800)], width=0.6, linewidth=0.75, color="#FA897B", edgecolor="black")
    pyplot.bar([1], [log10(480)], width=0.6, linewidth=0.75, color="#86E3CE", edgecolor="black")
    pyplot.text(0, log10(28800) + 0.02, str(28800), fontsize=10, va="bottom", ha="center")
    pyplot.text(1, log10(480) + 0.02, str(480), fontsize=10, va="bottom", ha="center")
    pyplot.xlabel("population name", fontsize=10)
    pyplot.ylabel("population size", fontsize=10)
    pyplot.xticks([0, 1], [" incoherent loop", "collider"], fontsize=10)
    pyplot.yticks([2, 3, 4, 5], ["1E+2", "1E+3", "1E+4", "1E+5"], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(2, 5)

    loop_data = load_data("../data/results/task01/lipschitz loop.npy")
    collider_data = load_data("../data/results/task01/lipschitz collider.npy")
    loop_data, collider_data = log10(loop_data[loop_data > 0]), log10(collider_data[collider_data > 0])

    pyplot.subplot(grid[:, 1])
    violin = pyplot.violinplot([loop_data], positions=[0], bw_method=0.5, showextrema=False)
    for patch in violin["bodies"]:
        patch.set_edgecolor("black")
        patch.set_facecolor("#FA897B")
        patch.set_linewidth(0.75)
        patch.set_alpha(1)
    violin = pyplot.violinplot([collider_data], positions=[1], bw_method=0.5, showextrema=False)
    for patch in violin["bodies"]:
        patch.set_edgecolor("black")
        patch.set_facecolor("#86E3CE")
        patch.set_linewidth(0.75)
        patch.set_alpha(1)
    lower_value, median_value, upper_value = percentile(loop_data, [25, 50, 75])
    pyplot.vlines(0, lower_value, upper_value, color="black", linewidth=8, zorder=2)
    pyplot.scatter([0], [median_value], color="white", s=5, zorder=3)
    lower_value, median_value, upper_value = percentile(collider_data, [25, 50, 75])
    pyplot.vlines(1, lower_value, upper_value, color="black", linewidth=8, zorder=2)
    pyplot.scatter([1], [median_value], color="white", s=5, zorder=3)
    pyplot.xlabel("population name", fontsize=10)
    pyplot.ylabel("Lipschitz constant", fontsize=10)
    pyplot.xticks([0, 1], ["incoherent loop", "collider"], fontsize=10)
    pyplot.yticks([-1, 0, 1, 2], ["1E\N{MINUS SIGN}1", "1E+0", "1E+1", "1E+2"], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(-1, 2)

    loop_data = load_data("../data/results/task01/propagation loop.npy")
    collider_data = load_data("../data/results/task01/propagation collider.npy")

    pyplot.subplot(grid[0, 2])
    pyplot.text(12.5, 12.5, "incoherent loop", color="white", va="top", ha="right", fontsize=10)
    mesh = pyplot.pcolormesh(arange(14), arange(14), loop_data[:13, :13], cmap="RdYlGn_r", vmin=0, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bar.set_label("maximum z error rate", fontsize=10)
    bar.ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    bar.ax.tick_params(labelsize=10)
    pyplot.xlabel("x error rate", fontsize=10)
    pyplot.ylabel("y error rate", fontsize=10)
    pyplot.xticks([0.5, 4.5, 8.5, 12.5], ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.yticks([0.5, 4.5, 8.5, 12.5], ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.xlim(0, 13)
    pyplot.ylim(0, 13)

    pyplot.subplot(grid[1, 2])
    pyplot.text(12.5, 12.5, "collider", color="white", va="top", ha="right", fontsize=10)
    mesh = pyplot.pcolormesh(arange(14), arange(14), collider_data[:13, :13], cmap="RdYlGn_r", vmin=0, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bar.set_label("maximum z error rate", fontsize=10)
    bar.ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    bar.ax.tick_params(labelsize=10)
    pyplot.xlabel("x error rate", fontsize=10)
    pyplot.ylabel("y error rate", fontsize=10)
    pyplot.xticks([0.5, 4.5, 8.5, 12.5], ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.yticks([0.5, 4.5, 8.5, 12.5], ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.xlim(0, 13)
    pyplot.ylim(0, 13)

    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.338, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.660, 0.99, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/main02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main03():
    figure = pyplot.figure(figsize=(10, 3.5), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"

    locations = load_data("../data/results/task01/locations.npy")
    pyplot.subplot(1, 3, 1)
    motif_types = hstack((zeros(shape=(28800,)), ones(shape=(480,))))
    color_map = colors.LinearSegmentedColormap.from_list("twp", ["#86E3CE", "#FA897B"], N=2)
    mesh = pyplot.scatter(locations[:, 0], locations[:, 1], c=motif_types, cmap=color_map, vmin=0, vmax=1, s=10)
    pyplot.vlines(0, -120, 120, color="black", linewidth=0.75, linestyle=":", zorder=2)
    pyplot.hlines(0, -120, 120, color="black", linewidth=0.75, linestyle=":", zorder=2)
    bar = pyplot.colorbar(mesh, ticks=[0.25, 0.75], orientation="horizontal")
    bar.set_label("motif type", fontsize=10)
    bar.ax.set_xticklabels(["incoherent loop", "collider"])
    bar.ax.tick_params(labelsize=10)
    pyplot.xlabel("t-SNE of output difference", fontsize=10)
    pyplot.ylabel("t-SNE of output difference", fontsize=10)
    pyplot.xlim(-120, 120)
    pyplot.ylim(-120, 120)
    pyplot.xticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.yticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)

    lipschitz_constants = load_data("../data/results/task01/lipschitz loop.npy")
    usages = where(lipschitz_constants > 0)
    locations_1 = load_data("../data/results/task01/location loop.npy")[usages]
    lipschitz_constants_1 = lipschitz_constants[usages]
    order_1 = argsort(lipschitz_constants_1)
    remain = where(lipschitz_constants == 0)
    locations_2 = load_data("../data/results/task01/location loop.npy")[remain]

    pyplot.subplot(1, 3, 2)
    pyplot.text(110, 110, "incoherent loop", fontsize=10, va="top", ha="right")
    mesh = pyplot.scatter(locations_1[order_1, 0], locations_1[order_1, 1], c=log10(lipschitz_constants_1[order_1]),
                          cmap="RdYlGn_r", s=10, vmin=-1, vmax=2)
    bar = pyplot.colorbar(mesh, ticks=[-1, 0, 1, 2], orientation="horizontal", extend="min", extendfrac=0.4)
    bar.set_label("Lipschitz constant", fontsize=10)
    bar.ax.set_xticklabels(["1E\N{MINUS SIGN}1", "1E+0", "1E+1", "1E+2"])
    bar.ax.tick_params(labelsize=10)
    pyplot.scatter(locations_2[:, 0], locations_2[:, 1], color="violet", s=10)
    pyplot.vlines(0, -120, 120, color="black", linewidth=0.75, linestyle=":", zorder=2)
    pyplot.hlines(0, -120, 120, color="black", linewidth=0.75, linestyle=":", zorder=2)
    pyplot.xlabel("t-SNE of output difference", fontsize=10)
    pyplot.ylabel("t-SNE of output difference", fontsize=10)
    pyplot.xlim(-120, 120)
    pyplot.ylim(-120, 120)
    pyplot.xticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.yticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    figure.text(0.3867, 0.135, "0", va="center", ha="center", fontsize=10)
    figure.add_artist(lines.Line2D([0.3867, 0.3867], [0.170, 0.193], color="black", linewidth=0.75))
    figure.patches.extend([patches.Polygon([[0.388, 0.193], [0.4585, 0.204], [0.4585, 0.182]], fill=True, zorder=5,
                                           facecolor="violet", transform=figure.transFigure, figure=figure)])

    lipschitz_constants = load_data("../data/results/task01/lipschitz collider.npy")
    usages = where(lipschitz_constants > 0)
    locations_1 = load_data("../data/results/task01/location collider.npy")[usages]
    lipschitz_constants_1 = lipschitz_constants[usages]
    order_1 = argsort(lipschitz_constants_1)
    remain = where(lipschitz_constants == 0)
    locations_2 = load_data("../data/results/task01/location collider.npy")[remain]

    pyplot.subplot(1, 3, 3)
    pyplot.text(110, 110, "collider", fontsize=10, va="top", ha="right")
    mesh = pyplot.scatter(locations_1[order_1, 0], locations_1[order_1, 1], c=log10(lipschitz_constants_1[order_1]),
                          cmap="RdYlGn_r", s=10, vmin=-1, vmax=2)
    bar = pyplot.colorbar(mesh, ticks=[-1, 0, 1, 2], orientation="horizontal", extend="min", extendfrac=0.4)
    bar.set_label("Lipschitz constant", fontsize=10)
    bar.ax.set_xticklabels(["1E\N{MINUS SIGN}1", "1E+0", "1E+1", "1E+2"])
    bar.ax.tick_params(labelsize=10)
    pyplot.scatter(locations_2[:, 0], locations_2[:, 1], color="violet", s=10)
    pyplot.vlines(0, -120, 120, color="black", linewidth=0.75, linestyle="--", zorder=2)
    pyplot.hlines(0, -120, 120, color="black", linewidth=0.75, linestyle="--", zorder=2)
    pyplot.xlabel("t-SNE of output difference", fontsize=10)
    pyplot.ylabel("t-SNE of output difference", fontsize=10)
    pyplot.xlim(-120, 120)
    pyplot.ylim(-120, 120)
    pyplot.xticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.yticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    figure.text(0.715, 0.135, "0", va="center", ha="center", fontsize=10)
    figure.add_artist(lines.Line2D([0.715, 0.715], [0.170, 0.193], color="black", linewidth=0.75))
    figure.patches.extend([patches.Polygon([[0.717, 0.193], [0.7875, 0.204], [0.7875, 0.182]], fill=True, zorder=5,
                                           facecolor="violet", transform=figure.transFigure, figure=figure)])

    figure.text(0.022, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.350, 0.990, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/main03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main04():
    figure = pyplot.figure(figsize=(10, 4.5), tight_layout=True)
    grid = pyplot.GridSpec(2, 4)
    rcParams["font.family"] = "Times New Roman"

    pyplot.subplot(grid[0, 0])
    info = draw_info["incoherent-loop"]
    for index, (px, py) in enumerate(zip(info[1], info[2])):
        if index + 1 in info[3]:
            pyplot.scatter(px, py, color="white", edgecolor="black", lw=1.5, s=80, zorder=2)
        elif index + 1 in info[4]:
            pyplot.scatter(px, py, color="black", edgecolor="black", lw=1.5, s=80, zorder=2)
        elif index + 1 in info[5]:
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                           color="white", edgecolor="black", lw=1.5, s=80, zorder=2)
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                           color="gray", edgecolor="black", lw=1.5, s=80, zorder=2)
        else:
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1.5, s=80, zorder=2)
    x, y = info[1], info[2]
    for former, latter in [(1, 2), (1, 3), (2, 3)]:
        pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        shrinkA=6, shrinkB=6, lw=1), zorder=2)
    pyplot.text(x=info[1][2], y=info[2][2] + 0.2, s="hardly replaced", fontsize=10, va="center", ha="center")
    pyplot.text(x=info[1][0], y=info[2][0] - 0.04, s="x", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[1][1], y=info[2][1] - 0.04, s="y", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[1][2], y=info[2][2] + 0.04, s="z", fontsize=9, va="bottom", ha="center")
    pyplot.text(x=(info[1][0] + info[1][1]) / 2.0, y=info[1][0] - 0.05,
                s="~inf", fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[1][0] + info[1][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                s="~0", fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[1][1] + info[1][2]) / 2.0 + 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                s="~0", fontsize=9, va="bottom", ha="left")
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.1, 1.0)
    pyplot.axis("off")

    pyplot.subplot(grid[0, 1])
    info = draw_info["incoherent-loop"]
    for index, (px, py) in enumerate(zip(info[1], info[2])):
        if index + 1 in info[3]:
            pyplot.scatter(px, py, color="white", edgecolor="black", lw=1.5, s=80, zorder=2)
        elif index + 1 in info[4]:
            pyplot.scatter(px, py, color="black", edgecolor="black", lw=1.5, s=80, zorder=2)
        elif index + 1 in info[5]:
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="right"),
                           color="white", edgecolor="black", lw=1.5, s=80, zorder=2)
            pyplot.scatter(px, py, marker=markers.MarkerStyle("o", fillstyle="left"),
                           color="gray", edgecolor="black", lw=1.5, s=80, zorder=2)
        else:
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1, s=80, zorder=2)
    x, y = info[1], info[2]
    for former, latter in [(1, 2), (1, 3), (2, 3)]:
        pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        shrinkA=6, shrinkB=6, lw=1), zorder=2)
    pyplot.text(x=info[1][2], y=info[2][2] + 0.2, s="easily replaced", fontsize=10, va="center", ha="center")
    pyplot.text(x=info[1][0], y=info[2][0] - 0.06, s="x", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[1][1], y=info[2][1] - 0.06, s="y", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[1][2], y=info[2][2] + 0.06, s="z", fontsize=9, va="bottom", ha="center")
    pyplot.text(x=(info[1][0] + info[1][1]) / 2.0, y=info[1][0] - 0.05,
                s="~0", fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[1][0] + info[1][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                s="~inf", fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[1][1] + info[1][2]) / 2.0 + 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                s="~inf", fontsize=9, va="bottom", ha="left")
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.1, 1.0)
    pyplot.axis("off")

    locations = load_data("../data/results/task02/locations.npy")
    cases = load_data("../data/results/task02/terminal cases.pkl")
    minimum_location = (locations[cases["min"][0] * 2, 0], locations[cases["min"][0] * 2, 1])
    maximum_location = (locations[cases["max"][0] * 2, 0], locations[cases["max"][0] * 2, 1])
    minimum_value, maximum_value = cases["min"][3], cases["max"][3]

    pyplot.subplot(grid[0, 2:])
    pyplot.scatter(locations[arange(0, 288, 2), 0], locations[arange(0, 288, 2), 1],
                   s=20, color="#86E3CE", label="incoherent loop")
    pyplot.scatter(locations[arange(1, 288, 2), 0], locations[arange(1, 288, 2), 1],
                   s=20, color="#FA897B", label="collider")

    pyplot.vlines(0, -32, 37, color="black", linewidth=0.75, linestyle=":", zorder=2)
    pyplot.hlines(2.5, -27.5, 27.5, color="black", linewidth=0.75, linestyle=":", zorder=2)
    pyplot.text(x=minimum_location[0], y=minimum_location[1] + 10,
                s=adjust_format("%.1E" % minimum_value), fontsize=8, va="bottom", ha="center")
    pyplot.annotate(s="", xy=(minimum_location[0], minimum_location[1]),
                    xytext=(minimum_location[0], minimum_location[1] + 10),
                    arrowprops=dict(arrowstyle="-|>", color="black", shrinkA=3, shrinkB=5, lw=1), zorder=2)
    pyplot.text(x=maximum_location[0] + 3.5, y=maximum_location[1] - 5,
                s=adjust_format("%.1E" % maximum_value), fontsize=8, va="center", ha="left")
    pyplot.annotate(s="", xy=(maximum_location[0], maximum_location[1]),
                    xytext=(maximum_location[0] + 4, maximum_location[1] - 5),
                    arrowprops=dict(arrowstyle="-|>", color="black", shrinkA=5, shrinkB=5, lw=1), zorder=2)
    pyplot.legend(loc="upper right", fontsize=8)
    pyplot.xlabel("t-SNE of output difference", fontsize=10)
    pyplot.ylabel("t-SNE of output difference", fontsize=10, labelpad=7)
    pyplot.xlim(-27.5, 27.5)
    pyplot.ylim(-32, 37)
    pyplot.xticks([-27.5, 0, 27.5], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.yticks([-32, 2.5, 37], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)

    minimum_losses = load_data(load_path="../data/results/task02/minimum losses.npy")
    max_min_losses = load_data(load_path="../data/results/task03/max-min losses.npy")

    print(mean(minimum_losses), mean(max_min_losses), mean(max_min_losses) - mean(minimum_losses))

    pyplot.subplot(grid[1, 0])
    pyplot.boxplot([minimum_losses], positions=[0], widths=0.4, showfliers=False, patch_artist=True,
                   boxprops=dict(color="black", facecolor="#EEEEEE", linewidth=1),
                   medianprops=dict(color="orange", linewidth=4))

    pyplot.boxplot([max_min_losses], positions=[1], widths=0.4, showfliers=False, patch_artist=True,
                   boxprops=dict(color="black", facecolor="#EEEEEE", linewidth=1),
                   medianprops=dict(color="orange", linewidth=4))
    pyplot.xlabel("strategy type", fontsize=10)
    pyplot.xticks([0, 1], ["intuition", "max-min"], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylabel("mean absolute error", fontsize=10, labelpad=5)
    pyplot.yticks([0, 0.25, 0.5], ["0.00", "0.25", "0.50"], fontsize=10)
    pyplot.ylim(-0.05, 0.55)

    minimum_lipschitz = load_data(load_path="../data/results/task02/lipschitz constants.npy")[:, 0]
    max_min_lipschitz = load_data(load_path="../data/results/task03/lipschitz constants.npy")[:, 1]

    print(mean(minimum_lipschitz), mean(max_min_lipschitz), mean(max_min_lipschitz) - mean(minimum_lipschitz))

    pyplot.subplot(grid[1, 1])
    pyplot.boxplot([minimum_lipschitz], positions=[0], widths=0.4, showfliers=False, patch_artist=True,
                   boxprops=dict(color="black", facecolor="#EEEEEE", linewidth=1),
                   medianprops=dict(color="cyan", linewidth=4))

    pyplot.boxplot([max_min_lipschitz], positions=[1], widths=0.4, showfliers=False, patch_artist=True,
                   boxprops=dict(color="black", facecolor="#EEEEEE", linewidth=1),
                   medianprops=dict(color="cyan", linewidth=4))
    pyplot.xlabel("strategy type", fontsize=10)
    pyplot.xticks([0, 1], ["intuition", "max-min"], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylabel("Lipschitz constant", fontsize=10)
    pyplot.yticks([0, 5, 10], [0, 5, 10], fontsize=10)
    pyplot.ylim(-1, 11)

    constants = load_data(load_path="../data/results/task03/lipschitz constants.npy")
    changes = constants[:, 1] - constants[:, 0]
    print(len(where(changes > 0)[0]))  # 112 samples > 0

    pyplot.subplot(grid[1, 2])
    x = linspace(min(changes), max(changes), 101)
    y = gaussian_kde(changes)(x)
    y /= sum(y)
    pyplot.plot(x, y, color="black", linewidth=1)
    pyplot.fill_between(x, 0, y, color="silver", alpha=0.5, zorder=0)
    pyplot.xlabel("Lipschitz constant change", fontsize=10)
    pyplot.xticks([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], fontsize=10)
    pyplot.xlim(-4, 44)
    pyplot.ylabel("proportion", fontsize=10)
    pyplot.yticks([0, 0.03, 0.06], ["0%", "3%", "6%"], fontsize=10)
    pyplot.ylim(-0.006, 0.066)

    losses = load_data(load_path="../data/results/task03/max-min losses.npy")
    constants = load_data(load_path="../data/results/task03/lipschitz constants.npy")[:, 1]
    print(spearmanr(losses, constants))  # correlation=0.7406713756440058, p-value=2.6744369606249177e-26

    pyplot.subplot(grid[1, 3])
    pyplot.scatter(losses, constants, s=10, color="silver", edgecolor="black")
    pyplot.xlabel("mean absolute error", fontsize=10)
    pyplot.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"], fontsize=10)
    pyplot.xlim(-0.05, 0.55)
    pyplot.ylabel("Lipschitz constant", fontsize=10)
    pyplot.yticks([0, 6, 12], [0, 6, 12], fontsize=10)
    pyplot.ylim(-1.2, 13.2)

    figure.text(0.022, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.275, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.515, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.022, 0.49, "d", va="center", ha="center", fontsize=12)
    figure.text(0.275, 0.49, "e", va="center", ha="center", fontsize=12)
    figure.text(0.515, 0.49, "f", va="center", ha="center", fontsize=12)
    figure.text(0.760, 0.49, "g", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/main04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main05():
    names = ["geometry", "baseline", "novelty"]

    figure = pyplot.figure(figsize=(10, 9), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"

    train_results = load_data(load_path="../data/results/task04/train results.npy")
    print(min(train_results[0], axis=1))
    print("%.4f" % min(train_results[1], axis=1)[-1])
    print(min(train_results[2], axis=1))

    for index in range(3):
        pyplot.subplot(3, 3, 1 + index)
        pyplot.title(names[index] + "-based search" if index != 1 else names[index] + " search", fontsize=10)
        pyplot.boxplot(train_results[index].tolist(), positions=[0.5, 1.5, 2.5, 3.5], showfliers=False,
                       boxprops=dict(color="black", facecolor="linen", linewidth=1),
                       medianprops=dict(color="orangered", linewidth=2), patch_artist=True)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.ylabel("train performance", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.yticks([80, 120, 160, 200], [80, 120, 160, 200], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylim(70, 210)

    performances = load_data(load_path="../data/results/task04/performances.npy")

    for index in range(3):
        pyplot.subplot(3, 3, 4 + index)
        pyplot.title(names[index] + "-based search" if index != 1 else names[index] + " search", fontsize=10)
        pyplot.pcolormesh(arange(5), arange(5), performances[index].T, vmin=100, vmax=200, cmap="spring")
        for i in range(4):
            for j in range(4):
                pyplot.text(i + 0.5, j + 0.5, "%.2f" % performances[index, i, j], va="center", ha="center", fontsize=10)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.ylabel("evaluating error scale", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.yticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylim(0, 4)

    motif_counts = load_data(load_path="../data/results/task04/final motifs.npy")

    for index in range(3):
        pyplot.subplot(3, 3, 7 + index)
        pyplot.title(names[index] + "-based search" if index != 1 else names[index] + " search", fontsize=10)
        pyplot.bar(arange(4) + 0.3, motif_counts[index][:, 0], width=0.4,
                   linewidth=0.75, edgecolor="black", color="#FA897B", label="incoherent loop")
        pyplot.bar(arange(4) + 0.7, motif_counts[index][:, 1], width=0.4,
                   linewidth=0.75, edgecolor="black", color="#86E3CE", label="collider")
        for position, (value_1, value_2) in enumerate(zip(motif_counts[index][:, 0], motif_counts[index][:, 1])):
            pyplot.text(position + 0.3, value_1 + 0.2, "%.1f" % value_1, va="bottom", ha="center", fontsize=8)
            pyplot.text(position + 0.7, value_2 + 0.2, "%.1f" % value_2, va="bottom", ha="center", fontsize=8)
        pyplot.legend(loc="upper right", ncol=2, fontsize=8)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylabel("average motif number", fontsize=10)
        pyplot.yticks([0, 10, 20, 30], [0, 10, 20, 30], fontsize=10)
        pyplot.ylim(0, 30)

    figure.align_labels()

    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.349, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.680, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.66, "d", va="center", ha="center", fontsize=12)
    figure.text(0.349, 0.66, "e", va="center", ha="center", fontsize=12)
    figure.text(0.680, 0.66, "f", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.34, "g", va="center", ha="center", fontsize=12)
    figure.text(0.349, 0.34, "h", va="center", ha="center", fontsize=12)
    figure.text(0.680, 0.34, "i", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/main05.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main06():
    figure = pyplot.figure(figsize=(10, 6), tight_layout=True)
    grid = pyplot.GridSpec(2, 3)
    rcParams["font.family"] = "Times New Roman"

    generation_data = load_data("../data/results/task05/generations.npy")

    pyplot.subplot(grid[:, 0])
    values = [generation_data[0, _, :] for _ in range(7)]
    violin_1 = pyplot.violinplot(values, points=100, positions=arange(7) + 0.5,
                                 widths=0.8, showmeans=False, showextrema=False, showmedians=False)
    for body in violin_1["bodies"]:
        center = mean(body.get_paths()[0].vertices[:, 0])
        body.get_paths()[0].vertices[:, 0] = clip(body.get_paths()[0].vertices[:, 0], -inf, center)
        body.set_color("#FA897B")
        body.set_edgecolor("black")
        body.set_linewidth(1)
        body.set_alpha(1)
    pyplot.scatter(arange(7) + 0.4, [mean(value) for value in values],
                   s=16, linewidth=1, color="black", zorder=2)
    values = [generation_data[1, _, :] for _ in range(7)]
    violin_2 = pyplot.violinplot(values, points=100, positions=arange(7) + 0.5,
                                 widths=0.8, showmeans=False, showextrema=False, showmedians=False)
    for body in violin_2["bodies"]:
        center = mean(body.get_paths()[0].vertices[:, 0])
        body.get_paths()[0].vertices[:, 0] = clip(body.get_paths()[0].vertices[:, 0], center, inf)
        body.set_color("#86E3CE")
        body.set_edgecolor("black")
        body.set_linewidth(1)
        body.set_alpha(1)
    scatters = pyplot.scatter(arange(7) + 0.6, [mean(value) for value in values],
                              s=16, linewidth=1, color="black", zorder=2)

    pyplot.legend([violin_1["bodies"][0], violin_2["bodies"][0], scatters], ["baseline", "adjusted", "average"],
                  loc="upper left", fontsize=10)
    pyplot.xlabel("training error scale", fontsize=10)
    pyplot.ylabel("used generation", fontsize=10)
    pyplot.xticks(arange(7) + 0.5, ["0%", "5%", "10%", "15%", "20%", "25%", "30%"], fontsize=10)
    pyplot.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=10)
    pyplot.xlim(0, 7)
    pyplot.ylim(0, 100)

    motif_data = load_data(load_path="../data/results/task05/final loops.npy")
    pyplot.subplot(grid[:, 1])
    pyplot.bar(arange(7) + 0.5, mean(motif_data, axis=1), color="silver", edgecolor="black", linewidth=0.75)
    for index, values in enumerate(motif_data):
        if mean(values) == 0:
            pyplot.scatter(index + 0.5, 0.02, marker="x", s=20, color="black")
    pyplot.xlabel("training error scale", fontsize=10)
    pyplot.ylabel("average number of incoherent loops in the best agent", fontsize=10)
    pyplot.xticks(arange(7) + 0.5, ["0%", "5%", "10%", "15%", "20%", "25%", "30%"], fontsize=10)
    pyplot.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                  ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "1.8", "2.0"], fontsize=10)
    pyplot.xlim(0, 7)
    pyplot.ylim(0, 2)

    accesses = load_data(load_path="../data/results/task05/accesses.npy")

    pyplot.subplot(grid[0, 2])
    for i in range(7):
        for j in range(7):
            color = "green" if accesses[0, i, j] else "red"
            pyplot.fill_between([i, i + 1], j, j + 1, color=color, linewidth=0, alpha=0.5)
    pyplot.xlabel("training error scale", fontsize=10)
    pyplot.ylabel("evaluating error scale", fontsize=10)
    pyplot.xticks(arange(7) + 0.5, ["0%", "5%", "10%", "15%", "20%", "25%", "30%"], fontsize=10)
    pyplot.yticks(arange(7) + 0.5, ["0%", "5%", "10%", "15%", "20%", "25%", "30%"], fontsize=10)
    pyplot.xlim(0, 7)
    pyplot.ylim(0, 7)

    pyplot.subplot(grid[1, 2])
    for i in range(7):
        for j in range(7):
            color = "green" if accesses[1, i, j] else "red"
            pyplot.fill_between([i, i + 1], j, j + 1, color=color, linewidth=0, alpha=0.5)
    pyplot.xlabel("training error scale", fontsize=10)
    pyplot.ylabel("evaluating error scale", fontsize=10)
    pyplot.xticks(arange(7) + 0.5, ["0%", "5%", "10%", "15%", "20%", "25%", "30%"], fontsize=10)
    pyplot.yticks(arange(7) + 0.5, ["0%", "5%", "10%", "15%", "20%", "25%", "30%"], fontsize=10)
    pyplot.xlim(0, 7)
    pyplot.ylim(0, 7)

    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.352, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.677, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.677, 0.50, "d", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/main06.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    main01()
    main02()
    main03()
    main04()
    main05()
    main06()
