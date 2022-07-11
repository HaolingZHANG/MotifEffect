from matplotlib import pyplot, rcParams, markers, patches
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange, ones, linspace, meshgrid, where
from numpy import sqrt, log, exp, min, max, mean, argmax, sum, sin, cos, percentile, log10, pi
from scipy.stats import gaussian_kde, spearmanr

from effect import NeuralMotif, prepare_data

from works import load_data, draw_info, adjust_format


# noinspection PyArgumentList,PyTypeChecker
def manuscript01():
    figure = pyplot.figure(figsize=(10, 3), tight_layout=True)
    grid = pyplot.GridSpec(2, 5)
    rcParams["font.family"] = "Times New Roman"

    pyplot.subplot(grid[:, 0])
    pyplot.bar([0], [log10(28800)], width=0.6, linewidth=0.75, color="#86E3CE", edgecolor="black")
    pyplot.bar([1], [log10(480)], width=0.6, linewidth=0.75, color="#FA897B", edgecolor="black")
    pyplot.text(0, log10(28800) + 0.02, str(28800), fontsize=8, va="bottom", ha="center")
    pyplot.text(1, log10(480) + 0.02, str(480), fontsize=8, va="bottom", ha="center")
    pyplot.xlabel("population name", fontsize=10)
    pyplot.ylabel("population size", fontsize=10)
    pyplot.xticks([0, 1], [" loop", "collider"], fontsize=10)
    pyplot.yticks([2, 3, 4, 5], ["1E+2", "1E+3", "1E+4", "1E+5"], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(2, 5)

    locations = load_data("../data/results/task01/locations.npy")

    pyplot.subplot(grid[:, 1: 3])
    pyplot.scatter(locations[:28800, 0], locations[:28800, 1], s=10, color="#86E3CE", label="loop", zorder=1)
    pyplot.scatter(locations[28800:, 0], locations[28800:, 1], s=10, color="#FA897B", label="collider", zorder=1)
    pyplot.legend(loc="upper right", fontsize=8)
    pyplot.xlabel("t-SNE of output difference", fontsize=10)
    pyplot.ylabel("t-SNE of output difference", fontsize=10)
    pyplot.xlim(-120, 120)
    pyplot.ylim(-120, 120)
    pyplot.xticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.yticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)

    loop_data = load_data("../data/results/task01/lipschitz loop.npy")
    collider_data = load_data("../data/results/task01/lipschitz collider.npy")

    loop_data, collider_data = log10(loop_data[loop_data > 0]), log10(collider_data[collider_data > 0])
    pyplot.subplot(grid[:, 3])
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
    pyplot.vlines(0, lower_value, upper_value, color="black", linewidth=4, zorder=2)
    pyplot.scatter([0], [median_value], color="white", s=1, zorder=3)
    lower_value, median_value, upper_value = percentile(collider_data, [25, 50, 75])
    pyplot.vlines(1, lower_value, upper_value, color="black", linewidth=4, zorder=2)
    pyplot.scatter([1], [median_value], color="white", s=1, zorder=3)
    pyplot.xlabel("population name", fontsize=10)
    pyplot.ylabel("Lipschitz constant", fontsize=10)
    pyplot.xticks([0, 1], ["loop", "collider"], fontsize=10)
    pyplot.yticks([-1, 0, 1, 2], ["1E\N{MINUS SIGN}1", "1E+0", "1E+1", "1E+2"], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(-1, 2)

    loop_data = load_data("../data/results/task01/propagation loop.npy")
    collider_data = load_data("../data/results/task01/propagation collider.npy")

    pyplot.subplot(grid[0, 4])
    pyplot.text(11.5, 11.5, "loop", color="white", va="top", ha="right", fontsize=8)
    mesh = pyplot.pcolormesh(arange(22), arange(22), loop_data[:21, :21], cmap="RdYlGn_r", vmin=0, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[0, 0.5, 1.0])
    bar.set_label("maximum\nz error rate", fontsize=10)
    bar.ax.set_yticklabels(["0%", "50%", "100%"])
    bar.ax.tick_params(labelsize=10)
    pyplot.xlabel("x error rate", fontsize=10)
    pyplot.ylabel("y error rate", fontsize=10)
    pyplot.xticks([0, 6, 12], ["0%", "15%", "30%"], fontsize=10)
    pyplot.yticks([0, 6, 12], ["0%", "15%", "30%"], fontsize=10)
    pyplot.xlim(0, 12)
    pyplot.ylim(0, 12)

    pyplot.subplot(grid[1, 4])
    pyplot.text(11.5, 11.5, "collider", color="white", va="top", ha="right", fontsize=8)
    mesh = pyplot.pcolormesh(arange(22), arange(22), collider_data[:21, :21], cmap="RdYlGn_r", vmin=0, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[0, 0.5, 1.0])
    bar.set_label("maximum\nz error rate", fontsize=10)
    bar.ax.set_yticklabels(["0%", "50%", "100%"])
    bar.ax.tick_params(labelsize=10)
    pyplot.xlabel("x error rate", fontsize=10)
    pyplot.ylabel("y error rate", fontsize=10)
    pyplot.xticks([0, 6, 12], ["0%", "15%", "30%"], fontsize=10)
    pyplot.yticks([0, 6, 12], ["0%", "15%", "30%"], fontsize=10)
    pyplot.xlim(0, 12)
    pyplot.ylim(0, 12)

    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.223, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.580, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.771, 0.99, "d", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/manuscript01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def manuscript02():
    figure = pyplot.figure(figsize=(10, 7), tight_layout=True)
    grid = pyplot.GridSpec(3, 4)
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
            pyplot.scatter(px, py, color="gray", edgecolor="black", lw=1.5, s=80, zorder=2)
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
                   s=20, color="#86E3CE", label="loop")
    pyplot.scatter(locations[arange(1, 288, 2), 0], locations[arange(1, 288, 2), 1],
                   s=20, color="#FA897B", label="collider")
    pyplot.text(x=minimum_location[0], y=minimum_location[1] + 10,
                s=adjust_format("%.1E" % minimum_value), fontsize=8, va="bottom", ha="center")
    pyplot.annotate(s="", xy=(minimum_location[0], minimum_location[1]),
                    xytext=(minimum_location[0], minimum_location[1] + 10),
                    arrowprops=dict(arrowstyle="-|>", color="black", shrinkA=3, shrinkB=5, lw=1), zorder=2)
    pyplot.text(x=maximum_location[0] + 3, y=maximum_location[1] - 6,
                s=adjust_format("%.1E" % maximum_value), fontsize=8, va="center", ha="left")
    pyplot.annotate(s="", xy=(maximum_location[0], maximum_location[1]),
                    xytext=(maximum_location[0] + 3, maximum_location[1] - 6),
                    arrowprops=dict(arrowstyle="-|>", color="black", shrinkA=5, shrinkB=5, lw=1), zorder=2)
    pyplot.legend(loc="upper right", fontsize=8)
    pyplot.xlabel("t-SNE of output difference", fontsize=10)
    pyplot.ylabel("t-SNE of output difference", fontsize=10)
    pyplot.xlim(-27.5, 27.5)
    pyplot.ylim(-32, 37)
    pyplot.xticks([-25, 0, 25], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.yticks([-25, 2.5, 30], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)

    minimum_losses = load_data(load_path="../data/results/task02/minimum losses.npy")
    max_min_losses = load_data(load_path="../data/results/task03/max-min losses.npy")

    print(mean(minimum_losses), mean(max_min_losses), mean(max_min_losses) - mean(minimum_losses))

    pyplot.subplot(grid[1, 0])
    pyplot.boxplot([minimum_losses], positions=[0], widths=0.4, showfliers=False, patch_artist=True,
                   boxprops=dict(color="black", facecolor="#EEEEEE", linewidth=1),
                   medianprops=dict(color="black", linewidth=4))

    pyplot.boxplot([max_min_losses], positions=[1], widths=0.4, showfliers=False, patch_artist=True,
                   boxprops=dict(color="black", facecolor="#EEEEEE", linewidth=1),
                   medianprops=dict(color="black", linewidth=4))
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
                   medianprops=dict(color="black", linewidth=4))

    pyplot.boxplot([max_min_lipschitz], positions=[1], widths=0.4, showfliers=False, patch_artist=True,
                   boxprops=dict(color="black", facecolor="#EEEEEE", linewidth=1),
                   medianprops=dict(color="black", linewidth=4))
    pyplot.xlabel("strategy type", fontsize=10)
    pyplot.xticks([0, 1], ["intuition", "max-min"], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylabel("Lipschitz constant", fontsize=10)
    pyplot.yticks([0, 5, 10], [0, 5, 10], fontsize=10)
    pyplot.ylim(-1, 11)

    params = load_data(load_path="../data/results/task03/max-min params.npy")

    pyplot.subplot(grid[1, 2:])
    colors = ["#005CCE", "#83B934", "#F78E4E"]
    location_groups = [[0.75, 1.75, 2.75, 3.75], [1.0, 2.0, 3.0, 4.0], [1.25, 2.25, 3.25, 4.25]]
    for arc_type, (color, locations) in enumerate(zip(colors, location_groups)):
        violin = pyplot.violinplot(params[:, :, arc_type].tolist(), positions=locations, widths=0.2, bw_method=0.5,
                                   showextrema=False)
        for patch in violin["bodies"]:
            patch.set_edgecolor(color)
            patch.set_facecolor(color)
            patch.set_linewidth(1)
            patch.set_alpha(1)
    legends = [patches.Patch(facecolor=color, edgecolor=color, linewidth=1, label=label)
               for color, label in zip(colors, ["x to y", "x to z", "y to z"])]
    for value in [1.5, 2.5, 3.5]:
        pyplot.vlines(value, -1.2, 1.2, linewidth=0.75, linestyle="--")
    pyplot.legend(handles=legends, loc="upper right", fontsize=8)
    pyplot.xlabel("motif type", fontsize=10)
    pyplot.xticks([1, 2, 3, 4], [1, 2, 3, 4], fontsize=10)
    pyplot.xlim(0.5, 4.5)
    pyplot.ylabel("weight", fontsize=10)
    pyplot.yticks([-1, 0, 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.ylim(-1.2, 1.2)

    constants = load_data(load_path="../data/results/task03/lipschitz constants.npy")
    changes = constants[:, 1] - constants[:, 0]
    print(len(where(changes > 0)[0]))  # 112 samples > 0

    pyplot.subplot(grid[2, :2])
    x = linspace(min(changes), max(changes), 101)
    y = gaussian_kde(changes)(x)
    y /= sum(y)
    pyplot.plot(x, y, color="black", linewidth=1)
    pyplot.fill_between(x, 0, y, color="silver", alpha=0.5, zorder=0)
    pyplot.xlabel("Lipschitz constant change of trained incoherent loops", fontsize=10)
    pyplot.xticks([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], fontsize=10)
    pyplot.xlim(-2, 42)
    pyplot.ylabel("proportion", fontsize=10, labelpad=10)
    pyplot.yticks([0, 0.03, 0.06], ["0%", "3%", "6%"], fontsize=10)
    pyplot.ylim(-0.0075, 0.0675)

    losses = load_data(load_path="../data/results/task03/max-min losses.npy")
    constants = load_data(load_path="../data/results/task03/lipschitz constants.npy")[:, 1]
    print(spearmanr(losses, constants))  # correlation=0.7406713756440058, p-value=2.6744369606249177e-26

    pyplot.subplot(grid[2, 2:])
    pyplot.scatter(losses, constants, s=10, color="silver", edgecolor="black")
    pyplot.xlabel("mean absolute error (maximum-minimum loss)", fontsize=10)
    pyplot.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"], fontsize=10)
    pyplot.xlim(-0.02, 0.52)
    pyplot.ylabel("Lipschitz constant", fontsize=10, labelpad=5)
    pyplot.yticks([0, 6, 12], [0, 6, 12], fontsize=10)
    pyplot.ylim(-1.5, 13.5)

    figure.text(0.022, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.277, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.520, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.022, 0.66, "d", va="center", ha="center", fontsize=12)
    figure.text(0.277, 0.66, "e", va="center", ha="center", fontsize=12)
    figure.text(0.520, 0.66, "f", va="center", ha="center", fontsize=12)
    figure.text(0.022, 0.33, "g", va="center", ha="center", fontsize=12)
    figure.text(0.520, 0.33, "h", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/manuscript02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def manuscript03():
    change_record = load_data(load_path="../data/results/task03/maximum case.pkl")
    loss_change, lipschitz_change, rugosity_change, param_change, _ = change_record

    figure = pyplot.figure(figsize=(10, 7), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"

    pyplot.subplot(4, 1, 1)
    pyplot.plot(arange(len(loss_change)), loss_change, color="#00B8D7")
    pyplot.fill_between(arange(len(loss_change)), 0, loss_change, color="#00B8D7", alpha=0.5)
    pyplot.xlabel("search iteration", fontsize=10)
    pyplot.xticks(arange(0, 1001, 100), arange(0, 1001, 100), fontsize=10)
    pyplot.xlim(0, 1000)
    pyplot.ylabel("L1 loss", fontsize=10)
    pyplot.yticks([0.0, 0.2, 0.4, 0.6], ["0.0", "0.2", "0.4", "0.6"], fontsize=10)
    pyplot.ylim(0, 0.6)

    pyplot.subplot(4, 1, 2)
    pyplot.plot(arange(len(lipschitz_change)), lipschitz_change, color="#00B583")
    pyplot.fill_between(arange(len(lipschitz_change)), 0, lipschitz_change, color="#00B583", alpha=0.5)
    pyplot.xlabel("search iteration", fontsize=10)
    pyplot.xticks(arange(0, 1001, 100), arange(0, 1001, 100), fontsize=10)
    pyplot.xlim(0, 1000)
    pyplot.ylabel("Lipschitz constant", fontsize=10)
    pyplot.yticks([0, 2, 4, 6], ["0.0", "2.0", "4.0", "6.0"], fontsize=10)
    pyplot.ylim(0, 6)

    pyplot.subplot(4, 1, 3)
    pyplot.plot(arange(len(rugosity_change)), rugosity_change, color="#FD8A7D")
    pyplot.fill_between(arange(len(rugosity_change)), 0, rugosity_change, color="#FD8A7D", alpha=0.5)
    pyplot.xlabel("search iteration", fontsize=10)
    pyplot.xticks(arange(0, 1001, 100), arange(0, 1001, 100), fontsize=10)
    pyplot.xlim(0, 1000)
    pyplot.ylabel("rugosity index", fontsize=10)
    pyplot.yticks([0, 1, 2, 3], ["0.0", "1.0", "2.0", "3.0"], fontsize=10)
    pyplot.ylim(0, 3)

    pyplot.subplot(4, 1, 4)
    names = ["weight x to y", "weight x to z", "weight y to z", "bias y", "bias z"]
    for index, name in enumerate(names):
        if "bias" not in name:
            pyplot.plot(arange(1000), param_change[:, index], label=name, zorder=4 - index, linewidth=2)
        else:
            pyplot.plot(arange(1000), param_change[:, index], linestyle="--", label=name, zorder=4 - index, linewidth=2)
    pyplot.legend(loc="upper right", ncol=2, fontsize=8)
    pyplot.xlabel("search iteration", fontsize=10)
    pyplot.xticks(arange(0, 1001, 100), arange(0, 1001, 100), fontsize=10)
    pyplot.xlim(0, 1000)
    pyplot.ylabel("parameter value", fontsize=10, labelpad=5)
    pyplot.yticks([-1, 0, 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.ylim(-1.1, 1.1)

    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.75, "b", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.50, "c", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.25, "d", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/manuscript03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def manuscript04():
    names = ["baseline search", "geometry-based search", "novelty search"]
    performances = load_data(load_path="../data/results/task04/performances.npy")
    minimum_value = min(performances)

    figure = pyplot.figure(figsize=(10, 8), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"
    grid = pyplot.GridSpec(3, 3)

    pyplot.subplot(grid[0, 0])
    pyplot.pcolormesh(arange(5), arange(5), performances[0].T, vmin=minimum_value, vmax=200, cmap="spring")
    for i in range(4):
        for j in range(4):
            pyplot.text(i + 0.5, j + 0.5, "%.2f" % performances[0, i, j], va="center", ha="center", fontsize=10)
    pyplot.xlabel("train error rate (" + names[0] + ")", fontsize=10)
    pyplot.ylabel("test error rate", fontsize=10)
    pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.yticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.xlim(0, 4)
    pyplot.ylim(0, 4)

    pyplot.subplot(grid[0, 1])
    pyplot.pcolormesh(arange(5), arange(5), performances[1].T, vmin=minimum_value, vmax=200, cmap="spring")
    for i in range(4):
        for j in range(4):
            pyplot.text(i + 0.5, j + 0.5, "%.2f" % performances[1, i, j], va="center", ha="center", fontsize=10)
    pyplot.xlabel("train error rate (" + names[1] + ")", fontsize=10)
    pyplot.ylabel("test error rate", fontsize=10)
    pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.yticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.xlim(0, 4)
    pyplot.ylim(0, 4)

    pyplot.subplot(grid[0, 2])
    pyplot.pcolormesh(arange(5), arange(5), performances[2].T, vmin=minimum_value, vmax=200, cmap="spring")
    for i in range(4):
        for j in range(4):
            pyplot.text(i + 0.5, j + 0.5, "%.2f" % performances[2, i, j], va="center", ha="center", fontsize=10)
    pyplot.xlabel("train error rate (" + names[2] + ")", fontsize=10)
    pyplot.ylabel("test error rate", fontsize=10)
    pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.yticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.xlim(0, 4)
    pyplot.ylim(0, 4)

    motif_counts = load_data(load_path="../data/results/task04/final motifs.npy")

    pyplot.subplot(grid[1, 0])
    pyplot.bar(arange(4) + 0.5, motif_counts[0][:, 1], color="#FA897B", label="collider")
    pyplot.bar(arange(4) + 0.5, motif_counts[0][:, 0], bottom=motif_counts[0][:, 1], color="#86E3CE", label="loop")
    pyplot.legend(loc="upper right", fontsize=8)
    pyplot.xlabel("train error rate (" + names[0] + ")", fontsize=10)
    pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.xlim(0, 4)
    pyplot.ylabel("motif count", fontsize=10, labelpad=11)
    pyplot.yticks([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], fontsize=10)
    pyplot.ylim(0, 40)

    pyplot.subplot(grid[1, 1])
    pyplot.bar(arange(4) + 0.5, motif_counts[1][:, 1], color="#FA897B", label="collider")
    pyplot.bar(arange(4) + 0.5, motif_counts[1][:, 0], bottom=motif_counts[1][:, 1], color="#86E3CE", label="loop")
    pyplot.legend(loc="upper right", fontsize=8)
    pyplot.xlabel("train error rate (" + names[1] + ")", fontsize=10)
    pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.xlim(0, 4)
    pyplot.ylabel("motif count", fontsize=10, labelpad=11)
    pyplot.yticks([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], fontsize=10)
    pyplot.ylim(0, 40)

    pyplot.subplot(grid[1, 2])
    pyplot.bar(arange(4) + 0.5, motif_counts[2][:, 1], color="#FA897B", label="collider")
    pyplot.bar(arange(4) + 0.5, motif_counts[2][:, 0], bottom=motif_counts[2][:, 1], color="#86E3CE", label="loop")
    pyplot.legend(loc="upper right", fontsize=8)
    pyplot.xlabel("train error rate (" + names[2] + ")", fontsize=10)
    pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.xlim(0, 4)
    pyplot.ylabel("motif count", fontsize=10, labelpad=11)
    pyplot.yticks([0, 10, 20, 30, 40], [0, 10, 20, 30, 40], fontsize=10)
    pyplot.ylim(0, 40)

    pyplot.subplot(grid[2, :])

    pyplot.xlabel("average training performance (fitness)", fontsize=10)
    pyplot.ylabel("average robustness score (GRACE)", fontsize=10, labelpad=9)
    pyplot.xticks(linspace(0, 200, 11, dtype=int), linspace(0, 200, 11, dtype=int), fontsize=10)
    pyplot.yticks(linspace(0, 0.4, 5), ["0.0", "0.1", "0.2", "0.3", "0.4"], fontsize=10)
    pyplot.xlim(0, 200)
    pyplot.ylim(0, 0.4)

    figure.text(0.022, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.346, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.671, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.022, 0.66, "d", va="center", ha="center", fontsize=12)
    figure.text(0.346, 0.66, "e", va="center", ha="center", fontsize=12)
    figure.text(0.671, 0.66, "f", va="center", ha="center", fontsize=12)
    figure.text(0.022, 0.34, "g", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/manuscript04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


# noinspection PyUnresolvedReferences
def manuscript05():
    figure = pyplot.figure(figsize=(10, 8), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"

    pyplot.subplot(2, 2, 1)
    loop_x = linspace(-2, +2, 101)
    loop_y = sqrt(16 - (2 * loop_x) ** 2)
    for location_x, info in zip([0, 6, 12], ["initialization", "minimum search", "maximum search"]):
        pyplot.plot(location_x + loop_x, 10 + loop_y, color="#FA897B", zorder=1)
        pyplot.plot(location_x + loop_x, 10 - loop_y, color="#FA897B", zorder=1)
        pyplot.fill_between(location_x + loop_x, 10 - loop_y, 10 + loop_y, color="#FCBBAE", zorder=1)
        pyplot.text(location_x, 5, info, ha="center", va="top", fontsize=10)
    collider_x = linspace(-1.5, +1.5, 101)
    collider_y = sqrt(9 - (2 * collider_x) ** 2)
    for location_x in [0.3, 6.3, 12.3]:
        pyplot.plot(location_x + collider_x, 10 + collider_y, color="#81B8DF", zorder=2)
        pyplot.plot(location_x + collider_x, 10 - collider_y, color="#81B8DF", zorder=2)
        pyplot.fill_between(location_x + collider_x, 10 - collider_y, 10 + collider_y, color="#B1CCDF", zorder=2)
    pyplot.plot(9 + loop_x, -5 + loop_y, color="#FA897B")
    pyplot.plot(9 + loop_x, -5 - loop_y, color="#FA897B")
    pyplot.fill_between(9 + loop_x, -5 - loop_y, -5 + loop_y, color="#FCBBAE")
    pyplot.plot(9.3 + collider_x, -5 + collider_y, color="#81B8DF")
    pyplot.plot(9.3 + collider_x, -5 - collider_y, color="#81B8DF")
    pyplot.fill_between(9.3 + collider_x, -5 - collider_y, -5 + collider_y, color="#B1CCDF")
    pyplot.scatter([7], [-5], s=20, color="#FA897B", edgecolor="black", zorder=3)
    pyplot.scatter([7.8], [-5], s=20, color="#81B8DF", edgecolor="black", zorder=3)
    pyplot.scatter([-1.4, 4.6, 10.6, 10.4], [11.4, 11.4, 11.4, 9.5], s=20, color="#FA897B", edgecolor="black", zorder=4)
    pyplot.scatter([0.5, 4.9, 6.5, 10.9], [10, 11, 10, 11], s=20, color="#81B8DF", edgecolor="black", zorder=4)
    pyplot.annotate("", (10.4, 9.5), (10.6, 11.4), arrowprops=dict(arrowstyle="-|>", color="black",
                                                                   shrinkA=0, shrinkB=0, lw=1), zorder=3)
    pyplot.annotate("", (5.7, 9.5), (6.5, 10), arrowprops=dict(arrowstyle="-|>", color="black",
                                                               shrinkA=0, shrinkB=0, lw=1), zorder=3)
    pyplot.annotate("", (5.6, 11.4), (5.7, 9.5), arrowprops=dict(arrowstyle="-|>", color="black",
                                                                 shrinkA=0, shrinkB=0, lw=1), zorder=3)
    pyplot.annotate("", (4.9, 11), (5.6, 11.4), arrowprops=dict(arrowstyle="-|>", color="black",
                                                                shrinkA=0, shrinkB=0, lw=1), zorder=3)
    pyplot.annotate("", (3.8, 10), (2.2, 10), arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    pyplot.annotate("", (9.8, 11), (8.2, 11), arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    pyplot.annotate("", (8.2, 9), (9.8, 9), arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    pyplot.annotate("", (9, 0), (9, 3), arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    pyplot.annotate("", (-1, 13.5), (-1, 17.8), arrowprops=dict(arrowstyle="-|>", color="black",
                                                                shrinkA=0, shrinkB=0, lw=1), zorder=3)
    pyplot.annotate("", (1, 12.6), (1, 15.3), arrowprops=dict(arrowstyle="-|>", color="black",
                                                              shrinkA=0, shrinkB=0, lw=1), zorder=3)
    pyplot.vlines(7, -5, -10, color="gray", linewidth=1, linestyle="--", zorder=2)
    pyplot.vlines(7.8, -5, -10, color="gray", linewidth=1, linestyle="--", zorder=2)
    pyplot.plot([7, 7.8], [-10, -10], color="red", linewidth=3, zorder=2)
    pyplot.text(-1.5, 18, "population 1", va="bottom", ha="left", fontsize=10)
    pyplot.text(0.5, 15.5, "population 2", va="bottom", ha="left", fontsize=10)
    pyplot.text(4.4, 10, "fix", va="center", ha="center", fontsize=10)
    pyplot.text(11.6, 11, "fix", va="center", ha="center", fontsize=10)
    pyplot.xlim(-3.5, 14.5)
    pyplot.ylim(-12, 20)
    pyplot.axis("off")

    ax = pyplot.subplot(2, 2, 2, projection="3d")
    value_range, points = (-1, +1), 41
    input_data = prepare_data(value_range=value_range, points=points)
    motif = NeuralMotif(motif_type="incoherent-loop", motif_index=1,
                        activations=("relu", "tanh"), aggregations=("max", "sum"),
                        weights=[-1.00e+00, +6.71e-01, +1.00e+00], biases=[+4.47e-01, -9.08e-01])
    output_data = motif(input_data).reshape(points, points).detach().numpy()
    x_axis, y_axis = meshgrid(arange(points), arange(points))
    ax.plot_surface(x_axis, y_axis, output_data, rstride=1, cstride=1, cmap="rainbow", alpha=0.8)
    ax.plot_surface(x_axis, y_axis, ones(shape=(points, points)) * -2, rstride=1, cstride=1, color="silver", alpha=1)
    ax.set_xlabel("input x", labelpad=-15, fontsize=10)
    ax.set_ylabel("input y", labelpad=-15, fontsize=10)
    ax.set_zlabel("output z", labelpad=-15, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=140)
    ax.set_zlim(-2, 1)
    figure.text(0.76, 0.75, "surface area (z)", va="center", ha="center", fontsize=10)
    figure.text(0.76, 0.62, "base area (x,y)", va="center", ha="center", fontsize=10)

    pyplot.subplot(2, 2, 3)
    pyplot.hlines(1.0, 0.0, 2.0, color="gray", linewidth=1, linestyle="--")
    pyplot.text(1.0, 1.0 - 0.05, "input samples", va="top", ha="center", fontsize=10)
    pyplot.hlines(0.0, 3.0, 5.0, color="gray", linewidth=1, linestyle="--")
    pyplot.text(4.0, 0.0 - 0.05, "input samples", va="top", ha="center", fontsize=10)
    pyplot.annotate("", (3.0, 0.5), (2.0, 1.5), arrowprops=dict(arrowstyle="-|>", color="black",
                                                                shrinkA=35, shrinkB=35, lw=1))
    x = linspace(1, 4, 401)
    y1 = ((1.0 / (1.0 + exp(-x)) + cos(1.5 * x) + 0.5) / 8.0 + 0.15)[::-1]
    y2 = ((log(1 + exp(x)) + sin(2 * x) * x * 0.5 + 0.5) / 8.0 + 0.2)[::-1]
    y2[:150] -= linspace(0, 0.2, 150)[::-1]
    pyplot.plot(linspace(0, 2, 401), y1 + 1, linewidth=2, color="black")
    pyplot.plot(linspace(0, 2, 401), y2 + 1, linewidth=2, color="black")
    pyplot.fill_between(linspace(0, 2, 401), y1 + 1, y2 + 1, color="silver")
    pyplot.text(2.1, y2[-1] + 1, "i-th value in output vector", va="center", ha="left", fontsize=10)
    pyplot.text(2.1, y1[-1] + 1, "j-th value in output vector", va="center", ha="left", fontsize=10)
    differences = y2 - y1 + 0.2
    pyplot.plot(linspace(3, 5, 401), differences, linewidth=2, color="black")
    pyplot.text(2.9, differences[0], "value difference", va="center", ha="right", fontsize=10)
    pyplot.scatter([4.0], [differences[201]], color="black", s=40, edgecolor="black", linewidth=1, zorder=3)
    pyplot.scatter([4.9], [differences[380]], color="white", s=40, edgecolor="black", linewidth=1, zorder=3)
    pyplot.plot([3.0, 5.0], [differences[201] - 0.3, differences[201] + 0.3], color="gray", linewidth=1, linestyle="--")
    pyplot.plot([3.0, 5.0], [differences[201] + 0.3, differences[201] - 0.3], color="gray", linewidth=1, linestyle="--")
    checks = [linspace(differences[201] - 0.3, differences[201] + 0.3, 401)[380],
              linspace(differences[201] + 0.3, differences[201] - 0.3, 401)[380]]
    pyplot.scatter([4.9, 4.9], checks, color="white", s=40, edgecolor="black", linewidth=1, zorder=3)
    pyplot.vlines(4.9, sum(checks) / 2.0, checks[0], color="blue", linewidth=3, zorder=2)
    pyplot.vlines(4.9, 0, checks[0], color="gray", linewidth=1, linestyle="--", zorder=1)
    pyplot.vlines(4.0, 0, differences[201], color="gray", linewidth=1, linestyle="--", zorder=1)
    pyplot.hlines(0, 4.0, 4.9, color="red", linewidth=3, zorder=2)
    pyplot.xlim(-0.5, 5.5)
    pyplot.ylim(-0.25, 2.25)
    pyplot.axis("off")

    pyplot.subplot(2, 2, 4)
    loop_x = linspace(0, 3.8, 501)
    loop_y = (loop_x ** 2 - 10 * cos(2 * pi * loop_x)) * loop_x
    pyplot.plot(loop_x, loop_y, color="black", linewidth=2, zorder=1)
    pyplot.plot(loop_x, ones(shape=(len(loop_x),)) * -40, color="gray", linestyle="--", linewidth=1, zorder=2)
    pyplot.vlines(loop_x[100], -40, loop_y[100], color="gray", linestyle="--", linewidth=1, zorder=2)
    pyplot.scatter([loop_x[100]], [loop_y[100]], color="black", s=40, edgecolor="black", linewidth=1, zorder=3)
    for index, location in enumerate([137, 175, 250, 400]):
        pyplot.scatter([loop_x[location]], [loop_y[location]], color="white", s=40, edgecolor="black",
                       linewidth=1, zorder=3)
        pyplot.plot([loop_x[100], loop_x[location]], [loop_y[100], loop_y[location]], color="gray", linestyle="--",
                    linewidth=1, zorder=1)
        pyplot.vlines(loop_x[location], loop_y[location], -40, color="gray", linestyle="--", linewidth=1, zorder=1)
    pyplot.annotate("", (loop_x[138], -45), (loop_x[400], -45), arrowprops=dict(arrowstyle="-|>", color="black", lw=1))
    pyplot.plot([loop_x[101], loop_x[136]], [-40, -40], color="red", linewidth=3)
    pyplot.text(loop_x[288], -47, "bisection method", va="top", ha="center", fontsize=10)
    pyplot.text(loop_x[argmax(loop_y)], loop_y[argmax(loop_y)] + 10, "output curve", va="top", ha="center", fontsize=10)
    pyplot.xlim(-0.5, 4.5)
    pyplot.ylim(-55, 110)
    pyplot.axis("off")

    figure.text(0.00, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.51, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.00, 0.47, "c", va="center", ha="center", fontsize=12)
    figure.text(0.51, 0.47, "d", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/manuscript05.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    manuscript01()
    manuscript02()
    manuscript03()
    manuscript04()
    manuscript05()
