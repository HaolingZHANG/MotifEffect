from matplotlib import pyplot, rcParams, markers, patches
from mpl_toolkits.mplot3d import Axes3D
from numpy import load, array, arange, ones, random, linspace, meshgrid, where
from numpy import sqrt, cos, log, exp, min, max, median, mean, argmax, sum, sin, abs, cos, percentile, log10, pi
from scipy.stats import gaussian_kde, pearsonr

from effect import NeuralMotif, prepare_data

from works import acyclic_motifs, draw_info, load_data


# noinspection PyArgumentList,PyTypeChecker
def manuscript_01():
    figure = pyplot.figure(figsize=(10, 3.2), tight_layout=True)
    grid = pyplot.GridSpec(2, 5)
    rcParams["font.family"] = "Times New Roman"

    pyplot.subplot(grid[:, 0])
    pyplot.bar([0], [log10(18000)], width=0.6, linewidth=0.75, color="#86E3CE", edgecolor="black")
    pyplot.bar([1], [log10(600)], width=0.6, linewidth=0.75, color="#FA897B", edgecolor="black")
    pyplot.text(0, log10(18000) + 0.05, str(18000), fontsize=10, va="bottom", ha="center")
    pyplot.text(1, log10(600) + 0.05, str(600), fontsize=10, va="bottom", ha="center")
    pyplot.xlabel("population name", fontsize=10)
    pyplot.ylabel("population size", fontsize=10)
    pyplot.xticks([0, 1], [" loop", "collider"], fontsize=10)
    pyplot.yticks([1, 2, 3, 4, 5], ["1E+1", "1E+2", "1E+3", "1E+4", "1E+5"], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(1, 5)

    locations = load_data("../data/results/task01/locations.npy")

    pyplot.subplot(grid[:, 1: 3])
    random.seed(2022)
    choices = arange(18000)
    random.shuffle(choices)
    random.seed(None)
    pyplot.scatter(locations[choices[:1200], 0], locations[choices[:1200], 1], s=10, color="#86E3CE", label="loop")
    pyplot.scatter(locations[18000:, 0], locations[18000:, 1], s=10, color="#FA897B", label="collider")
    pyplot.legend(loc="upper right", ncol=2, fontsize=10)
    pyplot.xlabel("tSNE of output landscape difference", fontsize=10, labelpad=20)
    pyplot.ylabel("tSNE of output landscape difference", fontsize=10, labelpad=20)
    pyplot.xlim(-110, 110)
    pyplot.ylim(-110, 140)
    pyplot.xticks([])
    pyplot.yticks([])

    loop_data = load_data("../data/results/task01/rugosity loop.npy")
    collider_data = load_data("../data/results/task01/rugosity collider.npy")

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
    pyplot.ylabel("rugosity index", fontsize=10)
    pyplot.xticks([0, 1], ["loop", "collider"], fontsize=10)
    pyplot.yticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(1, 5)

    loop_data = load("../data/results/task01/propagation extreme loop.npy")
    collider_data = load("../data/results/task01/propagation extreme collider.npy")

    pyplot.subplot(grid[0, 4])
    pyplot.text(1, 20, "loop", color="white", va="top", ha="left", fontsize=8)
    mesh = pyplot.pcolormesh(arange(22), arange(22), loop_data[:21, :21], cmap="RdYlGn_r", vmin=0, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[0, 0.5, 1.0])
    bar.set_label("maximum\nz error rate", fontsize=10)
    bar.ax.set_yticklabels(["0%", "50%", "100%"])
    bar.ax.tick_params(labelsize=10)
    pyplot.xlabel("x error rate", fontsize=10)
    pyplot.ylabel("y error rate", fontsize=10)
    pyplot.xticks([0, 10.5, 21], ["0%", "25%", "50%"], fontsize=10)
    pyplot.yticks([0, 10.5, 21], ["0%", "25%", "50%"], fontsize=10)
    pyplot.xlim(0, 21)
    pyplot.ylim(0, 21)

    pyplot.subplot(grid[1, 4])
    pyplot.text(1, 20, "collider", color="white", va="top", ha="left", fontsize=8)
    mesh = pyplot.pcolormesh(arange(22), arange(22), collider_data[:21, :21], cmap="RdYlGn_r", vmin=0, vmax=1)
    bar = pyplot.colorbar(mesh, ticks=[0, 0.5, 1.0])
    bar.set_label("maximum\nz error rate", fontsize=10)
    bar.ax.set_yticklabels(["0%", "50%", "100%"])
    bar.ax.tick_params(labelsize=10)
    pyplot.xlabel("x error rate", fontsize=10)
    pyplot.ylabel("y error rate", fontsize=10)
    pyplot.xticks([0, 10.5, 21], ["0%", "25%", "50%"], fontsize=10)
    pyplot.yticks([0, 10.5, 21], ["0%", "25%", "50%"], fontsize=10)
    pyplot.xlim(0, 21)
    pyplot.ylim(0, 21)

    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.223, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.595, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.762, 0.99, "d", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/manuscript01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


# noinspection PyArgumentList,PyTypeChecker
def manuscript_02():
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
    pyplot.text(x=info[1][2], y=info[2][2] + 0.2, s="most similar", fontsize=10, va="center", ha="center")
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
    pyplot.text(x=info[1][2], y=info[2][2] + 0.2, s="most dissimilar", fontsize=10, va="center", ha="center")
    pyplot.text(x=info[1][0], y=info[2][0] - 0.04, s="x", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[1][1], y=info[2][1] - 0.04, s="y", fontsize=9, va="top", ha="center")
    pyplot.text(x=info[1][2], y=info[2][2] + 0.04, s="z", fontsize=9, va="bottom", ha="center")
    pyplot.text(x=(info[1][0] + info[1][1]) / 2.0, y=info[1][0] - 0.05,
                s="~inf", fontsize=9, va="top", ha="center")
    pyplot.text(x=(info[1][0] + info[1][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                s="~0", fontsize=9, va="bottom", ha="right")
    pyplot.text(x=(info[1][1] + info[1][2]) / 2.0 + 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                s="~inf", fontsize=9, va="bottom", ha="left")
    pyplot.xlim(0.0, 1.0)
    pyplot.ylim(0.1, 1.0)
    pyplot.axis("off")

    minimum_losses = load(file="../data/results/task02/minimum losses.npy") / 2.0
    max_min_losses = load(file="../data/results/task03/max-min losses.npy") / 2.0

    pyplot.subplot(grid[0, 2])
    violin = pyplot.violinplot([minimum_losses], bw_method=0.2, positions=[0], widths=0.8, showextrema=False)
    for patch in violin["bodies"]:
        patch.set_edgecolor("black")
        patch.set_facecolor("silver")
        patch.set_linewidth(1)
        patch.set_alpha(1)
    lower_bound, median_value, upper_value = percentile(minimum_losses, [25, 50, 75])
    pyplot.vlines(0, lower_bound, upper_value, linewidth=4, zorder=2)
    pyplot.scatter([0], median(median_value), color="white", s=10, zorder=3)
    violin = pyplot.violinplot([max_min_losses], bw_method=0.2, positions=[1], widths=0.8, showextrema=False)
    for patch in violin["bodies"]:
        patch.set_edgecolor("black")
        patch.set_facecolor("silver")
        patch.set_linewidth(1)
        patch.set_alpha(1)
    lower_bound, median_value, upper_value = percentile(max_min_losses, [25, 50, 75])
    pyplot.vlines(1, lower_bound, upper_value, linewidth=4, zorder=2)
    pyplot.scatter([1], median(median_value), color="white", s=10, zorder=3)
    pyplot.xlabel("strategy type", fontsize=10)
    pyplot.xticks([0, 1], ["intuition", "max-min"], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylabel("representation error rate", fontsize=10)
    pyplot.yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25], ["0%", "5%", "10%", "15%", "20%", "25%"], fontsize=10)
    pyplot.ylim(0, 0.25)

    minimum_indices = load(file="../data/results/task02/rugosity indices.npy")
    max_min_indices = load(file="../data/results/task03/rugosity indices.npy")

    pyplot.subplot(grid[0, 3])
    violin = pyplot.violinplot([minimum_indices], bw_method=0.2, positions=[0], widths=0.8, showextrema=False)
    for patch in violin["bodies"]:
        patch.set_edgecolor("black")
        patch.set_facecolor("silver")
        patch.set_linewidth(1)
        patch.set_alpha(1)
    lower_bound, median_value, upper_value = percentile(minimum_indices, [25, 50, 75])
    pyplot.vlines(0, lower_bound, upper_value, linewidth=4, zorder=2)
    pyplot.scatter([0], median(median_value), color="white", s=10, zorder=3)
    violin = pyplot.violinplot([max_min_indices], bw_method=0.2, positions=[1], widths=0.8, showextrema=False)
    for patch in violin["bodies"]:
        patch.set_edgecolor("black")
        patch.set_facecolor("silver")
        patch.set_linewidth(1)
        patch.set_alpha(1)
    lower_bound, median_value, upper_value = percentile(max_min_indices, [25, 50, 75])
    pyplot.vlines(1, lower_bound, upper_value, linewidth=4, zorder=2)
    pyplot.scatter([1], median(median_value), color="white", s=10, zorder=3)
    pyplot.xlabel("strategy type", fontsize=10)
    pyplot.xticks([0, 1], ["intuition", "max-min"], fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylabel("rugosity index", fontsize=10, labelpad=15)
    pyplot.yticks([1, 2, 3, 4, 5, 6], ["1", "2", "3", "4", "5", "6"], fontsize=10)
    pyplot.ylim(1, 6)

    max_min_params = load(file="../data/results/task03/max-min params.npy")

    pyplot.subplot(grid[1, 0])
    pyplot.boxplot(max_min_params[0], positions=[0, 1, 2], widths=0.4, showfliers=True, patch_artist=True,
                   boxprops=dict(color="black", facecolor="silver", linewidth=1),
                   medianprops=dict(color="orange", linewidth=3),
                   flierprops=dict(marker=".", color="black"))
    pyplot.xlabel("direction [type 1]", fontsize=10)
    pyplot.xticks([0, 1, 2], ["x to y", "x to z", "y to z"], fontsize=10)
    pyplot.xlim(-0.5, 2.5)
    pyplot.ylabel("weight", fontsize=10, labelpad=10)
    pyplot.yticks([-1, 0, 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.ylim(-1.1, +1.1)

    pyplot.subplot(grid[1, 1])
    pyplot.boxplot(max_min_params[1], positions=[0, 1, 2], widths=0.4, showfliers=True, patch_artist=True,
                   boxprops=dict(color="black", facecolor="silver", linewidth=1),
                   medianprops=dict(color="orange", linewidth=3),
                   flierprops=dict(marker=".", color="black"))
    pyplot.xlabel("effect direction [type 2]", fontsize=10)
    pyplot.xticks([0, 1, 2], ["x to y", "x to z", "y to z"], fontsize=10)
    pyplot.xlim(-0.5, 2.5)
    pyplot.ylabel("weight", fontsize=10, labelpad=10)
    pyplot.yticks([-1, 0, 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.ylim(-1.1, +1.1)

    pyplot.subplot(grid[1, 2])
    pyplot.boxplot(max_min_params[2], positions=[0, 1, 2], widths=0.4, showfliers=True, patch_artist=True,
                   boxprops=dict(color="black", facecolor="silver", linewidth=1),
                   medianprops=dict(color="orange", linewidth=3),
                   flierprops=dict(marker=".", color="black"))
    pyplot.xlabel("effect direction [type 3]", fontsize=10)
    pyplot.xticks([0, 1, 2], ["x to y", "x to z", "y to z"], fontsize=10)
    pyplot.xlim(-0.5, 2.5)
    pyplot.ylabel("weight", fontsize=10, labelpad=10)
    pyplot.yticks([-1, 0, 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.ylim(-1.1, +1.1)

    pyplot.subplot(grid[1, 3])
    pyplot.boxplot(max_min_params[3], positions=[0, 1, 2], widths=0.4, showfliers=True, patch_artist=True,
                   boxprops=dict(color="black", facecolor="silver", linewidth=1),
                   medianprops=dict(color="orange", linewidth=3),
                   flierprops=dict(marker=".", color="black"))
    pyplot.xlabel("effect direction [type 4]", fontsize=10)
    pyplot.xticks([0, 1, 2], ["x to y", "x to z", "y to z"], fontsize=10)
    pyplot.xlim(-0.5, 2.5)
    pyplot.ylabel("weight", fontsize=10, labelpad=10)
    pyplot.yticks([-1, 0, 1], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.ylim(-1.1, +1.1)

    changes = load(file="../data/results/task03/rugosity changes.npy")

    pyplot.subplot(grid[2, :2])
    start, stop = abs(changes[0][:, 0] - changes[0][:, 1]), abs(changes[1][:, 0] - changes[1][:, 1])
    minimum_value, maximum_value = min([start, stop]), max([start, stop])
    x = linspace(minimum_value, maximum_value, 101)
    y = gaussian_kde(start)(x)
    y /= sum(y)
    pyplot.plot(x, y, color="green", linewidth=2)
    pyplot.fill_between(x, 0, y, color="green", alpha=0.5, zorder=0)
    y = gaussian_kde(stop)(x)
    y /= sum(y)
    pyplot.plot(x, y, color="blue", linewidth=2)
    pyplot.fill_between(x, 0, y, color="blue", alpha=0.5, zorder=0)
    legends = [patches.Patch(facecolor="#7FBF7F", edgecolor="green", linewidth=1, label="start"),
               patches.Patch(facecolor="#7F7FFF", edgecolor="blue", linewidth=1, label="stop")]
    pyplot.legend(handles=legends, loc="upper right", fontsize=10)
    pyplot.xlabel("rugosity index difference", fontsize=10)
    pyplot.xticks([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], fontsize=10)
    pyplot.xlim(0, 5)
    pyplot.ylabel("proportion", fontsize=10)
    pyplot.yticks([0, 0.1, 0.2, 0.3], ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.ylim(0, 0.3)

    pyplot.subplot(grid[2, 2:])
    pyplot.xlabel("representation error rate", fontsize=10)
    pyplot.xticks([0, 0.1, 0.2, 0.3], ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.xlim(0, 0.3)
    pyplot.ylabel("input error rate", fontsize=10)
    pyplot.yticks([0, 0.1, 0.2, 0.3], ["0%", "10%", "20%", "30%"], fontsize=10)
    pyplot.ylim(0, 0.3)

    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.510, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.758, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.66, "d", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.33, "e", va="center", ha="center", fontsize=12)
    figure.text(0.510, 0.33, "f", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/manuscript02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def manuscript_03():
    min_losses = load(file="../data/results/task02/minimum losses.npy") / 2.0
    indices = load(file="../data/results/task02/rugosity indices.npy")
    maxmin_losses = load(file="../data/results/task03/max-minimum losses.npy") / 2.0
    changes = load(file="../data/results/task03/rugosity changes.npy")[where(maxmin_losses > 0.0)]

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"

    pyplot.subplot(2, 4, 1)
    maximum_loss = max(maxmin_losses)
    loss_range = linspace(0, maximum_loss, 101)
    proportions = gaussian_kde(maxmin_losses)(loss_range)
    proportions /= sum(proportions)
    pyplot.plot(loss_range, proportions, color="#FE817D", linewidth=1, zorder=0)
    pyplot.fill_between(loss_range, 0, proportions, color="#FE817D", alpha=0.5, zorder=0)
    pyplot.vlines(maximum_loss, 0, 0.01, color="#FE817D", linewidth=1)
    pyplot.hlines(0.01, maximum_loss - 0.005, maximum_loss + 0.005, color="#FE817D", linewidth=1)
    pyplot.scatter([maximum_loss], [proportions[-1]], color="#FCBBAE", edgecolor="#FE817D", s=20, zorder=2)
    pyplot.text(maximum_loss, 0.0105, "%.2f" % (maximum_loss * 100) + "%", va="bottom", ha="center", fontsize=10)
    maximum_loss = max(min_losses)
    loss_range = linspace(0, maximum_loss, 101)
    proportions = gaussian_kde(min_losses)(loss_range)
    proportions /= sum(proportions)
    pyplot.plot(loss_range, proportions, color="#81B8DF", linewidth=1, zorder=1)
    pyplot.fill_between(loss_range, 0, proportions, color="#81B8DF", alpha=0.5, zorder=1)
    pyplot.vlines(maximum_loss, 0, 0.01, color="#81B8DF", linewidth=1)
    pyplot.hlines(0.01, maximum_loss - 0.005, maximum_loss + 0.005, color="#81B8DF", linewidth=1)
    pyplot.scatter([maximum_loss], [proportions[-1]], color="#B1CCDF", edgecolor="#81B8DF", s=20, zorder=2)
    pyplot.text(maximum_loss, 0.0105, "%.2f" % (maximum_loss * 100) + "%", va="bottom", ha="center", fontsize=10)
    legends = [patches.Patch(facecolor="#B1CCDF", edgecolor="#81B8DF", linewidth=1, label="threshold"),
               patches.Patch(facecolor="#FCBBAE", edgecolor="#FE817D", linewidth=1, label="max-min")]
    pyplot.legend(handles=legends, loc="upper right", fontsize=10)
    pyplot.xlabel("representation error", fontsize=10)
    pyplot.xlim(0, 0.2)
    pyplot.xticks([0, 0.04, 0.08, 0.12, 0.16, 0.20], ["0%", "4%", "8%", "12%", "16%", "20%"], fontsize=10)
    pyplot.ylabel("proportion", fontsize=10)
    pyplot.ylim(0, 0.04)
    pyplot.yticks([0, 0.01, 0.02, 0.03, 0.04], ["0%", "1%", "2%", "3%", "4%"], fontsize=10)

    pyplot.subplot(2, 4, 2)
    # violin = pyplot.violinplot([indices[where(min_losses > 0)]], positions=[0], widths=0.4, showextrema=False)
    # for patch in violin["bodies"]:
    #     patch.set_edgecolor("#81B8DF")
    #     patch.set_facecolor("#B1CCDF")
    #     patch.set_linewidth(1)
    #     patch.set_alpha(1)
    violin = pyplot.violinplot([changes[:, 1]], positions=[1], widths=0.4, showextrema=False)
    for patch in violin["bodies"]:
        patch.set_edgecolor("#FE817D")
        patch.set_facecolor("#FCBBAE")
        patch.set_linewidth(1)
        patch.set_alpha(1)
    # lower_value, median_value, upper_value = percentile(indices[where(min_losses > 0)], [25, 50, 75])
    # pyplot.vlines(0, lower_value, upper_value, color="#81B8DF", linewidth=5)
    # pyplot.scatter([0], [median_value], c="white", s=10, zorder=2)
    lower_value, median_value, upper_value = percentile(changes[:, 1] - changes[:, 0], [25, 50, 75])
    pyplot.vlines(1, lower_value, upper_value, color="#FE817D", linewidth=5)
    pyplot.scatter([1], [median_value], c="white", s=10, zorder=2)
    pyplot.xlabel("different search type", fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.xticks([0, 1], ["threshold", "max-min"], fontsize=10)
    pyplot.ylabel("rugosity index", fontsize=10)
    # pyplot.ylim(1.0, 1.5)
    # pyplot.yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5"], fontsize=10)

    pyplot.subplot(2, 4, 3)
    print(maxmin_losses.shape, changes.shape)
    used_locations = where(maxmin_losses > 0)
    violin = pyplot.violinplot([changes[used_locations, 0]], positions=[0], widths=0.4, showextrema=False)
    for patch in violin["bodies"]:
        patch.set_edgecolor("#81B8DF")
        patch.set_facecolor("#B1CCDF")
        patch.set_linewidth(1)
        patch.set_alpha(1)
    violin = pyplot.violinplot([changes[used_locations, 1]], positions=[1], widths=0.4, showextrema=False)
    for patch in violin["bodies"]:
        patch.set_edgecolor("#FE817D")
        patch.set_facecolor("#FCBBAE")
        patch.set_linewidth(1)
        patch.set_alpha(1)
    # lower_value, median_value, upper_value = percentile(indices[where(min_losses > 0)], [25, 50, 75])
    # pyplot.vlines(0, lower_value, upper_value, color="#81B8DF", linewidth=5)
    # pyplot.scatter([0], [median_value], c="white", s=10, zorder=2)
    # lower_value, median_value, upper_value = percentile(changes[:, 1] - changes[:, 0], [25, 50, 75])
    # pyplot.vlines(1, lower_value, upper_value, color="#FE817D", linewidth=5)
    # pyplot.scatter([1], [median_value], c="white", s=10, zorder=2)
    pyplot.xlabel("motif similarity", fontsize=10)
    pyplot.xlim(-0.5, 1.5)
    pyplot.xticks([0, 1], ["nearest", "fastest"], fontsize=10)
    pyplot.ylabel("rugosity index", fontsize=10)
    # pyplot.ylim(1.0, 1.5)
    # pyplot.yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5"], fontsize=10)

    pyplot.subplot(2, 4, 4)

    params, param_groups = load("../data/results/task03/max-min params.npy"), {1: [], 2: [], 3: [], 4: []}
    for param in params:
        param_groups[param[0]].append(param[1:4])

    info = draw_info["incoherent-loop"]
    for motif_index, motif in enumerate(acyclic_motifs["incoherent-loop"]):
        pyplot.subplot(2, 4, 5 + motif_index)
        pyplot.text(0.2, 0.2 - 0.06, "x", fontsize=9, va="top", ha="center")
        pyplot.text(0.8, 0.2 - 0.06, "y", fontsize=9, va="top", ha="center")
        pyplot.text(0.5, 0.7 + 0.06, "z", fontsize=9, va="bottom", ha="center")

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

        pyplot.vlines(0.4, -0.1, 0.1, linewidth=0.75)
        pyplot.hlines(-0.08, 0.38, 0.4, linewidth=0.75)
        pyplot.hlines(0.00, 0.38, 0.4, linewidth=0.75)
        pyplot.hlines(0.08, 0.38, 0.4, linewidth=0.75)
        pyplot.text(0.37, -0.08, "\N{MINUS SIGN}1", va="center", ha="right", fontsize=8)
        pyplot.text(0.3645, 0.00, "0", va="center", ha="right", fontsize=8)
        pyplot.text(0.37, 0.08, "+1", va="center", ha="right", fontsize=8)
        used_values = array(param_groups[motif_index + 1])[:, 0] * 0.08
        violin = pyplot.violinplot([used_values], positions=[0.5], widths=0.1, bw_method=0.5, showextrema=False)
        for patch in violin["bodies"]:
            patch.set_edgecolor("black")
            patch.set_facecolor("silver")
            patch.set_linewidth(1)
            patch.set_alpha(1)
        pyplot.scatter([0.5], [median(used_values)], color="black", s=8, zorder=3)

        pyplot.vlines(0.1, 0.4, 0.6, linewidth=0.75)
        pyplot.hlines(0.58, 0.08, 0.1, linewidth=0.75)
        pyplot.hlines(0.50, 0.08, 0.1, linewidth=0.75)
        pyplot.hlines(0.42, 0.08, 0.1, linewidth=0.75)
        pyplot.text(0.07, 0.42, "\N{MINUS SIGN}1", va="center", ha="right", fontsize=8)
        pyplot.text(0.0645, 0.50, "0", va="center", ha="right", fontsize=8)
        pyplot.text(0.07, 0.58, "+1", va="center", ha="right", fontsize=8)
        used_values = array(param_groups[motif_index + 1])[:, 1] * 0.08 + 0.5
        violin = pyplot.violinplot([used_values], positions=[0.2], widths=0.1, bw_method=0.5, showextrema=False)
        for patch in violin["bodies"]:
            patch.set_edgecolor("black")
            patch.set_facecolor("silver")
            patch.set_linewidth(1)
            patch.set_alpha(1)
        pyplot.scatter([0.2], [median(used_values)], color="black", s=8, zorder=3)

        pyplot.vlines(0.85, 0.4, 0.6, linewidth=0.75)
        pyplot.hlines(0.58, 0.83, 0.85, linewidth=0.75)
        pyplot.hlines(0.50, 0.83, 0.85, linewidth=0.75)
        pyplot.hlines(0.42, 0.83, 0.85, linewidth=0.75)
        pyplot.text(0.82, 0.42, "\N{MINUS SIGN}1", va="center", ha="right", fontsize=8)
        pyplot.text(0.8145, 0.50, "0", va="center", ha="right", fontsize=8)
        pyplot.text(0.82, 0.58, "+1", va="center", ha="right", fontsize=8)
        used_values = array(param_groups[motif_index + 1])[:, 2] * 0.08 + 0.5
        violin = pyplot.violinplot([used_values], positions=[0.95], widths=0.1, bw_method=0.5, showextrema=False)
        for patch in violin["bodies"]:
            patch.set_edgecolor("black")
            patch.set_facecolor("silver")
            patch.set_linewidth(1)
            patch.set_alpha(1)
        pyplot.scatter([0.95], [median(used_values)], color="black", s=8, zorder=3)

        pyplot.xlim(-0.07, 1.07)
        pyplot.ylim(-0.2, 1.0)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.axis("off")

    # figure.text(0.021, 0.97, "a", va="center", ha="center", fontsize=12)
    # figure.text(0.519, 0.97, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig("../results/figures/manuscript02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def manuscript_04():
    pass


def manuscript_05():
    pass


# noinspection PyUnresolvedReferences
def manuscript_06():
    figure = pyplot.figure(figsize=(10, 8), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"

    ax = pyplot.subplot(2, 2, 1, projection="3d")
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
    figure.text(0.26, 0.78, "surface area (z)", va="center", ha="center", fontsize=10)
    figure.text(0.26, 0.63, "base area (x,y)", va="center", ha="center", fontsize=10)

    pyplot.subplot(2, 2, 2)
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
    figure.text(0.00, 0.48, "c", va="center", ha="center", fontsize=12)
    figure.text(0.51, 0.48, "d", va="center", ha="center", fontsize=12)

    pyplot.savefig("./results/figures/manuscript06.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    # manuscript_01()
    manuscript_02()
    # manuscript_03()
    # manuscript_04()
    # manuscript_05()
    # manuscript_06()
