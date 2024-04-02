"""
@Author      : Haoling Zhang
@Description : Plot all the figures in the main text.
"""
from logging import getLogger, CRITICAL
from matplotlib import pyplot, patches, rcParams
from numpy import array, arange, meshgrid, linspace, sum, min, max, abs, percentile, nan, where
from scipy.stats import spearmanr
from warnings import filterwarnings

from practice import acyclic_motifs

from works import load_data, draw_info

filterwarnings("ignore")

getLogger("matplotlib").setLevel(CRITICAL)

rcParams["font.family"] = "Arial"
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Linux Libertine"
rcParams["mathtext.cal"] = "Lucida Calligraphy:italic"
rcParams["mathtext.it"] = "Linux Libertine:italic"
rcParams["mathtext.bf"] = "Linux Libertine:bold"

sort_path, save_path = "./data/", "./show/"


def main_01():
    """
    Create Figure 1 in the main text.
    """
    task_data = load_data(sort_path + "main01.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    grid = pyplot.GridSpec(8, 10)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[:, :4])
    motif_types = ["incoherent-loop", "coherent-loop", "collider"]
    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]
    for type_index, motif_type in enumerate(motif_types):
        motifs = acyclic_motifs[motif_type]
        info = draw_info[motif_type]
        info[2][2] = 0.45
        pyplot.text(3.0, 3.76 - type_index, motif_type.replace("-", " ") + " " + math_orders[type_index],
                    va="center", ha="center", fontsize=8)
        pyplot.fill_between([1, 2, 4, 5], 3.05 - type_index,
                            [3.75 - type_index, 3.86 - type_index, 3.86 - type_index, 3.75 - type_index],
                            color=info[0], lw=0, zorder=1)
        for motif_index, motif in enumerate(motifs):
            pyplot.text(1.5 + motif_index, 3.65 - type_index, str(motif_index + 1),
                        va="center", ha="center", fontsize=8)
            pyplot.plot([1.5 + motif_index - 0.35, 1.5 + motif_index - 0.35,
                         1.5 + motif_index + 0.35, 1.5 + motif_index + 0.35],
                        [3.65 - type_index - 0.08, 3.65 - type_index - 0.05,
                         3.65 - type_index - 0.05, 3.65 - type_index - 0.08],
                        lw=0.75, color="k")
            bias_x, bias_y = motif_index + 1, 3 - type_index
            for index, (px, py) in enumerate(zip(info[1], info[2])):
                if index + 1 in info[3]:
                    pyplot.scatter(px + bias_x, py + bias_y, fc="w", ec="k", lw=0.75, s=30, zorder=2)
                    pyplot.text(px + bias_x, py + bias_y - 0.03, "$x$", va="top", ha="center", fontsize=8)
                elif index + 1 in info[4]:
                    pyplot.scatter(px + bias_x, py + bias_y, fc="k", ec="k", lw=0.75, s=30, zorder=2)
                    pyplot.text(px + bias_x, py + bias_y + 0.03, "$z$", va="bottom", ha="center", fontsize=8)
                elif index + 1 in info[5]:
                    pyplot.scatter(px + bias_x, py + bias_y, fc="silver", ec="k", lw=0.75, s=30, zorder=2)
                    pyplot.text(px + bias_x, py + bias_y - 0.03, "$y$", va="top", ha="center", fontsize=8)
                else:
                    pyplot.scatter(px + bias_x, py + bias_y, fc="silver", ec="k", lw=0.75, s=30, zorder=2)
                    pyplot.text(px + bias_x, py + bias_y - 0.03, "$y$", va="top", ha="center", fontsize=8)
            x, y = array(info[1]) + bias_x, array(info[2]) + bias_y
            for former, latter in motif.edges:
                location_x, location_y = (x[former - 1] + x[latter - 1]) / 2.0, (y[former - 1] + y[latter - 1]) / 2.0
                flag = "+" if motif.get_edge_data(former, latter)["weight"] == 1 else "\N{MINUS SIGN}"
                # noinspection PyTypeChecker
                pyplot.annotate("", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                                arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                shrinkA=4, shrinkB=4, lw=0.75, ls=("-" if flag == "+" else ":")))
                if (former, latter) == (1, 2):
                    pyplot.text(location_x, location_y - 0.07, flag, va="center", ha="center", fontsize=8)
                if (former, latter) == (1, 3):
                    pyplot.text(location_x - 0.1, location_y, flag, va="center", ha="center", fontsize=8)
                if (former, latter) == (2, 3):
                    pyplot.text(location_x + 0.1, location_y, flag, va="center", ha="center", fontsize=8)

    pyplot.xlim(0.72, 5.00)
    pyplot.ylim(1.05, 3.86)
    pyplot.axis("off")

    results = task_data["b"]

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[:2, 4:7])
    for location, motif_type in enumerate(motif_types[:-1]):
        x = results[motif_type][0]
        y = results[motif_type][1] / sum(results[motif_type][1])
        pyplot.plot(x, y, color=draw_info[motif_type][0], lw=2, label=math_orders[location])
    pyplot.legend(loc="upper right", fontsize=7)
    pyplot.xlabel(r"minimum L2 norm difference for $\mathcal{C}$", fontsize=8)
    pyplot.ylabel("proportion", fontsize=8)
    pyplot.xticks(linspace(0.00, 0.03, 7),
                  ["%.3f" % v for v in linspace(0.00, 0.03, 7)], fontsize=7)
    pyplot.yticks(linspace(0.00, 0.08, 3),
                  ["%d" % (v * 100) + "%" for v in linspace(0.00, 0.08, 3)], fontsize=7)
    pyplot.xlim(0, 0.03)
    pyplot.ylim(0, 0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    results = task_data["c"]

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[:2, 7:])
    for location, (motif_type, color) in enumerate(zip(motif_types[:-1], ["#C27C77", "#C2A976"])):
        x = results[motif_type][0]
        y = results[motif_type][1] / sum(results[motif_type][1])
        pyplot.plot(x, y, color=draw_info[motif_type][0], lw=2, label=math_orders[location])
    pyplot.legend(loc="upper right", fontsize=7)
    pyplot.xlabel("best Lipschitz constant", fontsize=8)
    pyplot.ylabel("proportion", fontsize=8)
    pyplot.xticks(linspace(0.00, 4.00, 6),
                  ["%.2f" % v for v in linspace(0.00, 4.00, 6)], fontsize=7)
    pyplot.yticks(linspace(0.00, 0.08, 3),
                  ["%d" % (v * 100) + "%" for v in linspace(0.00, 0.08, 3)], fontsize=7)
    pyplot.xlim(0, 4.00)
    pyplot.ylim(0, 0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    results = task_data["d"]

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:5, 4:7])
    matrix = results[motif_types[0]]
    pyplot.pcolormesh(linspace(0.00, 0.03, 100), linspace(0.50, 2.50, 100),
                      matrix.T, vmin=0, vmax=1, cmap="cool", shading="gouraud")
    pyplot.text(0.0275, 1.65, "density", va="center", ha="center", fontsize=7)
    colors, locations = pyplot.get_cmap("cool")(linspace(0, 1, 41)), linspace(0.7, 1.5, 41)
    for color, former, latter in zip(colors, locations[:-1], locations[1:]):
        pyplot.fill_between([0.0265, 0.0285], former, latter, fc=color, lw=0, zorder=1)
    for location, info in zip(linspace(0.7, 1.5, 5), linspace(0, 1, 5)):
        pyplot.hlines(location, 0.0260, 0.0265, lw=0.75, color="k", zorder=2)
        pyplot.text(0.0255, location, ("%d" % (info * 100)) + "%", va="center", ha="right", fontsize=6)
    pyplot.plot([0.0265, 0.0265, 0.0285, 0.0285, 0.0265], [0.7, 1.5, 1.5, 0.7, 0.7], lw=0.75, color="k", zorder=2)
    pyplot.text(0.0275, 2.30, math_orders[0], va="center", ha="center", fontsize=8)
    pyplot.xlabel(r"minimum L2 norm difference for $\mathcal{C}$", fontsize=8)
    pyplot.ylabel("best Lipschitz constant", fontsize=8)
    pyplot.xlim(0.00, 0.03)
    pyplot.ylim(0.50, 2.50)
    pyplot.xticks(linspace(0.00, 0.03, 7),
                  ["%.3f" % v for v in linspace(0.00, 0.03, 7)], fontsize=7)
    pyplot.yticks(linspace(0.50, 2.50, 5),
                  ["%.1f" % v for v in linspace(0.50, 2.50, 5)], fontsize=7)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[5:8, 4:7])
    matrix = results[motif_types[1]]
    pyplot.pcolormesh(linspace(0.00, 0.03, 100), linspace(0.50, 2.50, 100),
                      matrix.T, vmin=0, vmax=1, cmap="cool", shading="gouraud")
    pyplot.text(0.0275, 1.65, "density", va="center", ha="center", fontsize=7)
    colors, locations = pyplot.get_cmap("cool")(linspace(0, 1, 41)), linspace(0.7, 1.5, 41)
    for color, former, latter in zip(colors, locations[:-1], locations[1:]):
        pyplot.fill_between([0.0265, 0.0285], former, latter, fc=color, lw=0, zorder=1)
    for location, info in zip(linspace(0.7, 1.5, 5), linspace(0, 1, 5)):
        pyplot.hlines(location, 0.0260, 0.0265, lw=0.75, color="k", zorder=2)
        pyplot.text(0.0255, location, ("%d" % (info * 100)) + "%", va="center", ha="right", fontsize=6)
    pyplot.plot([0.0265, 0.0265, 0.0285, 0.0285, 0.0265], [0.7, 1.5, 1.5, 0.7, 0.7], lw=0.75, color="k", zorder=2)
    pyplot.text(0.0275, 2.30, math_orders[1], va="center", ha="center", fontsize=8)
    pyplot.xlabel(r"minimum L2 norm difference for $\mathcal{C}$", fontsize=8)
    pyplot.ylabel("best Lipschitz constant", fontsize=8)
    pyplot.xlim(0.00, 0.03)
    pyplot.ylim(0.50, 2.50)
    pyplot.xticks(linspace(0.00, 0.03, 7),
                  ["%.3f" % v for v in linspace(0.00, 0.03, 7)], fontsize=7)
    pyplot.yticks(linspace(0.50, 2.50, 5),
                  ["%.1f" % v for v in linspace(0.50, 2.50, 5)], fontsize=7)

    results = task_data["e"]

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[2:5, 7:])
    matrix, count = results[motif_types[0]]
    matrix[matrix == 0] = nan
    pyplot.pcolormesh(linspace(0.00, 0.15, 50), linspace(0.00, 4.00, 50),
                      matrix.T, vmin=0, vmax=3, cmap="Greys")
    pyplot.text(0.015, 1.5, "%.2f" % (count * 100) + "%", va="center", ha="center", fontsize=7)
    pyplot.plot([0.00, 0.03, 0.03, 0.00], [0.5, 0.5, 2.5, 2.5], lw=0.75, ls="--", color="k")
    pyplot.text(0.165, 3.60, math_orders[0], va="center", ha="center", fontsize=8)
    pyplot.xlabel(r"minimum L2 norm difference for $\mathcal{C}$", fontsize=8)
    pyplot.ylabel("best Lipschitz constant", fontsize=8)
    pyplot.xlim(0.00, 0.18)
    pyplot.ylim(0.00, 4.00)
    pyplot.xticks(linspace(0.00, 0.18, 7),
                  ["%.2f" % v for v in linspace(0.00, 0.18, 7)], fontsize=7)
    pyplot.yticks(linspace(0.00, 4.00, 5),
                  ["%.1f" % v for v in linspace(0.00, 4.00, 5)], fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[5:8, 7:])
    matrix, count = results[motif_types[1]]
    matrix[matrix == 0] = nan
    pyplot.pcolormesh(linspace(0.00, 0.15, 50), linspace(0.00, 4.00, 50),
                      matrix.T, vmin=0, vmax=3, cmap="Greys")
    pyplot.text(0.015, 1.5, "%.2f" % (count * 100) + "%", va="center", ha="center", fontsize=7)
    pyplot.plot([0.00, 0.03, 0.03, 0.00], [0.5, 0.5, 2.5, 2.5], lw=0.75, ls="--", color="k")
    pyplot.text(0.165, 3.60, math_orders[1], va="center", ha="center", fontsize=8)
    pyplot.xlabel(r"minimum L2 norm difference for $\mathcal{C}$", fontsize=8)
    pyplot.ylabel("best Lipschitz constant", fontsize=8)
    pyplot.xlim(0.00, 0.18)
    pyplot.ylim(0.00, 4.00)
    pyplot.xticks(linspace(0.00, 0.18, 7),
                  ["%.2f" % v for v in linspace(0.00, 0.18, 7)], fontsize=7)
    pyplot.yticks(linspace(0.00, 4.00, 5),
                  ["%.1f" % v for v in linspace(0.00, 4.00, 5)], fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.386, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.698, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.386, 0.75, "d", va="center", ha="center", fontsize=12)
    figure.text(0.698, 0.75, "e", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "main01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_02():
    """
    Create Figure 2 in the main text.
    """
    task_data = load_data(sort_path + "main02.pkl")

    figure = pyplot.figure(figsize=(10, 7), tight_layout=True)
    grid = pyplot.GridSpec(3, 2)

    # noinspection PyTypeChecker
    panel = pyplot.subplot(grid[0, :])

    flag = 0
    for index in arange(4) * 1.5:
        if flag % 2 == 0:
            pyplot.text(index + 1.25, 1.65, "round " + str(flag // 2 + 1), va="center", ha="center", fontsize=8)
            pyplot.plot([index - 0.1, index - 0.1, index + 2.6, index + 2.6], [1.4, 1.5, 1.5, 1.4], lw=0.75, color="k")
            pyplot.text(index + 0.5, 1.2, "minimize difference by\ntraining the sample in " + r"$\mathcal{C}$",
                        va="center", ha="center", fontsize=7)
        else:
            pyplot.text(index + 0.5, 1.2,
                        "maximize difference by\ntraining the sample in " + r"$\mathcal{L}_i$ or $\mathcal{L}_c$",
                        va="center", ha="center", fontsize=7)
        pyplot.annotate("", xy=(index + 1.0, 0.5), xytext=(index + 1.5, 0.5),
                        arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=4, shrinkB=4, lw=0.75))
        panel.add_patch(patches.Ellipse(xy=(index + 0.5, 0.5), width=1.0, height=0.9, fc="#E8A575"))
        panel.add_patch(patches.Ellipse(xy=(index + 0.5, 0.5), width=0.4, height=0.3, angle=45, fc="#88CCF8"))
        flag += 1
    pyplot.text(6.25, 0.49, r"$\cdots$", va="center", ha="center", fontsize=20)

    pyplot.text(8.25, 1.65, "round n", va="center", ha="center", fontsize=8)
    pyplot.plot([6.9, 6.9, 9.6, 9.6], [1.4, 1.5, 1.5, 1.4], lw=0.75, color="k")
    pyplot.text(7.5, 1.2, "minimize difference by\ntraining the sample in " + r"$\mathcal{C}$",
                va="center", ha="center", fontsize=7)
    pyplot.text(9.0, 1.2,
                "maximize difference by\ntraining the sample in " + r"$\mathcal{L}_i$ or $\mathcal{L}_c$",
                va="center", ha="center", fontsize=7)
    for index in [7.0, 8.5]:
        pyplot.annotate("", xy=(index - 0.5, 0.5), xytext=(index, 0.5),
                        arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=4, shrinkB=4, lw=0.75))
        panel.add_patch(patches.Ellipse(xy=(index + 0.5, 0.5), width=1.0, height=0.9, fc="#E8A575"))
        panel.add_patch(patches.Ellipse(xy=(index + 0.5, 0.5), width=0.4, height=0.3, angle=45, fc="#88CCF8"))

    pyplot.scatter([0.50, 2.02], [0.50, 0.65], s=20, ec="k", fc="w", lw=0.75, zorder=2)
    pyplot.scatter([0.52], [0.65], s=20, ec="k", fc="grey", lw=0.75, zorder=2)
    pyplot.plot([0.50, 0.52], [0.50, 0.65], color="k", lw=0.75, zorder=1)

    pyplot.scatter([0.52, 2.03], [0.80, 0.80], s=20, marker="^", ec="k", fc="w", lw=0.75, zorder=2)
    pyplot.scatter([2.15], [0.92], s=20, marker="^", lw=0.75, ec="k", fc="grey", zorder=2)
    pyplot.plot([2.03, 2.15], [0.80, 0.92], color="k", lw=0.75, zorder=1)

    pyplot.scatter([3.50, 5.15], [0.50, 0.64], s=20, ec="k", fc="w", lw=0.75, zorder=2)
    # pyplot.scatter([3.52, 5.15], [0.65, 0.64], s=20, ec="k", fc="w", lw=0.75, zorder=2)
    pyplot.scatter([3.65], [0.64], s=20, ec="k", fc="grey", lw=0.75, zorder=2)
    pyplot.plot([3.50, 3.65], [0.50, 0.64], color="k", lw=0.75, zorder=1)
    # pyplot.plot([3.52, 3.65], [0.65, 0.64], color="k", lw=0.75, zorder=1)

    pyplot.scatter([3.65, 5.15], [0.92, 0.92], s=20, marker="^", lw=0.75, ec="k", fc="w", zorder=2)
    pyplot.scatter([5.40], [0.80], s=20, marker="^", lw=0.75, ec="k", fc="grey", zorder=2)
    pyplot.plot([5.15, 5.40], [0.92, 0.80], color="k", lw=0.75, zorder=1)

    pyplot.scatter([7.50, 9.18], [0.50, 0.54], s=20, ec="k", fc="w", lw=0.75, zorder=2)
    # pyplot.scatter([7.65, 9.18], [0.44, 0.54], s=20, ec="k", fc="w", lw=0.75, zorder=2)
    pyplot.scatter([7.68], [0.54], s=20, ec="k", fc="grey", lw=0.75, zorder=2)
    pyplot.plot([7.50, 7.68], [0.50, 0.54], color="k", lw=0.75, zorder=1)
    # pyplot.plot([7.65, 7.68], [0.44, 0.54], color="k", lw=0.75, zorder=1)

    pyplot.scatter([7.97, 9.47], [0.35, 0.35], s=20, marker="^", lw=0.75, ec="k", fc="w", zorder=2)
    pyplot.scatter([9.50], [0.49], s=20, marker="^", lw=0.75, ec="k", fc="grey", zorder=2)
    pyplot.plot([9.47, 9.50], [0.35, 0.49], color="k", lw=0.75, zorder=1)

    pyplot.plot([9.18, 9.50], [0.54, 0.49], lw=1.5, color="k", ls=":", zorder=1)

    pyplot.fill_between([0.00, 9.50], -0.65, -0.05, lw=0, fc="#EEEEEE", zorder=1)
    pyplot.scatter([0.14], [-0.20], s=50, marker="s", lw=0, fc="#88CCF8")
    pyplot.text(0.25, -0.20, r"hypothesized representational hyperspace of $\mathcal{C}$",
                va="center", ha="left", fontsize=8)
    pyplot.scatter([0.14], [-0.50], s=50, marker="s", lw=0, fc="#E8A575")
    pyplot.text(0.25, -0.50, r"hypothesized representational hyperspace of $\mathcal{L}_i$ or $\mathcal{L}_c$",
                va="center", ha="left", fontsize=8)

    pyplot.scatter([3.3], [-0.2], s=20, ec="k", fc="w", lw=0.75)
    pyplot.text(3.4, -0.2, r"current catcher: sample in $\mathcal{C}$",
                va="center", ha="left", fontsize=8)
    pyplot.scatter([3.3], [-0.5], s=20, marker="^", ec="k", fc="w", lw=0.75)
    pyplot.text(3.4, -0.5, r"current escaper: sample in $\mathcal{L}_i$ or $\mathcal{L}_c$",
                va="center", ha="left", fontsize=8)

    pyplot.scatter([5.5], [-0.2], s=20, ec="k", fc="grey", lw=0.75)
    pyplot.text(5.6, -0.2, r"trained catcher: sample in $\mathcal{C}$",
                va="center", ha="left", fontsize=8)
    pyplot.scatter([5.5], [-0.5], s=20, marker="^", ec="k", fc="grey", lw=0.75)
    pyplot.text(5.6, -0.5, r"trained escaper: sample in $\mathcal{L}_i$ or $\mathcal{L}_c$",
                va="center", ha="left", fontsize=8)

    pyplot.plot([7.65, 7.80], [-0.2, -0.2], color="k", lw=0.75)
    pyplot.text(7.85, -0.2, "output landscape change path",
                va="center", ha="left", fontsize=8)
    pyplot.plot([7.65, 7.80], [-0.5, -0.5], color="k", lw=1.50, ls=":")
    pyplot.text(7.85, -0.5, "maximum-minimum distance",
                va="center", ha="left", fontsize=8)

    pyplot.xlim(-0.10, 9.70)
    pyplot.ylim(-0.80, 1.80)
    pyplot.axis("off")

    x, y = linspace(0.2, 0.8, 41), linspace(0.2, 0.8, 41)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[1, 0])
    former, latter = task_data["b"][0], task_data["b"][1]
    pyplot.text(1.50, 1.7, "+", va="center", ha="center", fontsize=12)
    pyplot.text(3.50, 1.7, "=", va="center", ha="center", fontsize=12)
    pyplot.annotate("", xy=(0.50, 1.15), xytext=(0.50, 0.95),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(4.50, 1.15), xytext=(4.50, 0.95),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.60, 1.05, "change to\ncontour map", va="center", ha="left", fontsize=7)
    pyplot.text(4.60, 1.05, "change to\ncontour map", va="center", ha="left", fontsize=7)
    pyplot.text(2.50, 0.40,
                r"The escape method of $\mathcal{L}_i$ lies in" + "\n" +
                "extending the \"ridge/valley\" line", color="red",
                va="center", ha="center", fontsize=7)
    # pyplot.text(2.50, 0.40,
    #             r"The escape method of $\mathcal{L}_i$ lies in" + "\n" +
    #             "symmetry of one diagonal plane and\n" +
    #             "monotonicity of another diagonal plane",
    #             va="center", ha="center", fontsize=7)
    # pyplot.text(2.50, 0.60,
    #             r"The escape method of $\mathcal{L}_i$ lies in" + "\n" +
    #             "symmetry of one diagonal plane and\n" +
    #             "monotonicity of another diagonal plane",
    #             va="center", ha="center", fontsize=7)
    # pyplot.text(2.50, 0.00,
    #             "\"slope\" to \"ridge\" or \"valley\"", color="red",
    #             va="center", ha="center", fontsize=7)
    for index, landscape, info in zip([0.0, 2.0, 4.0],
                                      [former, latter - former, latter],
                                      ["former\nlandscape", "change\nfor escape", "latter\nlandscape"]):

        pyplot.pcolormesh(x + index, y + 1.2, landscape, vmin=-1, vmax=1, cmap="PRGn", shading="gouraud")
        pyplot.plot([0.2 + index, 0.8 + index, 0.8 + index, 0.2 + index, 0.2 + index],
                    [1.4, 1.4, 2.0, 2.0, 1.4], lw=0.75, color="k", zorder=2)
        pyplot.text(index + 0.50, 2.15, info, va="center", ha="center", fontsize=7)
        pyplot.text(index + 0.50, 1.30, "$x$", va="center", ha="center", fontsize=8)
        pyplot.text(index + 0.10, 1.70, "$y$", va="center", ha="center", fontsize=8)

        xx, yy = meshgrid(linspace(0.2, 0.8, 41) + index, linspace(0.1, 0.7, 41))
        if index in [0.0, 4.0]:
            print(index, landscape.shape, max(landscape), where(landscape == 1.0))
            ridge_path, previous_max_value = [(where(landscape == 1.0)[0][0], where(landscape == 1.0)[1][0])], 1.0
            while True:
                current_x, current_y = ridge_path[-1]
                used_x, used_y, max_value = None, None, -1
                for move_x in [-1, 0, +1]:
                    for move_y in [-1, 0, +1]:
                        if 0 <= current_x + move_x <= 40 and 0 <= current_y + move_y <= 40:
                            if (current_x + move_x, current_y + move_y) not in ridge_path:
                                if landscape[current_x + move_x, current_y + move_y] <= max_value:
                                    continue
                                if previous_max_value < max_value:
                                    continue
                                used_x, used_y = current_x + move_x, current_y + move_y
                                max_value = landscape[current_x + move_x, current_y + move_y]
                previous_max_value = max_value
                if used_x is not None:
                    ridge_path.append((used_x, used_y))
                else:
                    break
            print(ridge_path)
            pyplot.contour(xx, yy, landscape, linewidths=0.75, vmin=-1, vmax=1, cmap="PRGn")
            pyplot.plot([0.2 + index, 0.8 + index, 0.8 + index, 0.2 + index, 0.2 + index],
                        [0.1, 0.1, 0.7, 0.7, 0.1], lw=0.75, color="k", zorder=2)
            # used_x, used_y = [], []
            # for x_i, y_i in ridge_path:
            #     used_x.append((linspace(0.2, 0.8, 41) + index)[x_i])
            #     used_y.append((linspace(0.1, 0.7, 41))[y_i])
            #
            # pyplot.plot(used_x, used_y, color="k", lw=0.75)
            pyplot.text(index + 0.50, 0.00, "$x$", va="center", ha="center", fontsize=8)
            pyplot.text(index + 0.10, 0.40, "$y$", va="center", ha="center", fontsize=8)

    #     if index in [0.0, 4.0]:
    #         pyplot.plot([index + 0.20, index + 0.20, index + 0.80], [0.9, 0.5, 0.5], lw=0.75, color="k", zorder=2)
    #         pyplot.plot([index + 0.20, index + 0.20, index + 0.80], [0.2, -0.2, -0.2], lw=0.75, color="k", zorder=2)
    #         values = []
    #         for location in range(len(landscape)):
    #             values.append(landscape[location, location])
    #         values = (array(values) + 1.0) * 0.20 + 0.50
    #         pyplot.fill_between(index + linspace(0.20, 0.80, len(values)), 0.5, values, fc="silver", lw=0)
    #         pyplot.plot(index + linspace(0.20, 0.80, len(values)), values, lw=0.75, color="k")
    #         pyplot.text(index + 1.05, 0.80, "$x=y$", va="center", ha="center", fontsize=7, zorder=2)
    #         pyplot.text(index + 0.10, 0.70, "$z$", va="center", ha="center", fontsize=8)
    #         pyplot.text(index + 0.50, 0.35, "tangent plane", va="center", ha="center", fontsize=7)
    #         values = []
    #         for location in range(len(landscape)):
    #             values.append(landscape[location, len(landscape) - location - 1])
    #         values = (array(values) + 1.0) * 0.20 - 0.20
    #         pyplot.fill_between(index + linspace(0.20, 0.80, len(values)), -0.20, values, fc="silver", lw=0)
    #         pyplot.plot(index + linspace(0.20, 0.80, len(values)), values, lw=0.75, color="k")
    #         pyplot.text(index + 1.05, 0.10, "$x+y=1$", va="center", ha="center", fontsize=7, zorder=2)
    #         pyplot.text(index + 0.10, 0.00, "$z$", va="center", ha="center", fontsize=8)
    #         pyplot.text(index + 0.50, -0.35, "tangent plane", va="center", ha="center", fontsize=7)
    locations, colors = linspace(0.0, 1.9, 51), pyplot.get_cmap("PRGn")(linspace(0, 1, 50))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([5.7, 5.8], former, latter, fc=color, lw=0, zorder=1)
    pyplot.text(5.75, 2.00, "$z$", va="center", ha="center", fontsize=8)
    pyplot.plot([5.7, 5.7, 5.8, 5.8, 5.7], [0.0, 1.9, 1.9, 0.0, 0.0], lw=0.75, color="k", zorder=2)
    for location, info in zip([0.00, 0.95, 1.90], ["-1", "0", "+1"]):
        pyplot.hlines(location, 5.65, 5.70, lw=0.75, color="k")
        pyplot.text(5.60, location, info, va="center", ha="right", fontsize=7)
    pyplot.xlim(0.10, 6.0)
    pyplot.ylim(-0.5, 2.3)
    pyplot.axis("off")

    # noinspection PyTypeChecker
    pyplot.subplot(grid[1, 1])
    former, latter = task_data["c"][0], task_data["c"][1]
    pyplot.plot([0.2, 0.2, 2.8, 2.8], [1.15, 1.10, 1.10, 1.15], lw=0.75, color="k")
    pyplot.annotate("", xy=(1.50, 1.10), xytext=(1.50, 0.95),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(4.20, 0.40,
                r"The escape method of $\mathcal{L}_c$ lies in" + "\n" +
                "adjusting the high gradient region", color="red",
                va="center", ha="center", fontsize=7)
    # pyplot.text(4.20, 0.60,
    #             r"The escape method of $\mathcal{L}_c$ lies in" + "\n" +
    #             "adjusting high gradient region",
    #             va="center", ha="center", fontsize=7)
    # pyplot.text(4.20, 0.00,
    #             "seek steeper \"slope\"", color="red",
    #             va="center", ha="center", fontsize=7)
    pyplot.text(1.50, 1.7, "+", va="center", ha="center", fontsize=12)
    pyplot.text(3.50, 1.7, "=", va="center", ha="center", fontsize=12)
    for index, landscape, info in zip([0.0, 2.0, 4.0],
                                      [former, latter - former, latter],
                                      ["former\nlandscape", "change\nfor escape", "latter\nlandscape"]):
        pyplot.pcolormesh(x + index, y + 1.2, landscape, vmin=-1, vmax=1, cmap="PRGn")
        pyplot.plot([0.2 + index, 0.8 + index, 0.8 + index, 0.2 + index, 0.2 + index],
                    [1.4, 1.4, 2.0, 2.0, 1.4], lw=0.75, color="k", zorder=2)
        pyplot.text(index + 0.50, 2.15, info, va="center", ha="center", fontsize=7)
        pyplot.text(index + 0.50, 1.30, "$x$", va="center", ha="center", fontsize=8)
        pyplot.text(index + 0.10, 1.70, "$y$", va="center", ha="center", fontsize=8)
    values_x = task_data["c"][2].reshape(-1)
    values_y = abs(latter - former).reshape(-1)
    correlation, p_value = spearmanr(values_x, values_y)
    if p_value > 1e-10:
        info = "Spearman = %.2f\nP-value     = %.2e" % (correlation, p_value)
        pyplot.text(2.90, 1.12, info, va="center", ha="left", fontsize=7)
    else:
        info = "Spearman = %.2f\nP-value     < 1E-10" % correlation
        pyplot.text(2.90, 1.12, info, va="center", ha="left", fontsize=7)

    bounds = percentile(values_x, [25, 75])
    lower_bound, upper_bound = bounds[0] - 1.5 * (bounds[1] - bounds[0]), bounds[1] + 1.5 * (bounds[1] - bounds[0])

    used_x, used_y = [], []
    for x, y in zip(values_x, values_y):
        if lower_bound <= x <= upper_bound:
            used_x.append(x)
            used_y.append(y)
    used_x, used_y = array(used_x), array(used_y)
    used_x -= min(used_x)
    used_x /= max(used_x)
    used_y -= min(used_y)
    used_y /= max(used_y)
    used_x, used_y = used_x * 2.2 + 0.4, used_y * 0.8

    pyplot.scatter(used_x, used_y, color="k", alpha=0.1)

    locations, colors = linspace(0.0, 1.9, 51), pyplot.get_cmap("PRGn")(linspace(0, 1, 50))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([5.7, 5.8], former, latter, fc=color, lw=0, zorder=1)
    pyplot.text(5.75, 2.00, "$z$", va="center", ha="center", fontsize=8)
    pyplot.plot([5.7, 5.7, 5.8, 5.8, 5.7], [0.0, 1.9, 1.9, 0.0, 0.0], lw=0.75, color="k", zorder=2)
    for location, info in zip([0.00, 0.95, 1.90], ["-1", "0", "+1"]):
        pyplot.hlines(location, 5.65, 5.70, lw=0.75, color="k")
        pyplot.text(5.60, location, info, va="center", ha="right", fontsize=7)
    pyplot.plot([0.2, 0.2, 2.8], [0.9, -0.2, -0.2], lw=0.75, color="k", zorder=2)
    pyplot.text(0.12, 0.35, r"normalized $z$ change", rotation=90, va="center", ha="center", fontsize=7)
    pyplot.text(1.5, -0.35, "normalized gradient in former landscape", va="center", ha="center", fontsize=7)
    pyplot.xlim(0.10, 6.0)
    pyplot.ylim(-0.5, 2.3)
    pyplot.axis("off")

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2, 0])
    pyplot.text(3.1, 0.05, "length of \"ridge/valley\" line", va="center", ha="center", fontsize=7)
    pyplot.hlines(0.3, 0.2, 6.0, lw=0.75, color="k")
    pyplot.xlim(0.1, 6.2)
    pyplot.ylim(0.0, 2.1)
    # pyplot.axis("off")

    pyplot.xticks([])
    pyplot.yticks([])

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2, 1])
    pyplot.text(3.1, 0.05, "Spearman's rank correlation coefficient", va="center", ha="center", fontsize=7)
    pyplot.hlines(0.3, 0.2, 6.0, lw=0.75, color="k")

    values = task_data["e"][:, 0]

    labels = ["-0.3 ~ 0.0", "0.0 ~ 0.2", "0.2 ~ 0.4", "0.4 ~ 0.6", "0.6 ~ 0.8", "0.8 ~ 1.0"]
    for location, label in zip(linspace(0.4 + 0.45, 5.8 - 0.45, 6), labels):
        pyplot.vlines(location, 0.25, 0.30, lw=0.75, color="k")
        pyplot.text(location, 0.15, label, va="center", ha="center", fontsize=7)

    counts = [
        len(values[values < 0.0]),
        len(values[where((values >= 0.0) & (values < 0.2))]),
        len(values[where((values >= 0.2) & (values < 0.4))]),
        len(values[where((values >= 0.4) & (values < 0.6))]),
        len(values[where((values >= 0.6) & (values < 0.8))]),
        len(values[values >= 0.8])
    ]

    for index, (location, count) in enumerate(zip(linspace(0.4 + 0.45, 5.8 - 0.45, 6), counts)):
        height = count / 400.0 * 3.5
        if index > 0:
            pyplot.fill_between([location - 0.35, location + 0.35], 0.3, 0.3 + height,
                                ec="k", fc="#FA7F6F", lw=0.75)
        else:
            pyplot.fill_between([location - 0.35, location + 0.35], 0.3, 0.3 + height,
                                ec="k", fc="#BEB8DC", lw=0.75)

        if count >= 100:
            pyplot.text(location, 0.32 + height, str(count) + " / 400", va="bottom", ha="center", fontsize=7)
        else:
            pyplot.text(location, 0.32 + height, " " + str(count) + " / 400", va="bottom", ha="center", fontsize=7)

    pyplot.xlim(0.1, 6.2)
    pyplot.ylim(0.0, 2.1)
    pyplot.axis("off")

    figure.text(0.02, 0.98, "a", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.67, "b", va="center", ha="center", fontsize=12)
    figure.text(0.51, 0.67, "c", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.34, "d", va="center", ha="center", fontsize=12)
    figure.text(0.51, 0.34, "e", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "main02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_03():
    """
    Create Figure 3 in the main text.
    """
    task_data = load_data(sort_path + "main03.pkl")

    figure = pyplot.figure(figsize=(10, 4.8), tight_layout=True)
    grid = pyplot.GridSpec(2, 4)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[0, :2])
    pyplot.plot([0.05, 0.95], [0.15, 0.15], color="silver", lw=10, zorder=0)
    pyplot.fill_between([0.60, 0.80], 0.03, 0.27, fc="gray", ec="k", lw=0, zorder=1)
    pyplot.text(0.70, -0.05, "cart", va="center", ha="center", fontsize=9)
    pyplot.plot([0.70, 0.50], [0.15, 0.95], color="#CC9966", lw=10, zorder=0)
    pyplot.text(0.46, 1.05, "pole", va="center", ha="center", fontsize=9)
    pyplot.scatter([0.70], [0.15], s=170, fc="silver", ec="k", lw=0, zorder=3)
    pyplot.annotate("", xy=(0.60, 0.15), xytext=(0.50, 0.15),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75))
    pyplot.text(0.50, 0.15, "push", va="center", ha="right", fontsize=8)
    pyplot.plot([0.61, 0.70, 0.70], [0.52, 0.15, 0.60], color="k", lw=0.75, ls="--", zorder=4)
    pyplot.annotate("", xy=(0.61, 0.52), xytext=(0.70, 0.60),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.75,
                                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.3"))
    pyplot.text(0.64, 0.67, "angle", va="center", ha="center", fontsize=8)
    pyplot.annotate("", xy=(0.38, 0.75), xytext=(0.50, 0.95),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75, shrinkA=0, shrinkB=0))
    pyplot.text(0.32, 0.70, "angular\nvelocity", va="center", ha="center", fontsize=8)
    pyplot.xlim(0, 1)
    pyplot.ylim(-0.1, 1.02)
    pyplot.axis("off")

    labels = ["default", r"$\mathcal{L}_c + \mathcal{C}$", r"$\mathcal{L}_i + \mathcal{C}$", r"$\mathcal{C}$"]

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[0, 2:])
    for index, (label, color) in enumerate(zip(labels, pyplot.get_cmap("binary")(linspace(0.0, 0.8, 4)))):
        locations = arange(5) - 0.3 + 0.2 * index
        pyplot.bar(locations, task_data["b"][index], width=0.2, fc=color, ec="k", lw=0.75, label=label)
    flag = True
    for index in range(4):
        locations = arange(5) - 0.3 + 0.2 * index
        for location, value in zip(locations, task_data["b"][index]):
            if value >= 195:
                if flag:
                    pyplot.scatter([location], [value + 2.0], marker="v", s=15, fc="lawngreen", ec="k",
                                   label="effective training")
                    flag = False
                else:
                    pyplot.scatter([location], [value + 2.0], marker="v", s=15, fc="lawngreen", ec="k")

    pyplot.text(4.45, 196, "pass (≥ 195)", va="bottom", ha="right", fontsize=8)
    pyplot.hlines(195, -0.5, 4.5, lw=0.75, ls="--", zorder=2)
    pyplot.legend(loc="lower left", framealpha=1, fontsize=7)
    pyplot.xlabel("training error scale", fontsize=8)
    pyplot.ylabel("average training performance", fontsize=8)
    pyplot.xticks(arange(5), ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
    pyplot.yticks(arange(150, 201, 10), arange(150, 201, 10), fontsize=7)
    pyplot.xlim(-0.5, 4.5)
    pyplot.ylim(150, 202.5)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    for index, (panel_index, label) in enumerate(zip(["c", "d", "e", "f"], labels)):
        # noinspection PyTypeChecker
        pyplot.subplot(grid[1, index])
        pyplot.title(label, fontsize=9)
        values = task_data[panel_index].copy()
        values[values >= 195] = nan
        pyplot.pcolormesh(arange(6), arange(6), values.T, vmin=100, vmax=195, cmap="inferno")
        for location_x in range(5):
            for location_y in range(5):
                value = task_data[panel_index][location_x, location_y]
                if value >= 195:
                    pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "pass",
                                va="center", ha="center", fontsize=8)
                elif value > 140:
                    pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.1f" % value,
                                va="center", ha="center", fontsize=8)
                else:
                    pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.1f" % value, color="w",
                                va="center", ha="center", fontsize=8)

        pyplot.xlabel("training error scale", fontsize=8)
        pyplot.ylabel("evaluating error scale", fontsize=8)
        pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
        pyplot.yticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
        pyplot.xlim(0, 5)
        pyplot.ylim(0, 5)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.513, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.49, "c", va="center", ha="center", fontsize=12)
    figure.text(0.266, 0.49, "d", va="center", ha="center", fontsize=12)
    figure.text(0.513, 0.49, "e", va="center", ha="center", fontsize=12)
    figure.text(0.759, 0.49, "f", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "main03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_04():
    """
    Create Figure 4 in the main text.
    """
    task_data = load_data(sort_path + "main04.pkl")

    figure = pyplot.figure(figsize=(10, 4), tight_layout=True)

    labels = ["default", r"$\mathcal{L}_c + \mathcal{C}$", r"$\mathcal{L}_i + \mathcal{C}$", r"$\mathcal{C}$"]
    colors = ["#BF33B5", "#845EC2", "#D73222"]

    for index, (label, panel_index) in enumerate(zip(labels, ["a", "b", "c", "d"])):
        ax = pyplot.subplot(1, 4, index + 1)
        pyplot.title(label, fontsize=9)
        pyplot.hlines(195, 0, 5, lw=0.75, ls="--", zorder=-1)
        pyplot.text(4.9, 196, "pass (≥ 195)", va="bottom", ha="right", fontsize=8)
        for case_index, (curve, have, total) in enumerate(task_data[panel_index]):
            if curve is not None:
                info = str(have) + " / " + str(total)
                if have < 10:
                    info = "  " + info
                pyplot.plot(arange(5) + 0.5, curve, color=colors[case_index],
                            lw=2.5, zorder=case_index, marker="o",
                            label=str(case_index + 1) + ": " + info, alpha=0.75)
            else:
                info = "  0 / " + str(total)
                pyplot.plot(arange(5) + 0.5, [0, 0, 0, 0, 0], color=colors[case_index],
                            lw=2.5, zorder=case_index, marker="o",
                            label=str(case_index + 1) + ": " + info, alpha=0.75)

        pyplot.legend(loc="lower left", fontsize=8, title="failure type", title_fontsize=7)
        pyplot.xlabel("evaluating error scale", fontsize=8)
        pyplot.ylabel("evaluating performance", fontsize=8)
        pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
        pyplot.yticks(arange(120, 201, 10), arange(120, 201, 10), fontsize=7)
        pyplot.xlim(0, 5)
        pyplot.ylim(118, 202)
        # noinspection PyUnresolvedReferences
        ax.spines["top"].set_visible(False)
        # noinspection PyUnresolvedReferences
        ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.266, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.513, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.759, 0.99, "d", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "main04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    main_01()
    main_02()
    main_03()
    main_04()
