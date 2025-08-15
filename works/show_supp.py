"""
@Author      : Haoling Zhang
@Description : Plot all the figures in the supplementary file.
"""
from logging import getLogger, CRITICAL
from matplotlib import pyplot, rcParams
from numpy import array, arange, linspace, random, meshgrid, abs, log10, sum, min, median, max, mean, argmax, where
from scipy.stats import gaussian_kde
from warnings import filterwarnings

from works import load_data, draw_info

filterwarnings("ignore")

getLogger("matplotlib").setLevel(CRITICAL)

rcParams["font.family"] = "Arial"
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Linux Libertine"
rcParams["mathtext.cal"] = "Lucida Calligraphy"
rcParams["mathtext.it"] = "Linux Libertine:italic"
rcParams["mathtext.bf"] = "Linux Libertine:bold"

raw_path, sort_path, save_path = "./raw/", "./data/", "./show/"


def supp_01():
    """
    Create Figure S1 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp01.pkl")

    figure = pyplot.figure(figsize=(10, 11), tight_layout=True)

    index_change = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    for index, (panel_label, (motif_type, motif_index, (x, y))) in enumerate(task_data.items()):
        pyplot.subplot(5, 2, index_change[index])

        for location in linspace(0.00, 0.04, 5)[1:-1]:
            pyplot.hlines(location, 0.6, 2.4, lw=0.75, ls="--", color="k", zorder=1)

        for location in linspace(0.6, 2.4, 10)[1:-1]:
            pyplot.vlines(location, 0.0, 0.04, lw=0.75, ls="--", color="k", zorder=1)

        if motif_index > 0:
            pyplot.title("samples in " + motif_type + " " + str(motif_index), fontsize=8)
        else:  # total
            pyplot.title("all samples in " + motif_type, fontsize=8)
        pyplot.plot(x, y / sum(y), draw_info[motif_type][0], lw=2, zorder=2)

        pyplot.xlabel("best Lipschitz constant", fontsize=8)
        pyplot.ylabel("proportion", fontsize=8)
        pyplot.xticks(linspace(0.6, 2.4, 10),
                      ["%.1f" % v for v in linspace(0.6, 2.4, 10)], fontsize=7)
        pyplot.yticks(linspace(0.00, 0.04, 5),
                      ["%d" % (v * 100) + "%" for v in linspace(0.00, 0.04, 5)], fontsize=7)
        pyplot.xlim(0.6, 2.4)
        pyplot.ylim(0, 0.04)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.80, "b", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.60, "c", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.40, "d", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.20, "e", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.99, "f", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.80, "g", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.60, "h", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.40, "i", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.20, "j", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_02():
    """
    Create Figure S2 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp02.pkl")

    figure = pyplot.figure(figsize=(10, 9), tight_layout=True)

    index_change = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    for index, (panel_label, (motif_type, motif_index, (x, y))) in enumerate(task_data.items()):
        pyplot.subplot(5, 2, index_change[index])

        for location in linspace(0.00, 0.06, 4)[1:-1]:
            pyplot.hlines(location, 0.00, 0.03, lw=0.75, ls="--", color="k", zorder=1)

        for location in linspace(0.00, 0.03, 10)[1:-1]:
            pyplot.vlines(location, 0.0, 0.06, lw=0.75, ls="--", color="k", zorder=1)

        if motif_index > 0:
            pyplot.title("samples in " + motif_type + " " + str(motif_index), fontsize=8)
        else:  # total
            pyplot.title("all samples in " + motif_type, fontsize=8)
        pyplot.plot(x, y / sum(y), draw_info[motif_type][0], lw=2, zorder=2)

        pyplot.xlabel("best Lipschitz constant", fontsize=8)
        pyplot.ylabel("proportion", fontsize=8)
        pyplot.xticks(linspace(0.6, 2.4, 10),
                      ["%.1f" % v for v in linspace(0.6, 2.4, 10)], fontsize=7)
        pyplot.yticks(linspace(0.00, 0.06, 4),
                      ["%d" % (v * 100) + "%" for v in linspace(0.00, 0.06, 4)], fontsize=7)
        pyplot.xlim(0, 0.03)
        pyplot.ylim(0, 0.06)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.80, "b", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.60, "c", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.40, "d", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.20, "e", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.99, "f", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.80, "g", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.60, "h", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.40, "i", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.20, "j", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_03():
    """
    Create Figure S3 in the supplementary file.
    """
    pyplot.figure(figsize=(10, 2), tight_layout=True)
    pyplot.subplot(1, 2, 1)
    pyplot.scatter([0.2, 0.5, 0.8, 1.8, 2.1, 2.4], [0.2, 0.8, 0.2, 0.2, 0.8, 0.2],
                   ec="k", fc="w", s=40, lw=0.75)

    for location in [0.20, 0.80, 1.80, 2.40]:
        pyplot.annotate("", xy=(location, 0.00), xytext=(location, 0.10),
                        arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="k",
                                        shrinkA=3.2, shrinkB=0.0, lw=0.75), zorder=0)
    for location in [0.50, 2.10]:
        pyplot.annotate("", xy=(location, 0.90), xytext=(location, 1.00),
                        arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="k",
                                        shrinkA=0.0, shrinkB=3.0, lw=0.75), zorder=0)
    pyplot.text(0.20, 0.13, r"$x$", va="center", ha="center", fontsize=9)
    pyplot.text(1.80, 0.13, r"$x$", va="center", ha="center", fontsize=9)
    pyplot.text(0.80, 0.13, r"$y$", va="center", ha="center", fontsize=9)
    pyplot.text(2.40, 0.13, r"$y$", va="center", ha="center", fontsize=9)
    pyplot.text(0.50, 0.87, r"$z$", va="center", ha="center", fontsize=9)
    pyplot.text(2.10, 0.87, r"$z$", va="center", ha="center", fontsize=9)
    for former_point, latter_point in zip([(0.2, 0.2), (0.2, 0.2), (1.8, 0.2), (0.8, 0.2), (2.4, 0.2)],
                                          [(0.8, 0.2), (0.5, 0.8), (2.1, 0.8), (0.5, 0.8), (2.1, 0.8)]):
        pyplot.annotate("", xy=former_point, xytext=latter_point,
                        arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="k",
                                        shrinkA=3.2, shrinkB=3.2, lw=0.75), zorder=0)
    pyplot.hlines(0.51, 0.9, 1.7, lw=0.5, color="k", zorder=1)
    pyplot.hlines(0.49, 0.9, 1.7, lw=0.5, color="k", zorder=1)
    pyplot.text(1.300, 0.570, r"$w \rightarrow 0$", va="center", ha="center", fontsize=9)
    pyplot.fill_between([0.370, 0.635], 0.15, 0.25, fc="w", lw=0, zorder=1)
    pyplot.text(0.500, 0.195, r"$w \cdot x + b$", va="center", ha="center", fontsize=9, zorder=2)
    pyplot.text(2.450, 0.195, r"$+\ b$", va="center", ha="left", fontsize=9)
    pyplot.xlim(0.15, 2.55)
    pyplot.ylim(0.00, 1.00)
    pyplot.axis("off")

    pyplot.savefig(save_path + "supp03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_04():
    """
    Create Figure S4 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp04.pkl")

    x, y = linspace(0.0, 0.6, 41), linspace(0.0, 0.6, 41)

    figure = pyplot.figure(figsize=(10, 5.5), tight_layout=True)
    pyplot.subplot(2, 1, 1)
    pyplot.title("case 43 of coherent loop 1", fontsize=8)
    points = array([[0.0, 1.5], [2.0, 1.5], [4.0, 1.5], [6.0, 1.5], [8.0, 1.5], [10.0, 1.5],
                    [2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0]])
    for index, (x_location, y_location) in enumerate(points):
        pyplot.text(0.5 + x_location, 1.2 + y_location, "iteration\n" + str(index * 10),
                    va="center", ha="center", fontsize=7)
        pyplot.plot([0.2 + x_location, 0.8 + x_location, 0.8 + x_location, 0.2 + x_location, 0.2 + x_location],
                    [0.4 + y_location, 0.4 + y_location, 1.0 + y_location, 1.0 + y_location, 0.4 + y_location],
                    lw=0.75, color="k", zorder=2)
        if index != 0:
            pyplot.hlines(0.7 + y_location, 0.2 + x_location, 0.8 + x_location, lw=0.5, ls=":", color="k", zorder=1)
            pyplot.vlines(0.5 + x_location, 0.4 + y_location, 1.0 + y_location, lw=0.5, ls=":", color="k", zorder=1)
        if 2.0 < x_location <= 10:
            pyplot.annotate("", xy=(x_location - 1.0, 0.7 + y_location), xytext=(x_location - 0.1, 0.7 + y_location),
                            arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
        if x_location == 2.0 and y_location > 0.0:
            pyplot.annotate("", xy=(x_location - 1.0, 0.7 + y_location), xytext=(x_location - 0.1, 0.7 + y_location),
                            arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
        pyplot.text(0.50 + x_location, 0.30 + y_location, "$x$", va="center", ha="center", fontsize=8)
        pyplot.text(0.10 + x_location, 0.70 + y_location, "$y$", va="center", ha="center", fontsize=8)
        pyplot.text(0.10 + x_location, 0.30 + y_location, "$-1$", va="center", ha="center", fontsize=8)
        pyplot.text(0.10 + x_location, 1.05 + y_location, "$+1$", va="center", ha="center", fontsize=8)
        pyplot.text(0.90 + x_location, 0.30 + y_location, "$+1$", va="center", ha="center", fontsize=8)
        pyplot.hlines(0.30 + y_location, 0.21 + x_location, 0.39 + x_location, color="k", lw=0.75, ls="--")
        pyplot.hlines(0.30 + y_location, 0.79 + x_location, 0.61 + x_location, color="k", lw=0.75, ls="--")
        pyplot.vlines(0.10 + x_location, 0.41 + y_location, 0.59 + y_location, color="k", lw=0.75, ls="--")
        pyplot.vlines(0.10 + x_location, 0.81 + y_location, 0.99 + y_location, color="k", lw=0.75, ls="--")
    for index, ((x_location, y_location), landscape) in enumerate(zip(points, task_data["a"])):
        x_bias, y_bias = x_location + 0.2, y_location + 0.4
        pyplot.pcolormesh(x + x_bias, y + y_bias, landscape, vmin=-1, vmax=1, cmap="PRGn", shading="gouraud", zorder=0)

    pyplot.plot([10.5, 10.5, 1.5, 1.5], [1.7, 1.55, 1.55, 0.7], lw=0.75, color="k", zorder=2)
    pyplot.annotate("", xy=(1.5, 0.7), xytext=(1.9, 0.7),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.5, 1.4, r"$z$", va="center", ha="center", fontsize=8)
    pyplot.plot([0.4, 0.6, 0.6, 0.4, 0.4], [0.5, 0.5, 1.3, 1.3, 0.5], lw=0.75, color="k", zorder=2)
    locations, colors = linspace(0.5, 1.3, 41), pyplot.get_cmap("PRGn")(linspace(0, 1, 40))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([0.4, 0.6], former, latter, fc=color, lw=0, zorder=1)
    for location, info in zip([0.5, 0.9, 1.3], ["$-1$", "$0$", "$+1$"]):
        pyplot.hlines(location, 0.35, 0.40, lw=0.75, color="k", zorder=2)
        pyplot.text(0.33, location, info, va="center", ha="right", fontsize=8)
    pyplot.xlim(0.0, 11.0)
    pyplot.ylim(0.2, 2.9)
    pyplot.axis("off")

    pyplot.subplot(2, 1, 2)
    pyplot.title("case 61 of coherent loop 1", fontsize=8)
    points = array([[0.0, 1.5], [2.0, 1.5], [4.0, 1.5], [6.0, 1.5], [8.0, 1.5], [10.0, 1.5],
                    [2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0]])
    for index, (x_location, y_location) in enumerate(points):
        pyplot.text(0.5 + x_location, 1.2 + y_location, "iteration\n" + str(index * 10),
                    va="center", ha="center", fontsize=7)
        pyplot.plot([0.2 + x_location, 0.8 + x_location, 0.8 + x_location, 0.2 + x_location, 0.2 + x_location],
                    [0.4 + y_location, 0.4 + y_location, 1.0 + y_location, 1.0 + y_location, 0.4 + y_location],
                    lw=0.75, color="k", zorder=2)
        if index != 0:
            pyplot.hlines(0.7 + y_location, 0.2 + x_location, 0.8 + x_location, lw=0.5, ls=":", color="k", zorder=1)
            pyplot.vlines(0.5 + x_location, 0.4 + y_location, 1.0 + y_location, lw=0.5, ls=":", color="k", zorder=1)
        if 2.0 < x_location <= 10:
            pyplot.annotate("", xy=(x_location - 1.0, 0.7 + y_location), xytext=(x_location - 0.1, 0.7 + y_location),
                            arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
        if x_location == 2.0 and y_location > 0.0:
            pyplot.annotate("", xy=(x_location - 1.0, 0.7 + y_location), xytext=(x_location - 0.1, 0.7 + y_location),
                            arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
        pyplot.text(0.50 + x_location, 0.30 + y_location, "$x$", va="center", ha="center", fontsize=8)
        pyplot.text(0.10 + x_location, 0.70 + y_location, "$y$", va="center", ha="center", fontsize=8)
        pyplot.text(0.10 + x_location, 0.30 + y_location, "$-1$", va="center", ha="center", fontsize=8)
        pyplot.text(0.10 + x_location, 1.05 + y_location, "$+1$", va="center", ha="center", fontsize=8)
        pyplot.text(0.90 + x_location, 0.30 + y_location, "$+1$", va="center", ha="center", fontsize=8)
        pyplot.hlines(0.30 + y_location, 0.21 + x_location, 0.39 + x_location, color="k", lw=0.75, ls="--")
        pyplot.hlines(0.30 + y_location, 0.79 + x_location, 0.61 + x_location, color="k", lw=0.75, ls="--")
        pyplot.vlines(0.10 + x_location, 0.41 + y_location, 0.59 + y_location, color="k", lw=0.75, ls="--")
        pyplot.vlines(0.10 + x_location, 0.81 + y_location, 0.99 + y_location, color="k", lw=0.75, ls="--")
    for index, ((x_location, y_location), landscape) in enumerate(zip(points, task_data["b"])):
        x_bias, y_bias = x_location + 0.2, y_location + 0.4
        pyplot.pcolormesh(x + x_bias, y + y_bias, landscape, vmin=-1, vmax=1, cmap="PRGn", shading="gouraud", zorder=0)

    pyplot.plot([10.5, 10.5, 1.5, 1.5], [1.7, 1.55, 1.55, 0.7], lw=0.75, color="k", zorder=2)
    pyplot.annotate("", xy=(1.5, 0.7), xytext=(1.9, 0.7),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.5, 1.4, r"$z$", va="center", ha="center", fontsize=8)
    pyplot.plot([0.4, 0.6, 0.6, 0.4, 0.4], [0.5, 0.5, 1.3, 1.3, 0.5], lw=0.75, color="k", zorder=2)
    locations, colors = linspace(0.5, 1.3, 41), pyplot.get_cmap("PRGn")(linspace(0, 1, 40))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([0.4, 0.6], former, latter, fc=color, lw=0, zorder=1)
    for location, info in zip([0.5, 0.9, 1.3], ["$-1$", "$0$", "$+1$"]):
        pyplot.hlines(location, 0.35, 0.40, lw=0.75, color="k", zorder=2)
        pyplot.text(0.33, location, info, va="center", ha="right", fontsize=8)
    pyplot.xlim(0.0, 11.0)
    pyplot.ylim(0.2, 2.9)
    pyplot.axis("off")

    figure.align_labels()
    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.49, "b", va="center", ha="center", fontsize=12)
    pyplot.savefig(save_path + "supp04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_05():
    """
    Create Figure S5 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp05.pkl")

    figure, mesh, all_ax = pyplot.figure(figsize=(10, 10)), None, []
    grid = pyplot.GridSpec(2, 2)
    for index, (_, matrix) in enumerate(task_data.items()):
        # noinspection PyTypeChecker
        ax = pyplot.subplot(grid[index // 2, index % 2])
        all_ax.append(ax)
        pyplot.title("100 samples in coherent-loop " + str(index + 1), fontsize=8)
        for sample_index in range(len(matrix)):
            matrix[sample_index] /= max(matrix[sample_index])
        sorted_matrix = array(sorted(matrix, key=lambda row: argmax(row)))
        mesh = pyplot.pcolormesh(linspace(0, 100, 102), linspace(0, 100, 101),
                                 sorted_matrix, cmap="rainbow", zorder=0)
        points = []
        for distribution in sorted_matrix:
            points.append(argmax(distribution))
        for location, (former_point, latter_point) in enumerate(zip(points[:-1], points[1:])):
            if former_point < 50 < latter_point:
                pyplot.hlines(location + 1, 0, 100, lw=0.75, ls="--", color="k", zorder=2)
                break
        pyplot.plot(points, linspace(0, 100, 100), lw=1, color="k", zorder=2)
        pyplot.vlines(50, 0, 100, lw=0.75, ls="--", color="k", zorder=2)
        pyplot.xlabel("proportion of Spearman's rank correlation coefficient (per round)", fontsize=8)
        pyplot.ylabel("sample index ordered by the peak position", fontsize=8)
        pyplot.xticks(linspace(0, 100, 11),
                      ["-1.0", "-0.8", "-0.6", "-0.4", "-0.2", "0.0", "+0.2", "+0.4", "+0.6", "+0.8", "+1.0"],
                      fontsize=7)
        pyplot.yticks([])
        pyplot.xlim(0, 100)
        pyplot.ylim(0, 100)

    # noinspection PyTypeChecker
    cbar = figure.colorbar(mesh, ax=all_ax, cax=figure.add_axes([0.1, 0.05, 0.8, 0.015]), orientation="horizontal")
    cbar.set_label("normalized proportion density", fontsize=8)
    cbar.set_ticks(linspace(0, 1, 21))
    cbar.set_ticklabels(["%.2f" % v for v in linspace(0, 1, 21)])
    cbar.ax.xaxis.set_tick_params(labelsize=7)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.53, "c", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.53, "d", va="center", ha="center", fontsize=12)

    # noinspection PyTypeChecker
    pyplot.tight_layout(rect=[0.00, 0.07, 1.00, 1.00])
    pyplot.savefig(save_path + "supp05.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_06():
    """
    Create Figure S6 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp06.pkl")

    figure = pyplot.figure(figsize=(10, 9.5), tight_layout=True)
    for index, (panel_index, values) in enumerate(task_data.items()):
        pyplot.subplot(2, 2, index + 1)
        count = sum((values[:, 1] - values[:, 0]) > 0)
        pyplot.title("samples in incoherent-loop " + str(index + 1), fontsize=8)
        pyplot.fill_between([0, 1], [0, 1], [1, 1], lw=0, fc="#FEB2B4", alpha=0.5, zorder=0,
                            label="increase (" + str(count) + " samples)")
        pyplot.fill_between([0, 1], [0, 0], [0, 1], lw=0, fc="#A5B6C5", alpha=0.5, zorder=0,
                            label="decrease (" + str(100 - count) + " samples)")
        for location in linspace(0.1, 0.9, 9):
            pyplot.hlines(location, 0, 1, lw=0.75, ls="--", color="k", zorder=1)
            pyplot.vlines(location, 0, 1, lw=0.75, ls="--", color="k", zorder=1)
        pyplot.legend(loc="lower right", fontsize=7, title="proportion change", title_fontsize=7, framealpha=1)
        pyplot.scatter(values[:, 0], values[:, 1], ec="k", fc="w", lw=0.75, zorder=2)
        pyplot.xlabel("predominant curvature proportion of landscape before escaping", fontsize=8)
        pyplot.ylabel("predominant curvature proportion of landscape after escaping", fontsize=8)
        pyplot.xticks(linspace(0.0, 1.0, 11), [("%d" % v) + "%" for v in arange(0, 101, 10)], fontsize=7)
        pyplot.yticks(linspace(0.0, 1.0, 11), [("%d" % v) + "%" for v in arange(0, 101, 10)], fontsize=7)
        pyplot.xlim(0.0, 1.0)
        pyplot.ylim(0.0, 1.0)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.50, "c", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.50, "d", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp06.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_07():
    """
    Create Figure S7 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp07.pkl")

    figure = pyplot.figure(figsize=(10, 8), tight_layout=True)
    grid = pyplot.GridSpec(3, 1)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[:2, 0])
    source_1, target_1, source_2, target_2, source_3, target_3 = task_data["a"]
    pyplot.text(0.50, 2.03, "curvature feature", va="center", ha="center", fontsize=8)
    locations, colors = linspace(0.2, 0.8, 4), pyplot.get_cmap("binary")(linspace(0, 1, 3))
    for former, latter, color, label in zip(locations[:-1], locations[1:], colors, ["concave", "unknown", "convex"]):
        pyplot.fill_between([former, latter], 1.95, 2.00, fc=color, lw=0, zorder=1)
        pyplot.text((former + latter) / 2.0, 1.92, label, va="center", ha="center", fontsize=7)
    pyplot.plot([0.2, 0.8, 0.8, 0.2, 0.2], [1.95, 1.95, 2.00, 2.00, 1.95], lw=0.75, color="k", zorder=2)
    pyplot.text(2.00, 2.03, "z value", va="center", ha="center", fontsize=8)
    locations, colors = linspace(1.2, 2.8, 100), pyplot.get_cmap("PRGn")(linspace(0, 1, 100))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 1.95, 2.00, fc=color, lw=0, zorder=1)
    pyplot.plot([1.2, 2.8, 2.8, 1.2, 1.2], [1.95, 1.95, 2.00, 2.00, 1.95], lw=0.75, color="k", zorder=2)
    pyplot.text(1.2, 1.92, "-1.0", va="center", ha="center", fontsize=7)
    pyplot.text(2.8, 1.92, "+1.0", va="center", ha="center", fontsize=7)
    pyplot.text(2.0, 1.92, "0.0", va="center", ha="center", fontsize=7)
    pyplot.text(0.50, 1.84, "former curvature feature", va="center", ha="center", fontsize=8)
    pyplot.text(0.50, 1.16, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(0.16, 1.50, "$y$", va="center", ha="center", fontsize=8)
    pyplot.pcolormesh(linspace(0.2, 0.8, 101), linspace(1.2, 1.8, 101),
                      source_2, vmin=-1, vmax=1, cmap="binary", shading="gouraud")
    pyplot.plot([0.2, 0.8, 0.8, 0.2, 0.2], [1.2, 1.2, 1.8, 1.8, 1.2], lw=0.75, c="k", zorder=1)
    pyplot.text(0.50, 0.84, "latter curvature feature", va="center", ha="center", fontsize=8)
    pyplot.text(0.50, 0.16, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(0.16, 0.50, "$y$", va="center", ha="center", fontsize=8)
    pyplot.pcolormesh(linspace(0.2, 0.8, 101), linspace(0.2, 0.8, 101),
                      target_2, vmin=-1, vmax=1, cmap="binary", shading="gouraud")
    pyplot.plot([0.2, 0.8, 0.8, 0.2, 0.2], [0.2, 0.2, 0.8, 0.8, 0.2], lw=0.75, c="k", zorder=1)
    pyplot.text(1.0, 1.52, "calculate\ncurvature feature", va="bottom", ha="center", fontsize=7)
    pyplot.text(1.0, 0.52, "calculate\ncurvature feature", va="bottom", ha="center", fontsize=7)
    pyplot.annotate("", xy=(0.9, 1.5), xytext=(1.1, 1.5),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1), zorder=2)
    pyplot.annotate("", xy=(0.9, 0.5), xytext=(1.1, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1), zorder=2)
    pyplot.text(1.50, 1.84, "former landscape (mesh)", va="center", ha="center", fontsize=8)
    pyplot.text(1.50, 1.16, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(1.16, 1.50, "$y$", va="center", ha="center", fontsize=8)
    pyplot.pcolormesh(linspace(1.2, 1.8, 101), linspace(1.2, 1.8, 101),
                      source_1, vmin=-1, vmax=1, cmap="PRGn", shading="gouraud")
    pyplot.plot([1.2, 1.8, 1.8, 1.2, 1.2], [1.2, 1.2, 1.8, 1.8, 1.2], lw=0.75, c="k", zorder=1)
    pyplot.annotate("", xy=(1.5, 0.9), xytext=(1.5, 1.1),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1), zorder=2)
    pyplot.text(1.52, 1.00, "escape", va="center", ha="left", fontsize=7)
    pyplot.text(1.50, 0.84, "latter landscape (mesh)", va="center", ha="center", fontsize=8)
    pyplot.text(1.50, 0.16, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(1.16, 0.50, "$y$", va="center", ha="center", fontsize=8)
    pyplot.pcolormesh(linspace(1.2, 1.8, 101), linspace(0.2, 0.8, 101),
                      target_1, vmin=-1, vmax=1, cmap="PRGn", shading="gouraud")
    pyplot.plot([1.2, 1.8, 1.8, 1.2, 1.2], [0.2, 0.2, 0.8, 0.8, 0.2], lw=0.75, c="k", zorder=1)
    pyplot.annotate("", xy=(1.9, 1.5), xytext=(2.1, 1.5),
                    arrowprops=dict(arrowstyle="<|-", color="black", lw=1), zorder=2)
    pyplot.annotate("", xy=(1.9, 0.5), xytext=(2.1, 0.5),
                    arrowprops=dict(arrowstyle="<|-", color="black", lw=1), zorder=2)
    pyplot.text(2.0, 1.52, "calculate\ncontour", va="bottom", ha="center", fontsize=7)
    pyplot.text(2.0, 0.52, "calculate\ncontour", va="bottom", ha="center", fontsize=7)
    pyplot.text(2.50, 1.84, "former landscape (contour)", va="center", ha="center", fontsize=8)
    pyplot.text(2.50, 1.16, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(2.16, 1.50, "$y$", va="center", ha="center", fontsize=8)
    # noinspection PyCompatibility
    pyplot.contour(*meshgrid(linspace(2.2, 2.8, 101), linspace(1.2, 1.8, 101)),
                   source_1, vmin=-1, vmax=1, cmap="PRGn", lw=2, zorder=0)
    pyplot.plot([2.2, 2.8, 2.8, 2.2, 2.2], [1.2, 1.2, 1.8, 1.8, 1.2], lw=0.75, c="k", zorder=1)
    pyplot.text(2.50, 0.84, "latter landscape (contour)", va="center", ha="center", fontsize=8)
    pyplot.text(2.50, 0.16, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(2.16, 0.50, "$y$", va="center", ha="center", fontsize=8)
    # noinspection PyCompatibility
    pyplot.contour(*meshgrid(linspace(2.2, 2.8, 101), linspace(0.2, 0.8, 101)),
                   target_1, vmin=-1, vmax=1, cmap="PRGn", lw=2, zorder=0)
    pyplot.plot([2.2, 2.8, 2.8, 2.2, 2.2], [0.2, 0.2, 0.8, 0.8, 0.2], lw=0.75, c="k", zorder=1)
    pyplot.annotate("", xy=(2.9, 1.5), xytext=(3.1, 1.5),
                    arrowprops=dict(arrowstyle="<|-", color="black", lw=1), zorder=2)
    pyplot.annotate("", xy=(2.9, 0.5), xytext=(3.1, 0.5),
                    arrowprops=dict(arrowstyle="<|-", color="black", lw=1), zorder=2)
    pyplot.text(3.00, 1.52, "let value > 0\nas the ridge", va="bottom", ha="center", fontsize=7)
    pyplot.text(3.00, 0.52, "let value > 0\nas the ridge", va="bottom", ha="center", fontsize=7)
    pyplot.text(3.50, 2.03, "region definition", va="center", ha="center", fontsize=8)
    pyplot.fill_between([3.20, 3.50], 1.95, 2.00, fc=pyplot.get_cmap("viridis")(linspace(0, 1, 100))[0],
                        lw=0, zorder=1)
    pyplot.fill_between([3.50, 3.80], 1.95, 2.00, fc=pyplot.get_cmap("viridis")(linspace(0, 1, 100))[-1],
                        lw=0, zorder=1)
    pyplot.plot([3.2, 3.8, 3.8, 3.2, 3.2], [1.95, 1.95, 2.00, 2.00, 1.95], lw=0.75, color="k", zorder=2)
    pyplot.text(3.35, 1.92, "unknown", va="center", ha="center", fontsize=7)
    pyplot.text(3.65, 1.92, "ridge", va="center", ha="center", fontsize=7)
    pyplot.text(3.50, 1.84, "ridge region in former landscape", va="center", ha="center", fontsize=8)
    pyplot.text(3.50, 1.16, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(3.16, 1.50, "$y$", va="center", ha="center", fontsize=8)
    pyplot.pcolormesh(linspace(3.2, 3.8, 101), linspace(1.2, 1.8, 101),
                      source_3, vmin=0, vmax=1, cmap="viridis", shading="gouraud", zorder=1)
    pyplot.plot([3.2, 3.8, 3.8, 3.2, 3.2], [1.2, 1.2, 1.8, 1.8, 1.2], lw=0.75, c="k", zorder=2)
    locations_x, locations_y = where(source_3 == 1)
    pyplot.text(linspace(3.2, 3.8, 101)[int(mean(locations_y))], linspace(1.2, 1.8, 101)[int(mean(locations_x))],
                "%.2f" % (sum(source_3) / (101 ** 2)), va="center", ha="center", fontsize=10, zorder=3)
    pyplot.text(3.50, 0.84, "ridge region in latter landscape", va="center", ha="center", fontsize=8)
    pyplot.text(3.50, 0.16, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(3.16, 0.50, "$y$", va="center", ha="center", fontsize=8)
    pyplot.pcolormesh(linspace(3.2, 3.8, 101), linspace(0.2, 0.8, 101),
                      target_3, vmin=0, vmax=1, cmap="viridis", shading="gouraud", zorder=1)
    pyplot.plot([3.2, 3.8, 3.8, 3.2, 3.2], [0.2, 0.2, 0.8, 0.8, 0.2], lw=0.75, c="k", zorder=2)
    locations_x, locations_y = where(target_3 == 1)
    pyplot.text(linspace(3.2, 3.8, 101)[int(mean(locations_y))], linspace(0.2, 0.8, 101)[int(mean(locations_x))],
                "%.2f" % (sum(target_3) / (101 ** 2)), va="center", ha="center", fontsize=10, zorder=3)
    pyplot.xlim(0.15, 3.85)
    pyplot.ylim(0.10, 2.10)
    pyplot.axis("off")

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[2, 0])
    origin, changed = task_data["b"][:, 0], task_data["b"][:, 1]
    x = linspace(min(origin), max(origin), 100)
    y = gaussian_kde(origin)(x)
    y /= sum(y)
    pyplot.plot(x, y, lw=0.75, color="k", zorder=2)
    pyplot.fill_between(x, 0, y, ec="k", fc="#BEB8DC", lw=0.75, zorder=2, label="former landscape")
    x = linspace(min(changed), max(changed), 100)
    y = gaussian_kde(changed)(x)
    y /= sum(y)
    pyplot.fill_between(x, 0, y, ec="k", fc="#FA7F6F", lw=0.75, zorder=2, label="latter landscape")
    pyplot.legend(loc="upper right", ncol=2, fontsize=7)
    pyplot.xlabel("proportion of predominant ridge/valley region", fontsize=8)
    pyplot.xticks(linspace(0.2, 0.5, 7), ["%.2f" % v for v in linspace(0.2, 0.5, 7)], fontsize=7)
    pyplot.yticks([])
    pyplot.xlim(0.2, 0.5)
    pyplot.ylim(0.00, 0.023)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.35, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp07.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_08():
    """
    Create Figure S8 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp08.pkl")

    figure = pyplot.figure(figsize=(10, 9.5), tight_layout=True)
    for index, (panel_index, values) in enumerate(task_data.items()):
        pyplot.subplot(2, 2, index + 1)
        count = sum((values[:, 1] - values[:, 0]) > 0)
        pyplot.title("samples in coherent-loop " + str(index + 1), fontsize=8)
        pyplot.fill_between([0, 1], [0, 1], [1, 1], lw=0, fc="#FEB2B4", alpha=0.5, zorder=0,
                            label="increase (" + str(count) + " samples)")
        pyplot.fill_between([0, 1], [0, 0], [0, 1], lw=0, fc="#A5B6C5", alpha=0.5, zorder=0,
                            label="decrease (" + str(100 - count) + " samples)")
        for location in linspace(0.1, 0.9, 9):
            pyplot.hlines(location, 0, 1, lw=0.75, ls="--", color="k", zorder=1)
            pyplot.vlines(location, 0, 1, lw=0.75, ls="--", color="k", zorder=1)
        pyplot.legend(loc="lower right", fontsize=7, title="proportion change", title_fontsize=7, framealpha=1)
        pyplot.scatter(values[:, 0], values[:, 1], ec="k", fc="w", lw=0.75, zorder=2)
        pyplot.xlabel("predominant curvature proportion of landscape before escaping", fontsize=8)
        pyplot.ylabel("predominant curvature proportion of landscape after escaping", fontsize=8)
        pyplot.xticks(linspace(0.0, 1.0, 11), [("%d" % v) + "%" for v in arange(0, 101, 10)], fontsize=7)
        pyplot.yticks(linspace(0.0, 1.0, 11), [("%d" % v) + "%" for v in arange(0, 101, 10)], fontsize=7)
        pyplot.xlim(0.0, 1.0)
        pyplot.ylim(0.0, 1.0)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.50, "c", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.50, "d", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp08.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_09():
    """
    Create Figure S9 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp09.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)

    grid = pyplot.GridSpec(8, 5)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[:3, :])

    for location in linspace(0.0, 0.6, 7):
        if abs(location - 0.5) > 0.01:
            pyplot.fill_between([location + 0.01, location + 0.03, location + 0.05], 0.40, [0.40, 0.70, 0.40],
                                lw=0.75, ec="k", fc="silver")
            pyplot.annotate("", xy=(location + 0.03, 0.70), xytext=(location + 0.03, 0.85),
                            arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0.0, shrinkB=0.0, lw=0.75), zorder=0)
            pyplot.scatter([location + 0.03], [0.89], ec="k", fc="w", lw=0.75, zorder=1)

            pyplot.annotate("", xy=(location + 0.01, 0.20), xytext=(location + 0.01, 0.40), zorder=2,
                            arrowprops=dict(arrowstyle="<|-", color="#F1B1AC", shrinkA=0.0, shrinkB=0.0, lw=0.75))
            pyplot.scatter([location + 0.01], [0.27], s=10, color="w", zorder=1)
            pyplot.annotate("", xy=(location + 0.05, 0.27), xytext=(location + 0.05, 0.40), zorder=0,
                            arrowprops=dict(arrowstyle="<|-", color="#90C9EC", shrinkA=0.0, shrinkB=0.0, lw=0.75))
        else:
            pyplot.scatter([location + 0.02, location + 0.03, location + 0.04], [0.55, 0.55, 0.55], color="k", s=10)

    pyplot.scatter([0.35], [0.20], s=10, color="w", zorder=1)
    pyplot.hlines(0.20, 0.01, 0.61, lw=0.75, color="#F1B1AC", zorder=2)
    pyplot.hlines(0.27, 0.05, 0.65, lw=0.75, color="#90C9EC", zorder=0)

    pyplot.vlines(0.31, 0.15, 0.20, lw=0.75, color="#F1B1AC", zorder=2)
    pyplot.vlines(0.35, 0.15, 0.27, lw=0.75, color="#90C9EC", zorder=0)

    pyplot.text(0.31, 0.08, "x", va="center", ha="center", fontsize=10)
    pyplot.text(0.35, 0.08, "y", va="center", ha="center", fontsize=10)

    pyplot.text(0.010, 0.895, "{", va="center", ha="center", fontsize=10)
    pyplot.text(0.650, 0.895, "}", va="center", ha="center", fontsize=10)
    pyplot.text(0.660, 0.895, "sum = z", va="center", ha="left", fontsize=10)

    pyplot.fill_between([0.73, 1.00], 0.00, 1.00, color="#EEEEEE", zorder=0)
    pyplot.fill_between([0.75, 0.77, 0.79], 0.40, [0.40, 0.70, 0.40], lw=0.75, ec="k", fc="silver")
    pyplot.hlines(0.50, 0.81, 0.83, lw=0.75, color="k")
    pyplot.plot([0.83, 0.82, 0.82, 0.83], [0.25, 0.25, 0.75, 0.75], lw=0.75, color="k")
    pyplot.text(0.85, 0.75, "incoherent loop", va="center", ha="left", fontsize=10)
    pyplot.text(0.85, 0.50, "coherent loop", va="center", ha="left", fontsize=10)
    pyplot.text(0.85, 0.25, "collider", va="center", ha="left", fontsize=10)

    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    pyplot.axis("off")

    for panel_index in range(1, 6):
        pyplot.subplot(grid[4:7, panel_index - 1])
        landscape_name, z_values = task_data[chr(ord("a") + panel_index)]
        pyplot.title(landscape_name, fontsize=8)
        pyplot.pcolormesh(linspace(-1, 1, 41), linspace(-1, 1, 41), z_values,
                          shading="gouraud", cmap="plasma", vmin=-1, vmax=+1)

        pyplot.xlabel("input (x) signal", fontsize=7)
        pyplot.ylabel("input (y) signal", fontsize=7)
        pyplot.xticks(linspace(-1, 1, 5), ["%.1f" % v for v in linspace(-1, 1, 5)], fontsize=7)
        pyplot.yticks(linspace(-1, 1, 5), ["%.1f" % v for v in linspace(-1, 1, 5)], fontsize=7)
        pyplot.xlim(-1, +1)
        pyplot.ylim(-1, +1)

    pyplot.subplot(grid[7, :])

    locations, colors = linspace(-1, 1, 101), pyplot.get_cmap("plasma")(linspace(0, 1, 100))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 0, 1, ec="none", fc=color, lw=0)

    pyplot.xlabel("output (z) signal", fontsize=7)
    pyplot.xticks(linspace(-1, 1, 11), ["%.1f" % v for v in linspace(-1, 1, 11)], fontsize=7)
    pyplot.yticks([])
    pyplot.xlim(-1, 1)
    pyplot.ylim(0, 1)

    figure.align_labels()

    figure.text(0.012, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.012, 0.540, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp09.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_10():
    """
    Create Figure S10 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp10.pkl")

    figure = pyplot.figure(figsize=(10, 4), tight_layout=True)

    pyplot.subplot(1, 3, 1)

    panel_data = task_data["a"]
    for index, (values, color) in enumerate(zip(panel_data, pyplot.get_cmap("binary")(linspace(0, 1, 10)))):
        pyplot.scatter([log10(values[1])], [log10(values[0])], ec="k", fc=color, s=20, lw=0.75, label=str(index + 1),
                       zorder=1)
    for former_values, latter_values in zip(panel_data[:-1], panel_data[1:]):
        pyplot.plot([log10(former_values[1]), log10(latter_values[1])],
                    [log10(former_values[0]), log10(latter_values[0])],
                    lw=0.50, color="k", zorder=0)
    pyplot.legend(loc="lower left", ncol=2, title="motif number", handletextpad=0.3, columnspacing=0.9,
                  fontsize=7, title_fontsize=7)
    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.title("collider network performance", fontsize=7)
    pyplot.xlabel("iteration", fontsize=7)
    pyplot.ylabel("loss", fontsize=7)
    pyplot.xticks([3.0, 3.5, 4.0, 4.5], ["1E+3", "5E+3", "1E+4", "5E+4"],
                  fontsize=7)
    pyplot.yticks([-3.0, -2.0, -1.0, 0.0], ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0"],
                  fontsize=7)
    pyplot.xlim(2.9, 4.6)
    pyplot.ylim(-3.18, 0.18)

    pyplot.subplot(1, 3, 2)

    panel_data = task_data["b"]
    for index, (values, color) in enumerate(zip(panel_data, pyplot.get_cmap("binary")(linspace(0, 1, 10)))):
        pyplot.scatter([log10(values[1])], [log10(values[0])], ec="k", fc=color, s=20, lw=0.75, label=str(index + 1),
                       zorder=1)
    for former_values, latter_values in zip(panel_data[:-1], panel_data[1:]):
        pyplot.plot([log10(former_values[1]), log10(latter_values[1])],
                    [log10(former_values[0]), log10(latter_values[0])],
                    lw=0.50, color="k", zorder=0)
    pyplot.legend(loc="lower left", ncol=2, title="motif number", handletextpad=0.3, columnspacing=0.9,
                  fontsize=7, title_fontsize=7)
    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.title("loop network performance", fontsize=7)
    pyplot.xlabel("iteration", fontsize=7)
    pyplot.ylabel("loss", fontsize=7)
    pyplot.xticks([3.0, 3.5, 4.0, 4.5], ["1E+3", "5E+3", "1E+4", "5E+4"],
                  fontsize=7)
    pyplot.yticks([-3.0, -2.0, -1.0, 0.0], ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0"],
                  fontsize=7)
    pyplot.xlim(2.9, 4.6)
    pyplot.ylim(-3.18, 0.18)

    ax = pyplot.subplot(1, 3, 3)

    panel_data = task_data["c"]
    pyplot.title("performance summary", fontsize=7)

    x_locations, y_locations = linspace(0, 1, 7), linspace(1, 0, 25)
    pyplot.text(x_locations[3], y_locations[1], "collider-only", va="center", ha="center", fontsize=7)
    pyplot.text(x_locations[5], y_locations[1], "loop-only", va="center", ha="center", fontsize=7)
    pyplot.text(x_locations[1], y_locations[2], "motif number", va="center", ha="center", fontsize=7)

    pyplot.vlines([x_locations[2], x_locations[4]], 0, 1, color="k", lw=0.50, ls="--", zorder=0)
    pyplot.vlines([x_locations[3], x_locations[5]], 0, y_locations[2],
                  color="k", lw=0.50, ls="--", zorder=0)

    pyplot.text((x_locations[2] + x_locations[3]) / 2.0, y_locations[3],
                "loss", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[3] + x_locations[4]) / 2.0, y_locations[3],
                "iteration", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[4] + x_locations[5]) / 2.0, y_locations[3],
                "loss", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[5] + x_locations[6]) / 2.0, y_locations[3],
                "iteration", va="center", ha="center", fontsize=7)
    pyplot.hlines(y_locations[2], x_locations[2], x_locations[6], color="k", lw=0.50, ls="--", zorder=0)

    for index, location in enumerate(y_locations[::2][1:-1][1:]):
        if index == 0:
            pyplot.hlines(location, 0, 1, color="k", lw=0.75, zorder=0)
        else:
            pyplot.hlines(location, 0, 1, color="k", lw=0.50, ls="--", zorder=0)

    for index, location in enumerate(y_locations[1::2][2:]):
        pyplot.text(x_locations[1], location, str(index + 1), va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[2] + x_locations[3]) / 2.0, location, panel_data[index][0],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[3] + x_locations[4]) / 2.0, location, panel_data[index][1],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[4] + x_locations[5]) / 2.0, location, panel_data[index][2],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[5] + x_locations[6]) / 2.0, location, panel_data[index][3],
                    va="center", ha="center", fontsize=7)

    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()

    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.345, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.685, 0.990, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp10.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_11():
    """
    Create Figure S11 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp11.pkl")

    figure = pyplot.figure(figsize=(10, 4), tight_layout=True)

    pyplot.subplot(1, 3, 1)

    panel_data = task_data["a"]
    for index, (values, color) in enumerate(zip(panel_data, pyplot.get_cmap("binary")(linspace(0, 1, 10)))):
        pyplot.scatter([log10(values[1])], [log10(values[0])], ec="k", fc=color, s=20, lw=0.75, label=str(index + 1),
                       zorder=1)
    for former_values, latter_values in zip(panel_data[:-1], panel_data[1:]):
        pyplot.plot([log10(former_values[1]), log10(latter_values[1])],
                    [log10(former_values[0]), log10(latter_values[0])],
                    lw=0.50, color="k", zorder=0)
    pyplot.legend(loc="lower left", ncol=2, title="motif number", handletextpad=0.3, columnspacing=0.9,
                  fontsize=7, title_fontsize=7)
    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.title("collider network performance", fontsize=7)
    pyplot.xlabel("iteration", fontsize=7)
    pyplot.ylabel("loss", fontsize=7)
    pyplot.xticks([3.0, 3.5, 4.0, 4.5], ["1E+3", "5E+3", "1E+4", "5E+4"],
                  fontsize=7)
    pyplot.yticks([-3.0, -2.0, -1.0, 0.0], ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0"],
                  fontsize=7)
    pyplot.xlim(2.9, 4.6)
    pyplot.ylim(-3.18, 0.18)

    pyplot.subplot(1, 3, 2)

    panel_data = task_data["b"]
    for index, (values, color) in enumerate(zip(panel_data, pyplot.get_cmap("binary")(linspace(0, 1, 10)))):
        pyplot.scatter([log10(values[1])], [log10(values[0])], ec="k", fc=color, s=20, lw=0.75, label=str(index + 1),
                       zorder=1)
    for former_values, latter_values in zip(panel_data[:-1], panel_data[1:]):
        pyplot.plot([log10(former_values[1]), log10(latter_values[1])],
                    [log10(former_values[0]), log10(latter_values[0])],
                    lw=0.50, color="k", zorder=0)
    pyplot.legend(loc="lower left", ncol=2, title="motif number", handletextpad=0.3, columnspacing=0.9,
                  fontsize=7, title_fontsize=7)
    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.title("loop network performance", fontsize=7)
    pyplot.xlabel("iteration", fontsize=7)
    pyplot.ylabel("loss", fontsize=7)
    pyplot.xticks([3.0, 3.5, 4.0, 4.5], ["1E+3", "5E+3", "1E+4", "5E+4"],
                  fontsize=7)
    pyplot.yticks([-3.0, -2.0, -1.0, 0.0], ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0"],
                  fontsize=7)
    pyplot.xlim(2.9, 4.6)
    pyplot.ylim(-3.18, 0.18)

    ax = pyplot.subplot(1, 3, 3)

    panel_data = task_data["c"]
    pyplot.title("performance summary", fontsize=7)

    x_locations, y_locations = linspace(0, 1, 7), linspace(1, 0, 25)
    pyplot.text(x_locations[3], y_locations[1], "collider-only", va="center", ha="center", fontsize=7)
    pyplot.text(x_locations[5], y_locations[1], "loop-only", va="center", ha="center", fontsize=7)
    pyplot.text(x_locations[1], y_locations[2], "motif number", va="center", ha="center", fontsize=7)

    pyplot.vlines([x_locations[2], x_locations[4]], 0, 1, color="k", lw=0.50, ls="--", zorder=0)
    pyplot.vlines([x_locations[3], x_locations[5]], 0, y_locations[2],
                  color="k", lw=0.50, ls="--", zorder=0)

    pyplot.text((x_locations[2] + x_locations[3]) / 2.0, y_locations[3],
                "loss", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[3] + x_locations[4]) / 2.0, y_locations[3],
                "iteration", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[4] + x_locations[5]) / 2.0, y_locations[3],
                "loss", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[5] + x_locations[6]) / 2.0, y_locations[3],
                "iteration", va="center", ha="center", fontsize=7)
    pyplot.hlines(y_locations[2], x_locations[2], x_locations[6], color="k", lw=0.50, ls="--", zorder=0)

    for index, location in enumerate(y_locations[::2][1:-1][1:]):
        if index == 0:
            pyplot.hlines(location, 0, 1, color="k", lw=0.75, zorder=0)
        else:
            pyplot.hlines(location, 0, 1, color="k", lw=0.50, ls="--", zorder=0)

    for index, location in enumerate(y_locations[1::2][2:]):
        pyplot.text(x_locations[1], location, str(index + 1), va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[2] + x_locations[3]) / 2.0, location, panel_data[index][0],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[3] + x_locations[4]) / 2.0, location, panel_data[index][1],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[4] + x_locations[5]) / 2.0, location, panel_data[index][2],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[5] + x_locations[6]) / 2.0, location, panel_data[index][3],
                    va="center", ha="center", fontsize=7)

    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()

    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.345, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.685, 0.990, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp11.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_12():
    """
    Create Figure S12 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp12.pkl")

    figure = pyplot.figure(figsize=(10, 4), tight_layout=True)

    pyplot.subplot(1, 3, 1)

    panel_data = task_data["a"]
    for index, (values, color) in enumerate(zip(panel_data, pyplot.get_cmap("binary")(linspace(0, 1, 10)))):
        pyplot.scatter([log10(values[1])], [log10(values[0])], ec="k", fc=color, s=20, lw=0.75, label=str(index + 1),
                       zorder=1)
    for former_values, latter_values in zip(panel_data[:-1], panel_data[1:]):
        pyplot.plot([log10(former_values[1]), log10(latter_values[1])],
                    [log10(former_values[0]), log10(latter_values[0])],
                    lw=0.50, color="k", zorder=0)
    pyplot.legend(loc="lower left", ncol=2, title="motif number", handletextpad=0.3, columnspacing=0.9,
                  fontsize=7, title_fontsize=7)
    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.title("collider network performance", fontsize=7)
    pyplot.xlabel("iteration", fontsize=7)
    pyplot.ylabel("loss", fontsize=7)
    pyplot.xticks([3.0, 3.5, 4.0, 4.5], ["1E+3", "5E+3", "1E+4", "5E+4"],
                  fontsize=7)
    pyplot.yticks([-3.0, -2.0, -1.0, 0.0], ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0"],
                  fontsize=7)
    pyplot.xlim(2.9, 4.6)
    pyplot.ylim(-3.18, 0.18)

    pyplot.subplot(1, 3, 2)

    panel_data = task_data["b"]
    for index, (values, color) in enumerate(zip(panel_data, pyplot.get_cmap("binary")(linspace(0, 1, 10)))):
        pyplot.scatter([log10(values[1])], [log10(values[0])], ec="k", fc=color, s=20, lw=0.75, label=str(index + 1),
                       zorder=1)
    for former_values, latter_values in zip(panel_data[:-1], panel_data[1:]):
        pyplot.plot([log10(former_values[1]), log10(latter_values[1])],
                    [log10(former_values[0]), log10(latter_values[0])],
                    lw=0.50, color="k", zorder=0)
    pyplot.legend(loc="lower left", ncol=2, title="motif number", handletextpad=0.3, columnspacing=0.9,
                  fontsize=7, title_fontsize=7)
    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.title("loop network performance", fontsize=7)
    pyplot.xlabel("iteration", fontsize=7)
    pyplot.ylabel("loss", fontsize=7)
    pyplot.xticks([3.0, 3.5, 4.0, 4.5], ["1E+3", "5E+3", "1E+4", "5E+4"],
                  fontsize=7)
    pyplot.yticks([-3.0, -2.0, -1.0, 0.0], ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0"],
                  fontsize=7)
    pyplot.xlim(2.9, 4.6)
    pyplot.ylim(-3.18, 0.18)

    ax = pyplot.subplot(1, 3, 3)

    panel_data = task_data["c"]
    pyplot.title("performance summary", fontsize=7)

    x_locations, y_locations = linspace(0, 1, 7), linspace(1, 0, 25)
    pyplot.text(x_locations[3], y_locations[1], "collider-only", va="center", ha="center", fontsize=7)
    pyplot.text(x_locations[5], y_locations[1], "loop-only", va="center", ha="center", fontsize=7)
    pyplot.text(x_locations[1], y_locations[2], "motif number", va="center", ha="center", fontsize=7)

    pyplot.vlines([x_locations[2], x_locations[4]], 0, 1, color="k", lw=0.50, ls="--", zorder=0)
    pyplot.vlines([x_locations[3], x_locations[5]], 0, y_locations[2],
                  color="k", lw=0.50, ls="--", zorder=0)

    pyplot.text((x_locations[2] + x_locations[3]) / 2.0, y_locations[3],
                "loss", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[3] + x_locations[4]) / 2.0, y_locations[3],
                "iteration", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[4] + x_locations[5]) / 2.0, y_locations[3],
                "loss", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[5] + x_locations[6]) / 2.0, y_locations[3],
                "iteration", va="center", ha="center", fontsize=7)
    pyplot.hlines(y_locations[2], x_locations[2], x_locations[6], color="k", lw=0.50, ls="--", zorder=0)

    for index, location in enumerate(y_locations[::2][1:-1][1:]):
        if index == 0:
            pyplot.hlines(location, 0, 1, color="k", lw=0.75, zorder=0)
        else:
            pyplot.hlines(location, 0, 1, color="k", lw=0.50, ls="--", zorder=0)

    for index, location in enumerate(y_locations[1::2][2:]):
        pyplot.text(x_locations[1], location, str(index + 1), va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[2] + x_locations[3]) / 2.0, location, panel_data[index][0],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[3] + x_locations[4]) / 2.0, location, panel_data[index][1],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[4] + x_locations[5]) / 2.0, location, panel_data[index][2],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[5] + x_locations[6]) / 2.0, location, panel_data[index][3],
                    va="center", ha="center", fontsize=7)

    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()

    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.345, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.685, 0.990, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp12.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_13():
    """
    Create Figure S13 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp13.pkl")

    figure = pyplot.figure(figsize=(10, 4), tight_layout=True)

    pyplot.subplot(1, 3, 1)

    panel_data = task_data["a"]
    for index, (values, color) in enumerate(zip(panel_data, pyplot.get_cmap("binary")(linspace(0, 1, 10)))):
        pyplot.scatter([log10(values[1])], [log10(values[0])], ec="k", fc=color, s=20, lw=0.75, label=str(index + 1),
                       zorder=1)
    for former_values, latter_values in zip(panel_data[:-1], panel_data[1:]):
        pyplot.plot([log10(former_values[1]), log10(latter_values[1])],
                    [log10(former_values[0]), log10(latter_values[0])],
                    lw=0.50, color="k", zorder=0)
    pyplot.legend(loc="lower left", ncol=2, title="motif number", handletextpad=0.3, columnspacing=0.9,
                  fontsize=7, title_fontsize=7)
    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.title("collider network performance", fontsize=7)
    pyplot.xlabel("iteration", fontsize=7)
    pyplot.ylabel("loss", fontsize=7)
    pyplot.xticks([3.0, 3.5, 4.0, 4.5], ["1E+3", "5E+3", "1E+4", "5E+4"],
                  fontsize=7)
    pyplot.yticks([-3.0, -2.0, -1.0, 0.0], ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0"],
                  fontsize=7)
    pyplot.xlim(2.9, 4.6)
    pyplot.ylim(-3.18, 0.18)

    pyplot.subplot(1, 3, 2)

    panel_data = task_data["b"]
    for index, (values, color) in enumerate(zip(panel_data, pyplot.get_cmap("binary")(linspace(0, 1, 10)))):
        pyplot.scatter([log10(values[1])], [log10(values[0])], ec="k", fc=color, s=20, lw=0.75, label=str(index + 1),
                       zorder=1)
    for former_values, latter_values in zip(panel_data[:-1], panel_data[1:]):
        pyplot.plot([log10(former_values[1]), log10(latter_values[1])],
                    [log10(former_values[0]), log10(latter_values[0])],
                    lw=0.50, color="k", zorder=0)
    pyplot.legend(loc="lower left", ncol=2, title="motif number", handletextpad=0.3, columnspacing=0.9,
                  fontsize=7, title_fontsize=7)
    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.title("loop network performance", fontsize=7)
    pyplot.xlabel("iteration", fontsize=7)
    pyplot.ylabel("loss", fontsize=7)
    pyplot.xticks([3.0, 3.5, 4.0, 4.5], ["1E+3", "5E+3", "1E+4", "5E+4"],
                  fontsize=7)
    pyplot.yticks([-3.0, -2.0, -1.0, 0.0], ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0"],
                  fontsize=7)
    pyplot.xlim(2.9, 4.6)
    pyplot.ylim(-3.18, 0.18)

    ax = pyplot.subplot(1, 3, 3)

    panel_data = task_data["c"]
    pyplot.title("performance summary", fontsize=7)

    x_locations, y_locations = linspace(0, 1, 7), linspace(1, 0, 25)
    pyplot.text(x_locations[3], y_locations[1], "collider-only", va="center", ha="center", fontsize=7)
    pyplot.text(x_locations[5], y_locations[1], "loop-only", va="center", ha="center", fontsize=7)
    pyplot.text(x_locations[1], y_locations[2], "motif number", va="center", ha="center", fontsize=7)

    pyplot.vlines([x_locations[2], x_locations[4]], 0, 1, color="k", lw=0.50, ls="--", zorder=0)
    pyplot.vlines([x_locations[3], x_locations[5]], 0, y_locations[2],
                  color="k", lw=0.50, ls="--", zorder=0)

    pyplot.text((x_locations[2] + x_locations[3]) / 2.0, y_locations[3],
                "loss", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[3] + x_locations[4]) / 2.0, y_locations[3],
                "iteration", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[4] + x_locations[5]) / 2.0, y_locations[3],
                "loss", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[5] + x_locations[6]) / 2.0, y_locations[3],
                "iteration", va="center", ha="center", fontsize=7)
    pyplot.hlines(y_locations[2], x_locations[2], x_locations[6], color="k", lw=0.50, ls="--", zorder=0)

    for index, location in enumerate(y_locations[::2][1:-1][1:]):
        if index == 0:
            pyplot.hlines(location, 0, 1, color="k", lw=0.75, zorder=0)
        else:
            pyplot.hlines(location, 0, 1, color="k", lw=0.50, ls="--", zorder=0)

    for index, location in enumerate(y_locations[1::2][2:]):
        pyplot.text(x_locations[1], location, str(index + 1), va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[2] + x_locations[3]) / 2.0, location, panel_data[index][0],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[3] + x_locations[4]) / 2.0, location, panel_data[index][1],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[4] + x_locations[5]) / 2.0, location, panel_data[index][2],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[5] + x_locations[6]) / 2.0, location, panel_data[index][3],
                    va="center", ha="center", fontsize=7)

    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()

    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.345, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.685, 0.990, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp13.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_14():
    """
    Create Figure S14 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp14.pkl")

    figure = pyplot.figure(figsize=(10, 4), tight_layout=True)

    pyplot.subplot(1, 3, 1)

    panel_data = task_data["a"]
    for index, (values, color) in enumerate(zip(panel_data, pyplot.get_cmap("binary")(linspace(0, 1, 10)))):
        pyplot.scatter([log10(values[1])], [log10(values[0])], ec="k", fc=color, s=20, lw=0.75, label=str(index + 1),
                       zorder=1)
    for former_values, latter_values in zip(panel_data[:-1], panel_data[1:]):
        pyplot.plot([log10(former_values[1]), log10(latter_values[1])],
                    [log10(former_values[0]), log10(latter_values[0])],
                    lw=0.50, color="k", zorder=0)
    pyplot.legend(loc="lower left", ncol=2, title="motif number", handletextpad=0.3, columnspacing=0.9,
                  fontsize=7, title_fontsize=7)
    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.title("collider network performance", fontsize=7)
    pyplot.xlabel("iteration", fontsize=7)
    pyplot.ylabel("loss", fontsize=7)
    pyplot.xticks([3.0, 3.5, 4.0, 4.5], ["1E+3", "5E+3", "1E+4", "5E+4"],
                  fontsize=7)
    pyplot.yticks([-3.0, -2.0, -1.0, 0.0], ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0"],
                  fontsize=7)
    pyplot.xlim(2.9, 4.6)
    pyplot.ylim(-3.18, 0.18)

    pyplot.subplot(1, 3, 2)

    panel_data = task_data["b"]
    for index, (values, color) in enumerate(zip(panel_data, pyplot.get_cmap("binary")(linspace(0, 1, 10)))):
        pyplot.scatter([log10(values[1])], [log10(values[0])], ec="k", fc=color, s=20, lw=0.75, label=str(index + 1),
                       zorder=1)
    for former_values, latter_values in zip(panel_data[:-1], panel_data[1:]):
        pyplot.plot([log10(former_values[1]), log10(latter_values[1])],
                    [log10(former_values[0]), log10(latter_values[0])],
                    lw=0.50, color="k", zorder=0)
    pyplot.legend(loc="lower left", ncol=2, title="motif number", handletextpad=0.3, columnspacing=0.9,
                  fontsize=7, title_fontsize=7)
    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.vlines([3.0, 3.5, 4.0, 4.5], -3.18, 0.18, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-2, -1, 0], 2.9, 4.6, color="silver", lw=0.5, ls="--", zorder=0)
    pyplot.hlines([-3], 2.9, 4.6, color="k", lw=0.5, ls="--", zorder=0)

    pyplot.title("loop network performance", fontsize=7)
    pyplot.xlabel("iteration", fontsize=7)
    pyplot.ylabel("loss", fontsize=7)
    pyplot.xticks([3.0, 3.5, 4.0, 4.5], ["1E+3", "5E+3", "1E+4", "5E+4"],
                  fontsize=7)
    pyplot.yticks([-3.0, -2.0, -1.0, 0.0], ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0"],
                  fontsize=7)
    pyplot.xlim(2.9, 4.6)
    pyplot.ylim(-3.18, 0.18)

    ax = pyplot.subplot(1, 3, 3)

    panel_data = task_data["c"]
    pyplot.title("performance summary", fontsize=7)

    x_locations, y_locations = linspace(0, 1, 7), linspace(1, 0, 25)
    pyplot.text(x_locations[3], y_locations[1], "collider-only", va="center", ha="center", fontsize=7)
    pyplot.text(x_locations[5], y_locations[1], "loop-only", va="center", ha="center", fontsize=7)
    pyplot.text(x_locations[1], y_locations[2], "motif number", va="center", ha="center", fontsize=7)

    pyplot.vlines([x_locations[2], x_locations[4]], 0, 1, color="k", lw=0.50, ls="--", zorder=0)
    pyplot.vlines([x_locations[3], x_locations[5]], 0, y_locations[2],
                  color="k", lw=0.50, ls="--", zorder=0)

    pyplot.text((x_locations[2] + x_locations[3]) / 2.0, y_locations[3],
                "loss", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[3] + x_locations[4]) / 2.0, y_locations[3],
                "iteration", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[4] + x_locations[5]) / 2.0, y_locations[3],
                "loss", va="center", ha="center", fontsize=7)
    pyplot.text((x_locations[5] + x_locations[6]) / 2.0, y_locations[3],
                "iteration", va="center", ha="center", fontsize=7)
    pyplot.hlines(y_locations[2], x_locations[2], x_locations[6], color="k", lw=0.50, ls="--", zorder=0)

    for index, location in enumerate(y_locations[::2][1:-1][1:]):
        if index == 0:
            pyplot.hlines(location, 0, 1, color="k", lw=0.75, zorder=0)
        else:
            pyplot.hlines(location, 0, 1, color="k", lw=0.50, ls="--", zorder=0)

    for index, location in enumerate(y_locations[1::2][2:]):
        pyplot.text(x_locations[1], location, str(index + 1), va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[2] + x_locations[3]) / 2.0, location, panel_data[index][0],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[3] + x_locations[4]) / 2.0, location, panel_data[index][1],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[4] + x_locations[5]) / 2.0, location, panel_data[index][2],
                    va="center", ha="center", fontsize=7)
        pyplot.text((x_locations[5] + x_locations[6]) / 2.0, location, panel_data[index][3],
                    va="center", ha="center", fontsize=7)

    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()

    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.345, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.685, 0.990, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp14.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_15():
    """
    Create Figure S15 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp15.pkl")

    figure = pyplot.figure(figsize=(10, 3), tight_layout=True)

    for index in range(5):
        panel_data = task_data[chr(ord("a") + index)]

        ax = pyplot.subplot(1, 5, index + 1)
        pyplot.title(panel_data[0], fontsize=7)
        flag = True
        for motif_index in range(11):
            if panel_data[1][motif_index] > 0:
                if flag:
                    flag = False
                    pyplot.bar([motif_index], log10(panel_data[1][motif_index]), width=0.5, color="gray", zorder=1,
                               label="produced")
                else:
                    pyplot.bar([motif_index], log10(panel_data[1][motif_index]), width=0.5, color="gray", zorder=1)
        pyplot.bar(arange(11), log10(panel_data[2]), color="silver", zorder=0, label="successful")
        pyplot.legend(loc="upper right", fontsize=7)
        pyplot.xlabel("fraction of incoherent loops", fontsize=7)
        pyplot.ylabel("sample number", fontsize=7)
        pyplot.xticks(arange(11), [str(v) + "%" for v in arange(0, 101, 10)], rotation=90, fontsize=7)
        pyplot.yticks(arange(6), ["E+" + str(v) for v in arange(6)], fontsize=7)
        pyplot.xlim(-0.6, 10.6)
        pyplot.ylim(0, 5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    figure.align_labels()

    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.216, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.413, 0.990, "c", va="center", ha="center", fontsize=12)
    figure.text(0.611, 0.990, "d", va="center", ha="center", fontsize=12)
    figure.text(0.808, 0.990, "e", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp15.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_16():
    """
    Create Figure S16 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp16.pkl")
    milestones = [1e+0, 5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]
    labels = ["1E+1", "5E-1", "2E-1", "1E-1", "5E-2", "2E-2", "1E-2", "5E-3", "2E-3", "1E-3"]

    figure = pyplot.figure(figsize=(10, 3), tight_layout=True)

    ax = pyplot.subplot(1, 2, 1)

    panel_data = task_data["a"]

    pyplot.title("unrestricted loop-only networks (Quadratic Saddle)", fontsize=7)

    for index, (label, milestone) in enumerate(zip(labels, milestones)):
        parts = pyplot.violinplot([panel_data[label]], positions=[index], showextrema=False, vert=False)
        for body in parts["bodies"]:
            body.set_facecolor("silver")
            body.set_edgecolor("black")
            body.set_linewidth(0.75)
            body.set_alpha(1)
        maximum_value = max(panel_data[label])
        pyplot.vlines(maximum_value, index - 0.2, index + 0.2, lw=1.00, color="k")
        pyplot.text(maximum_value + 100, index - 0.05, maximum_value, va="center", ha="left", fontsize=7)

    pyplot.xlabel("used iteration during the training process", fontsize=7)
    pyplot.ylabel("training loss", fontsize=7)
    pyplot.xticks(arange(0, 10001, 1000), arange(0, 10001, 1000), fontsize=7)
    pyplot.yticks(arange(10),  [v.replace("-", "\N{MINUS SIGN}") for v in labels], fontsize=7)
    pyplot.xlim(-270, 10270)
    pyplot.ylim(-0.5, 9.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(1, 2, 2)

    panel_data = task_data["b"]

    pyplot.title("loop-only networks restricted by the fraction of incoherent loops (Quadratic Saddle)", fontsize=7)

    mesh = pyplot.pcolormesh(arange(11), arange(10), panel_data.T, cmap="RdYlGn_r", vmin=0, vmax=0.5)

    for index_1 in range(11):
        for index_2 in range(10):
            pyplot.text(index_1, index_2, ("%.1f" % (panel_data[index_1, index_2] * 100)) + "%",
                        va="center", ha="center", fontsize=6)

    pyplot.xlabel("fraction of incoherent loops", fontsize=7)
    pyplot.ylabel("training loss", fontsize=7)
    pyplot.xticks(arange(11), [str(v) + "%" for v in arange(0, 101, 10)], fontsize=7)
    pyplot.yticks(arange(10),  [v.replace("-", "\N{MINUS SIGN}") for v in labels], fontsize=7)
    pyplot.xlim(-0.5, 10.5)
    pyplot.ylim(-0.5, 9.5)

    figure.align_labels()

    cbar = figure.colorbar(mesh, ax=ax)
    cbar.set_label("proportion of cases worse than the unrestricted worst-case in (a)", fontsize=7)
    cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar.set_ticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    cbar.ax.tick_params(labelsize=7)

    figure.align_labels()
    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.990, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp16.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_17():
    """
    Create Figure S17 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp17.pkl")

    figure = pyplot.figure(figsize=(10, 3), tight_layout=True)

    ax = pyplot.subplot(1, 2, 1)

    panel_data = task_data["a"]
    pyplot.bar(arange(11), panel_data, ec="k", fc="silver", lw=0.75)
    for index, value in enumerate(panel_data):
        pyplot.text(index, value + 0.005, ("%.1f" % (value * 100)) + "%", va="bottom", ha="center", fontsize=7)
    pyplot.xlabel("fraction of incoherent loops", fontsize=7)
    pyplot.ylabel("failure rate", fontsize=7)
    pyplot.xticks(arange(11), [str(v) + "%" for v in arange(0, 101, 10)], fontsize=7)
    pyplot.yticks(linspace(0.0, 0.5, 6), [str(v) + "%" for v in arange(0, 51, 10)], fontsize=7)
    pyplot.xlim(-0.6, 10.6)
    pyplot.ylim(0, 0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(1, 2, 2)

    panel_data = task_data["b"]
    random.seed(2023)
    for index, y_values in enumerate(panel_data):
        x_values = random.random(len(y_values))
        x_values -= min(x_values)
        x_values /= max(x_values)
        x_values -= 0.5
        x_values *= 0.5
        pyplot.scatter(index + x_values, log10(y_values), s=5, c=log10(y_values), cmap="viridis", vmin=-3, vmax=0)
        pyplot.hlines(max(log10(y_values)), index - 0.4, index + 0.4, lw=1.00, color="k", zorder=-1)
        pyplot.vlines(index, -3.1, max(log10(y_values)), color="k", lw=0.50, ls="--", zorder=-1)
        pyplot.fill_between([index - 0.4, index + 0.4], -3.1, max(log10(y_values)), ec="none", fc="#EEEEEE", zorder=-2)
        pyplot.text(index, max(log10(y_values)) + 0.05, "%.3f" % max(y_values), va="bottom", ha="center", fontsize=7)

    pyplot.xlabel("fraction of incoherent loops", fontsize=7)
    pyplot.ylabel("final training loss (for failure cases)", fontsize=7)
    pyplot.xticks(arange(11), [str(v) + "%" for v in arange(0, 101, 10)], fontsize=7)
    values = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 1.000]
    labels = ["1E\N{MINUS SIGN}3", "2E\N{MINUS SIGN}3", "5E\N{MINUS SIGN}3",
              "1E\N{MINUS SIGN}2", "2E\N{MINUS SIGN}2", "5E\N{MINUS SIGN}2",
              "1E\N{MINUS SIGN}1", "2E\N{MINUS SIGN}1", "5E\N{MINUS SIGN}1",
              "1E+1"]
    pyplot.yticks(log10(values), labels, fontsize=7)
    pyplot.xlim(-0.6, 10.6)
    pyplot.ylim(-3.1, 0.1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.990, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp17.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_18():
    """
    Create Figure S18 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp18.pkl")

    figure = pyplot.figure(figsize=(10, 9), tight_layout=True)
    grid = pyplot.GridSpec(10, 6)
    orders = [grid[:2, 0], grid[:2, 1],
              grid[2:4, 0], grid[2:4, 1], grid[2:4, 2], grid[2:4, 3], grid[2:4, 4], grid[2:4, 5],
              grid[4:6, 0], grid[4:6, 1], grid[4:6, 2], grid[4:6, 3], grid[4:6, 4], grid[4:6, 5],
              grid[6:8, 0], grid[6:8, 1], grid[6:8, 2], grid[6:8, 3], grid[6:8, 4], grid[6:8, 5],
              grid[8:10, 4], grid[8:10, 5]]
    for panel_index, grid_order in enumerate(orders):
        pyplot.subplot(grid_order)
        info_1 = ("incoherent fraction = %d" % (panel_index // 2 * 10)) + "%"
        info_2 = "(for coherent loops)" if panel_index % 2 == 0 else "(for incoherent loops)"
        pyplot.title(info_1 + "\n" + info_2, fontsize=7)
        panel_data = task_data[chr(ord("a") + panel_index)]

        if panel_data[0] is not None:
            x_values, y_values = panel_data[0]
            if panel_index % 2 == 0:
                pyplot.scatter(x_values, y_values, s=10, ec="k", fc="#FCE0AB", lw=0.5)
            else:
                pyplot.scatter(x_values, y_values, s=10, ec="k", fc="#FCB1AB", lw=0.5)
            pyplot.text(-0.60, 0.03, "spearman\n%.2f" % panel_data[1], va="center", ha="center", fontsize=7)
        else:
            pyplot.text(-1.5, 0.5, "not applicable", va="center", ha="center", fontsize=7)

        if (panel_index // 2) % 2 == 0:
            pyplot.fill_between([-3.3, 0.3], -0.1, 1.1, color="#EEEEEE", lw=0, zorder=-1)

        pyplot.xlabel("final training loss", fontsize=7)
        pyplot.ylabel("final weight utilization", fontsize=7)
        pyplot.xticks([-3, -2, -1, 0],
                      ["1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+1"], fontsize=7)
        pyplot.yticks([0.00, 0.25, 0.50, 0.75, 1.00], ["0%", "25%", "50%", "75%", "100%"], fontsize=7)
        pyplot.xlim(-3.3, 0.3)
        pyplot.ylim(-0.1, 1.1)

    figure.align_labels()
    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.795, "b", va="center", ha="center", fontsize=12)
    figure.text(0.348, 0.795, "c", va="center", ha="center", fontsize=12)
    figure.text(0.677, 0.795, "d", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.600, "e", va="center", ha="center", fontsize=12)
    figure.text(0.348, 0.600, "f", va="center", ha="center", fontsize=12)
    figure.text(0.677, 0.600, "g", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.400, "h", va="center", ha="center", fontsize=12)
    figure.text(0.348, 0.400, "i", va="center", ha="center", fontsize=12)
    figure.text(0.677, 0.400, "j", va="center", ha="center", fontsize=12)
    figure.text(0.677, 0.200, "k", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp18.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_19():
    """
    Create Figure S19 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp19.pkl")

    figure = pyplot.figure(figsize=(10, 8), tight_layout=True)
    grid = pyplot.GridSpec(9, 5)

    panel_data = task_data["a"]
    ax = pyplot.subplot(grid[:2, :])

    pyplot.hlines(0, -0.6, 10.6, color="k", lw=0.75, ls="--", zorder=2)
    pyplot.scatter(arange(11), panel_data, ec="k", fc="w", lw=0.75, zorder=2)
    for index, value in enumerate(panel_data):
        pyplot.vlines(index, 0, value, lw=0.5, color="k", zorder=1)
        pyplot.text(index, 0.01, "%.2f" % value, va="bottom", ha="center", fontsize=7)

    pyplot.xlabel("fraction of incoherent loops", fontsize=7)
    pyplot.ylabel("correlation coefficient\n(delta loss v.s. iteration)", fontsize=7)
    pyplot.xticks(arange(11), [str(v) + "%" for v in arange(0, 101, 10)], fontsize=7)
    pyplot.yticks(linspace(-1, 1, 5), ["%.2f" % v for v in linspace(-1, 1, 5)], fontsize=7)
    pyplot.xlim(-0.6, 10.6)
    pyplot.ylim(-1.2, +1.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    orders = [grid[2:4, 0], grid[2:4, 1], grid[2:4, 2], grid[2:4, 3],
              grid[4:6, 1], grid[4:6, 2], grid[4:6, 3],
              grid[6:8, 1], grid[6:8, 2], grid[6:8, 3], grid[6:8, 4]]

    for incoherent_number, order in enumerate(orders):
        x_values, y_values, z_values, correlation = task_data[chr(ord("b") + incoherent_number)]
        z_values = log10(z_values / max(z_values))
        z_values[z_values < -10] = -10

        pyplot.subplot(order)
        pyplot.title(("incoherent fraction = %d" % (incoherent_number * 10)) + "%", fontsize=7)

        pyplot.pcolormesh(x_values, y_values, z_values.T, cmap="binary", vmin=-10, vmax=0)

        pyplot.text(0.059, -0.0039, "spearman = %.2f" % correlation, va="bottom", ha="right", fontsize=7)
        pyplot.hlines(0, 0.00, 0.06, color="r", lw=0.75, ls="--", zorder=2)
        pyplot.xlabel("delta loss", fontsize=7)
        pyplot.ylabel("delta lipschitz constant", fontsize=7)
        pyplot.xticks([0.00, 0.02, 0.04, 0.06], ["0E-2", "2E-2", "4E-2", "6E-2"], fontsize=7)
        pyplot.yticks([-0.004, 0.000, 0.004, 0.008], ["-4E-3", "0E-3", "4E-3", "8E-3"], fontsize=7)
        pyplot.xlim(0.00, 0.06)
        pyplot.ylim(-0.004, 0.008)

    pyplot.subplot(grid[8, :])

    locations, colors = linspace(-10, 1, 101), pyplot.get_cmap("binary")(linspace(0, 1, 100))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 0, 1, ec="none", fc=color, lw=0)

    pyplot.xlabel("normalized density", fontsize=7)
    labels = ["≤1E-10", "1E-9", "1E-8", "1E-7", "1E-6", "1E-5", "1E-4", "1E-3", "1E-2", "1E-1", "1E+0"]
    pyplot.xticks(arange(-10, 1), labels, fontsize=7)
    pyplot.yticks([])
    pyplot.xlim(-10, 0)
    pyplot.ylim(0, 1)

    figure.align_labels()
    figure.text(0.030, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.030, 0.780, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp19.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_20():
    """
    Create Figure S20 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp20.pkl")

    figure = pyplot.figure(figsize=(10, 8.5), tight_layout=True)
    grid = pyplot.GridSpec(10, 6)
    orders = [grid[:2, 0], grid[:2, 1],
              grid[2:4, 0], grid[2:4, 1], grid[2:4, 2], grid[2:4, 3], grid[2:4, 4], grid[2:4, 5],
              grid[4:6, 0], grid[4:6, 1], grid[4:6, 2], grid[4:6, 3], grid[4:6, 4], grid[4:6, 5],
              grid[6:8, 0], grid[6:8, 1], grid[6:8, 2], grid[6:8, 3], grid[6:8, 4], grid[6:8, 5],
              grid[8:10, 4], grid[8:10, 5]]
    for panel_index, grid_order in enumerate(orders):
        pyplot.subplot(grid_order)
        info_1 = ("incoherent fraction = %d" % (panel_index // 2 * 10)) + "%"
        info_2 = "(for coherent loops)" if panel_index % 2 == 0 else "(for incoherent loops)"
        pyplot.title(info_1 + "\n" + info_2, fontsize=7)
        panel_data = task_data[chr(ord("a") + panel_index)]

        if panel_data is not None:
            if panel_index % 2 == 0:
                for values in panel_data:
                    pyplot.plot(log10(arange(len(values)) + 1), values, color="#FCE0AB", alpha=0.2, zorder=1)
            else:
                for values in panel_data:
                    pyplot.plot(log10(arange(len(values)) + 1), values, color="#FCB1AB", alpha=0.2, zorder=1)
            pyplot.hlines(linspace(0.0, 2.4, 5)[1:-1], 0, 4, lw=0.75, ls="--", color="k", zorder=0)
            pyplot.vlines(arange(5)[1:-1], 0, 3, lw=0.75,  ls="--", color="k", zorder=0)
        else:
            pyplot.text(2.0, 1.2, "not applicable", va="center", ha="center", fontsize=7)

        if (panel_index // 2) % 2 == 0:
            pyplot.fill_between([0, 4], 0, 2.4, color="#EEEEEE", lw=0, zorder=-1)

        pyplot.xlabel("iteration", fontsize=7)
        pyplot.ylabel("average spectral norm", fontsize=7)
        pyplot.xticks(arange(5), ["1E" + str(v) for v in arange(5)], fontsize=7)
        pyplot.yticks(linspace(0, 2.4, 5), ["%.1f" % v for v in linspace(0, 2.4, 5)], fontsize=7)
        pyplot.xlim(0, 4)
        pyplot.ylim(0, 2.4)

    figure.align_labels()
    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.795, "b", va="center", ha="center", fontsize=12)
    figure.text(0.348, 0.795, "c", va="center", ha="center", fontsize=12)
    figure.text(0.677, 0.795, "d", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.600, "e", va="center", ha="center", fontsize=12)
    figure.text(0.348, 0.600, "f", va="center", ha="center", fontsize=12)
    figure.text(0.677, 0.600, "g", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.400, "h", va="center", ha="center", fontsize=12)
    figure.text(0.348, 0.400, "i", va="center", ha="center", fontsize=12)
    figure.text(0.677, 0.400, "j", va="center", ha="center", fontsize=12)
    figure.text(0.677, 0.200, "k", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp20.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_21():
    """
    Create Figure S21 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp21.pkl")

    pyplot.figure(figsize=(10, 5), tight_layout=True)

    for index, (x, y) in enumerate(task_data["a"]):
        pyplot.fill_betweenx(x, index - y * 0.4, index + y * 0.4, ec="k", fc="silver", lw=0.75)
    pyplot.xlabel("fraction of incoherent loops", fontsize=7)
    pyplot.ylabel("correlation coefficient\n(delta lipschitz constant v.s. delta spectral norm)", fontsize=7)
    pyplot.xticks(arange(11), [str(v) + "%" for v in arange(0, 101, 10)], fontsize=7)
    pyplot.yticks(linspace(-1, 1, 9), ["%.2f" % v for v in linspace(-1, 1, 9)], fontsize=7)
    pyplot.xlim(-0.6, 10.6)
    pyplot.ylim(-1.1, 1.1)

    pyplot.savefig(save_path + "supp21.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_22():
    """
    Create Figure S22 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp22.pkl")

    figure = pyplot.figure(figsize=(10, 6), tight_layout=True)
    grid = pyplot.GridSpec(7, 5)
    orders = [grid[0:2, 0], grid[0:2, 1], grid[0:2, 2], grid[0:2, 3],
              grid[2:4, 1], grid[2:4, 2], grid[2:4, 3],
              grid[4:6, 1], grid[4:6, 2], grid[4:6, 3], grid[4:6, 4]]
    x_values, y_values = linspace(-0.012, +0.002, 40), linspace(-0.008, +0.006, 40)
    for panel_index, grid_order in enumerate(orders):
        pyplot.subplot(grid_order)
        pyplot.title(("incoherent fraction = %d" % (panel_index * 10)) + "%", fontsize=7)
        densities, correlation = task_data[chr(ord("a") + panel_index)]
        values = log10(densities.T)
        values[values < -10] = -10
        pyplot.pcolormesh(x_values, y_values, values, cmap="Blues", vmin=-10, vmax=0, shading="gouraud")
        pyplot.text(-0.008, 0.004, "spearman\n%.2f" % correlation, va="center", ha="center", fontsize=7)

        pyplot.vlines(0, -0.008, +0.006, lw=0.75, ls="--", color="k", zorder=1)
        pyplot.hlines(0, -0.012, +0.002, lw=0.75, ls="--", color="k", zorder=1)

        pyplot.xlabel("delta gradient variance", fontsize=7)
        pyplot.ylabel("delta lipschitz constant", fontsize=7)
        pyplot.xticks([-0.012, -0.005, +0.002], ["-0.012", "-0.005", "+0.002"], fontsize=7)
        pyplot.yticks([-0.008, -0.001, +0.006], ["-0.008", "-0.001", "+0.006"], fontsize=7)
        pyplot.xlim(-0.012, +0.002)
        pyplot.ylim(-0.008, +0.006)

    pyplot.subplot(grid[6, :])

    locations, colors = linspace(-10, 0, 101), pyplot.get_cmap("Blues")(linspace(0, 1, 100))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 0, 1, ec="none", fc=color, lw=0)

    pyplot.xlabel("normalized density", fontsize=7)
    pyplot.xticks(linspace(-10, 0, 11),
                  ["≤1E-10", "1E-9", "1E-8", "1E-7", "1E-6", "1E-5", "1E-4", "1E-3", "1E-2", "1E-1", "1E+0"],
                  fontsize=7)
    pyplot.yticks([])
    pyplot.xlim(-10, 0)
    pyplot.ylim(0, 1)

    figure.align_labels()
    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.216, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.413, 0.990, "c", va="center", ha="center", fontsize=12)
    figure.text(0.610, 0.990, "d", va="center", ha="center", fontsize=12)
    figure.text(0.216, 0.700, "e", va="center", ha="center", fontsize=12)
    figure.text(0.413, 0.700, "f", va="center", ha="center", fontsize=12)
    figure.text(0.610, 0.700, "g", va="center", ha="center", fontsize=12)
    figure.text(0.216, 0.420, "h", va="center", ha="center", fontsize=12)
    figure.text(0.413, 0.420, "i", va="center", ha="center", fontsize=12)
    figure.text(0.610, 0.420, "j", va="center", ha="center", fontsize=12)
    figure.text(0.807, 0.420, "k", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp22.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_23():
    """
    Create Figure S23 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp23.pkl")

    figure = pyplot.figure(figsize=(10, 6), tight_layout=True)
    grid = pyplot.GridSpec(7, 5)
    orders = [grid[0:2, 0], grid[0:2, 1], grid[0:2, 2], grid[0:2, 3],
              grid[2:4, 1], grid[2:4, 2], grid[2:4, 3],
              grid[4:6, 1], grid[4:6, 2], grid[4:6, 3], grid[4:6, 4]]
    for panel_index, grid_order in enumerate(orders):
        panel_data = task_data[chr(ord("a") + panel_index)].T
        panel_data /= sum(panel_data)

        pyplot.subplot(grid_order)
        pyplot.title(("incoherent fraction = %d" % (panel_index * 10)) + "%", fontsize=7)

        show_data = panel_data.copy()
        show_data[show_data < 0.0001] = 0.0001
        pyplot.pcolormesh(arange(4), arange(4), log10(show_data), cmap="Blues", vmin=-4, vmax=0)
        for index_1 in range(3):
            for index_2 in range(3):
                value = panel_data[index_2, index_1]
                if panel_data[index_2, index_1] > 0.001:
                    info = ("%.2f" % (panel_data[index_2, index_1] * 100)) + "%"
                    if value > 0.5:
                        pyplot.text(index_1 + 0.5, index_2 + 0.5, info, color="w", va="center", ha="center", fontsize=6)
                    else:
                        pyplot.text(index_1 + 0.5, index_2 + 0.5, info, va="center", ha="center", fontsize=6)
                else:
                    pyplot.text(index_1 + 0.5, index_2 + 0.5, "< 0.01%", va="center", ha="center", fontsize=6)

        pyplot.xlabel("delta largest hessian eigenvalue", fontsize=7)
        pyplot.ylabel("delta Lipschitz constant", fontsize=7)
        pyplot.xticks(arange(3) + 0.5, ["neg", "con", "pos"], fontsize=7)
        pyplot.yticks(arange(3) + 0.5, ["neg", "con", "pos"], fontsize=7)
        pyplot.xlim(0, 3)
        pyplot.ylim(0, 3)

    pyplot.subplot(grid[6, :])

    locations, colors = linspace(-4, 0, 101), pyplot.get_cmap("Blues")(linspace(0, 1, 100))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 0, 1, ec="none", fc=color, lw=0)

    pyplot.xlabel("normalized density", fontsize=7)
    pyplot.xticks(linspace(-4, 0, 5), ["≤1E-4", "1E-3", "1E-2", "1E-1", "1E+0"], fontsize=7)
    pyplot.yticks([])
    pyplot.xlim(-4, 0)
    pyplot.ylim(0, 1)

    figure.align_labels()
    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.214, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.408, 0.990, "c", va="center", ha="center", fontsize=12)
    figure.text(0.604, 0.990, "d", va="center", ha="center", fontsize=12)
    figure.text(0.214, 0.700, "e", va="center", ha="center", fontsize=12)
    figure.text(0.408, 0.700, "f", va="center", ha="center", fontsize=12)
    figure.text(0.604, 0.700, "g", va="center", ha="center", fontsize=12)
    figure.text(0.214, 0.420, "h", va="center", ha="center", fontsize=12)
    figure.text(0.408, 0.420, "i", va="center", ha="center", fontsize=12)
    figure.text(0.604, 0.420, "j", va="center", ha="center", fontsize=12)
    figure.text(0.798, 0.420, "k", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp23.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_24():
    """
    Create Figure S24 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp24.pkl")

    labels = {
        "b": "baseline method",
        "i": r"[ $\mathcal{L}_c + \mathcal{C}$ ] - method",
        "c": r"[ $\mathcal{L}_i + \mathcal{C}$ ] - method",
        "a": r"$\mathcal{C}$ - method"
    }

    failure_types = ["failure type 1", "failure type 2", "failure type 3"]
    colors = ["#DABCBB", "#B2859B", "#A39EC0"]

    pyplot.figure(figsize=(10, 9), tight_layout=True)
    for index_1, (short_label, data) in enumerate(task_data.items()):
        label = labels[short_label]
        for index_2, (failure_type, values, color) in enumerate(zip(failure_types, data, colors)):
            pyplot.subplot(3, 4, index_2 * 4 + index_1 + 1)
            pyplot.title(label + "\n" + failure_type, fontsize=10)
            pyplot.fill_between([0, 5], 95, 195, fc="#EEEEEE", lw=0, zorder=-1)
            pyplot.hlines(195, 0, 5, lw=0.75, ls="--", color="silver", zorder=-1)
            pyplot.text(4.9, 196, "pass (≥ 195)", va="bottom", ha="right", fontsize=8)
            if values is not None:
                for index, value in enumerate(values.T):
                    pyplot.boxplot([value], positions=[index + 0.5], showfliers=False, showmeans=False,
                                   patch_artist=True,
                                   widths=0.3, boxprops=dict(lw=.75, ec="k", fc=color),
                                   medianprops=dict(lw=1.5, color="k"))
                links = median(values, axis=0)
                pyplot.plot(arange(5) + 0.5, links, color="k", lw=0.75, ls="--")
                pyplot.xlabel("evaluating noise level", fontsize=8)
                pyplot.ylabel("evaluating performance", fontsize=8)
                pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
                pyplot.yticks(arange(100, 201, 10), arange(100, 201, 10), fontsize=7)
                pyplot.xlim(0, 5)
                pyplot.ylim(95, 205)
            else:
                pyplot.text(2.5, 150, "no data", va="center", ha="center", fontsize=10)
                pyplot.xlabel("evaluating noise level", fontsize=8)
                pyplot.ylabel("evaluating performance", fontsize=8)
                pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
                pyplot.yticks(arange(100, 201, 10), arange(100, 201, 10), fontsize=7)
                pyplot.xlim(0, 5)
                pyplot.ylim(95, 205)

    pyplot.savefig(save_path + "supp24.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_25():
    """
    Create Figure S25 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp25.pkl")

    figure = pyplot.figure(figsize=(10, 10), tight_layout=True)
    location = 1
    for index in range(4):
        cases = task_data[chr(ord("a") + index)]
        for case_index, case in enumerate(cases):
            ax = pyplot.subplot(7, 5, location)
            pyplot.title("case " + str(case_index + 1) + " / " + str(len(cases)), fontsize=8)
            pyplot.plot(arange(5) + 0.5, case, color="silver", lw=2, marker="o", zorder=0)
            pyplot.scatter([argmax(case) + 0.5], [max(case)], color="k", zorder=1)
            pyplot.text(argmax(case) + 0.5, 220, "best", va="center", ha="center", fontsize=7)
            pyplot.xlabel("evaluating noise level", fontsize=8)
            pyplot.ylabel("performance", fontsize=8)
            pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
            pyplot.yticks(arange(50, 201, 50), arange(50, 201, 50), fontsize=7)
            pyplot.xlim(0, 5)
            pyplot.ylim(50, 230)
            # noinspection PyUnresolvedReferences
            ax.spines["top"].set_visible(False)
            # noinspection PyUnresolvedReferences
            ax.spines["right"].set_visible(False)
            location += 1
        while location % 5 != 1:
            location += 1

    figure.align_labels()
    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.43, "b", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.15, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp25.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_26():
    """
    Create Figure S26 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp26.pkl")

    figure = pyplot.figure(figsize=(10, 10), tight_layout=True)

    for index, (_, (description, values_1, values_2)) in enumerate(task_data.items()):
        ax = pyplot.subplot(3, 1, index + 1)
        pyplot.title(description.replace("-", " and "), fontsize=10)

        x, y_1, y_2 = arange(len(values_1), dtype=int), [], []
        for x_value in x:
            y_1.append(values_1[x_value])
            y_2.append(values_2[x_value])

        pyplot.bar(x - 0.2, y_1, ec="k", fc="#F5A889", width=0.4, lw=0.75, label="training set")
        pyplot.bar(x + 0.2, y_2, ec="k", fc="#ACD6EC", width=0.4, lw=0.75, label="testing set")
        for location, y_value in enumerate(y_1):
            pyplot.text(location - 0.2, y_value + 1, str(y_value), va="bottom", ha="center", fontsize=7)
        for location, y_value in enumerate(y_2):
            pyplot.text(location + 0.2, y_value + 1, str(y_value), va="bottom", ha="center", fontsize=7)

        pyplot.legend(loc="upper left", fontsize=7)
        pyplot.xlabel("label index", fontsize=8)
        pyplot.ylabel("number of instances", fontsize=8)
        pyplot.xticks(x, x, fontsize=7)
        pyplot.yticks(arange(0, 201, 20), arange(0, 201, 20), fontsize=7)
        pyplot.xlim(-0.6, len(values_1) - 0.4)
        pyplot.ylim(0, 200)

        # noinspection PyUnresolvedReferences
        ax.spines["top"].set_visible(False)
        # noinspection PyUnresolvedReferences
        ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.66, "b", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.33, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp26.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_27():
    """
    Create Figure S27 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp27.pkl")

    figure = pyplot.figure(figsize=(10, 10), tight_layout=True)

    pyplot.subplot(3, 1, 1)
    description, values_1, values_2 = task_data["a"]
    pyplot.title(description.replace("-", " and "), fontsize=10)
    random.seed(2023)
    for index in range(7):
        if index == 0:
            values = values_1[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index - 0.2, values, ec="k", fc="#F5A889", lw=0.75, label="training set", alpha=0.5)
            values = values_2[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index + 0.2, values, ec="k", fc="#ACD6EC", lw=0.75, label="testing set", alpha=0.5)
        else:
            values = values_1[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index - 0.2, values, ec="k", fc="#F5A889", lw=0.75, alpha=0.5)
            values = values_2[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index + 0.2, values, ec="k", fc="#ACD6EC", lw=0.75, alpha=0.5)
    pyplot.vlines(arange(6) + 0.5, -0.1, 1.1, color="k", lw=0.75, ls="--")
    pyplot.legend(loc="upper left", fontsize=7)
    pyplot.xlabel("feature index", fontsize=8)
    pyplot.ylabel("values of instances", fontsize=8)
    pyplot.xticks(arange(7), arange(7), fontsize=7)
    pyplot.yticks(linspace(0, 1, 6), ["%.1f" % v for v in linspace(0, 1, 6)], fontsize=7)
    pyplot.xlim(-0.5, 6.5)
    pyplot.ylim(-0.1, 1.1)

    pyplot.subplot(3, 1, 2)
    description, values_1, values_2 = task_data["b"]
    pyplot.title(description.replace("-", " and "), fontsize=10)
    random.seed(2023)
    for index in range(9):
        if index == 0:
            values = values_1[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index - 0.2, values, ec="k", fc="#F5A889", lw=0.75, label="training set", alpha=0.5)
            values = values_2[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index + 0.2, values, ec="k", fc="#ACD6EC", lw=0.75, label="testing set", alpha=0.5)
        else:
            values = values_1[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index - 0.2, values, ec="k", fc="#F5A889", lw=0.75, alpha=0.5)
            values = values_2[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index + 0.2, values, ec="k", fc="#ACD6EC", lw=0.75, alpha=0.5)
    pyplot.vlines(arange(8) + 0.5, -8, 88, color="k", lw=0.75, ls="--")
    pyplot.legend(loc="upper left", fontsize=7)
    pyplot.xlabel("feature index", fontsize=8)
    pyplot.ylabel("values of instances", fontsize=8)
    pyplot.xticks(arange(9), arange(9), fontsize=7)
    pyplot.yticks(arange(0, 81, 10), arange(0, 81, 10), fontsize=7)
    pyplot.xlim(-0.5, 8.5)
    pyplot.ylim(-8, 88)

    pyplot.subplot(3, 1, 3)
    description, values_1, values_2 = task_data["c"]
    pyplot.title(description.replace("-", " and "), fontsize=10)
    random.seed(2023)
    for index in range(6):
        if index == 0:
            values = values_1[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index - 0.2, values, ec="k", fc="#F5A889", lw=0.75, label="training set", alpha=0.5)
            values = values_2[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index + 0.2, values, ec="k", fc="#ACD6EC", lw=0.75, label="testing set", alpha=0.5)
        else:
            values = values_1[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index - 0.2, values, ec="k", fc="#F5A889", lw=0.75, alpha=0.5)
            values = values_2[:, index]
            bias = (random.random(size=(len(values))) - 0.50) * 0.30
            pyplot.scatter(bias + index + 0.2, values, ec="k", fc="#ACD6EC", lw=0.75, alpha=0.5)
    pyplot.vlines(arange(5) + 0.5, -50, 450, color="k", lw=0.75, ls="--")
    pyplot.legend(loc="upper left", fontsize=7)
    pyplot.xlabel("feature index", fontsize=8)
    pyplot.ylabel("values of instances", fontsize=8)
    pyplot.xticks(arange(6), arange(6), fontsize=7)
    pyplot.yticks(arange(-20, 421, 40), arange(-20, 421, 40), fontsize=7)
    pyplot.xlim(-0.5, 5.5)
    pyplot.ylim(-50, 450)

    figure.align_labels()
    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.66, "b", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.33, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp27.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_28():
    """
    Create Figure S28 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp28.pkl")

    labels = ["baseline method",
              r"[ $\mathcal{L}_c + \mathcal{C}$ ] - method",
              r"[ $\mathcal{L}_i + \mathcal{C}$ ] - method",
              r"$\mathcal{C}$ - method"]

    figure = pyplot.figure(figsize=(10, 3), tight_layout=True)

    ax = pyplot.subplot(1, 3, 1)

    pyplot.title("biology", fontsize=8)

    for index, (label, color) in enumerate(zip(labels, pyplot.get_cmap("binary")(linspace(0.0, 0.8, 4)))):
        locations = arange(5) - 0.3 + 0.2 * index
        pyplot.bar(locations, task_data["a"][index], width=0.2, fc=color, ec="k", lw=0.75, label=label)

    pyplot.legend(loc="lower left", framealpha=1, fontsize=7)
    pyplot.xlabel("training noise level", fontsize=8)
    pyplot.ylabel("average training performance (F1 score)", fontsize=8)
    pyplot.xticks(arange(5), ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
    pyplot.yticks(linspace(0, 1.0, 6), ["%.1f" % v for v in linspace(0, 1.0, 6)], fontsize=7)
    pyplot.xlim(-0.5, 4.5)
    pyplot.ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(1, 3, 2)
    pyplot.title("physics and chemistry", fontsize=8)

    for index, (label, color) in enumerate(zip(labels, pyplot.get_cmap("binary")(linspace(0.0, 0.8, 4)))):
        locations = arange(5) - 0.3 + 0.2 * index
        pyplot.bar(locations, task_data["b"][index], width=0.2, fc=color, ec="k", lw=0.75, label=label)

    pyplot.legend(loc="lower left", framealpha=1, fontsize=7)
    pyplot.xlabel("training noise level", fontsize=8)
    pyplot.ylabel("average training performance (F1 score)", fontsize=8)
    pyplot.xticks(arange(5), ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
    pyplot.yticks(linspace(0, 1.0, 6), ["%.1f" % v for v in linspace(0, 1.0, 6)], fontsize=7)
    pyplot.xlim(-0.5, 4.5)
    pyplot.ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(1, 3, 3)
    pyplot.title("health and medicine", fontsize=8)
    for index, (label, color) in enumerate(zip(labels, pyplot.get_cmap("binary")(linspace(0.0, 0.8, 4)))):
        locations = arange(5) - 0.3 + 0.2 * index
        pyplot.bar(locations, task_data["c"][index], width=0.2, fc=color, ec="k", lw=0.75, label=label)

    pyplot.legend(loc="lower left", framealpha=1, fontsize=7)
    pyplot.xlabel("training noise level", fontsize=8)
    pyplot.ylabel("average training performance (F1 score)", fontsize=8)
    pyplot.xticks(arange(5), ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
    pyplot.yticks(linspace(0, 1.0, 6), ["%.1f" % v for v in linspace(0, 1.0, 6)], fontsize=7)
    pyplot.xlim(-0.5, 4.5)
    pyplot.ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.348, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.676, 0.990, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp28.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_29():
    """
    Create Figure S27 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp29.pkl")

    labels = ["baseline method",
              r"[ $\mathcal{L}_c + \mathcal{C}$ ] - method",
              r"[ $\mathcal{L}_i + \mathcal{C}$ ] - method",
              r"$\mathcal{C}$ - method"]

    figure = pyplot.figure(figsize=(10, 7.5), tight_layout=True)

    for index, label in enumerate(labels):
        pyplot.subplot(3, 4, index + 1)
        pyplot.pcolormesh(arange(6), arange(6), task_data[chr(ord("a") + index)].T,
                          vmin=0, vmax=1, cmap="inferno")
        pyplot.plot([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
                    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0], lw=0.75, color="k")
        for location_x in range(5):
            for location_y in range(5):
                value = task_data[chr(ord("a") + index)][location_x, location_y]
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.3f" % value, color="w",
                            va="center", ha="center", fontsize=7)
        pyplot.title(label + " (biology)", fontsize=8)
        pyplot.xlabel("training noise level", fontsize=8)
        pyplot.ylabel("evaluating noise level", fontsize=8)
        pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
        pyplot.yticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
        pyplot.xlim(0, 5)
        pyplot.ylim(0, 5)

    for index, label in enumerate(labels):
        pyplot.subplot(3, 4, index + 5)
        pyplot.pcolormesh(arange(6), arange(6), task_data[chr(ord("a") + index + 4)].T,
                          vmin=0, vmax=1, cmap="inferno")
        pyplot.plot([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
                    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0], lw=0.75, color="k")
        for location_x in range(5):
            for location_y in range(5):
                value = task_data[chr(ord("a") + index + 4)][location_x, location_y]
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.3f" % value, color="w",
                            va="center", ha="center", fontsize=7)
        pyplot.title(label + " (physics & chemistry)", fontsize=8)
        pyplot.xlabel("training noise level", fontsize=8)
        pyplot.ylabel("evaluating noise level", fontsize=8)
        pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
        pyplot.yticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
        pyplot.xlim(0, 5)
        pyplot.ylim(0, 5)

    for index, label in enumerate(labels):
        pyplot.subplot(3, 4, index + 9)
        pyplot.pcolormesh(arange(6), arange(6), task_data[chr(ord("a") + index + 8)].T,
                          vmin=0, vmax=1, cmap="inferno")
        pyplot.plot([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
                    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0], lw=0.75, color="k")
        for location_x in range(5):
            for location_y in range(5):
                value = task_data[chr(ord("a") + index + 8)][location_x, location_y]
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.3f" % value, color="w",
                            va="center", ha="center", fontsize=7)
        pyplot.title(label + " (health & medicine)", fontsize=8)
        pyplot.xlabel("training noise level", fontsize=8)
        pyplot.ylabel("evaluating noise level", fontsize=8)
        pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
        pyplot.yticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
        pyplot.xlim(0, 5)
        pyplot.ylim(0, 5)

    figure.align_labels()
    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.267, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.513, 0.990, "c", va="center", ha="center", fontsize=12)
    figure.text(0.759, 0.990, "d", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.660, "e", va="center", ha="center", fontsize=12)
    figure.text(0.267, 0.660, "f", va="center", ha="center", fontsize=12)
    figure.text(0.513, 0.660, "g", va="center", ha="center", fontsize=12)
    figure.text(0.759, 0.660, "h", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.330, "i", va="center", ha="center", fontsize=12)
    figure.text(0.267, 0.330, "j", va="center", ha="center", fontsize=12)
    figure.text(0.513, 0.330, "k", va="center", ha="center", fontsize=12)
    figure.text(0.759, 0.330, "l", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp29.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_30():
    """
    Create Figure S30 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp30.pkl")

    labels = ["baseline",
              r"[ $\mathcal{L}_c + \mathcal{C}$ ]",
              r"[ $\mathcal{L}_i + \mathcal{C}$ ]",
              r"$\mathcal{C}$"]

    figure = pyplot.figure(figsize=(10, 3.5), tight_layout=True)

    pyplot.subplot(1, 3, 1)
    pyplot.pcolormesh(arange(5), arange(6), task_data["a"].T, vmin=0, vmax=100, cmap="viridis")
    for location_x in range(4):
        for location_y in range(5):
            if task_data["a"][location_x, location_y] < 100:
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01,
                            ("%d" % task_data["a"][location_x, location_y]) + "%",
                            color="w", va="center", ha="center", fontsize=7)
            else:
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01,
                            ("%d" % task_data["a"][location_x, location_y]) + "%",
                            color="k", va="center", ha="center", fontsize=7)
    pyplot.title("biology", fontsize=8)
    pyplot.xlabel("neuroevolution method", fontsize=8)
    pyplot.ylabel("training noise level", fontsize=8)
    pyplot.xticks(arange(4) + 0.5, labels, fontsize=7)
    pyplot.yticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
    pyplot.xlim(0, 4)
    pyplot.ylim(0, 5)

    pyplot.subplot(1, 3, 2)
    pyplot.pcolormesh(arange(5), arange(6), task_data["b"].T, vmin=0, vmax=100, cmap="viridis")
    for location_x in range(4):
        for location_y in range(5):
            if task_data["b"][location_x, location_y] < 100:
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01,
                            ("%d" % task_data["b"][location_x, location_y]) + "%",
                            color="w", va="center", ha="center", fontsize=7)
            else:
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01,
                            ("%d" % task_data["b"][location_x, location_y]) + "%",
                            color="k", va="center", ha="center", fontsize=7)
    pyplot.title("physics & chemistry", fontsize=8)
    pyplot.xlabel("neuroevolution method", fontsize=8)
    pyplot.ylabel("training noise level", fontsize=8)
    pyplot.xticks(arange(4) + 0.5, labels, fontsize=7)
    pyplot.yticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
    pyplot.xlim(0, 4)
    pyplot.ylim(0, 5)

    pyplot.subplot(1, 3, 3)
    pyplot.pcolormesh(arange(5), arange(6), task_data["c"].T, vmin=0, vmax=100, cmap="viridis")
    for location_x in range(4):
        for location_y in range(5):
            if task_data["c"][location_x, location_y] < 100:
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01,
                            ("%d" % task_data["c"][location_x, location_y]) + "%",
                            color="w", va="center", ha="center", fontsize=7)
            else:
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01,
                            ("%d" % task_data["c"][location_x, location_y]) + "%",
                            color="k", va="center", ha="center", fontsize=7)
    pyplot.title("health & medicine", fontsize=8)
    pyplot.xlabel("neuroevolution method", fontsize=8)
    pyplot.ylabel("training noise level", fontsize=8)
    pyplot.xticks(arange(4) + 0.5, labels, fontsize=7)
    pyplot.yticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
    pyplot.xlim(0, 4)
    pyplot.ylim(0, 5)

    figure.align_labels()
    figure.text(0.020, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.348, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.676, 0.990, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp30.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    supp_01()
    supp_02()
    supp_03()
    supp_04()
    supp_05()
    supp_06()
    supp_07()
    supp_08()
    supp_09()
    supp_10()
    supp_11()
    supp_12()
    supp_13()
    supp_14()
    supp_15()
    supp_16()
    supp_17()
    supp_18()
    supp_19()
    supp_20()
    supp_21()
    supp_22()
    supp_23()
    supp_24()
    supp_25()
    supp_26()
    supp_27()
    supp_28()
    supp_29()
    supp_30()
