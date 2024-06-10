"""
@Author      : Haoling Zhang
@Description : Plot all the figures in the supplementary file.
"""
from logging import getLogger, CRITICAL
from matplotlib import pyplot, rcParams
from numpy import array, arange, linspace, meshgrid, abs, sum, min, median, max, mean, argmax, where
from scipy.stats import gaussian_kde
from warnings import filterwarnings

from effect import NeuralMotif, calculate_landscape

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
    task_data = load_data(sort_path + "supp03.pkl")

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

    pyplot.savefig(save_path + "supp03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_04():
    """
    Create Figure S4 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp04.pkl")

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

    pyplot.savefig(save_path + "supp04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_05():
    """
    Create Figure S5 in the supplementary file.
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

    pyplot.savefig(save_path + "supp05.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_06():
    """
    Create Figure S6 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp06.pkl")

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
    pyplot.savefig(save_path + "supp06.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_07():
    """
    Create Figure S7 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp07.pkl")

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
    pyplot.savefig(save_path + "supp07.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_08():
    """
    Create Figure S8 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp08.pkl")

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

    pyplot.savefig(save_path + "supp08.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_09():
    """
    Create Figure S9 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp09.pkl")

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

    pyplot.savefig(save_path + "supp09.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_10():
    """
    Create Figure S10 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp10.pkl")

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

    pyplot.savefig(save_path + "supp10.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_11():
    """
    Create Figure S11 in the supplementary file.
    """
    motif_1 = NeuralMotif(motif_type="incoherent-loop", motif_index=1,
                          activations=["relu", "tanh"], aggregations=["max", "sum"],
                          weights=[-1.0, +0.7, +1.0], biases=[0.0, 0.0])
    motif_2 = NeuralMotif(motif_type="collider", motif_index=1,
                          activations=["tanh"], aggregations=["sum"],
                          weights=[+0.3, +0.5], biases=[0.0])
    value_range, points = (-1, +1), 101

    pyplot.figure(figsize=(10, 5.5), tight_layout=True)
    pyplot.text(0.925, 2.10, "motif information", va="center", ha="center", fontsize=12)
    pyplot.fill_between([0.05, 1.80], 1.98, 2.02, color="#DDDDDD", zorder=0)
    for matrix_index in range(2):
        pyplot.scatter([0.05], [matrix_index + 0.20], fc="w", ec="k", lw=1.5, s=80, zorder=2)
        pyplot.scatter([0.45], [matrix_index + 0.20], fc="silver", ec="k", lw=1.5, s=80, zorder=2)
        pyplot.scatter([0.25], [matrix_index + 0.80], fc="k", ec="k", lw=1.5, s=80, zorder=2)
        pyplot.text(0.05, matrix_index + 0.15, "$x$", va="top", ha="center", fontsize=12)
        pyplot.text(0.45, matrix_index + 0.15, "$y$", va="top", ha="center", fontsize=12)
        pyplot.text(0.25, matrix_index + 0.85, "$z$", va="bottom", ha="center", fontsize=12)
    pyplot.annotate("", xy=(0.05, 0.20), xytext=(0.25, 0.80),
                    arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="black",
                                    shrinkA=6, shrinkB=6, lw=1.5,
                                    ls=("-" if motif_2.w[0].value() > 0 else ":")))
    pyplot.annotate("", xy=(0.45, 0.20), xytext=(0.25, 0.80),
                    arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="black",
                                    shrinkA=6, shrinkB=6, lw=1.5,
                                    ls=("-" if motif_2.w[1].value() > 0 else ":")))
    pyplot.annotate("", xy=(0.05, 1.20), xytext=(0.45, 1.20),
                    arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="black",
                                    shrinkA=6, shrinkB=6, lw=1.5,
                                    ls=("-" if motif_1.w[0].value() > 0 else ":")))
    pyplot.annotate("", xy=(0.05, 1.20), xytext=(0.25, 1.80),
                    arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="black",
                                    shrinkA=6, shrinkB=6, lw=1.5,
                                    ls=("-" if motif_1.w[1].value() > 0 else ":")))
    pyplot.annotate("", xy=(0.45, 1.20), xytext=(0.25, 1.80),
                    arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="black",
                                    shrinkA=6, shrinkB=6, lw=1.5,
                                    ls=("-" if motif_1.w[2].value() > 0 else ":")))
    pyplot.text(0.25, 0.60, motif_2.a[0] + " / " + motif_2.g[0], va="center", ha="center",
                bbox=dict(facecolor="#FFCCCC", edgecolor="#FF8787", boxstyle="round"), fontsize=12, zorder=3)
    pyplot.text(0.25, 1.20, " " + motif_1.a[0] + " / " + motif_1.g[0] + " ", va="center", ha="center",
                bbox=dict(facecolor="#FFCCCC", edgecolor="#FF8787", boxstyle="round"), fontsize=12, zorder=3)
    pyplot.text(0.25, 1.60, motif_1.a[1] + " / " + motif_1.g[1], va="center", ha="center",
                bbox=dict(facecolor="#FFCCCC", edgecolor="#FF8787", boxstyle="round"), fontsize=12, zorder=3)
    for location, info_1, info_2, value in zip([0.65, 0.50, 0.35],
                                               ["weight", "weight", "bias"],
                                               [r"$x \rightarrow z$", r"$y \rightarrow z$",
                                                r"$x,y \rightarrow z$"],
                                               [motif_2.w[0].value(), motif_2.w[1].value(),
                                                motif_2.b[0].value()]):
        pyplot.text(0.65, location, info_1, va="center", ha="left", fontsize=10)
        pyplot.text(0.85, location, info_2, va="center", ha="left", fontsize=12)
        pyplot.plot([1.10, 1.10, 1.70, 1.70],
                    [location - 0.03, location, location, location - 0.03], color="k", lw=1)
        pyplot.vlines(1.40, location - 0.03, location, lw=1, color="k")
        color = pyplot.get_cmap("binary")([abs(value)])
        pyplot.scatter([(value + 1.0) / 2.0 * 0.6 + 1.10], location + 0.05,
                       marker="v", color=color, ec="k", s=30, lw=1)
    pyplot.text(1.10, 0.25, "\N{MINUS SIGN}1", va="center", ha="center", fontsize=9)
    pyplot.text(1.40, 0.25, "0", va="center", ha="center", fontsize=9)
    pyplot.text(1.70, 0.25, "+1", va="center", ha="center", fontsize=9)
    for location, info_1, info_2, value in zip([1.80, 1.65, 1.50, 1.35, 1.20],
                                               ["weight", "weight", "weight", "bias", "bias"],
                                               [r"$x \rightarrow y$", r"$x \rightarrow z$",
                                                r"$y \rightarrow z$", r"$x \rightarrow y$",
                                                r"$x,y \rightarrow z$"],
                                               [motif_1.w[0].value(), motif_1.w[1].value(),
                                                motif_1.w[2].value(), motif_1.b[0].value(),
                                                motif_1.b[1].value()]):
        pyplot.text(0.65, location, info_1, va="center", ha="left", fontsize=10)
        pyplot.text(0.85, location, info_2, va="center", ha="left", fontsize=12)
        pyplot.plot([1.10, 1.10, 1.70, 1.70],
                    [location - 0.03, location, location, location - 0.03],
                    color="k", lw=1)
        pyplot.vlines(1.40, location - 0.03, location, lw=1, color="k")
        color = pyplot.get_cmap("binary")([abs(value)])
        pyplot.scatter([(value + 1.0) / 2.0 * 0.6 + 1.10], location + 0.05, marker="v",
                       color=color, ec="k", s=24, lw=1)
    pyplot.text(1.10, 1.10, "\N{MINUS SIGN}1", va="center", ha="center", fontsize=9)
    pyplot.text(1.40, 1.10, "0", va="center", ha="center", fontsize=9)
    pyplot.text(1.70, 1.10, "+1", va="center", ha="center", fontsize=9)
    pyplot.text(2.325, 2.10, "output landscape", va="center", ha="center", fontsize=12)
    pyplot.fill_between([1.95, 2.70], 1.98, 2.02, color="#DDDDDD", zorder=0)
    for matrix_index, matrix in enumerate([calculate_landscape(value_range, points, motif_2),
                                           calculate_landscape(value_range, points, motif_1)]):
        pyplot.pcolormesh(linspace(2.10, 2.50, 101), linspace(matrix_index + 0.2, matrix_index + 0.8, 101), matrix,
                          cmap="PRGn", vmin=-1, vmax=1, shading="gouraud", zorder=1)
        pyplot.plot([2.10, 2.50, 2.50, 2.10, 2.10],
                    [matrix_index + 0.20, matrix_index + 0.20,
                     matrix_index + 0.80, matrix_index + 0.80, matrix_index + 0.20],
                    color="k", lw=1, zorder=2)
        pyplot.text(2.10, matrix_index + 0.10, "\N{MINUS SIGN}1", va="center", ha="center", fontsize=9)
        pyplot.text(2.30, matrix_index + 0.10, "0", va="center", ha="center", fontsize=9)
        pyplot.text(2.50, matrix_index + 0.10, "+1", va="center", ha="center", fontsize=9)
        for value in [2.10, 2.30, 2.50]:
            pyplot.vlines(value, matrix_index + 0.17, matrix_index + 0.20, lw=1, color="k")
        pyplot.text(2.04, matrix_index + 0.20, "\N{MINUS SIGN}1", va="center", ha="center", fontsize=9)
        pyplot.text(2.04, matrix_index + 0.50, "0", va="center", ha="center", fontsize=9)
        pyplot.text(2.04, matrix_index + 0.80, "+1", va="center", ha="center", fontsize=9)
        for value in [matrix_index + 0.20, matrix_index + 0.50, matrix_index + 0.80]:
            pyplot.hlines(value, 2.08, 2.10, lw=1, color="k")
        pyplot.text(2.30, matrix_index + 0.02, "$x$", va="center", ha="center", fontsize=12)
        pyplot.text(1.98, matrix_index + 0.50, "$y$", va="center", ha="center", fontsize=12)
        for former, latter, color in zip(linspace(matrix_index + 0.2, matrix_index + 0.8, 41)[:-1],
                                         linspace(matrix_index + 0.2, matrix_index + 0.8, 41)[1:],
                                         pyplot.get_cmap("PRGn")(linspace(0, 1, 40))):
            pyplot.fill_between([2.55, 2.58], former, latter, fc=color, lw=0, zorder=1)
        for location, value in zip(linspace(matrix_index + 0.2, matrix_index + 0.8, 3),
                                   ["\N{MINUS SIGN}1", "0", "+1"]):
            pyplot.hlines(location, 2.58, 2.60, lw=1, color="k")
            pyplot.text(2.65, location, value, va="center", ha="center", fontsize=9)
        pyplot.plot([2.55, 2.58, 2.58, 2.55, 2.55],
                    [matrix_index + 0.20, matrix_index + 0.20,
                     matrix_index + 0.80, matrix_index + 0.80, matrix_index + 0.20],
                    color="k", lw=1, zorder=2)
        pyplot.text(2.565, matrix_index + 0.88, "$z$", va="center", ha="center", fontsize=12)
    pyplot.xlim(0.00, 2.70)
    pyplot.ylim(0.00, 2.20)
    pyplot.axis("off")

    pyplot.savefig(save_path + "supp11.pdf", format="pdf", bbox_inches="tight", dpi=600)
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
