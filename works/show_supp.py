"""
@Author      : Haoling Zhang
@Description : Plot all the figures in the supplementary file.
"""
from logging import getLogger, CRITICAL
from matplotlib import pyplot, rcParams, markers
from numpy import array, ones, arange, linspace, meshgrid, sin, abs, sum, min, max, mean, argmax, where, pi
from scipy.stats import gaussian_kde
from warnings import filterwarnings

from effect import calculate_landscape, NeuralMotif, estimate_lipschitz
from practice import acyclic_motifs

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
        pyplot.xlabel("predominant curvature proportion in the former landscape", fontsize=8)
        pyplot.ylabel("predominant curvature proportion in the latter landscape", fontsize=8)
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
    pyplot.contour(*meshgrid(linspace(2.2, 2.8, 101), linspace(1.2, 1.8, 101)),
                   source_1, vmin=-1, vmax=1, cmap="PRGn", lw=2, zorder=0)
    pyplot.plot([2.2, 2.8, 2.8, 2.2, 2.2], [1.2, 1.2, 1.8, 1.8, 1.2], lw=0.75, c="k", zorder=1)
    pyplot.text(2.50, 0.84, "latter landscape (contour)", va="center", ha="center", fontsize=8)
    pyplot.text(2.50, 0.16, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(2.16, 0.50, "$y$", va="center", ha="center", fontsize=8)
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

    pyplot.savefig(save_path + "supp04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_05():
    """
    Create Figure S5 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp05.pkl")

    labels = ["-0.3 ~ 0.0", "0.0 ~ 0.2", "0.2 ~ 0.4", "0.4 ~ 0.6", "0.6 ~ 0.8", "0.8 ~ 1.0"]

    figure = pyplot.figure(figsize=(10, 9.5), tight_layout=True)
    for index, (panel_index, values) in enumerate(task_data.items()):
        pyplot.subplot(2, 2, index + 1)
        counts = [len(values[values < 0.0]),
                  len(values[where((values >= 0.0) & (values < 0.2))]),
                  len(values[where((values >= 0.2) & (values < 0.4))]),
                  len(values[where((values >= 0.4) & (values < 0.6))]),
                  len(values[where((values >= 0.6) & (values < 0.8))]),
                  len(values[values >= 0.8])]
        pyplot.title("samples in coherent-loop " + str(index + 1), fontsize=8)
        pyplot.bar(arange(len(counts)), counts, ec="k", fc="#F2FEDC", lw=0.75)
        for location, count in enumerate(counts):
            pyplot.text(location, count, str(count), va="bottom", ha="center", fontsize=7)
        pyplot.xlabel("Spearman's rank correlation coefficient", fontsize=8)
        pyplot.ylabel("number of sample", fontsize=8)
        pyplot.xticks(arange(len(labels)), labels, fontsize=7)
        pyplot.yticks(arange(0, 61, 10), arange(0, 61, 10), fontsize=7)
        pyplot.xlim(-0.6, 5.6)
        pyplot.ylim(0, 60)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.50, "c", va="center", ha="center", fontsize=12)
    figure.text(0.512, 0.50, "d", va="center", ha="center", fontsize=12)

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
        pyplot.title(str(len(values)) + " samples in coherent-loop " + str(index + 1), fontsize=8)

        for location in linspace(0.011, 0.013, 11)[1:-1]:
            pyplot.hlines(location, 0.00, 0.008, lw=0.75, ls="--", color="k", zorder=1)

        for location in linspace(0.00, 0.008, 9)[1:-1]:
            pyplot.vlines(location, 0.011, 0.013, lw=0.75, ls="--", color="k", zorder=1)

        pyplot.scatter(values[:, 0], values[:, 1], ec="k", fc="w", lw=0.75, zorder=2)
        pyplot.xlabel("L2-norm loss before escaping", fontsize=8)
        pyplot.ylabel("L2-norm loss after escaping", fontsize=8)
        pyplot.xticks(linspace(0, 0.008, 9),
                      ["%.3f" % v for v in linspace(0, 0.008, 9)], fontsize=7)
        pyplot.yticks(linspace(0.011, 0.013, 11),
                      ["%.4f" % v for v in linspace(0.011, 0.013, 11)], fontsize=7)
        pyplot.xlim(0.000, 0.008)
        pyplot.ylim(0.011, 0.013)

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
    figure = pyplot.figure(figsize=(10, 6), tight_layout=True)
    pyplot.subplot(2, 1, 1)
    source_landscape, target_landscape, correlation, correlations = task_data["a"]
    pyplot.text(2.00, 1.03, "$z$", va="center", ha="center", fontsize=8)
    locations, colors = linspace(0.20, 3.80, 100), pyplot.get_cmap("PRGn")(linspace(0, 1, 100))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 0.97, 1.00, fc=color, lw=0, zorder=1)
    for location, label in zip(linspace(0.20, 3.80, 6), linspace(0, 1, 6)):
        pyplot.vlines(location, 0.95, 0.97, lw=0.75, color="k", zorder=1)
        pyplot.text(location, 0.92, "%.1f" % label, va="center", ha="center", fontsize=7)
    pyplot.plot([0.2, 3.8, 3.8, 0.2, 0.2], [0.97, 0.97, 1.0, 1.0, 0.97], lw=0.75, color="k", zorder=2)
    pyplot.text(0.50, 0.83, "former landscape", va="center", ha="center", fontsize=7)
    pyplot.text(0.50, 0.17, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(0.17, 0.50, "$y$", va="center", ha="center", fontsize=8)
    pyplot.pcolormesh(linspace(0.2, 0.8, 41), linspace(0.2, 0.8, 41), source_landscape,
                      vmin=-1, vmax=1, cmap="PRGn", shading="gouraud")
    pyplot.plot([0.2, 0.8, 0.8, 0.2, 0.2], [0.2, 0.2, 0.8, 0.8, 0.2], color="k", lw=0.75, zorder=2)
    pyplot.text(3.50, 0.83, "latter landscape", va="center", ha="center", fontsize=7)
    pyplot.text(3.50, 0.17, "$x$", va="center", ha="center", fontsize=8)
    pyplot.text(3.17, 0.50, "$y$", va="center", ha="center", fontsize=8)
    pyplot.pcolormesh(linspace(3.2, 3.8, 41), linspace(0.2, 0.8, 41), target_landscape,
                      vmin=-1, vmax=1, cmap="PRGn", shading="gouraud")
    pyplot.plot([3.2, 3.8, 3.8, 3.2, 3.2], [0.2, 0.2, 0.8, 0.8, 0.2], color="k", lw=0.75, zorder=2)
    pyplot.plot([1.2, 1.2, 2.8, 2.8], [0.8, 0.2, 0.2, 0.8], lw=0.75, color="k", zorder=2)
    for location, label in zip(linspace(1.20, 2.80, 11), arange(0, 101, 10)):
        pyplot.vlines(location, 0.18, 0.20, lw=0.75, color="k", zorder=1)
        pyplot.text(location, 0.15, label, va="center", ha="center", fontsize=7)
    for location, label in zip(linspace(0.2, 0.8, 6), linspace(-1.0, 1.0, 6)):
        pyplot.hlines(location, 1.18, 1.20, lw=0.75, color="k", zorder=1)
        pyplot.text(1.16, location, "%.1f" % label, va="center", ha="right", fontsize=7)
    pyplot.text(1.05, 0.5, "Spearman's rank correlation coefficient",
                va="center", ha="center", rotation=90, fontsize=7)
    for location, label in zip(linspace(0.2, 0.8, 6), linspace(-1.0, 1.0, 6)):
        pyplot.hlines(location, 2.80, 2.82, lw=0.75, color="k", zorder=1)
        pyplot.text(2.90, location, "%.1f" % label, va="center", ha="right", fontsize=7)
    pyplot.text(2.96, 0.5, "Spearman's rank correlation coefficient",
                va="center", ha="center", rotation=90, fontsize=7)
    for location in linspace(0.2, 0.8, 6)[1:]:
        pyplot.hlines(location, 1.2, 2.8, lw=0.75, ls="--", color="silver", zorder=0)
    correlations = (array(correlations) + 1.0) / 2.0 * 0.6 + 0.2
    pyplot.plot(linspace(1.21, 2.79, 100)[:len(correlations)],
                correlations, lw=2, color="k", zorder=2)
    location = linspace(1.21, 2.79, 100)[32]
    pyplot.vlines(location, 0.2, 0.8, lw=0.75, ls="--", color="silver", zorder=1)
    pyplot.plot([1.2, 1.2, location, location], [0.80, 0.83, 0.83, 0.80], lw=0.75, color="k", zorder=2)
    pyplot.plot([2.8, 2.8, location, location], [0.80, 0.83, 0.83, 0.80], lw=0.75, color="k", zorder=2)
    pyplot.text((1.2 + location) / 2, 0.86, "rotate", va="center", ha="center", fontsize=7)
    pyplot.text((2.8 + location) / 2, 0.86, "adjust", va="center", ha="center", fontsize=7)
    pyplot.text(2.00, 0.10, "round", va="center", ha="center", fontsize=8)
    pyplot.xlim(0.19, 3.81)
    pyplot.ylim(0.10, 1.05)
    pyplot.axis("off")

    ax = pyplot.subplot(2, 1, 2)
    origin, changed = task_data["b"][:, 0], task_data["b"][:, 1]
    x = linspace(min(origin), max(origin), 100)
    y = gaussian_kde(origin)(x)
    y /= sum(y)
    pyplot.plot(x, y, lw=0.75, color="k", zorder=2)
    pyplot.fill_between(x, 0, y, ec="k", fc="#BEB8DC", lw=0.75, zorder=2, label="calculation of entire process")
    x = linspace(min(changed), max(changed), 100)
    y = gaussian_kde(changed)(x)
    y /= sum(y)
    pyplot.fill_between(x, 0, y, ec="k", fc="#FA7F6F", lw=0.75, zorder=2, label="calculation per round")
    pyplot.legend(loc="upper right", ncol=2, fontsize=7)
    pyplot.xlabel("Spearman's rank correlation coefficient", fontsize=8)
    pyplot.xticks(linspace(0.0, 0.6, 13), ["%.2f" % v for v in linspace(0.0, 0.6, 13)], fontsize=7)
    pyplot.yticks([])
    pyplot.xlim(0.0, 0.6)
    pyplot.ylim(0.00, 0.015)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.02, 0.50, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "supp07.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_08():
    """
    Create Figure S8 in the supplementary file.
    """
    task_data = load_data(sort_path + "supp08.pkl")
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
            pyplot.xlabel("evaluating error scale", fontsize=8)
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

    pyplot.savefig(save_path + "supp08.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_09():
    """
    Create Figure S9 in the supplementary file.
    """
    pyplot.figure(figsize=(10, 4))
    grid = pyplot.GridSpec(4, 10)
    pyplot.subplots_adjust(wspace=0, hspace=0)

    # noinspection PyTypeChecker
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
            pyplot.annotate("", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
        elif motif.get_edge_data(former, latter)["weight"] == -1:
            pyplot.annotate("", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
    pyplot.text(x=info[1][0], y=info[2][0] - 0.06, s="x", fontsize=10, va="top", ha="center")
    pyplot.text(x=info[1][1], y=info[2][1] - 0.06, s="y", fontsize=10, va="top", ha="center")
    pyplot.text(x=info[1][2], y=info[2][2] + 0.06, s="z", fontsize=10, va="bottom", ha="center")
    pyplot.text(-0.05, 0.45, "incoherent loop", va="center", ha="center", fontsize=10, rotation=90)
    pyplot.xlim(-0.10, 1.0)
    pyplot.ylim(-0.05, 1.0)
    pyplot.axis("off")

    # noinspection PyTypeChecker
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
            pyplot.annotate("", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            shrinkA=6, shrinkB=6, lw=1), zorder=2)
        elif motif.get_edge_data(former, latter)["weight"] == -1:
            pyplot.annotate("", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
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

    # noinspection PyTypeChecker
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
    pyplot.annotate("", xy=(0.45, 0.5), xytext=(0.35, 0.5),
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

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:4, 2:6])
    pyplot.text(0.09, 0.675, "x molecule", fontsize=7, va="center", ha="right", rotation=90)
    pyplot.text(0.09, 0.325, "y molecule", fontsize=7, va="center", ha="right", rotation=90)
    pyplot.text(0.20, 0.18, "time", fontsize=7, va="top", ha="center")
    pyplot.text(0.20, 0.53, "time", fontsize=7, va="top", ha="center")
    pyplot.plot([0.1, 0.1, 0.3], [0.45, 0.20, 0.20], color="black", linewidth=0.75)
    pyplot.plot([0.1, 0.1, 0.3], [0.80, 0.55, 0.55], color="black", linewidth=0.75)
    values = 0.60 + inputs_x * 0.15
    pyplot.plot(linspace(0.1, 0.3, len(values)), values, color="k", linewidth=0.75)
    values = 0.25 + inputs_yi * 0.15
    pyplot.plot(linspace(0.1, 0.3, len(values))[20: 60], values[20: 60], color="k", linewidth=0.75, linestyle=":")
    values = 0.25 + inputs_yc * 0.15
    pyplot.plot(linspace(0.1, 0.3, len(values)), values, color="k", linewidth=0.75)
    pyplot.text(0.4, 0.53, "reaction", fontsize=7, va="bottom", ha="center")
    pyplot.annotate("", xy=(0.45, 0.5), xytext=(0.35, 0.5),
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
    lipschitz1 = estimate_lipschitz(value_range=(-1, +1), points=41, output=matrix1, norm_type="L-2")
    lipschitz2 = estimate_lipschitz(value_range=(-1, +1), points=41, output=matrix2, norm_type="L-2")

    # noinspection PyTypeChecker
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
    pyplot.annotate("", xy=(0.55, 0.5), xytext=(0.45, 0.5),
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

    # noinspection PyTypeChecker
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
    pyplot.annotate("", xy=(0.55, 0.5), xytext=(0.45, 0.5),
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

    pyplot.savefig(save_path + "supp09.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_10():
    """
    Create Figure S10 in the supplementary file.
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

    pyplot.savefig(save_path + "supp10.pdf", format="pdf", bbox_inches="tight", dpi=600)
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
