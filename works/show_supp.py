from logging import getLogger, CRITICAL
from matplotlib import pyplot, rcParams
from numpy import arange, linspace, array, zeros, random, percentile, where, nan
from numpy import log2, log10, sqrt, min, max, abs, sum, argmax, std
from os import mkdir
from scipy.stats import gaussian_kde, spearmanr
from shutil import rmtree
from warnings import filterwarnings

from practice import acyclic_motifs

from works import load_data, draw_info, save_data

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
    task_data = load_data(sort_path + "supp01.pkl")

    figure = pyplot.figure(figsize=(10, 6), tight_layout=True)

    motif_types = ["incoherent-loop", "coherent-loop", "collider"]
    activations, aggregations = ["tanh", "sigmoid", "relu"], ["sum", "max"]

    pyplot.subplot(2, 1, 1)
    for motif_location in arange(3):
        pyplot.text(motif_location * 8 + 5.0, 5.55, "propagation style", va="center", ha="center", fontsize=10)
        pyplot.text(motif_location * 8 + 0.6, 2.00, motif_types[motif_location] + " structure",
                    va="center", ha="center", fontsize=10, rotation=90)
        for structure_location in arange(4):
            info = draw_info[motif_types[motif_location]]
            info[2][2] = 0.8
            motif = acyclic_motifs[motif_types[motif_location]][structure_location]
            bias_x, bias_y = motif_location * 8 + 0.875, 4 - structure_location - 1.0
            for index, (px, py) in enumerate(zip(info[1], info[2])):
                if index + 1 in info[3]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="w", edgecolor="k", lw=0.75, s=15, zorder=2)
                elif index + 1 in info[4]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="k", edgecolor="k", lw=0.75, s=15, zorder=2)
                elif index + 1 in info[5]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="k", lw=0.75, s=15, zorder=2)
                else:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="k", lw=0.75, s=15, zorder=2)
            x, y = info[1], info[2]
            for former, latter in motif.edges:
                if motif.get_edge_data(former, latter)["weight"] == 1:
                    pyplot.annotate("", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
                else:
                    pyplot.annotate("", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle=":"))
        for function_location in arange(6):
            x = motif_location * 8 + function_location + 2.0 + 0.5
            if function_location % 2 == 0:
                pyplot.plot([x, x, x + 0.3, x + 0.3], [4.6, 4.7, 4.7, 4.8], color="k", lw=0.75)
                pyplot.text(x, 4.4, aggregations[function_location % 2], va="center", ha="center", fontsize=8)
                pyplot.text(x + 0.5, 5.0, activations[(function_location // 2) % 3], va="center", ha="center",
                            fontsize=8)
            else:
                pyplot.plot([x, x, x - 0.3, x - 0.3], [4.6, 4.7, 4.7, 4.8], color="k", lw=0.75)
                pyplot.text(x, 4.4, aggregations[function_location % 2], va="center", ha="center", fontsize=8)

            for structure_location in arange(4):
                y = 4 - structure_location - 0.5
                count = task_data["a"][motif_location * 6 + function_location, structure_location]
                count_color = pyplot.get_cmap("RdYlGn")([(log10(count) - 2.0) / 4.0])
                pyplot.fill_between([x - 0.4, x + 0.4], y - 0.4, y + 0.4, fc=count_color, ec="k", lw=0.75)
                pyplot.text(x, y - 0.05, str(int(count)), va="center", ha="center", fontsize=6)

    pyplot.text(13, -0.35, "number of collected samples (difference is greater than 0.01 - L1 loss)",
                va="center", ha="center", fontsize=10)
    pyplot.plot([2.1, 23.9, 23.9, 2.1, 2.1], [-0.6, -0.6, -0.8, -0.8, -0.6], color="k", lw=0.75)
    colors = pyplot.get_cmap("RdYlGn")(linspace(0, 1, 100))
    interval = (23.9 - 2.1) / 100.0
    for index in range(100):
        pyplot.fill_between([2.1 + index * interval, 2.1 + (index + 1) * interval], -0.8, -0.6, color=colors[index],
                            lw=0)
    interval = (23.9 - 2.1) / 4.0
    for index in range(5):
        pyplot.vlines(2.1 + index * interval, -0.9, -0.8, lw=0.75)
        pyplot.text(2.1 + index * interval, -1.1, "1E+" + str(index + 2), va="center", ha="center", fontsize=8)
    pyplot.axis("off")
    pyplot.xlim(0, 24)
    pyplot.ylim(-1.2, 6)

    pyplot.subplot(2, 1, 2)
    for motif_location in arange(3):
        pyplot.text(motif_location * 8 + 5.0, 5.55, "propagation style", va="center", ha="center", fontsize=10)
        pyplot.text(motif_location * 8 + 0.6, 2, motif_types[motif_location] + " structure",
                    va="center", ha="center", fontsize=10, rotation=90)
        for structure_location in arange(4):
            info = draw_info[motif_types[motif_location]]
            info[2][2] = 0.8
            motif = acyclic_motifs[motif_types[motif_location]][structure_location]
            bias_x, bias_y = motif_location * 8 + 0.875, 4 - structure_location - 1.0
            for index, (px, py) in enumerate(zip(info[1], info[2])):
                if index + 1 in info[3]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="w", edgecolor="k", lw=0.75, s=15, zorder=2)
                elif index + 1 in info[4]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="k", edgecolor="k", lw=0.75, s=15, zorder=2)
                elif index + 1 in info[5]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="k", lw=0.75, s=15, zorder=2)
                else:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="k", lw=0.75, s=15, zorder=2)
            x, y = info[1], info[2]
            for former, latter in motif.edges:
                if motif.get_edge_data(former, latter)["weight"] == 1:
                    pyplot.annotate("", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
                else:
                    pyplot.annotate("", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle=":"))
        for function_location in arange(6):
            x = motif_location * 8 + function_location + 2.0 + 0.5
            if function_location % 2 == 0:
                pyplot.plot([x, x, x + 0.3, x + 0.3], [4.6, 4.7, 4.7, 4.8], color="k", lw=0.75)
                pyplot.text(x, 4.4, aggregations[function_location % 2], va="center", ha="center", fontsize=8)
                pyplot.text(x + 0.5, 5.0, activations[(function_location // 2) % 3], va="center", ha="center",
                            fontsize=8)
            else:
                pyplot.plot([x, x, x - 0.3, x - 0.3], [4.6, 4.7, 4.7, 4.8], color="k", lw=0.75)
                pyplot.text(x, 4.4, aggregations[function_location % 2], va="center", ha="center", fontsize=8)

            for structure_location in arange(4):
                y = 4 - structure_location - 0.5
                rate = task_data["b"][motif_location * 6 + function_location, structure_location]
                rate_color = pyplot.get_cmap("plasma")([(log10(rate) + 3.0) / 3.0 / 2.0 + 0.5])
                pyplot.fill_between([x - 0.4, x + 0.4], y - 0.4, y + 0.4, fc=rate_color, ec="k", lw=0.75)
                pyplot.text(x, y - 0.05, "%.1f" % (rate * 100) + "%", va="center", ha="center", fontsize=6)

    pyplot.text(13, -0.35, "proportion of collected samples in generated samples",
                va="center", ha="center", fontsize=10)
    pyplot.plot([2.1, 23.9, 23.9, 2.1, 2.1], [-0.6, -0.6, -0.8, -0.8, -0.6], color="k", lw=0.75)
    colors = pyplot.get_cmap("plasma")(linspace(0.5, 1, 100))
    interval = (23.9 - 2.1) / 100.0
    for index in range(100):
        pyplot.fill_between([2.1 + index * interval, 2.1 + (index + 1) * interval], -0.8, -0.6, color=colors[index],
                            lw=0)
    interval = (23.9 - 2.1) / 3.0
    for index, info in enumerate(["0.1%", "1%", "10%", "100%"]):
        pyplot.vlines(2.1 + index * interval, -0.9, -0.8, lw=0.75)
        pyplot.text(2.1 + index * interval, -1.1, info, va="center", ha="center", fontsize=8)
    pyplot.axis("off")
    pyplot.xlim(0, 24)
    pyplot.ylim(-1.2, 6)

    figure.text(0.022, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.022, 0.48, "b", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_02():
    figure = pyplot.figure(figsize=(10, 3.5), tight_layout=True)

    # noinspection PyTypeChecker
    random.seed(2023)
    # noinspection PyArgumentList
    samples_1 = random.random(size=(10, 2)) - 0.5
    # noinspection PyArgumentList
    samples_2 = random.random(size=(25, 2)) - 0.5

    pyplot.subplot(1, 3, 1)
    pyplot.title("hyper-spatial location distribution of samples", fontsize=9)
    pyplot.scatter(samples_1[:, 0], samples_1[:, 1], fc="#88CCF8", ec="k", s=24, label="population 1 (10 samples)")
    pyplot.scatter(samples_2[:, 0], samples_2[:, 1], fc="#FCB1AB", ec="k", s=24, label="population 2 (25 samples)")
    pyplot.legend(loc="upper right", fontsize=8)
    pyplot.xlim(-0.7, 0.7)
    pyplot.ylim(-0.6, 0.8)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.subplot(1, 3, 2)
    pyplot.title("replacement rate of population 1 by 2", fontsize=9)
    select_1 = array([0, 1, 4, 5, 6, 7])
    select_2 = array([2, 3, 8, 9])
    pyplot.scatter(samples_1[:, 0], 0.0 + samples_1[:, 1], fc="#88CCF8", ec="k", s=16, label="population 1")
    pyplot.scatter(samples_2[:, 0], 1.2 + samples_2[:, 1], fc="silver", s=16, label="population 2")
    pyplot.scatter(samples_1[select_1, 0], 1.2 + samples_1[select_1, 1], fc="k", ec="k", s=16, label="replaceable")
    pyplot.scatter(samples_1[select_2, 0], 1.2 + samples_1[select_2, 1], fc="w", ec="k", s=16, label="irreplaceable")
    pyplot.hlines(0.6, -0.6, 0.6, lw=1)
    pyplot.text(0.8, 0.58, "=", va="center", ha="center", fontsize=12)
    pyplot.hlines(0.6, 1.0, 1.3, lw=1)
    pyplot.text(1.15, 0.70, str(len(select_1)), va="center", ha="center", fontsize=12)
    pyplot.text(1.15, 0.45, "10", va="center", ha="center", fontsize=12)
    pyplot.legend(loc="upper right", fontsize=8)
    pyplot.xlim(-0.7, 1.4)
    pyplot.ylim(-0.8, 2.0)
    pyplot.xticks([])
    pyplot.yticks([])

    pyplot.subplot(1, 3, 3)
    pyplot.title("replacement rate of population 2 by 1", fontsize=9)
    select_1 = array([0, 3, 4, 11, 23])
    select_2 = array([1, 2, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24])
    pyplot.scatter(samples_1[:, 0], 1.2 + samples_1[:, 1], fc="silver", s=16, label="population 1")
    pyplot.scatter(samples_2[:, 0], 0.0 + samples_2[:, 1], fc="#FCB1AB", ec="k", s=16, label="population 2")
    pyplot.scatter(samples_2[select_1, 0], 1.2 + samples_2[select_1, 1], fc="k", ec="k", s=16, label="replaceable")
    pyplot.scatter(samples_2[select_2, 0], 1.2 + samples_2[select_2, 1], fc="w", ec="k", s=16, label="irreplaceable")
    pyplot.hlines(0.6, -0.6, 0.6, lw=1)
    pyplot.text(0.8, 0.58, "=", va="center", ha="center", fontsize=12)
    pyplot.hlines(0.6, 1.0, 1.3, lw=1)
    pyplot.text(1.15, 0.70, str(len(select_1)), va="center", ha="center", fontsize=12)
    pyplot.text(1.15, 0.45, "25", va="center", ha="center", fontsize=12)
    pyplot.legend(loc="upper right", fontsize=8)
    pyplot.xlim(-0.7, 1.4)
    pyplot.ylim(-0.8, 2.0)
    pyplot.xticks([])
    pyplot.yticks([])

    figure.text(0.012, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.342, 0.99, "b", va="center", ha="center", fontsize=14)
    figure.text(0.672, 0.99, "c", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_03():
    task_data = load_data(sort_path + "supp03.pkl")

    pyplot.figure(figsize=(10, 8.5), tight_layout=True)

    pyplot.hlines(8, 0, 72, lw=0.75)
    pyplot.text(36, 9, "replaced motif", va="center", ha="center", fontsize=10)
    pyplot.vlines(-8, 0, -72, lw=0.75)
    pyplot.text(-9, -36, "motif to be replaced", va="center", ha="center", fontsize=10, rotation=90)

    pyplot.text(80, 0, "motif style", va="center", ha="center", fontsize=10)
    pyplot.scatter([77], [-3], marker="o", fc="w", ec="k", lw=0.75, s=25)
    pyplot.text(82, -3, "input x", va="center", ha="center", fontsize=8)
    pyplot.scatter([77], [-6], marker="o", fc="silver", ec="k", lw=0.75, s=25)
    pyplot.text(82, -6, "input y", va="center", ha="center", fontsize=8)
    pyplot.scatter([77], [-9], marker="o", fc="k", ec="k", lw=0.75, s=25)
    pyplot.text(82, -9, "output z", va="center", ha="center", fontsize=8)
    pyplot.annotate("", xy=(78, -12), xytext=(76, -12),
                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
    pyplot.text(82, -12, "positive\nweight", va="center", ha="center", fontsize=8)
    pyplot.annotate("", xy=(78, -15), xytext=(76, -15),
                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle=":"))
    pyplot.text(82, -15, "negative\nweight", va="center", ha="center", fontsize=8)

    pyplot.text(80, -24, "propagation style", va="center", ha="center", fontsize=10)
    pyplot.scatter([77], [-27], marker="s", fc="w", ec="k", lw=0.75, s=25)
    pyplot.text(82, -27, "tanh", va="center", ha="center", fontsize=8)
    pyplot.scatter([77], [-30], marker="s", fc="gray", ec="k", lw=0.75, s=25)
    pyplot.text(82, -30, "sigmoid", va="center", ha="center", fontsize=8)
    pyplot.scatter([77], [-33], marker="s", fc="k", ec="k", lw=0.75, s=25)
    pyplot.text(82, -33, "relu", va="center", ha="center", fontsize=8)
    pyplot.scatter([75.5], [-36], marker="^", fc="w", ec="k", lw=0.75, s=25)
    pyplot.text(77, -36, "or", va="center", ha="center", fontsize=8)
    pyplot.scatter([78.5], [-36], marker="<", fc="w", ec="k", lw=0.75, s=25)
    pyplot.text(82, -36, "sum", va="center", ha="center", fontsize=8)
    pyplot.scatter([75.5], [-39], marker="^", fc="k", ec="k", lw=0.75, s=25)
    pyplot.text(77, -39, "or", va="center", ha="center", fontsize=8)
    pyplot.scatter([78.5], [-39], marker="<", fc="k", ec="k", lw=0.75, s=25)
    pyplot.text(82, -39, "max", va="center", ha="center", fontsize=8)

    pyplot.text(80, -48, "replacement rate", va="center", ha="center", fontsize=10)
    for index, color in enumerate(pyplot.get_cmap("RdYlBu")(linspace(0, 1, 50))):
        pyplot.fill_between([75.5, 77.0], -72 + index * (22 / 50), -72 + (index + 1) * (22 / 50), fc=color,
                            ec="k", lw=0, zorder=1)
    pyplot.plot([75.5, 75.5, 77.0, 77.0, 75.5], [-72, -50, -50, -72, -72], color="k", lw=0.75, zorder=2)
    for index, color in enumerate(pyplot.get_cmap("RdYlBu")(linspace(0, 1, 6))):
        pyplot.scatter([78.0], [-72 + index * (22 / 5)], marker="D", fc=color, ec="k", lw=0.75, s=25, zorder=3)
        pyplot.hlines(-72 + index * (22 / 5), 78.0, 80.0, lw=0.75, ls="--", zorder=2)
        pyplot.text(82, -72 + index * (22 / 5), str(index * 20) + "%", va="center", ha="center", fontsize=8)

    motif_types = ["incoherent-loop", "coherent-loop", "collider"]
    for motif_location in arange(3):
        for structure_location in arange(4):
            info = list(draw_info[motif_types[motif_location]])
            info[2][2] = 0.7
            info[1], info[2] = array(info[1]) * 5, array(info[2]) * 5
            motif = acyclic_motifs[motif_types[motif_location]][structure_location]
            bias_x, bias_y = motif_location * 24 + structure_location * 6 + 0.5, 3.0
            for index, (px, py) in enumerate(zip(info[1], info[2])):
                if index + 1 in info[3]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="w", edgecolor="k", lw=0.75, s=20, zorder=2)
                elif index + 1 in info[4]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="k", edgecolor="k", lw=0.75, s=20, zorder=2)
                elif index + 1 in info[5]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="k", lw=0.75, s=20, zorder=2)
                else:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="k", lw=0.75, s=20, zorder=2)
            x, y = info[1], info[2]
            for former, latter in motif.edges:
                if motif.get_edge_data(former, latter)["weight"] == 1:
                    pyplot.annotate("", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
                else:
                    pyplot.annotate("", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle=":"))

            info = list(draw_info[motif_types[motif_location]])
            info[2][2] = 0.75
            info[1], info[2] = array(info[1]) * 5, array(info[2]) * 5
            bias_x, bias_y = -8.0, -motif_location * 24 - structure_location * 6 - 5.5
            for index, (px, py) in enumerate(zip(info[1], info[2])):
                if index + 1 in info[3]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="w", edgecolor="k", lw=0.75, s=20, zorder=2)
                elif index + 1 in info[4]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="k", edgecolor="k", lw=0.75, s=20, zorder=2)
                elif index + 1 in info[5]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="k", lw=0.75, s=20, zorder=2)
                else:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="k", lw=0.75, s=20, zorder=2)
            x, y = info[1], info[2]
            for former, latter in motif.edges:
                if motif.get_edge_data(former, latter)["weight"] == 1:
                    pyplot.annotate("", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
                else:
                    pyplot.annotate("", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle=":"))

    for idx1 in range(12):
        for idx2 in range(12):
            pyplot.fill_between([idx1 * 6 + 0.5, (idx1 + 1) * 6 - 0.5], -idx2 * 6 - 0.5, -(idx2 + 1) * 6 + 0.5,
                                fc="#CCCCCC", ec="#CCCCCC", lw=0.75, zorder=1)

    for idx in range(72):
        pyplot.hlines(-idx - 0.5, -0.5, +71.5, color="#444444", lw=0.75, zorder=2)
        pyplot.vlines(+idx + 0.5, +0.5, -71.5, color="#444444", lw=0.75, zorder=2)
        if idx % 2 == 0:
            pyplot.plot([0.5, -1.0, -1.0, 0.5], [-idx - 0.5, -idx - 0.5, -idx - 1.5, -idx - 1.5],
                        color="k", lw=0.75, zorder=2)
            pyplot.hlines(-idx - 1.0, -2.0, -1.0, color="k", lw=0.75, zorder=2)
            if idx % 6 == 0:
                pyplot.scatter([-2.0], [-idx - 1.0], marker="s", fc="w", ec="k", lw=0.75, s=20, zorder=3)
                pyplot.plot([-2.0, -3.0, -3.0, -2.0], [-idx - 1.0, -idx - 1.0, -idx - 5.0, -idx - 5.0],
                            color="k", lw=0.75, zorder=2)
                pyplot.hlines(-idx - 3.0, -3.0, -2.0, color="k", lw=0.75, zorder=2)
            elif idx % 6 == 2:
                pyplot.scatter([-2.0], [-idx - 1.0], marker="s", fc="gray", ec="k", lw=0.75, s=20, zorder=3)
            elif idx % 6 == 4:
                pyplot.scatter([-2.0], [-idx - 1.0], marker="s", fc="k", ec="k", lw=0.75, s=20, zorder=3)
            pyplot.scatter([-1.0], [-idx - 0.5], marker="<", fc="w", ec="k", lw=0.75, s=25, zorder=3)
            pyplot.scatter([-1.0], [-idx - 1.5], marker="<", fc="k", ec="k", lw=0.75, s=25, zorder=3)
            pyplot.plot([+idx + 0.5, +idx + 0.5, +idx + 1.5, +idx + 1.5], [-0.5, 1.0, 1.0, -0.5],
                        color="k", lw=0.75, zorder=2)
            pyplot.vlines(+idx + 1.0, 1.0, 2.0, color="k", lw=0.75, zorder=2)
            if idx % 6 == 0:
                pyplot.scatter([+idx + 1.0], [2.0], marker="s", fc="w", ec="k", lw=0.75, s=20, zorder=3)
                pyplot.plot([+idx + 1.0, +idx + 1.0, +idx + 5.0, +idx + 5.0], [2.0, 3.0, 3.0, 2.0],
                            color="k", lw=0.75, zorder=2)
                pyplot.vlines(+idx + 3.0, 2.0, 3.0, color="k", lw=0.75, zorder=2)
            elif idx % 6 == 2:
                pyplot.scatter([+idx + 1.0], [2.0], marker="s", fc="gray", ec="k", lw=0.75, s=20, zorder=3)
            elif idx % 6 == 4:
                pyplot.scatter([+idx + 1.0], [2.0], marker="s", fc="k", ec="k", lw=0.75, s=20, zorder=3)
            pyplot.scatter([+idx + 0.5], [1.0], marker="^", fc="w", ec="k", lw=0.75, s=25, zorder=3)
            pyplot.scatter([+idx + 1.5], [1.0], marker="^", fc="k", ec="k", lw=0.75, s=25, zorder=3)

    for idx1 in range(task_data["a"].shape[0]):
        for idx2 in range(task_data["a"].shape[1]):
            if idx1 in list(range(0, 24)) and idx2 in list(range(0, 24)):
                continue
            if idx1 in list(range(24, 48)) and idx2 in list(range(24, 48)):
                continue
            if idx1 in list(range(48, 72)) and idx2 in list(range(48, 72)):
                continue

            if task_data["a"][idx1, idx2] >= 0:
                color = pyplot.get_cmap("RdYlBu")(task_data["a"][idx1, idx2])
                pyplot.scatter([idx1 + 0.5], [-idx2 - 0.5], marker="D", fc=color, ec="k", lw=0.75, s=40, zorder=3)

    pyplot.xlim(-10, 71 + 16)
    pyplot.ylim(-73, 0 + 10)
    pyplot.axis("off")
    pyplot.savefig(save_path + "supp03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_04():
    task_data = load_data(sort_path + "supp04.pkl")

    pyplot.figure(figsize=(10, 3), tight_layout=True)

    motif_types = ["incoherent-loop", "coherent-loop", "collider"]
    activations, aggregations = ["tanh", "sigmoid", "relu"], ["sum", "max"]

    for motif_location in arange(3):
        pyplot.text(motif_location * 8 + 5.0, 5.55, "propagation style", va="center", ha="center", fontsize=10)
        pyplot.text(motif_location * 8 + 0.6, 2, motif_types[motif_location] + " structure",
                    va="center", ha="center", fontsize=10, rotation=90)
        for structure_location in arange(4):
            info = draw_info[motif_types[motif_location]]
            info[2][2] = 0.8
            motif = acyclic_motifs[motif_types[motif_location]][structure_location]
            bias_x, bias_y = motif_location * 8 + 0.875, 4 - structure_location - 1.0
            for index, (px, py) in enumerate(zip(info[1], info[2])):
                if index + 1 in info[3]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="w", edgecolor="k", lw=0.75, s=15, zorder=2)
                elif index + 1 in info[4]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="k", edgecolor="k", lw=0.75, s=15, zorder=2)
                elif index + 1 in info[5]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="k", lw=0.75, s=15, zorder=2)
                else:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="k", lw=0.75, s=15, zorder=2)
            x, y = info[1], info[2]
            for former, latter in motif.edges:
                if motif.get_edge_data(former, latter)["weight"] == 1:
                    pyplot.annotate("", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
                else:
                    pyplot.annotate("", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle=":"))
        for function_location in arange(6):
            x = motif_location * 8 + function_location + 2.0 + 0.5
            if function_location % 2 == 0:
                pyplot.plot([x, x, x + 0.3, x + 0.3], [4.6, 4.7, 4.7, 4.8], color="k", lw=0.75)
                pyplot.text(x, 4.4, aggregations[function_location % 2], va="center", ha="center", fontsize=8)
                pyplot.text(x + 0.5, 5.0, activations[(function_location // 2) % 3], va="center", ha="center",
                            fontsize=8)
            else:
                pyplot.plot([x, x, x - 0.3, x - 0.3], [4.6, 4.7, 4.7, 4.8], color="k", lw=0.75)
                pyplot.text(x, 4.4, aggregations[function_location % 2], va="center", ha="center", fontsize=8)

            for structure_location in arange(4):
                y = 4 - structure_location - 0.5
                value = task_data["a"][motif_location * 6 + function_location, structure_location]
                pyplot.fill_between([x - 0.4, x + 0.4], y - 0.4, y + 0.4,
                                    fc=pyplot.get_cmap("RdYlGn_r")([value / 5.0]), ec="k", lw=0.75)
                pyplot.text(x, y - 0.05, "%.2f" % value, va="center", ha="center", fontsize=6)
    pyplot.text(13, -0.35, "median Lipschitz constant for all collected samples",
                va="center", ha="center", fontsize=10)
    pyplot.plot([2.1, 23.9, 23.9, 2.1, 2.1], [-0.6, -0.6, -0.8, -0.8, -0.6], color="k", lw=0.75)
    colors = pyplot.get_cmap("RdYlGn_r")(linspace(0, 0.8, 100))
    interval = (23.9 - 2.1) / 100.0
    for index in range(100):
        pyplot.fill_between([2.1 + index * interval, 2.1 + (index + 1) * interval], -0.8, -0.6, color=colors[index],
                            lw=0)
    interval = (23.9 - 2.1) / 4.0
    for index in range(5):
        pyplot.vlines(2.1 + index * interval, -0.9, -0.8, lw=0.75)
        pyplot.text(2.1 + index * interval, -1.1, "%.1f" % index, va="center", ha="center", fontsize=8)
    pyplot.axis("off")
    pyplot.xlim(0, 24)
    pyplot.ylim(-1.2, 6)

    pyplot.savefig(save_path + "supp04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_05():
    task_data = load_data(sort_path + "supp05.pkl")["a"]
    math_orders = [r"$\mathcal{L}_i$ in $\mathcal{L}_i \cap \mathcal{L}_c \cap \mathcal{C}$",
                   r"$\mathcal{L}_c$ in $\mathcal{L}_i \cap \mathcal{L}_c \cap \mathcal{C}$"]
    used_colors = [draw_info["incoherent-loop"][0], draw_info["coherent-loop"][0]]
    # fix the comparison starting point problem of the set.
    if len(task_data[(3, 1)]) == 0 or len(task_data[(3, 2)]) == 0:
        task_data[6, 1] = array(task_data[6, 1].tolist() + task_data[3, 1].tolist())
        task_data[3, 1] = array([])
        task_data[6, 2] = array(task_data[6, 2].tolist() + task_data[3, 2].tolist())
        task_data[3, 2] = array([])
    if len(task_data[(4, 0)]) == 0 or len(task_data[(4, 2)]) == 0:
        task_data[6, 0] = array(task_data[6, 1].tolist() + task_data[4, 0].tolist())
        task_data[4, 0] = array([])
        task_data[6, 2] = array(task_data[6, 2].tolist() + task_data[4, 2].tolist())
        task_data[4, 2] = array([])
    if len(task_data[(5, 0)]) == 0 or len(task_data[(5, 1)]) == 0:
        task_data[6, 0] = array(task_data[6, 0].tolist() + task_data[5, 0].tolist())
        task_data[5, 0] = array([])
        task_data[6, 1] = array(task_data[6, 1].tolist() + task_data[5, 1].tolist())
        task_data[5, 1] = array([])

    figure = pyplot.figure(figsize=(10, 6), tight_layout=True)
    ax = pyplot.subplot(2, 1, 1)
    pyplot.title(math_orders[0], fontsize=12)
    values = task_data[(6, 0)]
    constants = linspace(min(values), max(values), 100)
    radios = gaussian_kde(values)(constants)
    radios /= sum(radios)
    pyplot.fill_between(constants, 0, radios, fc=used_colors[0], ec="k", lw=0.75)
    for value in linspace(0.25, 3.75, 15):
        pyplot.vlines(value, 0, 1, color="silver", lw=0.75, ls="--")
    for value in linspace(0.004, 0.036, 9):
        pyplot.hlines(value, 0, 4, color="silver", lw=0.75, ls="--")
    pyplot.xlabel("Lipschitz constant", fontsize=9)
    pyplot.ylabel("proportion", fontsize=9)
    pyplot.xticks(linspace(0.00, 4.00, 21), ["%.2f" % v for v in linspace(0.00, 4.00, 21)], fontsize=8)
    pyplot.yticks(linspace(0.00, 0.04, 11), ["%.1f" % (v * 100) + "%" for v in linspace(0, 0.04, 11)], fontsize=8)
    pyplot.xlim(0.00, 4.00)
    pyplot.ylim(0.00, 0.04)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(2, 1, 2)
    pyplot.title(math_orders[1], fontsize=12)
    values = task_data[(6, 1)]
    constants = linspace(min(values), max(values), 100)
    radios = gaussian_kde(values)(constants)
    radios /= sum(radios)
    pyplot.fill_between(constants, 0, radios, fc=used_colors[1], ec="k", lw=0.75)
    for value in linspace(0.25, 3.75, 15):
        pyplot.vlines(value, 0, 1, color="silver", lw=0.75, ls="--")
    for value in linspace(0.004, 0.036, 9):
        pyplot.hlines(value, 0, 4, color="silver", lw=0.75, ls="--")
    pyplot.xlabel("Lipschitz constant", fontsize=9)
    pyplot.ylabel("proportion", fontsize=9)
    pyplot.xticks(linspace(0.00, 4.00, 21), ["%.2f" % v for v in linspace(0.00, 4.00, 21)], fontsize=8)
    pyplot.yticks(linspace(0.00, 0.04, 11), ["%.1f" % (v * 100) + "%" for v in linspace(0.00, 0.04, 11)], fontsize=8)
    pyplot.xlim(0.00, 4.00)
    pyplot.ylim(0.00, 0.04)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.020, 0.50, "b", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp05.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_06():
    task_data = load_data(sort_path + "supp06.pkl")

    pyplot.figure(figsize=(10, 6), tight_layout=True)
    activation_selection, aggregation_selection = ["tanh", "sigmoid", "relu"], ["sum", "max"]
    colors = [draw_info["incoherent-loop"][0], draw_info["coherent-loop"][0], draw_info["collider"][0]]
    index = 0
    for aggregation in aggregation_selection:
        for activation in activation_selection:
            record = task_data[chr(ord("a") + index)]
            # fix the comparison starting point problem of the set.
            if len(record[(3, 1)]) == 0 or len(record[(3, 2)]) == 0:
                record[6, 1] = array(record[6, 1].tolist() + record[3, 1].tolist())
                record[3, 1] = array([])
                record[6, 2] = array(record[6, 2].tolist() + record[3, 2].tolist())
                record[3, 2] = array([])
            if len(record[(4, 0)]) == 0 or len(record[(4, 2)]) == 0:
                record[6, 0] = array(record[6, 1].tolist() + record[4, 0].tolist())
                record[4, 0] = array([])
                record[6, 2] = array(record[6, 2].tolist() + record[4, 2].tolist())
                record[4, 2] = array([])
            if len(record[(5, 0)]) == 0 or len(record[(5, 1)]) == 0:
                record[6, 0] = array(record[6, 0].tolist() + record[5, 0].tolist())
                record[5, 0] = array([])
                record[6, 1] = array(record[6, 1].tolist() + record[5, 1].tolist())
                record[5, 1] = array([])

            pyplot.subplot(2, 3, index + 1)
            pyplot.text(-1.8, 7.2, chr(ord("a") + index), va="center", ha="center", fontsize=14)
            pyplot.text(1, 7.2, activation + " + " + aggregation, va="center", ha="center", fontsize=9)
            pyplot.plot([-1, -1, 3], [7, 0, 0], lw=0.75, color="k")
            for idx, info in enumerate(["0", "1", "2", "3", "4"]):
                pyplot.vlines(idx - 1, 0, -0.1, lw=0.75, color="k")
                pyplot.text(idx - 1, -0.4, info, va="center", ha="center", fontsize=8)
            pyplot.text(1.0, -0.8, "Lipschitz constant", va="center", ha="center", fontsize=9)
            location, x = 0, [-1.42, -1.27, -1.12]
            pyplot.text(-1.8, 3.5, "intersection", va="center", ha="center", fontsize=9, rotation=90)
            for flag_1, flag_2, flag_3 in [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]:
                left, right, y = [], [], 6.5 - location
                if [flag_1, flag_2, flag_3] != [1, 1, 1]:
                    pyplot.hlines(y - 0.5, -1, 3, lw=0.75)
                    pyplot.vlines(0, y - 0.50, y - 0.60, lw=0.75, color="k")
                    pyplot.vlines(1, y - 0.50, y - 0.60, lw=0.75, color="k")
                    pyplot.vlines(2, y - 0.50, y - 0.60, lw=0.75, color="k")
                    pyplot.vlines(3, y - 0.50, y - 0.60, lw=0.75, color="k")
                if flag_1:
                    pyplot.scatter([x[0]], [y], s=15, fc=colors[0], ec="k", lw=0.75, zorder=3)
                else:
                    pyplot.scatter([x[0]], [y], s=15, marker="x", ec="k", lw=0.75, zorder=3)
                if flag_2:
                    pyplot.scatter([x[1]], [y], s=15, fc=colors[1], ec="k", lw=0.75, zorder=3)
                else:
                    pyplot.scatter([x[1]], [y], s=15, marker="x", ec="k", lw=0.75, zorder=3)
                if flag_3:
                    pyplot.scatter([x[2]], [y], s=15, fc=colors[2], ec="k", lw=0.75, zorder=3)
                else:
                    pyplot.scatter([x[2]], [y], s=15, marker="x", ec="k", lw=0.75, zorder=3)
                pyplot.hlines(y, x[0], x[2], lw=0.75, zorder=2)
                pyplot.hlines(y, -1.05, -1.00, lw=0.75, zorder=2)
                location += 1

            for location in range(7):
                flag_1, flag_2 = False, False
                if (location, 2) in record and len(record[(location, 2)]) > 0:
                    flag_1 = True
                    values = record[(location, 2)]
                    constants = linspace(min(values), max(values), 100)
                    radios = gaussian_kde(values)(constants)
                    radios = radios / max(radios) * 0.65
                    pyplot.plot(constants - 1, 6 - location + radios, color="k", lw=0.75)
                    pyplot.fill_between(constants - 1, 6 - location, 6 - location + radios, color=colors[2], lw=0)
                if (location, 1) in record and len(record[(location, 1)]) > 0:
                    flag_2 = True
                    values = record[(location, 1)]
                    constants = linspace(min(values), max(values), 100)
                    radios = gaussian_kde(values)(constants)
                    radios = radios / max(radios) * 0.65
                    pyplot.plot(constants - 1, 6 - location + radios, color="k", lw=0.75)
                    if flag_1:
                        pyplot.fill_between(constants - 1, 6 - location, 6 - location + radios,
                                            color="#FF8B5C", lw=0, alpha=0.5)
                    else:
                        pyplot.fill_between(constants - 1, 6 - location, 6 - location + radios, color=colors[1], lw=0)
                if (location, 0) in record and len(record[(location, 0)]) > 0:
                    values = record[(location, 0)]
                    constants = linspace(min(values), max(values), 100)
                    radios = gaussian_kde(values)(constants)
                    radios = radios / max(radios) * 0.65
                    pyplot.plot(constants - 1, 6 - location + radios, color="k", lw=0.75)
                    if flag_1 or flag_2:
                        pyplot.fill_between(constants - 1, 6 - location, 6 - location + radios,
                                            color="#EE6462", lw=0, alpha=0.5)
                    else:
                        pyplot.fill_between(constants - 1, 6 - location, 6 - location + radios, color=colors[0], lw=0)

            pyplot.xlim(-2.0, 3.01)
            pyplot.ylim(-1.0, 7.0)
            pyplot.axis("off")

            index += 1

    pyplot.savefig(save_path + "supp06.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_07():
    task_data = load_data(sort_path + "supp07.pkl")
    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)

    for motif_type in range(2):
        for motif_index in range(4):
            ax = pyplot.subplot(2, 4, motif_type * 4 + motif_index + 1)
            robust_change, loss_change = task_data[chr(ord("a") + motif_type * 4 + motif_index)]
            robust_change, loss_change = array(robust_change), array(loss_change)
            pyplot.title("train " + str(math_orders[motif_type]) + " (" + str(motif_index + 1) + ") "
                                                                                                 "to escape from " +
                         math_orders[-1], fontsize=9)
            location = where(robust_change > 0)[0]
            pyplot.scatter(loss_change[location], robust_change[location], s=12, color="tomato",
                           label=str(len(location)) + " / 100", alpha=0.75)
            location = where(robust_change <= 0)[0]
            if len(location) >= 10:
                pyplot.scatter(loss_change[location], robust_change[location], s=12, color="royalblue",
                               label=str(len(location)) + " / 100", alpha=0.75)
            else:
                pyplot.scatter(loss_change[location], robust_change[location], s=12, color="royalblue",
                               label="  " + str(len(location)) + " / 100", alpha=0.75)
            # noinspection PyTypeChecker
            corr, p = spearmanr(loss_change, robust_change)
            pyplot.text(0.048, -0.80, "spearman", va="center", ha="left", fontsize=8)
            if corr < 0:
                pyplot.text(0.086, -0.80, ("= %.2f" % corr).replace("-", "\N{MINUS SIGN}"),
                            va="center", ha="left", fontsize=8)
            else:
                pyplot.text(0.086, -0.80, "= +%.2f" % corr, va="center", ha="left", fontsize=8)
            pyplot.text(0.048, -1.00, "p-value", va="center", ha="left", fontsize=8)
            pyplot.text(0.086, -1.00, ("= %.2e" % p).upper().replace("-", "\N{MINUS SIGN}"),
                        va="center", ha="left", fontsize=8)
            pyplot.legend(loc="upper left", fontsize=8)
            pyplot.xlabel("L1 loss growth after training", fontsize=9)
            pyplot.ylabel("variation of Lipschitz constant", fontsize=9)
            pyplot.xticks(linspace(0, 0.12, 7), ["%.2f" % v for v in linspace(0, 0.12, 7)], fontsize=8)
            pyplot.yticks(linspace(-1.0, 1.0, 11),
                          ["\N{MINUS SIGN}1.0", "\N{MINUS SIGN}0.8", "\N{MINUS SIGN}0.6", "\N{MINUS SIGN}0.4",
                           "\N{MINUS SIGN}0.2", "0.0", "+0.2", "+0.4", "+0.6", "+0.8", "+1.0"], fontsize=8)
            pyplot.xlim(-0.005, 0.125)
            pyplot.ylim(-1.105, 1.105)
            # noinspection PyUnresolvedReferences
            ax.spines["top"].set_visible(False)
            # noinspection PyUnresolvedReferences
            ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.266, 0.99, "b", va="center", ha="center", fontsize=14)
    figure.text(0.513, 0.99, "c", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.99, "d", va="center", ha="center", fontsize=14)
    figure.text(0.020, 0.50, "e", va="center", ha="center", fontsize=14)
    figure.text(0.266, 0.50, "f", va="center", ha="center", fontsize=14)
    figure.text(0.513, 0.50, "g", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.50, "i", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp07.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_08():
    task_data = load_data(sort_path + "supp08.pkl")
    param_names = ["weight\n" + r"$x \rightarrow y$",
                   "weight\n" + r"$x \rightarrow z$",
                   "weight\n" + r"$y \rightarrow z$",
                   "bias\n" + r"$x \rightarrow y$",
                   "bias\n" + r"$x|y \rightarrow z$"]
    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    for motif_type in range(2):
        for motif_index in range(4):
            pyplot.subplot(2, 4, motif_type * 4 + motif_index + 1)
            record = array(task_data[chr(ord("a") + motif_type * 4 + motif_index)]).T
            pyplot.title("train " + str(math_orders[motif_type]) + " (" + str(motif_index + 1) + ") "
                                                                                                 "to escape from " +
                         math_orders[-1], fontsize=9)
            for index in arange(4) + 0.5:
                pyplot.vlines(index, -1.1, 1.1, lw=0.75)
            for index, params in enumerate(record):
                violin = pyplot.violinplot(dataset=[params], positions=[index], showextrema=False, showmeans=False)
                for patch in violin["bodies"]:
                    patch.set_edgecolor("k")
                    if index < 3:
                        patch.set_facecolor("#A8E7AD")
                    else:
                        patch.set_facecolor("#CCC9FA")
                    patch.set_linewidth(0.75)
                    patch.set_alpha(1)
            pyplot.xlabel("motif parameters after training", fontsize=9)
            pyplot.ylabel("value", fontsize=9)
            pyplot.xticks([0, 1, 2, 3, 4], param_names, fontsize=8)
            pyplot.yticks(linspace(-1, 1, 5), ["\N{MINUS SIGN}1.0", "\N{MINUS SIGN}0.5", "0.0", "+0.5", "+1.0"],
                          fontsize=8)
            pyplot.xlim(-0.5, 4.5)
            pyplot.ylim(-1.1, 1.1)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.266, 0.99, "b", va="center", ha="center", fontsize=14)
    figure.text(0.513, 0.99, "c", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.99, "d", va="center", ha="center", fontsize=14)
    figure.text(0.020, 0.50, "e", va="center", ha="center", fontsize=14)
    figure.text(0.266, 0.50, "f", va="center", ha="center", fontsize=14)
    figure.text(0.513, 0.50, "g", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.50, "i", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp08.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_09():
    task_data = load_data(sort_path + "supp09.pkl")
    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]
    selections = {(0, 0): array([69, 25, 48, 34]), (1, 0): array([84, 90, 25, 17])}
    figure = pyplot.figure(figsize=(10, 8), tight_layout=True)

    for motif_type in range(2):
        for motif_index in range(4):
            pyplot.subplot(2, 4, motif_type * 4 + motif_index + 1)
            differences, locations = task_data[chr(ord("a") + motif_type * 4 + motif_index)]
            pyplot.title("train " + str(math_orders[motif_type]) + " (" + str(motif_index + 1) + ") "
                                                                                                 "to escape from " +
                         math_orders[-1], fontsize=9)
            differences = differences.reshape(-1)
            differences = differences[differences > 0]
            proportions = gaussian_kde(differences)(linspace(0, 1.5, 200))
            proportions = proportions / max(proportions) * 0.7
            x, y, c = linspace(0.15, 0.85, 200), 1.15 + proportions, linspace(0, 1, 200)
            pyplot.scatter(x, y, c=c, cmap="RdYlBu_r", alpha=0.75)
            pyplot.plot([0.10, 0.10, 0.90], [1.90, 1.10, 1.10], color="k", lw=0.75)
            pyplot.plot([1.00, 1.05, 1.05, 1.00, 1.00], [1.10, 1.10, 1.90, 1.90, 1.10], lw=0.75, color="k", zorder=2)
            for idx, color in enumerate(pyplot.get_cmap("RdYlBu_r")(linspace(0, 1, 50))):
                pyplot.fill_between([1.00, 1.05], 1.10 + idx * 0.016, 1.10 + (idx + 1) * 0.016, fc=color, lw=0)
            for location, value in zip(linspace(1.1, 1.9, 4), linspace(0, 1.5, 4)):
                pyplot.hlines(location, 1.05, 1.08, lw=0.75)
                pyplot.text(1.16, location, "%.1f" % value, va="center", ha="center", fontsize=8)
            pyplot.text(1.025, 1.95, "L1 loss", va="center", ha="center", fontsize=8)
            pyplot.annotate("", xy=(0.5, 0.9), xytext=(0.5, 1.0),
                            arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75, shrinkA=0, shrinkB=0))
            pyplot.text(0.50, 1.05, "L1 loss between samples", va="center", ha="center", fontsize=8)
            pyplot.text(0.05, 1.50, "proportion", va="center", ha="center", fontsize=8, rotation=90)
            locations[:, 0] -= min(locations[:, 0])
            locations[:, 1] -= min(locations[:, 1])
            locations = locations / max(locations) * 0.7 + 0.15
            pyplot.scatter(locations[:, 0], locations[:, 1], color="k", alpha=0.1)
            pyplot.text(1.025, 0.95, "aggregation level", va="center", ha="center", fontsize=8)
            if (motif_type, motif_index) in selections:
                indices = selections[(motif_type, motif_index)]
                pyplot.scatter(locations[indices, 0], locations[indices, 1], color="tomato", zorder=2)
                pyplot.text(0.88, 0.5, "(   ) selection", va="center", ha="right", fontsize=8)
                pyplot.scatter([0.569], [0.500], color="tomato", zorder=2)

            for idx, color in enumerate(pyplot.get_cmap("binary")(linspace(0, 0.8, 50))):
                pyplot.fill_between([1.00, 1.00 + idx * 0.001, 1.00 + (idx + 1) * 0.001],
                                    [0.10 + idx * 0.016, 0.10 + idx * 0.016, 0.10 + (idx + 1) * 0.016],
                                    [0.10 + (idx + 1) * 0.016, 0.10 + (idx + 1) * 0.016, 0.10 + (idx + 1) * 0.016],
                                    fc=color, lw=0)
            pyplot.plot([1.00, 1.05, 1.00, 1.00], [0.10, 0.90, 0.90, 0.10], lw=0.75, color="k", zorder=2)
            pyplot.plot([0.10, 0.10, 0.90], [0.90, 0.10, 0.10], color="k", lw=0.75)
            pyplot.text(0.50, 0.05, "UMAP dimension 1", va="center", ha="center", fontsize=8)
            pyplot.text(0.05, 0.50, "UMAP dimension 2", va="center", ha="center", fontsize=8, rotation=90)

            pyplot.xlim(0.0, 1.2)
            pyplot.ylim(0.0, 2.0)
            pyplot.axis("off")

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.266, 0.99, "b", va="center", ha="center", fontsize=14)
    figure.text(0.513, 0.99, "c", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.99, "d", va="center", ha="center", fontsize=14)
    figure.text(0.020, 0.50, "e", va="center", ha="center", fontsize=14)
    figure.text(0.266, 0.50, "f", va="center", ha="center", fontsize=14)
    figure.text(0.513, 0.50, "g", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.50, "i", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp09.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_10():
    task_data = load_data(sort_path + "supp10.pkl")
    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$"]

    figure = pyplot.figure(figsize=(10, 6), tight_layout=True)

    for index_1, motif_type in enumerate(math_orders):
        for index_2, change_flag in enumerate(["increase", "decrease"]):
            for case_index in range(2):
                record = task_data[chr(ord("a") + index_1 * 4 + index_2 * 2 + case_index)]
                pyplot.subplot(2, 4, index_1 * 4 + index_2 * 2 + case_index + 1)
                pyplot.title(change_flag + " case " + str(case_index + 1) + " of " + motif_type + " escape", fontsize=9)
                x1, x2, x3 = linspace(0.2, 0.8, 41), linspace(1.2, 1.8, 41), linspace(2.2, 2.8, 41)
                y = linspace(2.2, 2.8, 41)
                if index_1 == 0:
                    pyplot.plot([0.1, 0.9], [2.9, 2.1], color="k", lw=0.75, ls="--", zorder=2)
                    pyplot.plot([2.1, 2.9], [2.9, 2.1], color="k", lw=0.75, ls="--", zorder=2)
                    pyplot.scatter([0.1, 2.1], [2.9, 2.9], fc="w", ec="k", s=12, zorder=3)
                    pyplot.scatter([0.9, 2.9], [2.1, 2.1], fc="k", ec="k", s=12, zorder=3)
                pyplot.pcolormesh(x1, y, record[0], cmap="PRGn", vmin=-1, vmax=1)
                pyplot.pcolormesh(x2, y, record[1] - record[0], cmap="PRGn", vmin=-1, vmax=1)
                pyplot.pcolormesh(x3, y, record[1], cmap="PRGn", vmin=-1, vmax=1)
                pyplot.text(0.95, 2.5, "+", va="center", ha="center", fontsize=8)
                pyplot.text(1.95, 2.5, "=", va="center", ha="center", fontsize=8)
                for value, info in zip([0.50, 1.50, 2.50],
                                       ["former\nlandscape", "change\nfor escape", "latter\nlandscape"]):
                    pyplot.text(value, 2.95, info, va="center", ha="center", fontsize=7)
                for value in [0.50, 1.50, 2.50]:
                    pyplot.text(value, 2.1, "$x$", va="center", ha="center", fontsize=9)
                for value in [0.1, 1.1, 2.1]:
                    pyplot.text(value, 2.5, "$y$", va="center", ha="center", fontsize=9)
                for index, color in enumerate(pyplot.get_cmap("PRGn")(linspace(0, 1, 50))):
                    pyplot.fill_between([0.20 + index * 0.052, 0.20 + (index + 1) * 0.052], 3.3, 3.4, fc=color, lw=0)
                pyplot.plot([0.2, 2.8, 2.8, 0.2, 0.2], [3.3, 3.3, 3.4, 3.4, 3.3], color="k", lw=0.75, zorder=2)
                for location, value in zip(linspace(0.2, 2.8, 5), linspace(-1, 1, 5)):
                    pyplot.vlines(location, 3.25, 3.30, lw=0.75)
                    if value == 0:
                        pyplot.text(location, 3.15, "0.0", va="center", ha="center", fontsize=7)
                    elif value > 0:
                        pyplot.text(location, 3.15, "+%.1f" % value, va="center", ha="center", fontsize=7)
                    else:
                        pyplot.text(location, 3.15, "\N{MINUS SIGN}%.1f" % -value, va="center", ha="center", fontsize=7)
                pyplot.text(1.5, 3.48, "$z$", va="center", ha="center", fontsize=9)

                if index_1 == 0:
                    if case_index == 0:
                        pyplot.text(1.50, 1.05, "\"slope\" to \"valley\"",
                                    color="red", va="center", ha="center", fontsize=8)
                    else:
                        pyplot.text(1.50, 1.05, "\"slope\" to \"ridge\"",
                                    color="red", va="center", ha="center", fontsize=8)
                    pyplot.hlines(0.3, 0.1, 0.9, lw=0.75)
                    pyplot.hlines(0.3, 2.1, 2.9, lw=0.75)
                    pyplot.scatter([0.1, 2.1], [0.3, 0.3], fc="w", ec="k", s=12, zorder=3)
                    pyplot.scatter([0.9, 2.9], [0.3, 0.3], fc="k", ec="k", s=12, zorder=3)
                    pyplot.vlines(0.9, 0.3, 2.1, lw=0.75, ls="--")
                    pyplot.vlines(2.9, 0.3, 2.1, lw=0.75, ls="--")
                    pyplot.vlines(0.1, 0.3, 2.3, lw=0.75, ls="--")
                    pyplot.vlines(0.1, 2.7, 2.9, lw=0.75, ls="--")
                    pyplot.vlines(2.1, 0.3, 2.3, lw=0.75, ls="--")
                    pyplot.vlines(2.1, 2.7, 2.9, lw=0.75, ls="--")
                    pyplot.annotate("", xy=(2.05, 0.90), xytext=(0.95, 0.90),
                                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75, shrinkA=0, shrinkB=0))
                    x, y = linspace(-1, +1, 41), linspace(-1, +1, 41)
                    values_1, values_2 = [], []
                    for x_i, x_value in enumerate(x):
                        for y_i, y_value in enumerate(y):
                            if x_value == -y_value:
                                values_1.append(-(x_value - y_value) / sqrt(2))
                                values_2.append(record[0][x_i, y_i])
                    values_1, values_2 = array(values_1), array(values_2)
                    values_1 -= min(values_1)
                    values_1 = values_1 / max(values_1) * 0.6 + 0.2
                    values_2 -= min(values_2)
                    values_2 = values_2 / max(values_2) * 1.0 + 0.4
                    pyplot.fill_between(values_1, 0.3, values_2, fc="silver", ec="k", lw=0.75)
                    x, y = linspace(-1, +1, 41), linspace(-1, +1, 41)
                    values_1, values_2 = [], []
                    for x_i, x_value in enumerate(x):
                        for y_i, y_value in enumerate(y):
                            if x_value == -y_value:
                                values_1.append(-(x_value - y_value) / sqrt(2))
                                values_2.append(record[1][x_i, y_i])
                    values_1, values_2 = array(values_1), array(values_2)
                    values_1 -= min(values_1)
                    values_1 = values_1 / max(values_1) * 0.6 + 2.2
                    values_2 -= min(values_2)
                    values_2 = values_2 / max(values_2) * 1.0 + 0.4
                    pyplot.fill_between(values_1, 0.3, values_2, fc="silver", ec="k", lw=0.75)
                    pyplot.hlines(0.2, 0.2, 2.8, lw=0.75)
                    pyplot.text(1.5, 0.1, "tangent plane ($x+y=1$)", va="center", ha="center", fontsize=8)
                else:
                    pyplot.annotate("", xy=(1.5, 1.6), xytext=(1.5, 2.0),
                                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75, shrinkA=0, shrinkB=0))
                    pyplot.text(1.60, 1.85, "adjust high\ngradient region",
                                color="red", va="center", ha="left", fontsize=8)
                    gradient_values = record[2].reshape(-1)
                    percentiles = percentile(gradient_values, [25, 75])
                    iqr = percentiles[1] - percentiles[0]
                    limits = [percentiles[0] - iqr * 1.5, percentiles[1] + iqr * 1.5]
                    change_matrix = abs(record[1] - record[0])
                    gradients, changes = [], []
                    for x_i in range(41):
                        for y_i in range(41):
                            if limits[0] <= record[2][x_i, y_i] <= limits[1]:
                                gradients.append(record[2][x_i, y_i])
                                changes.append(change_matrix[x_i, y_i])
                    gradients, changes = array(gradients), array(changes)
                    # noinspection PyTypeChecker
                    corr, p = spearmanr(gradients, changes)
                    pyplot.text(0.78, 1.5, "spearman", va="center", ha="left", fontsize=8)
                    pyplot.text(0.78, 1.3, "p-value", va="center", ha="left", fontsize=8)
                    pyplot.text(1.46, 1.5, "= %.2f" % corr, va="center", ha="left", fontsize=8)
                    pyplot.text(1.46, 1.3, ("= %.2e" % p).upper().replace("-", "\N{MINUS SIGN}"),
                                va="center", ha="left", fontsize=8)
                    gradients = gradients / max(gradients) * 2.2 + 0.4
                    changes = changes / max(changes) * 0.6 + 0.4
                    pyplot.scatter(gradients, changes, color="k", alpha=0.1)
                    pyplot.plot([0.2, 0.2, 2.8], [1.2, 0.2, 0.2], color="k", lw=0.75)
                    pyplot.text(1.5, 0.1, "gradient in former landscape",
                                va="center", ha="center", fontsize=8)
                    pyplot.text(0.1, 0.7, "change", va="center", ha="center", fontsize=8, rotation=90)

                pyplot.xlim(0, 3.0)
                pyplot.ylim(0, 3.5)
                pyplot.axis("off")

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.266, 0.99, "b", va="center", ha="center", fontsize=14)
    figure.text(0.513, 0.99, "c", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.99, "d", va="center", ha="center", fontsize=14)
    figure.text(0.020, 0.50, "e", va="center", ha="center", fontsize=14)
    figure.text(0.266, 0.50, "f", va="center", ha="center", fontsize=14)
    figure.text(0.513, 0.50, "g", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.50, "i", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp10.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_11():
    task_data = load_data(sort_path + "supp11.pkl")
    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]
    labels = ["default", r"$\mathcal{L}_c + \mathcal{C}$", r"$\mathcal{L}_i + \mathcal{C}$", r"$\mathcal{C}$"]
    colors = [draw_info["incoherent-loop"][0], draw_info["coherent-loop"][0], draw_info["collider"][0]]
    bias = [-0.2, 0, 0.2]

    figure = pyplot.figure(figsize=(10, 10), tight_layout=True)
    for noise_index in range(5):
        ax = pyplot.subplot(6, 1, noise_index + 1)
        pyplot.title("maximum generation = 20 and training error scale = %d" % (noise_index * 10) + "%", fontsize=9)
        for index, values in enumerate(array(task_data["a"][noise_index]).T):
            pyplot.bar(arange(len(values)) + bias[index], values, width=0.2, lw=0.75,
                       fc=colors[index], ec="k", label=math_orders[index])
            for location, value in zip(arange(len(values)) + bias[index], values):
                if value > 0:
                    pyplot.text(location, value + 1.0, "%.1f" % value, va="center", ha="center", fontsize=8)
                else:
                    pyplot.text(location, value + 1.0, "N.A", va="center", ha="center", fontsize=8)
        pyplot.legend(loc="upper right", fontsize=8, ncol=3, title="motif type", title_fontsize=8)
        pyplot.xlabel("training setting", fontsize=9)
        pyplot.ylabel("average number", fontsize=9)
        pyplot.xticks(arange(4), labels, fontsize=8)
        pyplot.yticks(arange(0, 17, 4), arange(0, 17, 4), fontsize=8)
        pyplot.xlim(-0.5, 3.5)
        pyplot.ylim(0, 16)
        # noinspection PyUnresolvedReferences
        ax.spines["top"].set_visible(False)
        # noinspection PyUnresolvedReferences
        ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(6, 1, 6)
    pyplot.title("maximum generation = 100 and training error scale = 30%", fontsize=9)
    for index, values in enumerate(array(task_data["b"]).T):
        pyplot.bar(arange(len(values)) + bias[index], values, width=0.2, lw=0.75,
                   fc=colors[index], ec="k", label=math_orders[index])
        for location, value in zip(arange(len(values)) + bias[index], values):
            if value > 0:
                pyplot.text(location, value + 1.0, "%.1f" % value, va="center", ha="center", fontsize=8)
            else:
                pyplot.text(location, value + 1.0, "N.A", va="center", ha="center", fontsize=8)
    pyplot.legend(loc="upper right", fontsize=8, ncol=3, title="motif type", title_fontsize=8)
    pyplot.xlabel("training setting", fontsize=9)
    pyplot.ylabel("average number", fontsize=9)
    pyplot.xticks(arange(4), labels, fontsize=8)
    pyplot.yticks(arange(0, 17, 4), arange(0, 17, 4), fontsize=8)
    pyplot.xlim(-0.5, 3.5)
    pyplot.ylim(0, 16)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.02, 0.83, "b", va="center", ha="center", fontsize=14)
    figure.text(0.02, 0.67, "c", va="center", ha="center", fontsize=14)
    figure.text(0.02, 0.50, "d", va="center", ha="center", fontsize=14)
    figure.text(0.02, 0.33, "e", va="center", ha="center", fontsize=14)
    figure.text(0.02, 0.17, "f", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp11.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_12():
    task_data = load_data(sort_path + "supp12.pkl")
    labels = ["default", r"$\mathcal{L}_c + \mathcal{C}$", r"$\mathcal{L}_i + \mathcal{C}$", r"$\mathcal{C}$"]

    figure = pyplot.figure(figsize=(10, 4), tight_layout=True)
    ax = pyplot.subplot(1, 3, 1)
    for index, (value, color) in enumerate(zip(task_data["a"], pyplot.get_cmap("binary")(linspace(0.0, 0.8, 4)))):
        pyplot.bar([index], [value], width=0.6, fc=color, ec="k", lw=0.75)
        pyplot.text(index, value + 0.2, "%.1f" % value, va="center", ha="center", fontsize=8)
    pyplot.text(-0.42, 195.15, "pass (≥ 195)", va="bottom", ha="left", fontsize=8)
    pyplot.hlines(195, -0.5, 3.5, color="k", lw=0.75, ls="--")
    pyplot.xlabel("training setting", fontsize=9)
    pyplot.ylabel("average training performance", fontsize=9)
    pyplot.xticks(arange(4), labels, fontsize=8)
    pyplot.yticks(arange(192, 201, 2), arange(192, 201, 2), fontsize=8)
    pyplot.xlim(-0.5, 3.5)
    pyplot.ylim(192, 200)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(1, 3, 2)
    for index, (value, color) in enumerate(zip(task_data["b"], pyplot.get_cmap("binary")(linspace(0.0, 0.8, 4)))):
        pyplot.bar([index], [value], width=0.6, fc=color, ec="k", lw=0.75)
        pyplot.text(index, value + 2.0, "%.1f" % value, va="center", ha="center", fontsize=8)
    pyplot.xlabel("training setting", fontsize=9)
    pyplot.ylabel("median generation", fontsize=9)
    pyplot.xticks(arange(4), labels, fontsize=8)
    pyplot.yticks(arange(0, 81, 20), arange(0, 81, 20), fontsize=8)
    pyplot.xlim(-0.5, 3.5)
    pyplot.ylim(0, 80)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    pyplot.subplot(1, 3, 3)
    values = task_data["c"].copy()
    values[values >= 195] = nan
    pyplot.pcolormesh(arange(5), arange(6), values.T, vmin=100, vmax=195, cmap="inferno")
    for location_x in range(4):
        for location_y in range(5):
            value = task_data["c"][location_x, location_y]
            if value >= 195:
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "pass",
                            va="center", ha="center", fontsize=8)
            elif value > 140:
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.1f" % value,
                            va="center", ha="center", fontsize=8)
            else:
                pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.1f" % value, color="w",
                            va="center", ha="center", fontsize=8)

    pyplot.xlabel("training setting", fontsize=9)
    pyplot.ylabel("evaluating error scale", fontsize=9)
    pyplot.xticks(arange(4) + 0.5, labels, fontsize=8)
    pyplot.yticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=8)
    pyplot.xlim(0, 4)
    pyplot.ylim(0, 5)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.358, 0.99, "b", va="center", ha="center", fontsize=14)
    figure.text(0.677, 0.99, "c", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp12.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_13():
    task_data = load_data(sort_path + "supp13.pkl")
    figure = pyplot.figure(figsize=(10, 10), tight_layout=True)
    location = 1
    for index in range(4):
        cases = task_data[chr(ord("a") + index)]
        for case_index, case in enumerate(cases):
            ax = pyplot.subplot(7, 5, location)
            pyplot.title("case " + str(case_index + 1) + " / " + str(len(cases)), fontsize=9)
            pyplot.plot(arange(5) + 0.5, case, color="silver", lw=2, marker="o", zorder=0)
            pyplot.scatter([argmax(case) + 0.5], [max(case)], color="k", zorder=1)
            pyplot.text(argmax(case) + 0.5, 220, "best", va="center", ha="center", fontsize=8)
            pyplot.xlabel("evaluating error scale", fontsize=9)
            pyplot.ylabel("performance", fontsize=9)
            pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=8)
            pyplot.yticks(arange(50, 201, 50), arange(50, 201, 50), fontsize=8)
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
    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.02, 0.43, "b", va="center", ha="center", fontsize=14)
    figure.text(0.02, 0.15, "c", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp13.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_14():
    task_data = load_data(sort_path + "supp14.pkl")
    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]
    labels = ["default", r"$\mathcal{L}_c + \mathcal{C}$", r"$\mathcal{L}_i + \mathcal{C}$", r"$\mathcal{C}$"]
    colors = [draw_info["incoherent-loop"][0], draw_info["coherent-loop"][0], draw_info["collider"][0]]
    bias = [-0.2, 0, 0.2]

    figure = pyplot.figure(figsize=(10, 8), tight_layout=True)
    for index in range(4):
        ax = pyplot.subplot(4, 1, index + 1)
        record = task_data[chr(ord("a") + index)]
        pyplot.title("training setting: " + labels[index], fontsize=9)
        rates = []
        for result_type, (counts, number) in enumerate(record):
            rates.append(number)
            if counts is not None:
                for motif_type in range(3):
                    if result_type > 0:
                        pyplot.bar(bias[motif_type] + result_type, counts[motif_type], width=0.2,
                                   fc=colors[motif_type], ec="k", lw=0.75)
                    else:
                        pyplot.bar(bias[motif_type] + result_type, counts[motif_type], width=0.2,
                                   fc=colors[motif_type], ec="k", lw=0.75, label=math_orders[motif_type])
                    pyplot.text(bias[motif_type] + result_type, counts[motif_type] + 1.0, "%.1f" % counts[motif_type],
                                va="center", ha="center", fontsize=8)
            else:
                pyplot.text(result_type, 1.0, "N.A", va="center", ha="center", fontsize=8)
        pyplot.legend(loc="upper right", fontsize=8, ncol=3, title="motif type", title_fontsize=8)
        pyplot.xlabel("training result (100 samples)", fontsize=9)
        pyplot.ylabel("average number", fontsize=9)
        y_ticks = ["pass type ( " + str(rates[0]) + " / 100 )", "failure type 1 ( " + str(rates[1]) + " / 100 )",
                   "failure type 2 ( " + str(rates[2]) + " / 100 )", "failure type 3 ( " + str(rates[3]) + " / 100 )"]
        pyplot.xticks([0, 1, 2, 3], y_ticks, fontsize=8)
        pyplot.yticks(arange(0, 21, 4), arange(0, 21, 4), fontsize=8)
        pyplot.xlim(-0.5, 3.5)
        pyplot.ylim(0, 20)
        # noinspection PyUnresolvedReferences
        ax.spines["top"].set_visible(False)
        # noinspection PyUnresolvedReferences
        ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.02, 0.75, "b", va="center", ha="center", fontsize=14)
    figure.text(0.02, 0.50, "c", va="center", ha="center", fontsize=14)
    figure.text(0.02, 0.25, "d", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "supp14.pdf", format="pdf", bbox_inches="tight", dpi=600)
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
