from logging import getLogger, CRITICAL
from matplotlib import pyplot, rcParams
from numpy import arange, linspace, array, log10
from warnings import filterwarnings


from works import load_data, draw_info, acyclic_motifs

filterwarnings("ignore")

getLogger("matplotlib").setLevel(CRITICAL)

rcParams["font.family"] = "Arial"
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Linux Libertine"
rcParams["mathtext.cal"] = "Lucida Calligraphy"
rcParams["mathtext.it"] = "Linux Libertine:italic"
rcParams["mathtext.bf"] = "Linux Libertine:bold"

task_path = "./data/"
save_path = "./show/"


def supp01():
    task_data = load_data(task_path + "supp01.pkl")

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
    pyplot.annotate(s="", xy=(78, -12), xytext=(76, -12),
                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
    pyplot.text(82, -12, "positive\nweight", va="center", ha="center", fontsize=8)
    pyplot.annotate(s="", xy=(78, -15), xytext=(76, -15),
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
                    pyplot.annotate(s="", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
                else:
                    pyplot.annotate(s="", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
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
                    pyplot.annotate(s="", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
                else:
                    pyplot.annotate(s="", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
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
    pyplot.savefig(save_path + "supp01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp02():
    task_data = load_data(task_path + "supp02.pkl")

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
                    pyplot.annotate(s="", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
                else:
                    pyplot.annotate(s="", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
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
                    pyplot.annotate(s="", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
                else:
                    pyplot.annotate(s="", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
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

    pyplot.savefig(save_path + "supp02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp03():
    task_data = load_data(task_path + "supp03.pkl")

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
                    pyplot.annotate(s="", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
                                    xytext=(x[former - 1] + bias_x, y[former - 1] + bias_y),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                    shrinkA=3, shrinkB=2, lw=0.75, linestyle="-"))
                else:
                    pyplot.annotate(s="", xy=(x[latter - 1] + bias_x, y[latter - 1] + bias_y),
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

    pyplot.savefig(save_path + "supp03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp04():
    pyplot.figure(figsize=(10, 6), tight_layout=True)

    task_data = load_data(task_path + "supp04.pkl")
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
            location, x = 0, [-1.40, -1.25, -1.10]
            pyplot.text(-1.8, 3.5, "intersection", va="center", ha="center", fontsize=9, rotation=90)
            for flag_1, flag_2, flag_3 in [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]:
                left, right, y = [], [], 6.5 - location
                if [flag_1, flag_2, flag_3] != [1, 1, 1]:
                    pyplot.hlines(y - 0.5, -1, 3, lw=0.75)
                    pyplot.vlines(0, y - 0.50, y - 0.56, lw=0.75, color="k")
                    pyplot.vlines(1, y - 0.50, y - 0.56, lw=0.75, color="k")
                    pyplot.vlines(2, y - 0.50, y - 0.56, lw=0.75, color="k")
                    pyplot.vlines(3, y - 0.50, y - 0.56, lw=0.75, color="k")
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
                pyplot.hlines(y, -1.03, -1.00, lw=0.75, zorder=2)
                location += 1

            pyplot.xlim(-2.0, 3.01)
            pyplot.ylim(-1.0, 7.0)
            pyplot.axis("off")

            index += 1

    pyplot.savefig(save_path + "supp04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    # supp01()
    # supp02()
    # supp03()
    supp04()
