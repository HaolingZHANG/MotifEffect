from logging import getLogger, CRITICAL
from matplotlib import pyplot, rcParams
from numpy import array, arange, linspace, min, max, abs, std, log10, where, nan
from scipy.stats import gaussian_kde, spearmanr
from warnings import filterwarnings

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

task_path = "./data/"
save_path = "./show/"


def main01():
    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    grid = pyplot.GridSpec(1, 5)

    pyplot.subplot(grid[:, :2])
    for type_index, motif_type in enumerate(["incoherent-loop", "coherent-loop", "collider"]):
        motifs = acyclic_motifs[motif_type]
        info = draw_info[motif_type]
        info[2][2] = 0.45
        pyplot.text(3.0, 3.76 - type_index, motif_type.replace("-", " ") + " " + math_orders[type_index],
                    va="center", ha="center", fontsize=9)
        pyplot.fill_between([1, 2, 4, 5], 3.05 - type_index,
                            [3.75 - type_index, 3.86 - type_index, 3.86 - type_index, 3.75 - type_index],
                            color=info[0], lw=0, zorder=1)
        for motif_index, motif in enumerate(motifs):
            pyplot.text(1.5 + motif_index, 3.65 - type_index, "(" + str(motif_index + 1) + ")",
                        va="center", ha="center", fontsize=9)
            bias_x, bias_y = 4 - motif_index, 3 - type_index
            for index, (px, py) in enumerate(zip(info[1], info[2])):
                if index + 1 in info[3]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="white", edgecolor="black", lw=0.75, s=30, zorder=2)
                    pyplot.text(px + bias_x, py + bias_y - 0.05, "$x$", va="top", ha="center", fontsize=9)
                elif index + 1 in info[4]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="black", edgecolor="black", lw=0.75, s=30, zorder=2)
                    pyplot.text(px + bias_x, py + bias_y + 0.05, "$z$", va="bottom", ha="center", fontsize=9)
                elif index + 1 in info[5]:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="black", lw=0.75, s=30, zorder=2)
                    pyplot.text(px + bias_x, py + bias_y - 0.06, "$y$", va="top", ha="center", fontsize=9)
                else:
                    pyplot.scatter(px + bias_x, py + bias_y, color="silver", edgecolor="black", lw=0.75, s=30, zorder=2)
                    pyplot.text(px + bias_x, py + bias_y - 0.06, "$y$", va="top", ha="center", fontsize=9)
            x, y = array(info[1]) + bias_x, array(info[2]) + bias_y
            for former, latter in motif.edges:
                location_x, location_y = (x[former - 1] + x[latter - 1]) / 2.0, (y[former - 1] + y[latter - 1]) / 2.0
                flag = "+" if motif.get_edge_data(former, latter)["weight"] == 1 else "\N{MINUS SIGN}"
                pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                                arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="black",
                                                shrinkA=4, shrinkB=4, lw=0.75, ls=("-" if flag == "+" else ":")))
                if (former, latter) == (1, 2):
                    pyplot.text(location_x, location_y - 0.1, flag, va="center", ha="center", fontsize=9)
                if (former, latter) == (1, 3):
                    pyplot.text(location_x - 0.1, location_y, flag, va="center", ha="center", fontsize=9)
                if (former, latter) == (2, 3):
                    pyplot.text(location_x + 0.1, location_y, flag, va="center", ha="center", fontsize=9)

    pyplot.xlim(0.50, 5.50)
    pyplot.ylim(0.86, 3.86)
    pyplot.axis("off")

    pyplot.subplot(grid[:, 2:])
    task_data = load_data(task_path + "main01.pkl")["b"]

    activation_selection, aggregation_selection = ["tanh", "sigmoid", "relu"], ["sum", "max"]
    for index_1, activation in enumerate(activation_selection):
        for index_2, aggregation in enumerate(aggregation_selection):
            x, y = index_2 * 0.65 + 0.5, 2.5 - index_1
            pyplot.text(x, y + 0.37, activation + " + " + aggregation, va="center", ha="center", fontsize=9)
            location, colors = 0, ["#FCB1AB", "#FCE0AB", "#88CCF8"]
            pyplot.plot([x - 0.2, x - 0.2, x + 0.2, x + 0.2, x - 0.2],
                        [y + 0.285, y - 0.205, y - 0.205, y + 0.285, y + 0.285], lw=0.75, color="k")
            pyplot.text(x - 0.35, y + 0.04, "intersection", va="center", ha="center", fontsize=9, rotation=90)
            sub_x = [x - 0.30, x - 0.27, x - 0.24]
            for flag_1, flag_2, flag_3 in [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]:
                left, right, sub_y = [], [], y + 0.25 - location * 0.07
                if [flag_1, flag_2, flag_3] != [0, 0, 1]:
                    pyplot.hlines(sub_y + 0.035, x - 0.2, x + 0.2, lw=0.75)
                if flag_1:
                    pyplot.scatter([sub_x[0]], [sub_y], s=15, fc=colors[0], ec="k", lw=0.75, zorder=3)
                else:
                    pyplot.scatter([sub_x[0]], [sub_y], s=15, marker="x", ec="k", lw=0.75, zorder=3)
                if flag_2:
                    pyplot.scatter([sub_x[1]], [sub_y], s=15, fc=colors[1], ec="k", lw=0.75, zorder=3)
                else:
                    pyplot.scatter([sub_x[1]], [sub_y], s=15, marker="x", ec="k", lw=0.75, zorder=3)
                if flag_3:
                    pyplot.scatter([sub_x[2]], [sub_y], s=15, fc=colors[2], ec="k", lw=0.75, zorder=3)
                else:
                    pyplot.scatter([sub_x[2]], [sub_y], s=15, marker="x", ec="k", lw=0.75, zorder=3)
                pyplot.hlines(sub_y, sub_x[0], sub_x[2], lw=0.75, zorder=2)
                pyplot.hlines(sub_y, x - 0.215, x - 0.200, lw=0.75, zorder=2)
                location += 1

            # fix the comparison starting point problem of the set.
            if len(where(task_data[(activation, aggregation)][3] > 0)[0]) == 1:
                task_data[(activation, aggregation)][6, 1] += task_data[(activation, aggregation)][3, 1]
                task_data[(activation, aggregation)][3, 1] = 0
                task_data[(activation, aggregation)][6, 2] += task_data[(activation, aggregation)][3, 2]
                task_data[(activation, aggregation)][3, 2] = 0
            if len(where(task_data[(activation, aggregation)][4] > 0)[0]) == 1:
                task_data[(activation, aggregation)][6, 0] += task_data[(activation, aggregation)][4, 0]
                task_data[(activation, aggregation)][4, 0] = 0
                task_data[(activation, aggregation)][6, 2] += task_data[(activation, aggregation)][4, 2]
                task_data[(activation, aggregation)][4, 2] = 0
            if len(where(task_data[(activation, aggregation)][5] > 0)[0]) == 1:
                task_data[(activation, aggregation)][6, 0] += task_data[(activation, aggregation)][5, 0]
                task_data[(activation, aggregation)][5, 0] = 0
                task_data[(activation, aggregation)][6, 1] += task_data[(activation, aggregation)][5, 1]
                task_data[(activation, aggregation)][5, 1] = 0

            for intersect_index in range(7):
                sub_y = y + 0.25 - intersect_index * 0.07
                for type_index in range(3):
                    sub_x = x - 0.40 / 3.0 + type_index * 0.40 / 3.0
                    color_value = task_data[(activation, aggregation)][intersect_index, type_index]
                    color = pyplot.get_cmap("Greens")(color_value * 0.5)
                    pyplot.fill_between([sub_x - 0.20 / 3.0, sub_x + 0.20 / 3.0], sub_y - 0.035, sub_y + 0.035,
                                        fc=color, ec=color, lw=0.75, zorder=2)
                    if color_value == 1:
                        pyplot.text(sub_x, sub_y - 0.002, "100%", va="center", ha="center", fontsize=7.5)
                    elif color_value > 0:
                        pyplot.text(sub_x, sub_y - 0.002, ("%.2f" % (color_value * 100)) + "%",
                                    va="center", ha="center", fontsize=7.5)

            pyplot.vlines(x - 0.20 / 3.0, y - 0.205, y + 0.285, lw=0.75, zorder=2)
            pyplot.vlines(x + 0.20 / 3.0, y - 0.205, y + 0.285, lw=0.75, zorder=2)
            pyplot.vlines(x - 0.40 / 3.0, y - 0.205, y - 0.235, lw=0.75, zorder=2)
            pyplot.text(x - 0.40 / 3.0, y - 0.30, math_orders[0], va="center", ha="center", fontsize=9)
            pyplot.vlines(x + 0.00, y - 0.205, y - 0.235, lw=0.75, zorder=2)
            pyplot.text(x + 0.00, y - 0.30, math_orders[1], va="center", ha="center", fontsize=9)
            pyplot.vlines(x + 0.40 / 3.0, y - 0.205, y - 0.235, lw=0.75, zorder=2)
            pyplot.text(x + 0.40 / 3.0, y - 0.30, math_orders[2], va="center", ha="center", fontsize=9)
            pyplot.text(x + 0.00, y - 0.40, "population", va="center", ha="center", fontsize=9)

    mark = 1.6
    pyplot.text(mark, 2.75, "intersection size", va="center", ha="center", fontsize=9)
    pyplot.text(mark, 2.65, "population size", va="center", ha="center", fontsize=9)
    pyplot.hlines(2.7, mark - 0.14, mark + 0.14, lw=0.75)
    for index, color in enumerate(pyplot.get_cmap("Greens")(linspace(0.0, 0.5, 100))):
        pyplot.fill_between([mark - 0.02, mark + 0.02], 0.5 + index * 0.02, 0.5 + (index + 1) * 0.02,
                            fc=color, lw=0, zorder=1)
    pyplot.plot([mark - 0.02, mark + 0.02, mark + 0.02, mark - 0.02, mark - 0.02],
                [0.5, 0.5, 2.5, 2.5, 0.5], lw=0.75, color="k", zorder=2)
    for index in range(6):
        pyplot.hlines(0.5 + index * 0.4, mark + 0.02, mark + 0.04, lw=0.75)
        pyplot.text(mark + 0.06, 0.5 + index * 0.4, str(index * 20) + "%", va="center", ha="left", fontsize=9)
    pyplot.xlim(0.1, 1.8)
    pyplot.ylim(0.0, 3.0)
    pyplot.axis("off")

    figure.text(0.02, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.39, 0.99, "b", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "main01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main02():
    task_data = load_data(task_path + "main02.pkl")

    used_colors = [draw_info["incoherent-loop"][0], draw_info["coherent-loop"][0], draw_info["collider"][0]]
    activation_selection, aggregation_selection = ["tanh", "sigmoid", "relu"], ["sum", "max"]

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    pyplot.subplot(1, 2, 1)

    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]
    for index_1, activation in enumerate(activation_selection):
        for index_2, aggregation in enumerate(aggregation_selection):
            x, y = 0.5 + index_2, 2.5 - index_1
            pyplot.text(x + 0.05, y + 0.42, activation + " + " + aggregation, va="center", ha="center", fontsize=9)
            pyplot.plot([x - 0.300, x - 0.300, x + 0.400, x + 0.400, x - 0.300],
                        [y + 0.365, y - 0.265, y - 0.265, y + 0.365, y + 0.365], lw=0.75, color="k")
            pyplot.text(x - 0.50, y + 0.05, "population size",
                        va="center", ha="center", fontsize=9, rotation=90)
            location = 0
            for info in ["1E+6", "1E+5", "1E+4", "1E+3", "1E+2"]:
                sub_y = y + 0.32 - location * 0.135
                pyplot.hlines(sub_y, x - 0.315, x - 0.300, lw=0.75, zorder=2)
                pyplot.text(x - 0.39, sub_y - 0.01, info, va="center", ha="center", fontsize=8)
                location += 1

            collected_values = {}
            for index, (s, r) in enumerate(zip(task_data["a"][0][(activation, aggregation)],
                                               task_data["a"][1][(activation, aggregation)])):
                x_values, y_values = x - 0.27 + array(r) / 4.0 * 0.64, y - 0.22 + (log10(s) - 2) / 5.0 * 0.54
                pyplot.scatter(x_values, y_values, s=16, fc=used_colors[index], ec="k", lw=0.75, zorder=1)
                collected_values[index] = {}
                if std(r) <= 1.0 and std(r) < 1e-2:
                    point = (x - 0.27 + r[0] / 4.0 * 0.64, y - 0.22 + (log10(s[0]) - 2) / 5.0 * 0.54)
                    collected_values[index] = {math_orders[index] + " (1~4)": point}
                else:
                    collected_values[index] = {}
                    point = (x - 0.27 + r[0] / 4.0 * 0.64, y - 0.22 + (log10(s[0]) - 2) / 5.0 * 0.54)
                    collected_values[index][math_orders[index] + " (1,3)"] = point
                    point = (x - 0.27 + r[1] / 4.0 * 0.64, y - 0.22 + (log10(s[1]) - 2) / 5.0 * 0.54)
                    collected_values[index][math_orders[index] + " (2,4)"] = point
            if (activation, aggregation) == ("tanh", "sum"):
                key, point = list(collected_values[0].items())[0]
                pyplot.plot([point[0], point[0] - 0.12], [point[1], point[1] + 0.12],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] - 0.12, point[1] + 0.12, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[1].items())[0]
                pyplot.plot([point[0], point[0] + 0.18], [point[1], point[1]],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.18, point[1] - 0.004, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8)
                key, point = list(collected_values[2].items())[0]
                pyplot.plot([point[0], point[0] + 0.16], [point[1], point[1] - 0.08],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.16, point[1] - 0.08, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8)
            elif (activation, aggregation) == ("sigmoid", "sum"):
                key, point = list(collected_values[0].items())[0]
                pyplot.plot([point[0], point[0] - 0.04], [point[1], point[1] + 0.20],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] - 0.04, point[1] + 0.20, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
                key, point = list(collected_values[1].items())[0]
                pyplot.plot([point[0], point[0] + 0.20], [point[1], point[1]],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.20, point[1], key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
                key, point = list(collected_values[2].items())[0]
                pyplot.plot([point[0], point[0] + 0.20], [point[1], point[1] - 0.08],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.20, point[1] - 0.08, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
            elif (activation, aggregation) == ("relu", "sum"):
                key, point = list(collected_values[0].items())[0]
                pyplot.plot([point[0], point[0] - 0.12], [point[1], point[1] + 0.14],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] - 0.12, point[1] + 0.14, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[0].items())[1]
                pyplot.plot([point[0], point[0] + 0.14], [point[1], point[1] + 0.14],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.14, point[1] + 0.14, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[1].items())[0]
                pyplot.plot([point[0], point[0] - 0.12], [point[1], point[1] - 0.10],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] - 0.12, point[1] - 0.10, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
                key, point = list(collected_values[1].items())[1]
                pyplot.plot([point[0], point[0] + 0.20], [point[1], point[1]],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.20, point[1], key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
                key, point = list(collected_values[2].items())[0]
                pyplot.plot([point[0], point[0] + 0.20], [point[1], point[1] - 0.05],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.20, point[1] - 0.05, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
            elif (activation, aggregation) == ("tanh", "max"):
                key, point = list(collected_values[0].items())[0]
                pyplot.plot([point[0], point[0] + 0.12], [point[1], point[1] - 0.14],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.12, point[1] - 0.14, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[0].items())[1]
                pyplot.plot([point[0], point[0] - 0.14], [point[1], point[1] + 0.14],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] - 0.14, point[1] + 0.14, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[1].items())[0]
                pyplot.plot([point[0], point[0] - 0.20], [point[1], point[1] - 0.14],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] - 0.20, point[1] - 0.14, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
                key, point = list(collected_values[1].items())[1]
                pyplot.plot([point[0], point[0] + 0.15], [point[1], point[1] + 0.14],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.15, point[1] + 0.14, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
                key, point = list(collected_values[2].items())[0]
                pyplot.plot([point[0], point[0] + 0.20], [point[1], point[1] - 0.05],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.20, point[1] - 0.05, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
            elif (activation, aggregation) == ("sigmoid", "max"):
                key, point = list(collected_values[0].items())[0]
                pyplot.plot([point[0], point[0]], [point[1], point[1] + 0.22],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0], point[1] + 0.22, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[0].items())[1]
                pyplot.plot([point[0], point[0] - 0.15], [point[1], point[1] - 0.04],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] - 0.15, point[1] - 0.04, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[1].items())[0]
                pyplot.plot([point[0], point[0] - 0.02], [point[1], point[1] + 0.12],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] - 0.02, point[1] + 0.12, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[1].items())[1]
                pyplot.plot([point[0], point[0]], [point[1], point[1] + 0.20],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0], point[1] + 0.20, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[2].items())[0]
                pyplot.plot([point[0], point[0] + 0.20], [point[1], point[1] - 0.05],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.20, point[1] - 0.05, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
            elif (activation, aggregation) == ("relu", "max"):
                key, point = list(collected_values[0].items())[0]
                pyplot.plot([point[0], point[0]], [point[1], point[1] + 0.20],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0], point[1] + 0.20, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[0].items())[1]
                pyplot.plot([point[0], point[0] - 0.14], [point[1], point[1] - 0.04],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] - 0.14, point[1] - 0.04, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[1].items())[0]
                pyplot.plot([point[0], point[0] + 0.15], [point[1], point[1] - 0.04],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.15, point[1] - 0.04, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[1].items())[1]
                pyplot.plot([point[0], point[0] - 0.12], [point[1], point[1] + 0.12],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] - 0.12, point[1] + 0.12, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.02), fontsize=8, zorder=3)
                key, point = list(collected_values[2].items())[0]
                pyplot.plot([point[0], point[0] + 0.15], [point[1], point[1] - 0.05],
                            color="k", lw=0.75, ls="--", zorder=2)
                pyplot.text(point[0] + 0.15, point[1] - 0.05, key, va="center", ha="center",
                            bbox=dict(fc="w", ec="w", pad=0.03), fontsize=8, zorder=3)
            for index, value in enumerate(linspace(-0.27, +0.37, 5)):
                pyplot.vlines(x + value, y - 0.265, y - 0.290, lw=0.75, zorder=2)
                pyplot.text(x + value, y - 0.35, str(index), va="center", ha="center", fontsize=8)
            pyplot.text(x + 0.05, y - 0.45, "median Lipschitz constant", va="center", ha="center", fontsize=9)
    pyplot.xlim(0, 2)
    pyplot.ylim(0, 3)
    pyplot.axis("off")

    pyplot.subplot(1, 2, 2)
    pyplot.plot([-1, -1, 3], [4, 0, 0], lw=0.75, color="k")
    for index, info in enumerate(["0", "1", "2", "3", "4"]):
        pyplot.vlines(index - 1, 0, -0.04, lw=0.75, color="k")
        pyplot.text(index - 1, -0.12, info, va="center", ha="center", fontsize=8)
    pyplot.text(1.0, -0.26, "Lipschitz constant", va="center", ha="center", fontsize=9)
    colors = [draw_info["incoherent-loop"][0], draw_info["coherent-loop"][0], draw_info["collider"][0]]
    location, x = 0, [-1.26, -1.18, -1.10]
    pyplot.text(-1.4, 2.0, "intersection", va="center", ha="center", fontsize=9, rotation=90)
    for flag_1, flag_2, flag_3 in [[0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]:
        left, right, y = [], [], 3.5 - location
        if [flag_1, flag_2, flag_3] != [1, 1, 1]:
            pyplot.hlines(y - 0.5, -1, 3, lw=0.75)
            pyplot.vlines(0, y - 0.50, y - 0.54, lw=0.75, color="k")
            pyplot.vlines(1, y - 0.50, y - 0.54, lw=0.75, color="k")
            pyplot.vlines(2, y - 0.50, y - 0.54, lw=0.75, color="k")
            pyplot.vlines(3, y - 0.50, y - 0.54, lw=0.75, color="k")
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
    record = task_data["b"]
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

    values = record[(1, 1)]
    constants = linspace(min(values), max(values), 100)
    radios = gaussian_kde(values)(constants)
    radios = radios / max(radios) * 0.65
    pyplot.plot(constants - 1, 3 + radios, color="k", lw=0.75, zorder=3)
    pyplot.fill_between(constants - 1, 3, 3 + radios, fc=used_colors[1], lw=0)
    pyplot.text(0.70, 3.70, r"1.59% $\mathcal{L}_c$", va="center", ha="center", fontsize=9)
    values = record[(2, 0)]
    constants = linspace(min(values), max(values), 100)
    radios = gaussian_kde(values)(constants)
    radios = radios / max(radios) * 0.65
    pyplot.plot(constants - 1, 2 + radios, color="k", lw=0.75, zorder=3)
    pyplot.fill_between(constants - 1, 2, 2 + radios, fc=used_colors[0], lw=0)
    pyplot.text(0.85, 2.70, r"11.23% $\mathcal{L}_i$", va="center", ha="center", fontsize=9)
    values = record[(5, 0)]
    constants = linspace(min(values), max(values), 100)
    radios = gaussian_kde(values)(constants)
    radios = radios / max(radios) * 0.65
    pyplot.plot(constants - 1, 1 + radios, color="k", lw=0.75, zorder=3)
    pyplot.fill_between(constants - 1, 1, 1 + radios, fc=used_colors[0], lw=0)
    pyplot.text(-0.05, 1.70, r"6.32% $\mathcal{L}_i$", va="center", ha="center", fontsize=9)
    pyplot.text(1.50, 1.70, r"1.64% $\mathcal{L}_c$", va="center", ha="center", fontsize=9)
    values = record[(5, 1)]
    constants = linspace(min(values), max(values), 100)
    radios = gaussian_kde(values)(constants)
    radios = radios / max(radios) * 0.65
    pyplot.plot(constants - 1, 1 + radios, color="k", lw=0.75, zorder=3)
    pyplot.fill_between(constants - 1, 1, 1 + radios, fc="orange", lw=0, alpha=0.5)
    values = record[(6, 2)]
    constants = linspace(min(values), max(values), 100)
    radios = gaussian_kde(values)(constants)
    radios = radios / max(radios) * 0.65
    pyplot.plot(constants - 1, 0 + radios, color="k", lw=0.75, zorder=3)
    pyplot.fill_between(constants - 1, 0, 0 + radios, fc=used_colors[2], lw=0)
    pyplot.text(-0.10, 0.70, r"100% $\mathcal{C}$", va="center", ha="center", fontsize=9)
    values = array(record[(6, 0)].tolist() + record[(6, 1)].tolist())
    constants = linspace(min(values), max(values), 100)
    radios = gaussian_kde(values)(constants)
    radios = radios / max(radios) * 0.65
    pyplot.plot(constants - 1, 0 + radios, color="k", lw=0.75, zorder=3)
    pyplot.fill_between(constants - 1, 0, 0 + radios, fc="gray", lw=0, alpha=0.5)
    pyplot.text(1.45, 0.70,
                r"82.25% $\mathcal{L}_i$ + 96.77% $\mathcal{L}_c$  ($\mathcal{L}_i\ \approx\ \mathcal{L}_c$)",
                va="center", ha="center", fontsize=9)
    pyplot.xlim(-1.40, 3.00)
    pyplot.ylim(-0.35, 4.20)
    pyplot.axis("off")

    figure.text(0.022, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.510, 0.99, "b", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "main02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main03():
    task_data = load_data(load_path=task_path + "main03.pkl")
    # paint_data = load_data("paint_data.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    grid = pyplot.GridSpec(2, 3)

    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]

    ax = pyplot.subplot(grid[0, 0])
    pyplot.title("train " + str(math_orders[0]) + " to escape from " + math_orders[-1], fontsize=9)
    loss_change, robust_change = array(task_data["a"][0][0]), array(task_data["a"][1][0])
    # noinspection PyTypeChecker
    corr, p = spearmanr(loss_change, robust_change)
    pyplot.text(0.070, -0.65, "spearman", va="center", ha="left", fontsize=8)
    pyplot.text(0.096, -0.65, "= %.2f" % corr, va="center", ha="left", fontsize=8)
    pyplot.text(0.070, -0.80, "p-value", va="center", ha="left", fontsize=8)
    pyplot.text(0.096, -0.80, ("= %.2e" % p).upper().replace("-", "\N{MINUS SIGN}"), va="center", ha="left", fontsize=8)
    location = where(robust_change > 0)[0]
    pyplot.scatter(loss_change[location], robust_change[location], s=12, color="tomato",
                   label=str(len(location)) + " / 400", alpha=0.75)
    location = where(robust_change <= 0)[0]
    pyplot.scatter(loss_change[location], robust_change[location], s=12, color="royalblue",
                   label=str(len(location)) + " / 400", alpha=0.75)
    pyplot.legend(loc="upper left", fontsize=8)
    pyplot.xlabel("loss growth after training", fontsize=9)
    pyplot.ylabel("variation of Lipschitz constant", fontsize=9)
    pyplot.xticks(linspace(0, 0.12, 7), ["%.2f" % v for v in linspace(0, 0.12, 7)], fontsize=8)
    pyplot.yticks(linspace(-0.8, 0.8, 5), ["\N{MINUS SIGN}0.8", "\N{MINUS SIGN}0.4", "0.0", "+0.4", "+0.8"], fontsize=8)
    pyplot.xlim(-0.006, 0.126)
    pyplot.ylim(-0.91, 0.91)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(grid[0, 1])
    pyplot.title("train " + str(math_orders[1]) + " to escape from " + math_orders[-1], fontsize=9)
    loss_change, robust_change = array(task_data["a"][0][1]), array(task_data["a"][1][1])
    # noinspection PyTypeChecker
    corr, p = spearmanr(loss_change, robust_change)
    pyplot.text(0.070, -0.65, "spearman", va="center", ha="left", fontsize=8)
    pyplot.text(0.096, -0.65, "= %.2f" % corr, va="center", ha="left", fontsize=8)
    pyplot.text(0.070, -0.80, "p-value", va="center", ha="left", fontsize=8)
    pyplot.text(0.096, -0.80, ("= %.2e" % p).upper().replace("-", "\N{MINUS SIGN}"), va="center", ha="left", fontsize=8)
    location = where(robust_change > 0)[0]
    pyplot.scatter(loss_change[location], robust_change[location], s=12, color="tomato",
                   label=str(len(location)) + " / 400", alpha=0.75)
    location = where(robust_change <= 0)[0]
    pyplot.scatter(loss_change[location], robust_change[location], s=12, color="royalblue",
                   label="  " + str(len(location)) + " / 400", alpha=0.75)
    pyplot.legend(loc="upper left", fontsize=8)
    pyplot.xlabel("loss growth after training", fontsize=9)
    pyplot.ylabel("variation of Lipschitz constant", fontsize=9)
    pyplot.xticks(linspace(0, 0.12, 7), ["%.2f" % v for v in linspace(0, 0.12, 7)], fontsize=8)
    pyplot.yticks(linspace(-0.8, 0.8, 5), ["\N{MINUS SIGN}0.8", "\N{MINUS SIGN}0.4", "0.0", "+0.4", "+0.8"], fontsize=8)
    pyplot.xlim(-0.006, 0.126)
    pyplot.ylim(-0.91, 0.91)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    cases = task_data["b"]

    pyplot.subplot(grid[1, 0])
    pyplot.fill_between([0.2, 1.8], 2.80, 2.85, fc="tomato", ec="k", lw=0, zorder=0)
    pyplot.fill_between([2.2, 3.8], 2.80, 2.85, fc="royalblue", ec="k", lw=0, zorder=0)
    pyplot.text(1.0, 3.0, "maximum increase", va="center", ha="center", fontsize=7)
    pyplot.text(3.0, 3.0, "maximum decrease", va="center", ha="center", fontsize=7)
    pyplot.text(0.50, 2.57, math_orders[1], va="center", ha="center", fontsize=8)
    pyplot.text(1.50, 2.57, math_orders[2], va="center", ha="center", fontsize=8)
    pyplot.text(2.50, 2.57, math_orders[1], va="center", ha="center", fontsize=8)
    pyplot.text(3.50, 2.57, math_orders[2], va="center", ha="center", fontsize=8)
    pyplot.text(-0.9, 2.1, "Lipschitz\nvariation", va="center", ha="center", fontsize=7)
    pyplot.text(-0.9, 1.4, "former\nlandscape", va="center", ha="center", fontsize=7)
    pyplot.text(-0.9, 0.3, "latter\nlandscape", va="center", ha="center", fontsize=7)
    for index_1 in range(4):
        for index_2 in range(2):
            pyplot.text(index_1 + 0.5, index_2 * 1.1 - 0.1, "$x$", va="center", ha="center", fontsize=8)
            pyplot.text(index_1 + 0.1, index_2 * 1.1 + 0.3, "$y$", va="center", ha="center", fontsize=8)
        pyplot.scatter([index_1 + 0.5], [2.1], marker="v", color="k", s=10)
        pyplot.annotate(s="", xy=(index_1 + 0.5, 0.60), xytext=(index_1 + 0.5, 0.95),
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75))
    pyplot.text(2.0, -0.25, "$z$", va="center", ha="center", fontsize=8)
    pyplot.plot([0.2, 3.8, 3.8, 0.2, 0.2], [-0.6, -0.6, -0.4, -0.4, -0.6], color="k", lw=0.75, zorder=1)
    for index, color in enumerate(pyplot.get_cmap("PRGn")(linspace(0, 1, 100))):
        pyplot.fill_between([0.2 + index * 0.036, 0.2 + (index + 1) * 0.036], -0.6, -0.4, fc=color, lw=0, zorder=0)
    for location, value in zip(linspace(0.2, 3.8, 5), linspace(-1, 1, 5)):
        pyplot.vlines(location, -0.6, -0.7, lw=0.75)
        if value == 0:
            pyplot.text(location, -0.9, "%.1f" % value, va="center", ha="center", fontsize=7)
        elif value > 0:
            pyplot.text(location, -0.9, "+%.1f" % value, va="center", ha="center", fontsize=7)
        else:
            pyplot.text(location, -0.9, "\N{MINUS SIGN}%.1f" % abs(value), va="center", ha="center", fontsize=7)
    indices, location = [2, 0, 3, 1], 0
    for index_1 in range(2):
        for index_2 in range(2):
            landscape = cases["i"][0][indices[location]][0]
            x = linspace(index_1 + 0.2, index_1 + 0.8, 41)
            y = linspace(index_2 * 1.1 + 0.0, index_2 * 1.1 + 0.6, 41)
            pyplot.pcolormesh(x, y, landscape, cmap="PRGn", vmin=-1, vmax=1)
            if indices[location] >= 2:
                pyplot.text(index_1 + 0.5, 1.9, "%.3f" % cases["i"][0][indices[location]][1],
                            va="center", ha="center", fontsize=7)
            else:
                pyplot.text(index_1 + 0.5, 2.3, "%.3f" % cases["i"][0][indices[location]][1],
                            va="center", ha="center", fontsize=7)
            location += 1
    location = 0
    for index_1 in range(2):
        for index_2 in range(2):
            landscape = cases["i"][1][indices[location]][0]
            x = linspace(2 + index_1 + 0.2, 2 + index_1 + 0.8, 41)
            y = linspace(index_2 * 1.1 + 0.0, index_2 * 1.1 + 0.6, 41)
            pyplot.pcolormesh(x, y, landscape, cmap="PRGn", vmin=-1, vmax=1)
            if indices[location] >= 2:
                pyplot.text(index_1 + 2.5, 1.9, "%.3f" % cases["i"][1][indices[location]][1],
                            va="center", ha="center", fontsize=7)
            else:
                pyplot.text(index_1 + 2.5, 2.3, "%.3f" % cases["i"][1][indices[location]][1],
                            va="center", ha="center", fontsize=7)
            location += 1
    pyplot.xlim(-0.5, 4.5)
    pyplot.ylim(-1.0, 2.9)
    pyplot.axis("off")

    pyplot.subplot(grid[1, 1])
    pyplot.fill_between([0.2, 1.8], 2.80, 2.85, fc="tomato", ec="k", lw=0, zorder=0)
    pyplot.fill_between([2.2, 3.8], 2.80, 2.85, fc="royalblue", ec="k", lw=0, zorder=0)
    pyplot.text(1.0, 3.0, "maximum increase", va="center", ha="center", fontsize=7)
    pyplot.text(3.0, 3.0, "maximum decrease", va="center", ha="center", fontsize=7)
    pyplot.text(0.50, 2.57, math_orders[1], va="center", ha="center", fontsize=8)
    pyplot.text(1.50, 2.57, math_orders[2], va="center", ha="center", fontsize=8)
    pyplot.text(2.50, 2.57, math_orders[1], va="center", ha="center", fontsize=8)
    pyplot.text(3.50, 2.57, math_orders[2], va="center", ha="center", fontsize=8)
    pyplot.text(-0.9, 2.1, "Lipschitz\nvariation", va="center", ha="center", fontsize=7)
    pyplot.text(-0.9, 1.4, "former\nlandscape", va="center", ha="center", fontsize=7)
    pyplot.text(-0.9, 0.3, "latter\nlandscape", va="center", ha="center", fontsize=7)
    for index_1 in range(4):
        for index_2 in range(2):
            pyplot.text(index_1 + 0.5, index_2 * 1.1 - 0.1, "$x$", va="center", ha="center", fontsize=8)
            pyplot.text(index_1 + 0.1, index_2 * 1.1 + 0.3, "$y$", va="center", ha="center", fontsize=8)
        pyplot.scatter([index_1 + 0.5], [2.1], marker="v", color="k", s=10)
        pyplot.annotate(s="", xy=(index_1 + 0.5, 0.60), xytext=(index_1 + 0.5, 0.95),
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75))
    pyplot.text(2.0, -0.25, "$z$", va="center", ha="center", fontsize=8)
    pyplot.plot([0.2, 3.8, 3.8, 0.2, 0.2], [-0.6, -0.6, -0.4, -0.4, -0.6], color="k", lw=0.75, zorder=1)
    for index, color in enumerate(pyplot.get_cmap("PRGn")(linspace(0, 1, 100))):
        pyplot.fill_between([0.2 + index * 0.036, 0.2 + (index + 1) * 0.036], -0.6, -0.4, fc=color, lw=0, zorder=0)
    for location, value in zip(linspace(0.2, 3.8, 5), linspace(-1, 1, 5)):
        pyplot.vlines(location, -0.6, -0.7, lw=0.75)
        if value == 0:
            pyplot.text(location, -0.9, "%.1f" % value, va="center", ha="center", fontsize=7)
        elif value > 0:
            pyplot.text(location, -0.9, "+%.1f" % value, va="center", ha="center", fontsize=7)
        else:
            pyplot.text(location, -0.9, "\N{MINUS SIGN}%.1f" % abs(value), va="center", ha="center", fontsize=7)

    indices, location = [2, 0, 3, 1], 0
    for index_1 in range(2):
        for index_2 in range(2):
            landscape = cases["c"][0][indices[location]][0]
            x = linspace(index_1 + 0.2, index_1 + 0.8, 41)
            y = linspace(index_2 * 1.1 + 0.0, index_2 * 1.1 + 0.6, 41)
            pyplot.pcolormesh(x, y, landscape, cmap="PRGn", vmin=-1, vmax=1)
            if indices[location] >= 2:
                pyplot.text(index_1 + 0.5, 1.9, "%.3f" % cases["c"][0][indices[location]][1],
                            va="center", ha="center", fontsize=7)
            else:
                pyplot.text(index_1 + 0.5, 2.3, "%.3f" % cases["c"][0][indices[location]][1],
                            va="center", ha="center", fontsize=7)
            location += 1
    location = 0
    for index_1 in range(2):
        for index_2 in range(2):
            landscape = cases["c"][1][indices[location]][0]
            x = linspace(2 + index_1 + 0.2, 2 + index_1 + 0.8, 41)
            y = linspace(index_2 * 1.1 + 0.0, index_2 * 1.1 + 0.6, 41)
            pyplot.pcolormesh(x, y, landscape, cmap="PRGn", vmin=-1, vmax=1)
            if indices[location] >= 2:
                pyplot.text(index_1 + 2.5, 1.9, "%.3f" % cases["c"][1][indices[location]][1],
                            va="center", ha="center", fontsize=7)
            else:
                pyplot.text(index_1 + 2.5, 2.3, "%.3f" % cases["c"][1][indices[location]][1],
                            va="center", ha="center", fontsize=7)
            location += 1
    pyplot.xlim(-0.5, 4.5)
    pyplot.ylim(-1.0, 2.9)
    pyplot.axis("off")

    pyplot.subplot(grid[:, 2])
    pyplot.title("escape strategy of " + math_orders[0] + " and " + math_orders[1], fontsize=9)
    pyplot.text(-1.1, 7, math_orders[0], va="center", ha="center", fontsize=9)
    pyplot.text(-1.1, 2, math_orders[1], va="center", ha="center", fontsize=9)
    pyplot.vlines(-0.7, 7.2, 8.8, color="tomato", lw=3)
    pyplot.vlines(-0.7, 5.2, 6.8, color="royalblue", lw=3)
    pyplot.vlines(-0.7, 2.2, 3.8, color="tomato", lw=3)
    pyplot.vlines(-0.7, 0.2, 1.8, color="royalblue", lw=3)
    for index_1 in range(3):
        for index_2 in range(8):
            if index_1 == 0:
                pyplot.text(index_1 + 1.45, index_2 + 0.5 + (index_2 > 3), "+", va="center", ha="center", fontsize=8)
                pyplot.text(index_1 + 2.65, index_2 + 0.5 + (index_2 > 3), "=", va="center", ha="center", fontsize=8)
            pyplot.text(index_1 * 1.2 + 0.9, index_2 + 0.1 + (index_2 > 3), "$x$",
                        va="center", ha="center", fontsize=8)
            pyplot.text(index_1 * 1.2 + 0.5, index_2 + 0.5 + (index_2 > 3), "$y$",
                        va="center", ha="center", fontsize=8)
    pyplot.text(-0.1, 4.5, "Lipschitz\nvariation", va="center", ha="center", fontsize=7)
    pyplot.text(0.9, 4.5, "former\nlandscape", va="center", ha="center", fontsize=7)
    pyplot.text(3.3, 4.5, "latter\nlandscape", va="center", ha="center", fontsize=7)
    pyplot.annotate(s="", xy=(2.8, 4.5), xytext=(1.4, 4.5), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75))
    pyplot.text(4.1, 9.0, "$z$", va="center", ha="center", fontsize=9)
    pyplot.plot([4.0, 4.2, 4.2, 4.0, 4.0], [0.2, 0.2, 8.8, 8.8, 0.2], color="k", lw=0.75, zorder=2)
    for index, color in enumerate(pyplot.get_cmap("PRGn")(linspace(0, 1, 100))):
        pyplot.fill_between([4.0, 4.2], 0.2 + index * 0.086, 0.2 + (index + 1) * 0.086, fc=color, lw=0, zorder=0)
    for location, value in zip(linspace(0.2, 8.8, 5), linspace(-1, 1, 5)):
        pyplot.hlines(location, 4.2, 4.3, lw=0.75)
        if value == 0:
            pyplot.text(4.9, location - 0.02, "0.0", va="center", ha="right", fontsize=7)
        elif value > 0:
            pyplot.text(4.9, location - 0.02, "+%.1f" % value, va="center", ha="right", fontsize=7)
        else:
            pyplot.text(4.9, location - 0.02, "\N{MINUS SIGN}%.1f" % abs(value), va="center", ha="right", fontsize=7)

    record = task_data["c"]

    for index in range(4):
        former, difference, latter, former_value, latter_value = record["i"][index]
        pyplot.scatter([-0.1], [8 - index + 0.5], marker="v", color="k", s=10)
        pyplot.text(-0.1, 8 - index + 0.75, "%.3f" % former_value, va="center", ha="center", fontsize=7)
        pyplot.text(-0.1, 8 - index + 0.25, "%.3f" % latter_value, va="center", ha="center", fontsize=7)
        pyplot.pcolormesh(linspace(0.6, 1.2, 41), linspace(8 - index + 0.2, 8 - index + 0.8, 41), former,
                          cmap="PRGn", vmin=-1, vmax=1, zorder=3)
        pyplot.pcolormesh(linspace(1.8, 2.4, 41), linspace(8 - index + 0.2, 8 - index + 0.8, 41), difference,
                          cmap="PRGn", vmin=-1, vmax=1, zorder=3)
        pyplot.pcolormesh(linspace(3.0, 3.6, 41), linspace(8 - index + 0.2, 8 - index + 0.8, 41), latter,
                          cmap="PRGn", vmin=-1, vmax=1, zorder=3)
    for index in range(4):
        former, difference, latter, former_value, latter_value = record["c"][index]
        pyplot.scatter([-0.1], [3 - index + 0.5], marker="v", color="k", s=10)
        pyplot.text(-0.1, 3 - index + 0.75, "%.3f" % former_value, va="center", ha="center", fontsize=7)
        pyplot.text(-0.1, 3 - index + 0.25, "%.3f" % latter_value, va="center", ha="center", fontsize=7)
        pyplot.pcolormesh(linspace(0.6, 1.2, 41), linspace(3 - index + 0.2, 3 - index + 0.8, 41), former,
                          cmap="PRGn", vmin=-1, vmax=1, zorder=3)
        pyplot.pcolormesh(linspace(1.8, 2.4, 41), linspace(3 - index + 0.2, 3 - index + 0.8, 41), difference,
                          cmap="PRGn", vmin=-1, vmax=1, zorder=3)
        pyplot.pcolormesh(linspace(3.0, 3.6, 41), linspace(3 - index + 0.2, 3 - index + 0.8, 41), latter,
                          cmap="PRGn", vmin=-1, vmax=1, zorder=3)
    pyplot.xlim(-0.72, 4.9)
    pyplot.ylim(-0.20, 9.2)
    pyplot.axis("off")

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.351, 0.99, "b", va="center", ha="center", fontsize=14)
    figure.text(0.702, 0.99, "c", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "main03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main04():
    task_data = load_data(task_path + "main04.pkl")

    figure = pyplot.figure(figsize=(10, 7), tight_layout=True)
    grid = pyplot.GridSpec(3, 12)

    pyplot.subplot(grid[0, :4])
    pyplot.plot([0.05, 0.95], [0.15, 0.15], color="silver", lw=10, zorder=0)
    pyplot.fill_between([0.60, 0.80], 0.03, 0.27, fc="gray", ec="k", lw=0, zorder=1)
    pyplot.text(0.70, -0.05, "cart", va="center", ha="center", fontsize=9)
    pyplot.plot([0.70, 0.50], [0.15, 0.95], color="#CC9966", lw=10, zorder=0)
    pyplot.text(0.43, 1.05, "pole", va="center", ha="center", fontsize=9)
    pyplot.scatter([0.70], [0.15], s=170, fc="silver", ec="k", lw=0, zorder=3)
    pyplot.annotate(s="", xy=(0.60, 0.15), xytext=(0.50, 0.15),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75))
    pyplot.text(0.50, 0.15, "push", va="center", ha="right", fontsize=8)
    pyplot.plot([0.60, 0.70, 0.70], [0.55, 0.15, 0.60], color="k", lw=0.75, ls="--", zorder=4)
    pyplot.annotate(s="", xy=(0.60, 0.55), xytext=(0.70, 0.60),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.75,
                                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.3"))
    pyplot.text(0.67, 0.67, "angle", va="center", ha="center", fontsize=8)
    pyplot.annotate(s="", xy=(0.36, 0.85), xytext=(0.50, 0.95),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75, shrinkA=0, shrinkB=0))
    pyplot.text(0.30, 0.75, "angular\nvelocity", va="center", ha="center", fontsize=8)
    pyplot.xlim(0, 1)
    pyplot.ylim(-0.1, 1.02)
    pyplot.axis("off")

    labels = ["default", r"$\mathcal{L}_c + \mathcal{C}$", r"$\mathcal{L}_i + \mathcal{C}$", r"$\mathcal{C}$"]
    ax = pyplot.subplot(grid[0, 4:8])
    pyplot.text(4.45, 195.5, "pass (≥ 195)", va="bottom", ha="right", fontsize=8)
    pyplot.hlines(195, -0.5, 4.5, lw=0.75, ls="--", zorder=-1)
    for index, (label, color) in enumerate(zip(labels, pyplot.get_cmap("binary")(linspace(0.0, 0.8, 4)))):
        pyplot.plot(arange(5), task_data["b"][index], lw=0.75, color="k",
                    marker="o", mfc=color, mec="k", ms=5, label=label)
    pyplot.legend(loc="lower left", fontsize=8)
    pyplot.xlabel("training error scale", fontsize=9)
    pyplot.ylabel("average training performance", fontsize=9)
    pyplot.xticks(arange(5), ["0%", "10%", "20%", "30%", "40%"], fontsize=8)
    pyplot.yticks(arange(150, 201, 10), arange(150, 201, 10), fontsize=8)
    pyplot.xlim(-0.5, 4.5)
    pyplot.ylim(150, 202.5)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    ax = pyplot.subplot(grid[0, 8:])
    for index, (label, color) in enumerate(zip(labels, pyplot.get_cmap("binary")(linspace(0.0, 0.8, 4)))):
        locations = arange(5) - 0.3 + 0.2 * index
        pyplot.bar(locations, task_data["c"][index], width=0.2, fc=color, ec="k", lw=0.75, label=label)
    pyplot.legend(loc="upper left", fontsize=8)
    pyplot.xlabel("training error scale", fontsize=9)
    pyplot.ylabel("median generation", fontsize=9)
    pyplot.xticks(arange(5), ["0%", "10%", "20%", "30%", "40%"], fontsize=8)
    pyplot.yticks(arange(0, 21, 4), arange(0, 21, 4), fontsize=8)
    pyplot.xlim(-0.5, 4.5)
    pyplot.ylim(0, 21)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    for index in range(4):
        pyplot.subplot(grid[1, index * 3: (index + 1) * 3])
        pyplot.title(labels[index], fontsize=9)
        values = task_data["d"][index].copy()
        values[values >= 195] = nan
        pyplot.pcolormesh(arange(6), arange(6), values.T, vmin=100, vmax=195, cmap="inferno")
        for location_x in range(5):
            for location_y in range(5):
                value = task_data["d"][index][location_x, location_y]
                if value >= 195:
                    pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "pass",
                                va="center", ha="center", fontsize=8)
                elif value > 140:
                    pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.1f" % value,
                                va="center", ha="center", fontsize=8)
                else:
                    pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.1f" % value, color="w",
                                va="center", ha="center", fontsize=8)
        pyplot.xlabel("training error scale", fontsize=9)
        pyplot.ylabel("evaluating error scale", fontsize=9)
        pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=8)
        pyplot.yticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=8)
        pyplot.xlim(0, 5)
        pyplot.ylim(0, 5)

    ax = pyplot.subplot(grid[2, :4])
    pyplot.title("30% training error", fontsize=9)
    pyplot.bar(arange(4) - 0.2, task_data["e"][:, 0], width=0.4, fc="w", ec="k", lw=0.75, label="max =   20")
    pyplot.bar(arange(4) + 0.2, task_data["e"][:, 1], width=0.4, fc="lightblue", ec="k", lw=0.75, label="max = 100")
    for value_index, value in enumerate(task_data["e"]):
        pyplot.text(value_index - 0.2, value[0] + 0.04, "%d" % (value[0] * 100) + "%",
                    va="center", ha="center", fontsize=8)
        pyplot.text(value_index + 0.2, value[1] + 0.04, "%d" % (value[1] * 100) + "%",
                    va="center", ha="center", fontsize=8)
    pyplot.legend(loc="upper left", fontsize=8, title="generation", title_fontsize=8)
    pyplot.xlabel("training setting", fontsize=9)
    pyplot.ylabel("pass rate (resist ≤ 30% error)", fontsize=9)
    pyplot.xticks(arange(4), labels, fontsize=8)
    pyplot.yticks(linspace(0, 1, 6), [("%d" % (v * 100)) + "%" for v in linspace(0, 1, 6)], fontsize=8)
    pyplot.xlim(-0.5, 3.5)
    pyplot.ylim(0, 1)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    markers, colors = ["b", "i", "c", "a"], ["#BF33B5", "#845EC2", "#D73222"]
    for index in range(4):
        ax = pyplot.subplot(grid[2, 4 + index * 2: 4 + (index + 1) * 2])
        pyplot.title(labels[index], fontsize=9)
        pyplot.hlines(195, 0, 5, lw=0.75, ls="--", zorder=-1)
        pyplot.text(4.9, 196, "pass", va="bottom", ha="right", fontsize=8)
        for case_index, (curve, have, total) in enumerate(task_data["f"][markers[index]]):
            if curve is not None:
                info = str(have) + " / " + str(total)
                if have < 10:
                    info = "  " + info
                pyplot.plot(arange(5) + 0.5, curve, color=colors[case_index], lw=2.5, zorder=case_index, marker="o",
                            label=str(case_index + 1) + ": " + info, alpha=0.75)
        pyplot.legend(loc="lower left", fontsize=8, title="fail type", title_fontsize=8)
        pyplot.xlabel("evaluating error scale", fontsize=9)
        pyplot.ylabel("evaluating performance", fontsize=9)
        pyplot.xticks(arange(5) + 0.5, ["0%", "", "20%", "", "40%"], fontsize=8)
        pyplot.yticks(arange(50, 201, 30), arange(50, 201, 30), fontsize=8)
        pyplot.xlim(0, 5)
        pyplot.ylim(50, 207.5)
        # noinspection PyUnresolvedReferences
        ax.spines["top"].set_visible(False)
        # noinspection PyUnresolvedReferences
        ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.356, 0.99, "b", va="center", ha="center", fontsize=14)
    figure.text(0.682, 0.99, "c", va="center", ha="center", fontsize=14)
    figure.text(0.020, 0.66, "d", va="center", ha="center", fontsize=14)
    figure.text(0.020, 0.33, "e", va="center", ha="center", fontsize=14)
    figure.text(0.356, 0.33, "f", va="center", ha="center", fontsize=14)

    pyplot.savefig(save_path + "main04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    main01()
    main02()
    main03()
    main04()
