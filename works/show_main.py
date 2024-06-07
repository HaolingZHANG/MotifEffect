"""
@Author      : Haoling Zhang
@Description : Plot all the figures in the main text.
"""
from collections import Counter
from logging import getLogger, CRITICAL
from matplotlib import pyplot, patches, rcParams
from numpy import array, arange, linspace, sum, max, argmax, nan
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
    pyplot.figure(figsize=(10, 5), tight_layout=True)
    panel = pyplot.subplot(111)
    pyplot.text(0.10, 1.93, "a", va="center", ha="left", fontsize=12)
    pyplot.text(2.10, 1.93, "b", va="center", ha="left", fontsize=12)
    pyplot.text(0.90, 2.00,
                "local network properties of 3-node coherent and incoherent motif classes",
                va="bottom", ha="center", fontsize=8)
    pyplot.text(2.50, 2.00, "noise resilience in a neural network solving a control task",
                va="bottom", ha="center", fontsize=8)
    pyplot.fill_between([0.08, 1.72], 0.02, 1.98, fc="#EFEFEF", lw=0, zorder=-1)
    pyplot.fill_between([2.08, 2.92], 0.02, 1.98, fc="#EFEFEF", lw=0, zorder=-1)

    pyplot.text(0.90, 1.93, "3-node network motif for information superposition",
                va="center", ha="center", fontsize=7)
    pyplot.hlines(1.93, 0.36, 0.56, lw=0.75, color="k")
    pyplot.hlines(1.93, 1.24, 1.44, lw=0.75, color="k")
    pyplot.annotate("", xy=(0.36, 1.84), xytext=(0.36, 1.93),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(0.90, 1.84), xytext=(0.90, 1.90),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(1.44, 1.84), xytext=(1.44, 1.93),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    motif_types = ["incoherent-loop", "coherent-loop", "collider"]
    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]
    for type_index, (motif_type, motif_order) in enumerate(zip(motif_types, math_orders)):
        motifs, info, bias = acyclic_motifs[motif_type], draw_info[motif_type], 0.11 + 0.54 * type_index
        pyplot.text(bias + 0.25, 1.80, motif_type + " ( " + motif_order + " - 1,2,3,4) ",
                    va="center", ha="center", fontsize=7)
        pyplot.fill_between(array([0.00, 0.50]) + bias, 1.57, 1.77, color=info[0], lw=0, zorder=1)
        locations_x = array([0.000, 0.125, 0.250, 0.375]) + bias + 0.025
        points_x = array([locations_x.tolist(), [1.625 for _ in range(4)]])
        pyplot.scatter(points_x[0], points_x[1], fc="w", ec="k", lw=0.75, s=20, zorder=2)
        for location in points_x[0]:
            pyplot.text(location, 1.585, "$x$", va="center", ha="center", fontsize=7)
        locations_y = array([0.000, 0.125, 0.250, 0.375]) + bias + 0.100
        points_y = array([locations_y.tolist(), [1.625 for _ in range(4)]])
        pyplot.scatter(points_y[0], points_y[1], fc="silver", ec="k", lw=0.75, s=20, zorder=2)
        for location in points_y[0]:
            pyplot.text(location, 1.585, "$y$", va="center", ha="center", fontsize=7)
        locations_z = array([0.000, 0.125, 0.250, 0.375]) + bias + 0.0625
        points_z = array([locations_z.tolist(), [1.715 for _ in range(4)]])
        pyplot.scatter(points_z[0], points_z[1], fc="k", ec="k", lw=0.75, s=20, zorder=2)
        for location in points_z[0]:
            pyplot.text(location, 1.745, "$z$", va="center", ha="center", fontsize=7)
        for motif_index, motif in enumerate(motifs):
            points = [points_x[:, motif_index], points_y[:, motif_index], points_z[:, motif_index]]
            for former, latter in motif.edges:
                flag = "+" if motif.get_edge_data(former, latter)["weight"] == 1 else "\N{MINUS SIGN}"
                pyplot.annotate("", xy=points[latter - 1], xytext=points[former - 1],
                                arrowprops=dict(arrowstyle="-|>, head_length=0.2, head_width=0.15", color="k",
                                                shrinkA=3, shrinkB=3, lw=0.75, ls=("-" if flag == "+" else ":")))
                flag_location = (points[latter - 1] + points[former - 1]) / 2.0
                if (former, latter) == (1, 2):
                    pyplot.text(flag_location[0], flag_location[1] - 0.04, flag, va="center", ha="center", fontsize=7)
                if (former, latter) == (1, 3):
                    pyplot.text(flag_location[0] - 0.02, flag_location[1], flag, va="center", ha="center", fontsize=7)
                if (former, latter) == (2, 3):
                    pyplot.text(flag_location[0] + 0.02, flag_location[1], flag, va="center", ha="center", fontsize=7)

    pyplot.plot([0.36, 0.36, 0.90, 0.90], [1.56, 1.53, 1.53, 1.56], lw=0.75, color="k")
    pyplot.annotate("", xy=(0.63, 1.47), xytext=(0.63, 1.53),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(1.44, 1.47), xytext=(1.44, 1.56),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.63, 1.44, "experimental groups (    ,    )", va="center", ha="center", fontsize=7)
    pyplot.text(1.44, 1.44, "control group (    )", va="center", ha="center", fontsize=7)
    pyplot.scatter([0.745], [1.44], s=24, marker="^", ec="k", fc="#FCB1AB", lw=0.75)
    pyplot.scatter([0.790], [1.44], s=24, marker="^", ec="k", fc="#FCE0AB", lw=0.75)
    pyplot.scatter([1.530], [1.44], s=24, marker="^", ec="k", fc="#88CCF8", lw=0.75)
    pyplot.text(0.50, 1.32,
                "estimation of representation capacity and numerical stability\n"
                "using static experiments",
                va="center", ha="center", fontsize=7)
    pyplot.text(1.30, 1.32,
                "estimation of output signature and role of x-to-y edge\n"
                "using dynamic experiments",
                va="center", ha="center", fontsize=7)
    pyplot.vlines(0.90, 1.01, 1.20, lw=0.75, ls="--", color="k")
    pyplot.vlines(0.90, 0.22, 0.90, lw=0.75, ls="--", color="k")
    pyplot.scatter([0.48], [1.20], s=24, marker="^", ec="k", fc="#FCB1AB", lw=0.75)
    pyplot.scatter([0.52], [1.20], s=24, marker="^", ec="k", fc="#FCE0AB", lw=0.75)
    pyplot.vlines(0.50, 1.14, 1.17, lw=0.75, color="k")
    pyplot.text(0.50, 1.10, "generate loops uniformly from the parameter domain",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.annotate("", xy=(0.50, 1.00), xytext=(0.50, 1.06),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.50, 0.96, "441,000 " + r"$\mathcal{L}_i$" + " samples", va="center", ha="center", fontsize=7)
    pyplot.text(0.50, 0.90, "441,000 " + r"$\mathcal{L}_c$" + " samples", va="center", ha="center", fontsize=7)
    pyplot.scatter([0.15], [0.90], s=24, marker="^", ec="k", fc="#88CCF8", lw=0.75)
    pyplot.plot([0.15, 0.15, 0.30], [0.87, 0.85, 0.85], lw=0.75, color="k")
    pyplot.plot([0.30, 0.30, 0.45, 0.45], [0.82, 0.85, 0.85, 0.87], lw=0.75, color="k")
    pyplot.plot([0.55, 0.55, 0.70, 0.70], [0.87, 0.85, 0.85, 0.82], lw=0.75, color="k")
    pyplot.text(0.30, 0.78, "minimum L2-norm difference of " + r"$\mathcal{C}$",
                va="center", ha="center", fontsize=7)
    pyplot.text(0.30, 0.73, "(detailed in Figure 2a)", va="center", ha="center", fontsize=6)
    pyplot.text(0.70, 0.78, "best Lipschitz constant", va="center", ha="center", fontsize=7)
    pyplot.annotate("", xy=(0.30, 0.63), xytext=(0.30, 0.69),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.30, 0.59, "representational capacity", va="center", ha="center", fontsize=7)
    pyplot.annotate("", xy=(0.70, 0.63), xytext=(0.70, 0.74),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.70, 0.59, "numerical stability", va="center", ha="center", fontsize=7)
    pyplot.plot([0.30, 0.30, 0.70, 0.70], [0.55, 0.52, 0.52, 0.55], lw=0.75, color="k")
    pyplot.annotate("", xy=(0.50, 0.47), xytext=(0.50, 0.52),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.50, 0.43, "form distributions separately for two types of values",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.annotate("", xy=(0.50, 0.33), xytext=(0.50, 0.39),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.50, 0.27,
                "investigate the trade-off differences\nbetween two loop types based on the distributions",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.text(0.90, 0.96, "select by maximizing output difference",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.annotate("", xy=(1.18, 0.93), xytext=(0.65, 0.93),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(1.30, 0.96, "400 " + r"$\mathcal{L}_i$" + " samples", va="center", ha="center", fontsize=7)
    pyplot.text(1.30, 0.90, "400 " + r"$\mathcal{L}_c$" + " samples", va="center", ha="center", fontsize=7)
    pyplot.annotate("", xy=(1.30, 0.80), xytext=(1.30, 0.87),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.scatter([1.50], [0.90], s=24, marker="^", ec="k", fc="#88CCF8", lw=0.75)
    pyplot.plot([1.30, 1.50, 1.50], [0.85, 0.85, 0.87], lw=0.75, color="k")
    pyplot.text(1.30, 0.78, "apply the adversarial escape process for each loop sample",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.text(1.30, 0.73, "(detailed in Figure 3a)", va="center", ha="center", fontsize=6)
    pyplot.plot([1.10, 1.25, 1.25], [0.66, 0.66, 0.69], lw=0.75, color="k")
    pyplot.plot([1.35, 1.35, 1.50], [0.69, 0.66, 0.66], lw=0.75, color="k")
    pyplot.annotate("", xy=(1.10, 0.43), xytext=(1.10, 0.66),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(1.50, 0.60), xytext=(1.50, 0.66),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(1.50, 0.57, "predominant curvature proportion", va="center", ha="center", fontsize=7)
    pyplot.text(1.50, 0.52, "(detailed in Figure 4a,4b)", va="center", ha="center", fontsize=6)
    pyplot.annotate("", xy=(1.50, 0.43), xytext=(1.50, 0.49),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(1.10, 0.40, "observe parameter changes",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.text(1.50, 0.40, "observe output changes",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.plot([1.10, 1.10, 1.50, 1.50], [0.37, 0.35, 0.35, 0.37], lw=0.75, color="k")
    pyplot.annotate("", xy=(1.30, 0.29), xytext=(1.30, 0.35),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(1.30, 0.245, "analyze the role played by x-to-y edge in two loop types",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.vlines(0.50, 0.09, 0.21, lw=0.75, color="k")
    pyplot.vlines(1.30, 0.09, 0.21, lw=0.75, color="k")
    pyplot.annotate("", xy=(1.08, 0.09), xytext=(1.30, 0.09),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(0.72, 0.09), xytext=(0.50, 0.09),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.90, 0.09, "cross-validate the findings\nin two experiments",
                va="center", ha="center", color="#C0504D", fontsize=7)

    pyplot.annotate("", xy=(2.05, 1.0), xytext=(1.76, 1.0),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(1.90, 1.02, "local to global", va="bottom", ha="center", fontsize=8)

    centers = [(2.30, 1.80), (2.70, 1.80), (2.70, 1.40), (2.30, 1.40)]
    biases = array([[-0.042, +0.063], [+0.042, +0.063], [+0.042, -0.063], [-0.042, -0.063]])
    for center in centers:
        panel.add_patch(patches.Ellipse(xy=center, width=0.22, height=0.30, ec="k", fc="w", lw=0.75))
    pyplot.annotate("", xy=(2.41, 1.80), xytext=(2.59, 1.80),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(2.41, 1.40), xytext=(2.59, 1.40),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(2.70, 1.65), xytext=(2.70, 1.55),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(2.30, 1.65), xytext=(2.30, 1.55),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    for index, center in enumerate(centers):
        if index != 1:
            for bias in biases:
                sub_center = (center[0] + bias[0], center[1] + bias[1])
                pyplot.scatter([sub_center[0] - 0.02, sub_center[0] + 0.00, sub_center[0] + 0.02],
                               [sub_center[1] - 0.03, sub_center[1] - 0.03, sub_center[1] - 0.03],
                               ec="k", fc="#F4FCE9", lw=0.5, s=6, zorder=3)
                pyplot.scatter([sub_center[0] - 0.01, sub_center[0] + 0.01],
                               [sub_center[1] + 0.03, sub_center[1] + 0.03],
                               ec="k", fc="#55D17D", lw=0.5, s=6, zorder=3)
    for index, center in enumerate(centers):
        if index == 0 or index == 2:
            sub_center = (center[0] + biases[0, 0], center[1] + biases[0, 1])
            pyplot.plot([sub_center[0] - 0.02, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.00, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.00, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.02, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
            sub_center = (center[0] + biases[1, 0], center[1] + biases[1, 1])
            pyplot.scatter([sub_center[0] - 0.01], [sub_center[1]], ec="k", fc="#BCE292",
                           lw=0.5, s=6, zorder=3)
            pyplot.plot([sub_center[0] - 0.02, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.00, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] - 0.01, sub_center[0] - 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] - 0.01, sub_center[0] + 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.02, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
            sub_center = (center[0] + biases[2, 0], center[1] + biases[2, 1])
            pyplot.scatter([sub_center[0] + 0.00], [sub_center[1]], ec="k", fc="#BCE292",
                           lw=0.5, s=6, zorder=3)
            pyplot.plot([sub_center[0] - 0.02, sub_center[0] + 0.00], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.00, sub_center[0] + 0.00], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.02, sub_center[0] + 0.00], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.00, sub_center[0] - 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.00, sub_center[0] + 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
            sub_center = (center[0] + biases[3, 0], center[1] + biases[3, 1])
            pyplot.scatter([sub_center[0] + 0.01], [sub_center[1]], ec="k", fc="#BCE292",
                           lw=0.5, s=6, zorder=3)
            pyplot.plot([sub_center[0] - 0.02, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.00, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.02, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                        color="k", lw=0.5, zorder=2)
            pyplot.plot([sub_center[0] + 0.01, sub_center[0] + 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                        color="k", lw=0.5, zorder=2)
    center = centers[1]
    for bias, value in zip(biases, [0.2, 0.8, 0.5, 0.4]):
        if value < 0.8:
            pyplot.text(center[0] + bias[0], center[1] + bias[1], str(value),
                        va="center", ha="center", color="#999999", fontsize=7, zorder=3)
        else:
            pyplot.text(center[0] + bias[0], center[1] + bias[1], str(value),
                        va="center", ha="center", color="#000000", fontsize=7, zorder=3)
    center = centers[2]
    for index, bias in enumerate(biases):
        if index != 1:
            sub_center = (center[0] + bias[0], center[1] + bias[1])
            pyplot.fill_between([sub_center[0] - 0.028, sub_center[0] + 0.028],
                                sub_center[1] - 0.039, sub_center[1] + 0.039,
                                lw=0, fc="w", alpha=0.75, zorder=4)
    center = centers[3]
    sub_center = (center[0] + biases[0, 0], center[1] + biases[0, 1])
    pyplot.scatter([sub_center[0] + 0.01], [sub_center[1]], ec="k", fc="#BCE292",
                   lw=0.5, s=6, zorder=3)
    pyplot.plot([sub_center[0] - 0.02, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.03],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.00, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.03],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.00, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.02, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.01, sub_center[0] + 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                color="k", lw=0.5, zorder=2)
    sub_center = (center[0] + biases[1, 0], center[1] + biases[1, 1])
    pyplot.scatter([sub_center[0] - 0.01], [sub_center[1]], ec="k", fc="#BCE292",
                   lw=0.5, s=6, zorder=3)
    pyplot.plot([sub_center[0] - 0.02, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.00, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] - 0.01, sub_center[0] - 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] - 0.01, sub_center[0] + 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.02, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.03],
                color="k", lw=0.5, zorder=2)
    sub_center = (center[0] + biases[2, 0], center[1] + biases[2, 1])
    pyplot.scatter([sub_center[0] - 0.01], [sub_center[1]], ec="k", fc="#BCE292",
                   lw=0.5, s=6, zorder=3)
    pyplot.plot([sub_center[0] - 0.02, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.00, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] - 0.01, sub_center[0] - 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.02, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.03],
                color="k", lw=0.5, zorder=2)
    sub_center = (center[0] + biases[3, 0], center[1] + biases[3, 1])
    pyplot.scatter([sub_center[0] - 0.01], [sub_center[1]], ec="k", fc="#BCE292",
                   lw=0.5, s=6, zorder=3)
    pyplot.scatter([sub_center[0] + 0.01], [sub_center[1]], ec="k", fc="#BCE292",
                   lw=0.5, s=6, zorder=3)
    pyplot.plot([sub_center[0] - 0.02, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.00, sub_center[0] - 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.00, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.02, sub_center[0] + 0.01], [sub_center[1] - 0.03, sub_center[1] + 0.00],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] + 0.01, sub_center[0] + 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                color="k", lw=0.5, zorder=2)
    pyplot.plot([sub_center[0] - 0.01, sub_center[0] - 0.01], [sub_center[1] + 0.00, sub_center[1] + 0.03],
                color="k", lw=0.5, zorder=2)
    pyplot.text(2.50, 1.20, "workflow of NeuroEvolution", va="center", ha="center", fontsize=7)
    pyplot.text(2.50, 1.84, "evaluate", va="center", ha="center", fontsize=7)
    pyplot.text(2.72, 1.60, "select", va="center", ha="left", fontsize=7)
    pyplot.text(2.50, 1.36, "evolve", va="center", ha="center", fontsize=7)
    pyplot.text(2.28, 1.60, "repeat", va="center", ha="right", fontsize=7)
    pyplot.hlines(1.16, 2.33, 2.67, color="k", lw=0.75)
    pyplot.vlines(2.50, 1.12, 1.16, color="k", lw=0.75)
    pyplot.vlines(2.50, 1.00, 1.04, color="k", lw=0.75)
    pyplot.text(2.50, 1.08, "intervene motif composition at the evolution stage",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.plot([2.2, 2.2, 2.8, 2.8], [0.98, 1.00, 1.00, 0.98], color="k", lw=0.75)
    pyplot.annotate("", xy=(2.20, 0.95), xytext=(2.20, 1.00),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(2.40, 0.95), xytext=(2.40, 1.00),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(2.60, 0.95), xytext=(2.60, 1.00),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(2.80, 0.95), xytext=(2.80, 1.00),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(2.20, 0.92, "baseline", va="center", ha="center", fontsize=7)
    pyplot.text(2.40, 0.92, r"[ $\mathcal{L}_c + \mathcal{C}$ ]", va="center", ha="center", fontsize=7)
    pyplot.text(2.60, 0.92, r"[ $\mathcal{L}_i + \mathcal{C}$ ]", va="center", ha="center", fontsize=7)
    pyplot.text(2.80, 0.92, r"$\mathcal{C}$", va="center", ha="center", fontsize=7)
    pyplot.scatter([2.16], [0.87], s=24, marker="^", ec="k", fc="#FCB1AB", lw=0.75)
    pyplot.scatter([2.20], [0.87], s=24, marker="^", ec="k", fc="#FCE0AB", lw=0.75)
    pyplot.scatter([2.24], [0.87], s=24, marker="^", ec="k", fc="#88CCF8", lw=0.75)
    pyplot.scatter([2.38], [0.87], s=24, marker="^", ec="k", fc="#FCE0AB", lw=0.75)
    pyplot.scatter([2.42], [0.87], s=24, marker="^", ec="k", fc="#88CCF8", lw=0.75)
    pyplot.scatter([2.58], [0.87], s=24, marker="^", ec="k", fc="#FCB1AB", lw=0.75)
    pyplot.scatter([2.62], [0.87], s=24, marker="^", ec="k", fc="#88CCF8", lw=0.75)
    pyplot.scatter([2.80], [0.87], s=24, marker="^", ec="k", fc="#88CCF8", lw=0.75)
    pyplot.plot([2.20, 2.20, 2.80, 2.80], [0.84, 0.81, 0.81, 0.84], color="k", lw=0.75)
    pyplot.vlines([2.40, 2.60], 0.81, 0.84, color="k", lw=0.75)
    pyplot.vlines(2.50, 0.78, 0.81, color="k", lw=0.75)
    pyplot.text(2.50, 0.73, "produce agents for the control task in noisy environments",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.annotate("", xy=(2.50, 0.63), xytext=(2.50, 0.69),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.hlines(0.50, 2.15, 2.40, lw=2.0, color="silver", zorder=2)  # TODO
    pyplot.plot([2.30, 2.23], [0.50, 0.60], lw=3.0, color="#CC9966", zorder=3)
    pyplot.fill_between([2.27, 2.33], 0.48, 0.52, fc="gray", lw=0, zorder=4)
    pyplot.scatter([2.30], [0.50], fc="silver", s=20, lw=0, zorder=4)
    pyplot.plot([2.265, 2.300, 2.300], [0.550, 0.500, 0.570], lw=0.5, ls="--", color="k", zorder=4)
    pyplot.annotate("", xy=(2.265, 0.550), xytext=(2.300, 0.570),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.5,
                                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.3"))
    pyplot.text(2.277, 0.585, r"${\rm \theta}$", va="center", ha="center", fontsize=7)
    pyplot.text(2.275, 0.46, "cart-pole balance task", va="top", ha="center", fontsize=7)
    pyplot.text(2.275, 0.39, "(detailed in Figure 5a)", va="center", ha="center", fontsize=6)
    pyplot.vlines(2.500, 0.505, 0.545, color="k", lw=0.75)
    pyplot.hlines(0.525, 2.487, 2.513, color="k", lw=0.75)
    pyplot.fill_between([2.66, 2.69], 0.50, 0.52, ec="k", fc="w", lw=0.75)
    pyplot.fill_between([2.71, 2.74], 0.50, 0.54, ec="k", fc="w", lw=0.75)
    pyplot.fill_between([2.76, 2.79], 0.50, 0.56, ec="k", fc="w", lw=0.75)
    pyplot.fill_between([2.81, 2.84], 0.50, 0.58, ec="k", fc="w", lw=0.75)
    pyplot.text(2.625, 0.51, "0%", va="bottom", ha="center", fontsize=6)
    pyplot.text(2.674, 0.53, "10%", va="bottom", ha="center", fontsize=6)
    pyplot.text(2.725, 0.55, "20%", va="bottom", ha="center", fontsize=6)
    pyplot.text(2.775, 0.57, "30%", va="bottom", ha="center", fontsize=6)
    pyplot.text(2.825, 0.59, "40%", va="bottom", ha="center", fontsize=6)
    pyplot.hlines(0.50, 2.60, 2.85, lw=0.75, color="k")
    pyplot.text(2.725, 0.46, "noise level", va="top", ha="center", fontsize=7)
    pyplot.annotate("", xy=(2.50, 0.38), xytext=(2.50, 0.44),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(2.50, 0.30,
                "assess agents trained at a noise level from 0% to 40%\n"
                "under 0% - 40% noise levels at the evaluation stage",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.vlines(2.50, 0.21, 0.24, color="k", lw=0.75)
    pyplot.plot([2.30, 2.70], [0.21, 0.21], color="k", lw=0.75)
    pyplot.annotate("", xy=(2.30, 0.15), xytext=(2.30, 0.21),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(2.70, 0.15), xytext=(2.70, 0.21),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(2.30, 0.09, "agents learn the target and\nresist the introduced noise",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.text(2.70, 0.09, "agents learn the target and\nthe introduced noise together",
                va="center", ha="center", color="#C0504D", fontsize=7)
    pyplot.axis("off")
    pyplot.xlim(0.07, 2.93)
    pyplot.ylim(0.00, 2.03)

    pyplot.savefig(save_path + "main01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_02():
    """
    Create Figure 2 in the main text.
    """
    motif_types = ["incoherent-loop", "coherent-loop", "collider"]
    math_orders = [r"$\mathcal{L}_i$", r"$\mathcal{L}_c$", r"$\mathcal{C}$"]

    task_data = load_data(sort_path + "main02.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    grid = pyplot.GridSpec(8, 10)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[:, :4])
    locations = task_data["a"]

    # for better visual effects.
    locations[0, 0] -= 0.04
    locations[0, 1] += 0.05

    pyplot.text(locations[0, 0], locations[0, 1] + 0.03, math_orders[0] + "-1",
                va="center", ha="center", fontsize=7)
    pyplot.scatter([locations[0, 0]], [locations[0, 1]], ec="k", fc="#FCB1AB", zorder=3)
    pyplot.plot([locations[0, 0], locations[101, 0]], [locations[0, 1], locations[101, 1]],
                lw=0.75, color="k", zorder=1)
    pyplot.annotate("", xy=((locations[0, 0] + locations[101, 0]) / 2.0, (locations[0, 1] + locations[101, 1]) / 2.0),
                    xytext=(0.85, 0.92), arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=3, shrinkB=3, lw=0.75))
    pyplot.text(0.73, 0.88,
                "minimum L2-norm difference\n" + "of " + r"$\mathcal{C}$" +
                " for a fixed " + r"$\mathcal{L}_i$" + " sample",
                va="center", ha="center", fontsize=7)
    pyplot.scatter([locations[1, 0]], [locations[1, 1]], ec="k", fc="#88CCF8", zorder=2)
    pyplot.text(locations[1, 0] - 0.06, locations[1, 1], "start", va="center", ha="center", fontsize=7)
    pyplot.scatter([locations[101, 0]], [locations[101, 1]], ec="k", fc="#88CCF8", zorder=2)
    pyplot.text(locations[101, 0] + 0.06, locations[101, 1], "stop", va="center", ha="center", fontsize=7)
    pyplot.plot(locations[1: 101, 0], locations[1: 101, 1], color="k", lw=0.75, ls="--", zorder=1)
    pyplot.text(locations[51, 0] + 0.050, locations[51, 1], math_orders[-1] + "-1",
                va="center", ha="center", fontsize=7)
    pyplot.scatter([locations[102, 0]], [locations[102, 1]], ec="k", fc="#88CCF8", zorder=2)
    pyplot.text(locations[102, 0], locations[102, 1] - 0.04, "start", va="center", ha="center", fontsize=7)
    pyplot.scatter([locations[202, 0]], [locations[202, 1]], ec="k", fc="#88CCF8", zorder=2)
    pyplot.text(locations[202, 0], locations[202, 1] - 0.04, "stop", va="center", ha="center", fontsize=7)
    pyplot.plot(locations[102: 202, 0], locations[102: 202, 1], color="k", lw=0.75, ls="--", zorder=1)
    pyplot.text(locations[152, 0], locations[152, 1] + 0.025, math_orders[-1] + "-2",
                va="center", ha="center", fontsize=7)
    pyplot.scatter([locations[203, 0]], [locations[203, 1]], ec="k", fc="#88CCF8", zorder=2)
    pyplot.text(locations[203, 0], locations[203, 1] - 0.04, "start", va="center", ha="center", fontsize=7)
    pyplot.scatter([locations[303, 0]], [locations[303, 1]], ec="k", fc="#88CCF8", zorder=2)
    pyplot.text(locations[303, 0] - 0.060, locations[303, 1], "stop", va="center", ha="center", fontsize=7)
    pyplot.plot(locations[203: 303, 0], locations[203: 303, 1], color="k", lw=0.75, ls="--", zorder=1)
    pyplot.text(locations[253, 0] + 0.060, locations[253, 1], math_orders[-1] + "-3",
                va="center", ha="center", fontsize=7)
    pyplot.scatter([locations[304, 0]], [locations[304, 1]], ec="k", fc="#88CCF8", zorder=2)
    pyplot.text(locations[304, 0], locations[304, 1] - 0.04, "start", va="center", ha="center", fontsize=7)
    pyplot.scatter([locations[404, 0]], [locations[404, 1]], ec="k", fc="#88CCF8", zorder=2)
    pyplot.text(locations[404, 0], locations[404, 1] - 0.04, "stop", va="center", ha="center", fontsize=7)
    pyplot.plot(locations[304: 404, 0], locations[304: 404, 1], color="k", lw=0.75, ls="--", zorder=1)
    pyplot.text(locations[354, 0], locations[354, 1] + 0.025, math_orders[-1] + "-4",
                va="center", ha="center", fontsize=7)

    pyplot.fill_between([0.53, 0.94], -0.02, + 0.44, ec="k", fc="#EEEEEE", lw=0.75)
    pyplot.text(0.55, 0.41, "output similarity criterion:", va="center", ha="left", fontsize=7)
    pyplot.text(0.60, 0.38, "L2-norm (MSE) loss", va="center", ha="left", fontsize=7)
    pyplot.text(0.55, 0.31, "visualization:", va="center", ha="left", fontsize=7)
    pyplot.text(0.60, 0.28, "UMAP", va="center", ha="left", fontsize=7)
    pyplot.text(0.55, 0.21, "parameters in loop sample:", va="center", ha="left", fontsize=7)
    pyplot.text(0.60, 0.18, "motif type = " + math_orders[0], va="center", ha="left", fontsize=7)
    pyplot.text(0.60, 0.15, "motif structure index = " + r"$1$", va="center", ha="left", fontsize=7)
    pyplot.text(0.60, 0.12, "weight " + r"$x \rightarrow y$" + " = " + r"$-1.000$",
                va="center", ha="left", fontsize=7)
    pyplot.text(0.60, 0.09, "weight " + r"$x \rightarrow z$" + " = " + r"$+0.652$",
                va="center", ha="left", fontsize=7)
    pyplot.text(0.60, 0.06, "weight " + r"$y \rightarrow z$" + " = " + r"$+1.000$",
                va="center", ha="left", fontsize=7)
    pyplot.text(0.60, 0.03, "bias " + r"$x \rightarrow y$" + " = " + r"$+1.000$",
                va="center", ha="left", fontsize=7)
    pyplot.text(0.60, 0.00, "bias " + r"$x,y \rightarrow z$" + " = " + r"$+0.337$",
                va="center", ha="left", fontsize=7)

    pyplot.xlim(-0.02, +1.02)
    pyplot.ylim(-0.02, +1.02)
    pyplot.axis("off")

    results = task_data["b"]

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[:2, 4:7])
    for location, (motif_type, color) in enumerate(zip(motif_types[:-1], ["#C27C77", "#C2A976"])):
        x = results[motif_type][0]
        y = results[motif_type][1] / sum(results[motif_type][1])
        pyplot.plot(x, y, color=draw_info[motif_type][0], lw=2, label=math_orders[location])
    pyplot.legend(loc="upper right", fontsize=7)
    pyplot.xlabel("best Lipschitz constant", fontsize=8)
    pyplot.ylabel("proportion", fontsize=8)
    pyplot.xticks(linspace(0.6, 2.4, 7),
                  ["%.1f" % v for v in linspace(0.6, 2.4, 7)], fontsize=7)
    pyplot.yticks(linspace(0.00, 0.08, 3),
                  ["%d" % (v * 100) + "%" for v in linspace(0.00, 0.08, 3)], fontsize=7)
    pyplot.xlim(0.6, 2.4)
    pyplot.ylim(0, 0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    results = task_data["c"]

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[:2, 7:])
    for location, motif_type in enumerate(motif_types[:-1]):
        x = results[motif_type][0]
        y = results[motif_type][1] / sum(results[motif_type][1])
        pyplot.plot(x, y, color=draw_info[motif_type][0], lw=2, label=math_orders[location])
    pyplot.legend(loc="upper right", fontsize=7)
    pyplot.xlabel(r"minimum L2-norm difference of $\mathcal{C}$", fontsize=8)
    pyplot.ylabel("proportion", fontsize=8)
    pyplot.xticks(linspace(0.00, 0.03, 7),
                  ["%.3f" % v for v in linspace(0.00, 0.03, 7)], fontsize=7)
    pyplot.yticks(linspace(0.00, 0.08, 3),
                  ["%d" % (v * 100) + "%" for v in linspace(0.00, 0.08, 3)], fontsize=7)
    pyplot.xlim(0, 0.03)
    pyplot.ylim(0, 0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:, 4:7])
    pyplot.pcolormesh(linspace(0.00, 0.03, 100), linspace(0.50, 2.50, 100),
                      task_data["d"].T, vmin=0.02, vmax=1, cmap="Purples", shading="gouraud")
    x_values, y_values = linspace(0.00, 0.03, 100), []
    for values in task_data["d"]:
        if max(values) < 0.02:
            break
        y_values.append(linspace(0.50, 2.50, 100)[argmax(values)])
    pyplot.plot(x_values[:len(y_values)], y_values, color="k", lw=0.75, ls=":")
    pyplot.text(0.0278, 1.06, "density", va="center", ha="center", fontsize=7)
    colors, locations = pyplot.get_cmap("Purples")(linspace(0.02, 1, 41)), linspace(0.6, 1.0, 41)
    for color, former, latter in zip(colors, locations[:-1], locations[1:]):
        pyplot.fill_between([0.0275, 0.0285], former, latter, fc=color, lw=0, zorder=1)
    for location, info in zip(linspace(0.65, 1.0, 5), linspace(0, 1, 5)):
        pyplot.hlines(location, 0.0270, 0.0275, lw=0.75, color="k", zorder=2)
        pyplot.text(0.0268, location, ("%d" % (info * 100)) + "%", va="center", ha="right", fontsize=7)
    pyplot.plot([0.0275, 0.0285, 0.0285, 0.0275, 0.0275], [0.65, 0.65, 1.00, 1.00, 0.65], lw=0.75, color="k", zorder=3)
    pyplot.xlabel(r"minimum L2-norm difference of $\mathcal{C}$ for $\mathcal{L}_i$", fontsize=8)
    pyplot.ylabel("best Lipschitz constant", fontsize=8)
    pyplot.xlim(0.00, 0.03)
    pyplot.ylim(0.60, 2.40)
    pyplot.xticks(linspace(0.00, 0.03, 7),
                  ["%.3f" % v for v in linspace(0.00, 0.03, 7)], fontsize=7)
    pyplot.yticks(linspace(0.60, 2.40, 9),
                  ["%.1f" % v for v in linspace(0.60, 2.40, 9)], fontsize=7)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:, 7:])
    pyplot.pcolormesh(linspace(0.00, 0.03, 100), linspace(0.50, 2.50, 100),
                      task_data["e"].T, vmin=0.02, vmax=1, cmap="Purples", shading="gouraud")
    x_values, y_values = linspace(0.00, 0.03, 100), []
    for values in task_data["e"]:
        if max(values) < 0.02:
            break
        y_values.append(linspace(0.50, 2.50, 100)[argmax(values)])
    pyplot.plot(x_values[:len(y_values)], y_values, color="k", lw=0.75, ls=":")
    pyplot.text(0.0278, 1.06, "density", va="center", ha="center", fontsize=7)
    colors, locations = pyplot.get_cmap("Purples")(linspace(0.02, 1, 41)), linspace(0.6, 1.0, 41)
    for color, former, latter in zip(colors, locations[:-1], locations[1:]):
        pyplot.fill_between([0.0275, 0.0285], former, latter, fc=color, lw=0, zorder=1)
    for location, info in zip(linspace(0.65, 1.0, 5), linspace(0, 1, 5)):
        pyplot.hlines(location, 0.0270, 0.0275, lw=0.75, color="k", zorder=2)
        pyplot.text(0.0268, location, ("%d" % (info * 100)) + "%", va="center", ha="right", fontsize=7)
    pyplot.plot([0.0275, 0.0285, 0.0285, 0.0275, 0.0275], [0.65, 0.65, 1.00, 1.00, 0.65], lw=0.75, color="k", zorder=3)
    pyplot.xlabel(r"minimum L2-norm difference of $\mathcal{C}$ for $\mathcal{L}_c$", fontsize=8)
    pyplot.ylabel("best Lipschitz constant", fontsize=8)
    pyplot.xlim(0.00, 0.03)
    pyplot.ylim(0.60, 2.40)
    pyplot.xticks(linspace(0.00, 0.03, 7),
                  ["%.3f" % v for v in linspace(0.00, 0.03, 7)], fontsize=7)
    pyplot.yticks(linspace(0.60, 2.40, 9),
                  ["%.1f" % v for v in linspace(0.60, 2.40, 9)], fontsize=7)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.386, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.698, 0.99, "c", va="center", ha="center", fontsize=12)
    figure.text(0.386, 0.75, "d", va="center", ha="center", fontsize=12)
    figure.text(0.698, 0.75, "e", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "main02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_03():
    """
    Create Figure 3 in the main text.
    """
    task_data = load_data(sort_path + "main03.pkl")

    figure = pyplot.figure(figsize=(10, 6), tight_layout=True)
    grid = pyplot.GridSpec(5, 4)

    # noinspection PyTypeChecker
    panel = pyplot.subplot(grid[:2, :])

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

    pyplot.xlim(-0.11, 9.70)
    pyplot.ylim(-0.80, 1.80)
    pyplot.axis("off")

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:, 0])
    pyplot.fill_between([-0.5, 1.5], -0.03, 1.03, lw=0, fc="#FCB1AB", alpha=0.3, zorder=0)
    pyplot.boxplot([task_data["b"][0], task_data["b"][1]], positions=[0, 1],
                   showmeans=False, patch_artist=True, widths=0.3, notch=True, medianprops=dict(lw=1.5, color="k"),
                   flierprops=dict(mec="k", mfc="w", ms=3, lw=0.75), boxprops=dict(lw=0.75, ec="k", fc="w"))
    pyplot.xlabel("weight on " + r"$x \rightarrow y$" + " of " + r"$\mathcal{L}_i$", fontsize=8)
    pyplot.ylabel("utilization", fontsize=8)
    pyplot.xticks([0, 1], ["before escaping", "after escaping"], fontsize=7)
    pyplot.yticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)], fontsize=7)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(-0.03, 1.03)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:, 1])
    pyplot.fill_between([-0.5, 1.5], -0.03, 1.03, lw=0, fc="#FCB1AB", alpha=0.3, zorder=0)
    pyplot.boxplot([task_data["c"][0], task_data["c"][1]], positions=[0, 1],
                   showmeans=False, patch_artist=True, widths=0.3, notch=True, medianprops=dict(lw=1.5, color="k"),
                   flierprops=dict(mec="k", mfc="w", ms=3, lw=0.75), boxprops=dict(lw=0.75, ec="k", fc="w"))
    pyplot.xlabel("bias on " + r"$x \rightarrow y$" + " of " + r"$\mathcal{L}_i$", fontsize=8)
    pyplot.ylabel("utilization", fontsize=8)
    pyplot.xticks([0, 1], ["before escaping", "after escaping"], fontsize=7)
    pyplot.yticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)], fontsize=7)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(-0.03, 1.03)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:, 2])
    pyplot.fill_between([-0.5, 1.5], -0.03, 1.03, lw=0, fc="#FCE0AB", alpha=0.3, zorder=0)
    pyplot.boxplot([task_data["d"][0], task_data["d"][1]], positions=[0, 1],
                   showmeans=False, patch_artist=True, widths=0.3, notch=True, medianprops=dict(lw=1.5, color="k"),
                   flierprops=dict(mec="k", mfc="w", ms=3, lw=0.75), boxprops=dict(lw=0.75, ec="k", fc="w"))
    pyplot.xlabel("weight on " + r"$x \rightarrow y$" + " of " + r"$\mathcal{L}_c$", fontsize=8)
    pyplot.ylabel("utilization", fontsize=8)
    pyplot.xticks([0, 1], ["before escaping", "after escaping"], fontsize=7)
    pyplot.yticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)], fontsize=7)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(-0.03, 1.03)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:, 3])
    pyplot.fill_between([-0.5, 1.5], -0.03, 1.03, lw=0, fc="#FCE0AB", alpha=0.3, zorder=0)
    pyplot.boxplot([task_data["e"][0], task_data["e"][1]], positions=[0, 1],
                   showmeans=False, patch_artist=True, widths=0.3, notch=True, medianprops=dict(lw=1.5, color="k"),
                   flierprops=dict(mec="k", mfc="w", ms=3, lw=0.75), boxprops=dict(lw=0.75, ec="k", fc="w"))
    pyplot.xlabel("bias on " + r"$x \rightarrow y$" + " of " + r"$\mathcal{L}_c$", fontsize=8)
    pyplot.ylabel("utilization", fontsize=8)
    pyplot.xticks([0, 1], ["before escaping", "after escaping"], fontsize=7)
    pyplot.yticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)], fontsize=7)
    pyplot.xlim(-0.5, 1.5)
    pyplot.ylim(-0.03, 1.03)

    figure.align_labels()
    figure.text(0.020, 0.98, "a", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.63, "b", va="center", ha="center", fontsize=12)
    figure.text(0.275, 0.63, "c", va="center", ha="center", fontsize=12)
    figure.text(0.517, 0.63, "d", va="center", ha="center", fontsize=12)
    figure.text(0.757, 0.63, "e", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "main03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_04():
    """
    Create Figure 4 in the main text.
    """
    task_data = load_data(sort_path + "main04.pkl")

    x, y = linspace(0.2, 0.8, 101), linspace(0.2, 0.8, 101)

    figure = pyplot.figure(figsize=(10, 6.8), tight_layout=True)
    grid = pyplot.GridSpec(14, 2)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[:5, 0])
    former, latter = task_data["a"][0], task_data["a"][1]
    pyplot.text(1.50, 1.7, "+", va="center", ha="center", fontsize=12)
    pyplot.text(3.50, 1.7, "=", va="center", ha="center", fontsize=12)
    pyplot.annotate("", xy=(0.50, 1.20), xytext=(0.50, 1.00),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(4.50, 1.20), xytext=(4.50, 1.00),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.50, 0.80, "curvature\nfeature", va="center", ha="center", fontsize=7)
    pyplot.text(4.50, 0.80, "curvature\nfeature", va="center", ha="center", fontsize=7)
    pyplot.text(2.50, 0.80, "change in predominant curvature proportion",
                va="center", ha="center", fontsize=7)
    labels = ["landscape of " + r"$\mathcal{L}_i$" + "\nbefore escaping",
              "adjustment of " + r"$\mathcal{L}_i$" + "\nduring escaping",
              "landscape of " + r"$\mathcal{L}_i$" + "\nafter escaping"]
    for index, landscape, info in zip([0.0, 2.0, 4.0], [former, latter - former, latter], labels):
        pyplot.pcolormesh(x + index, y + 1.2, landscape, vmin=-1, vmax=1, cmap="PRGn", shading="gouraud")
        pyplot.plot([0.2 + index, 0.8 + index, 0.8 + index, 0.2 + index, 0.2 + index],
                    [1.4, 1.4, 2.0, 2.0, 1.4], lw=0.75, color="k", zorder=2)
        pyplot.text(index + 0.50, 2.30, info, va="center", ha="center", fontsize=7)
        pyplot.text(index + 0.50, 1.30, "$x$", va="center", ha="center", fontsize=8)
        pyplot.text(index + 0.10, 1.70, "$y$", va="center", ha="center", fontsize=8)
        pyplot.text(index + 0.10, 1.30, "$-1$", va="center", ha="center", fontsize=8)
        pyplot.text(index + 0.10, 2.05, "$+1$", va="center", ha="center", fontsize=8)
        pyplot.text(index + 0.90, 1.30, "$+1$", va="center", ha="center", fontsize=8)
        pyplot.hlines(1.30, index + 0.2, index + 0.4, color="k", lw=0.75, ls="--")
        pyplot.hlines(1.30, index + 0.6, index + 0.8, color="k", lw=0.75, ls="--")
        pyplot.vlines(index + 0.10, 1.4, 1.6, color="k", lw=0.75, ls="--")
        pyplot.vlines(index + 0.10, 1.8, 2.0, color="k", lw=0.75, ls="--")

    # noinspection PyArgumentEqualDefault
    locations, colors = linspace(1.2, 3.8, 51), pyplot.get_cmap("PRGn")(linspace(0, 1, 50))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 1.02, 1.16, fc=color, lw=0, zorder=1)
    pyplot.plot([1.2, 1.2, 3.8, 3.8, 1.2], [1.02, 1.16, 1.16, 1.02, 1.02], lw=0.75, color="k", zorder=2)
    pyplot.text(1.18, 1.09, "$-1$", va="center", ha="right", fontsize=8)
    pyplot.text(3.82, 1.09, "$+1$", va="center", ha="left", fontsize=8)
    pyplot.text(2.50, 1.09, "$z$", va="center", ha="center", fontsize=8)

    for index, landscape in zip([0.0, 4.0], [task_data["a"][2], task_data["a"][3]]):
        pyplot.pcolormesh(x + index, y - 0.15, landscape, vmin=-1, vmax=1, cmap="binary", shading="gouraud")
        pyplot.plot([0.2 + index, 0.8 + index, 0.8 + index, 0.2 + index, 0.2 + index],
                    [0.05, 0.05, 0.65, 0.65, 0.06], lw=0.75, color="k", zorder=2)
    pyplot.plot([1.20, 1.20, 3.80, 3.80, 1.20], [0.00, 0.70, 0.70, 0.00, 0.00], lw=0.75, color="k", zorder=2)
    pyplot.fill_between([1.20, 3.80], 0.48, 0.70, lw=0, fc="ivory", zorder=1)
    pyplot.vlines(2.10, 0.00, 0.70, lw=0.75, color="k", zorder=2)
    pyplot.vlines(2.60, 0.00, 0.70, lw=0.75, color="k", zorder=2)
    pyplot.hlines(0.48, 1.20, 3.80, lw=0.75, color="k", zorder=2)
    pyplot.text(1.25, 0.60, "label", va="center", ha="left", fontsize=7)
    pyplot.text(2.15, 0.60, "color", va="center", ha="left", fontsize=7)
    pyplot.text(2.65, 0.60, "change in proportion", va="center", ha="left", fontsize=7)
    pyplot.text(1.25, 0.35, "convex area", va="center", ha="left", fontsize=7)
    pyplot.text(1.25, 0.10, "concave area", va="center", ha="left", fontsize=7)
    pyplot.scatter([2.22], [0.35], ec="k", fc="k", marker="s", lw=0.75)
    pyplot.scatter([2.22], [0.10], ec="k", fc="w", marker="s", lw=0.75)
    pyplot.annotate("", xy=(3.05, 0.35), xytext=(3.30, 0.35),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(3.05, 0.10), xytext=(3.30, 0.10),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    for index, landscape in enumerate([task_data["a"][2], task_data["a"][3]]):
        counter = Counter(landscape.reshape(-1))
        # noinspection PyTypeChecker
        convex_rate, concave_rate = counter[1] / (101.0 * 101.0), counter[-1] / (101.0 * 101.0)
        value_1, value_2 = ("%.1f" % (convex_rate * 100.0)) + "%", ("%.1f" % (concave_rate * 100.0)) + "%"
        if convex_rate > concave_rate:
            pyplot.text(3.00 + index * 0.7, 0.35, value_1, va="center", ha="right",
                        weight="bold", color="r", fontsize=7)
            pyplot.text(3.00 + index * 0.7, 0.10, value_2, va="center", ha="right", fontsize=7)
        else:
            pyplot.text(3.00 + index * 0.7, 0.35, value_1, va="center", ha="right", fontsize=7)
            pyplot.text(3.00 + index * 0.7, 0.10, value_2, va="center", ha="right",
                        weight="bold", color="r", fontsize=7)

    pyplot.xlim(0.10, 5.20)
    pyplot.ylim(-0.20, 2.45)
    pyplot.axis("off")

    # noinspection PyTypeChecker
    pyplot.subplot(grid[:5, 1])
    former, latter = task_data["b"][0], task_data["b"][1]
    pyplot.text(1.50, 1.7, "+", va="center", ha="center", fontsize=12)
    pyplot.text(3.50, 1.7, "=", va="center", ha="center", fontsize=12)
    pyplot.annotate("", xy=(0.50, 1.20), xytext=(0.50, 1.00),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(4.50, 1.20), xytext=(4.50, 1.00),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.50, 0.80, "curvature\nfeature", va="center", ha="center", fontsize=7)
    pyplot.text(4.50, 0.80, "curvature\nfeature", va="center", ha="center", fontsize=7)
    pyplot.text(2.50, 0.80, "change in predominant curvature proportion", va="center", ha="center", fontsize=7)
    labels = ["landscape of " + r"$\mathcal{L}_c$" + "\nbefore escaping",
              "adjustment of " + r"$\mathcal{L}_c$" + "\nduring escaping",
              "landscape of " + r"$\mathcal{L}_c$" + "\nafter escaping"]
    for index, landscape, info in zip([0.0, 2.0, 4.0], [former, latter - former, latter], labels):
        pyplot.pcolormesh(x + index, y + 1.2, landscape, vmin=-1, vmax=1, cmap="PRGn", shading="gouraud")
        pyplot.plot([0.2 + index, 0.8 + index, 0.8 + index, 0.2 + index, 0.2 + index],
                    [1.4, 1.4, 2.0, 2.0, 1.4], lw=0.75, color="k", zorder=2)
        pyplot.text(index + 0.50, 2.30, info, va="center", ha="center", fontsize=7)
        pyplot.text(index + 0.50, 1.30, "$x$", va="center", ha="center", fontsize=8)
        pyplot.text(index + 0.10, 1.70, "$y$", va="center", ha="center", fontsize=8)
        pyplot.text(index + 0.10, 1.30, "$-1$", va="center", ha="center", fontsize=8)
        pyplot.text(index + 0.10, 2.05, "$+1$", va="center", ha="center", fontsize=8)
        pyplot.text(index + 0.90, 1.30, "$+1$", va="center", ha="center", fontsize=8)
        pyplot.hlines(1.30, index + 0.2, index + 0.4, color="k", lw=0.75, ls="--")
        pyplot.hlines(1.30, index + 0.6, index + 0.8, color="k", lw=0.75, ls="--")
        pyplot.vlines(index + 0.10, 1.4, 1.6, color="k", lw=0.75, ls="--")
        pyplot.vlines(index + 0.10, 1.8, 2.0, color="k", lw=0.75, ls="--")

    # noinspection PyArgumentEqualDefault
    locations, colors = linspace(1.2, 3.8, 51), pyplot.get_cmap("PRGn")(linspace(0, 1, 50))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 1.02, 1.16, fc=color, lw=0, zorder=1)
    pyplot.plot([1.2, 1.2, 3.8, 3.8, 1.2], [1.02, 1.16, 1.16, 1.02, 1.02], lw=0.75, color="k", zorder=2)
    pyplot.text(1.18, 1.09, "$-1$", va="center", ha="right", fontsize=8)
    pyplot.text(3.82, 1.09, "$+1$", va="center", ha="left", fontsize=8)
    pyplot.text(2.50, 1.09, "$z$", va="center", ha="center", fontsize=8)

    for index, landscape in zip([0.0, 4.0], [task_data["b"][2], task_data["b"][3]]):
        pyplot.pcolormesh(x + index, y - 0.15, landscape, vmin=-1, vmax=1, cmap="binary", shading="gouraud")
        pyplot.plot([0.2 + index, 0.8 + index, 0.8 + index, 0.2 + index, 0.2 + index],
                    [0.05, 0.05, 0.65, 0.65, 0.06], lw=0.75, color="k", zorder=2)
    pyplot.plot([1.20, 1.20, 3.80, 3.80, 1.20], [0.00, 0.70, 0.70, 0.00, 0.00], lw=0.75, color="k", zorder=2)
    pyplot.fill_between([1.20, 3.80], 0.48, 0.70, lw=0, fc="ivory", zorder=1)
    pyplot.vlines(2.10, 0.00, 0.70, lw=0.75, color="k", zorder=2)
    pyplot.vlines(2.60, 0.00, 0.70, lw=0.75, color="k", zorder=2)
    pyplot.hlines(0.48, 1.20, 3.80, lw=0.75, color="k", zorder=2)
    pyplot.text(1.25, 0.60, "label", va="center", ha="left", fontsize=7)
    pyplot.text(2.15, 0.60, "color", va="center", ha="left", fontsize=7)
    pyplot.text(2.65, 0.60, "change in proportion", va="center", ha="left", fontsize=7)
    pyplot.text(1.25, 0.35, "convex area", va="center", ha="left", fontsize=7)
    pyplot.text(1.25, 0.10, "concave area", va="center", ha="left", fontsize=7)
    pyplot.scatter([2.22], [0.35], ec="k", fc="k", marker="s", lw=0.75)
    pyplot.scatter([2.22], [0.10], ec="k", fc="w", marker="s", lw=0.75)
    pyplot.annotate("", xy=(3.05, 0.35), xytext=(3.30, 0.35),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(3.05, 0.10), xytext=(3.30, 0.10),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    for index, landscape in enumerate([task_data["b"][2], task_data["b"][3]]):
        counter = Counter(landscape.reshape(-1))
        # noinspection PyTypeChecker
        convex_rate, concave_rate = counter[1] / (101.0 * 101.0), counter[-1] / (101.0 * 101.0)
        value_1, value_2 = ("%.1f" % (convex_rate * 100.0)) + "%", ("%.1f" % (concave_rate * 100.0)) + "%"
        if convex_rate > concave_rate:
            pyplot.text(3.00 + index * 0.7, 0.35, value_1, va="center", ha="right",
                        weight="bold", color="r", fontsize=7)
            pyplot.text(3.00 + index * 0.7, 0.10, value_2, va="center", ha="right", fontsize=7)
        else:
            pyplot.text(3.00 + index * 0.7, 0.35, value_1, va="center", ha="right", fontsize=7)
            pyplot.text(3.00 + index * 0.7, 0.10, value_2, va="center", ha="right",
                        weight="bold", color="r", fontsize=7)
    pyplot.xlim(0.10, 5.20)
    pyplot.ylim(-0.20, 2.45)
    pyplot.axis("off")

    # noinspection PyTypeChecker
    pyplot.subplot(grid[5:, 0])
    change_data = task_data["c"]
    group_i_1, group_i_2, group_b_1, group_b_2 = [], [], [], []
    for former_i, latter_i, former_b, latter_b in change_data:
        if latter_i > former_i:
            group_i_1.append([former_i, latter_i])
        else:
            group_i_2.append([former_i, latter_i])
        if latter_b > former_b:
            group_b_1.append([former_b, latter_b])
        else:
            group_b_2.append([former_b, latter_b])
    group_i_1, group_i_2, group_b_1, group_b_2 = array(group_i_1), array(group_i_2), array(group_b_1), array(group_b_2)
    pyplot.scatter(group_i_1[:, 0], group_i_1[:, 1], ec="k", fc="#FCB1AB", marker="o", lw=0)
    pyplot.scatter(group_i_2[:, 0], group_i_2[:, 1], ec="k", fc="#FCB1AB", marker="o", lw=0)
    pyplot.scatter(group_b_1[:, 0], group_b_1[:, 1], ec="k", fc="#88CCF8", marker="o", lw=0)
    pyplot.scatter(group_b_2[:, 0], group_b_2[:, 1], ec="k", fc="#88CCF8", marker="o", lw=0)
    pyplot.text(0.525, 0.28, "motif type", va="center", ha="left", fontsize=7)
    pyplot.text(0.675, 0.28, "complex", va="center", ha="left", fontsize=7)
    pyplot.text(0.825, 0.28, "proportion", va="center", ha="left", fontsize=7)
    pyplot.scatter([0.54], [0.23], ec="k", fc="#FCB1AB", marker="o", lw=0, label="incoherent loop samples")
    pyplot.scatter([0.54], [0.18], ec="k", fc="#FCB1AB", marker="o", lw=0)
    pyplot.scatter([0.54], [0.13], ec="k", fc="#88CCF8", marker="o", lw=0, label="collider samples")
    pyplot.scatter([0.54], [0.08], ec="k", fc="#88CCF8", marker="o", lw=0)
    pyplot.annotate("", xy=(0.69, 0.21), xytext=(0.69, 0.25), arrowprops=dict(arrowstyle="<|-", color="k", lw=0.75))
    pyplot.annotate("", xy=(0.69, 0.16), xytext=(0.69, 0.20), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75))
    pyplot.annotate("", xy=(0.69, 0.11), xytext=(0.69, 0.15), arrowprops=dict(arrowstyle="<|-", color="k", lw=0.75))
    pyplot.annotate("", xy=(0.69, 0.06), xytext=(0.69, 0.10), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75))
    pyplot.text(0.825, 0.23, str(len(group_i_1)), va="center", ha="left", fontsize=7)
    pyplot.text(0.825, 0.18, str(len(group_i_2)), va="center", ha="left", fontsize=7)
    pyplot.text(0.825, 0.13, str(len(group_b_1)), va="center", ha="left", fontsize=7)
    pyplot.text(0.825, 0.08, str(len(group_b_2)), va="center", ha="left", fontsize=7)
    pyplot.plot([0.50, 0.95, 0.95, 0.50, 0.50], [0.31, 0.31, 0.05, 0.05, 0.31], lw=0.75, color="k", zorder=2)
    pyplot.text(0.725, 0.33, "statistics", va="center", ha="center", fontsize=8)
    pyplot.hlines(0.255, 0.50, 0.95, lw=0.75, color="k", zorder=2)
    pyplot.fill_between([0.50, 0.95], 0.255, 0.31, lw=0, fc="#EEEEEE", zorder=1)
    pyplot.vlines(0.65, 0.05, 0.31, lw=0.75, color="k", zorder=2)
    pyplot.vlines(0.80, 0.05, 0.31, lw=0.75, color="k", zorder=2)
    pyplot.vlines(change_data[55, 0], change_data[55, 1], 0.93, lw=0.75, ls="--", color="k", zorder=2)
    pyplot.text(change_data[55, 0], 0.95, "detailed in (a)", va="center", ha="center", fontsize=7)
    pyplot.legend(loc="upper left", fontsize=7)
    pyplot.plot([0.0, 1.0], [0.0, 1.0], color="k", lw=0.75, ls="--", zorder=3)
    pyplot.xlabel("predominant curvature proportion of landscape before escaping", fontsize=8)
    pyplot.ylabel("predominant curvature proportion of landscape after escaping", fontsize=8)
    pyplot.xticks(linspace(0, 1, 11), [str(v) + "%" for v in arange(0, 101, 10)], fontsize=7)
    pyplot.yticks(linspace(0, 1, 11), [str(v) + "%" for v in arange(0, 101, 10)], fontsize=7)
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[5:, 1])
    change_data = task_data["d"]
    group_c_1, group_c_2, group_b_1, group_b_2 = [], [], [], []
    for former_c, latter_c, former_b, latter_b in change_data:
        if latter_c > former_c:
            group_c_1.append([former_c, latter_c])
        else:
            group_c_2.append([former_c, latter_c])
        if latter_b > former_b:
            group_b_1.append([former_b, latter_b])
        else:
            group_b_2.append([former_b, latter_b])
    group_c_1, group_c_2, group_b_1, group_b_2 = array(group_c_1), array(group_c_2), array(group_b_1), array(group_b_2)
    pyplot.scatter(group_c_1[:, 0], group_c_1[:, 1], ec="k", fc="#FCE0AB", marker="o", lw=0)
    pyplot.scatter(group_c_2[:, 0], group_c_2[:, 1], ec="k", fc="#FCE0AB", marker="o", lw=0)
    pyplot.scatter(group_b_1[:, 0], group_b_1[:, 1], ec="k", fc="#88CCF8", marker="o", lw=0)
    pyplot.scatter(group_b_2[:, 0], group_b_2[:, 1], ec="k", fc="#88CCF8", marker="o", lw=0)
    pyplot.text(0.525, 0.28, "motif type", va="center", ha="left", fontsize=7)
    pyplot.text(0.675, 0.28, "complex", va="center", ha="left", fontsize=7)
    pyplot.text(0.825, 0.28, "proportion", va="center", ha="left", fontsize=7)
    pyplot.scatter([0.54], [0.23], ec="k", fc="#FCE0AB", marker="o", lw=0, label="coherent loop samples")
    pyplot.scatter([0.54], [0.18], ec="k", fc="#FCE0AB", marker="o", lw=0)
    pyplot.scatter([0.54], [0.13], ec="k", fc="#88CCF8", marker="o", lw=0, label="collider samples")
    pyplot.scatter([0.54], [0.08], ec="k", fc="#88CCF8", marker="o", lw=0)
    pyplot.annotate("", xy=(0.69, 0.21), xytext=(0.69, 0.25), arrowprops=dict(arrowstyle="<|-", color="k", lw=0.75))
    pyplot.annotate("", xy=(0.69, 0.16), xytext=(0.69, 0.20), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75))
    pyplot.annotate("", xy=(0.69, 0.11), xytext=(0.69, 0.15), arrowprops=dict(arrowstyle="<|-", color="k", lw=0.75))
    pyplot.annotate("", xy=(0.69, 0.06), xytext=(0.69, 0.10), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75))
    pyplot.text(0.825, 0.23, str(len(group_c_1)), va="center", ha="left", fontsize=7)
    pyplot.text(0.825, 0.18, str(len(group_c_2)), va="center", ha="left", fontsize=7)
    pyplot.text(0.825, 0.13, str(len(group_b_1)), va="center", ha="left", fontsize=7)
    pyplot.text(0.825, 0.08, str(len(group_b_2)), va="center", ha="left", fontsize=7)
    pyplot.plot([0.50, 0.95, 0.95, 0.50, 0.50], [0.31, 0.31, 0.05, 0.05, 0.31], lw=0.75, color="k", zorder=2)
    pyplot.text(0.725, 0.33, "statistics", va="center", ha="center", fontsize=8)
    pyplot.hlines(0.255, 0.50, 0.95, lw=0.75, color="k", zorder=2)
    pyplot.fill_between([0.50, 0.95], 0.255, 0.31, lw=0, fc="#EEEEEE", zorder=1)
    pyplot.vlines(0.65, 0.05, 0.31, lw=0.75, color="k", zorder=2)
    pyplot.vlines(0.80, 0.05, 0.31, lw=0.75, color="k", zorder=2)
    pyplot.vlines(change_data[42, 0], 0.09, change_data[42, 1], lw=0.75, ls="--", color="k", zorder=2)
    pyplot.text(change_data[42, 0], 0.06, "detailed in (b)", va="center", ha="center", fontsize=7)
    pyplot.legend(loc="upper left", fontsize=7)
    pyplot.plot([0.0, 1.0], [0.0, 1.0], color="k", lw=0.75, ls="--")
    pyplot.xlabel("predominant curvature proportion of landscape before escaping", fontsize=8)
    pyplot.ylabel("predominant curvature proportion of landscape after escaping", fontsize=8)
    pyplot.xticks(linspace(0, 1, 11), [str(v) + "%" for v in arange(0, 101, 10)], fontsize=7)
    pyplot.yticks(linspace(0, 1, 11), [str(v) + "%" for v in arange(0, 101, 10)], fontsize=7)
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.513, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.67, "c", va="center", ha="center", fontsize=12)
    figure.text(0.513, 0.67, "d", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "main04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_05():
    """
    Create Figure 5 in the main text.
    """
    task_data = load_data(sort_path + "main05.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
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
    pyplot.annotate("", xy=(0.80, 0.15), xytext=(0.90, 0.15),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75))
    pyplot.text(0.50, 0.15, "push", va="center", ha="right", fontsize=8)
    pyplot.text(0.90, 0.15, "push", va="center", ha="left", fontsize=8)
    pyplot.plot([0.61, 0.70, 0.70], [0.52, 0.15, 0.60], color="k", lw=0.75, ls="--", zorder=4)
    pyplot.annotate("", xy=(0.61, 0.52), xytext=(0.70, 0.60),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.75,
                                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.3"))
    pyplot.text(0.64, 0.67, "angle", va="center", ha="center", fontsize=8)
    pyplot.annotate("", xy=(0.40, 0.75), xytext=(0.50, 0.95),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75, shrinkA=0, shrinkB=0))
    pyplot.text(0.34, 0.72, "angular\nvelocity", va="center", ha="center", fontsize=8)
    pyplot.fill_between([0.005, 0.270], 0.24, 1.01, lw=0.75, ec="k", fc="w", ls="--")
    pyplot.text(0.02, 0.95, "observation:", va="center", ha="left", fontsize=7)
    pyplot.text(0.05, 0.85, "cart position", va="center", ha="left", fontsize=7)
    pyplot.text(0.05, 0.75, "cart velocity", va="center", ha="left", fontsize=7)
    pyplot.text(0.05, 0.65, "pole angle", va="center", ha="left", fontsize=7)
    pyplot.text(0.05, 0.55, "pole angular velocity", va="center", ha="left", fontsize=7)
    pyplot.text(0.02, 0.40, "action:", va="center", ha="left", fontsize=7)
    pyplot.text(0.05, 0.30, "push left or right", va="center", ha="left", fontsize=7)
    pyplot.xlim(0, 1)
    pyplot.ylim(-0.1, 1.02)
    pyplot.axis("off")

    labels = ["baseline method",
              r"[ $\mathcal{L}_c + \mathcal{C}$ ] - method",
              r"[ $\mathcal{L}_i + \mathcal{C}$ ] - method",
              r"$\mathcal{C}$ - method"]

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[0, 2:])
    for index, (label, color) in enumerate(zip(labels, pyplot.get_cmap("binary")(linspace(0.0, 0.8, 4)))):
        locations = arange(5) - 0.3 + 0.2 * index
        pyplot.bar(locations, task_data["b"][index], width=0.2, fc=color, ec="k", lw=0.75, label=label)

    pyplot.text(4.45, 196, "pass (≥ 195)", va="bottom", ha="right", fontsize=8)
    pyplot.hlines(195, -0.5, 4.5, color="black", lw=0.75, ls="--", zorder=2)
    pyplot.legend(loc="lower left", framealpha=1, fontsize=7)
    pyplot.xlabel("training noise level", fontsize=8)
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
        pyplot.title(label, fontsize=8)
        values = task_data[panel_index].copy()
        values[values >= 195] = nan
        pyplot.pcolormesh(arange(6), arange(6), values.T, vmin=100, vmax=195, cmap="inferno")
        pyplot.plot([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
                    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0], lw=0.75, color="k")
        for location_x in range(5):
            for location_y in range(5):
                value = task_data[panel_index][location_x, location_y]
                if location_x >= location_y:
                    if value >= 195:
                        pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "pass",
                                    va="center", ha="center", fontsize=7)
                    elif value > 140:
                        pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.1f" % value,
                                    va="center", ha="center", fontsize=7)
                    else:
                        pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.1f" % value, color="w",
                                    va="center", ha="center", fontsize=7)
                else:
                    if value >= 195:
                        pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "pass",
                                    va="center", ha="center", fontsize=7)
                    elif value > 140:
                        pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.1f" % value,
                                    va="center", ha="center", fontsize=7)
                    else:
                        pyplot.text(location_x + 0.5, location_y + 0.5 - 0.01, "%.1f" % value, color="w",
                                    va="center", ha="center", fontsize=7)

        pyplot.xlabel("training noise level", fontsize=8)
        pyplot.ylabel("evaluating noise level", fontsize=8)
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

    pyplot.savefig(save_path + "main05.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_06():
    """
    Create Figure 6 in the main text.
    """
    task_data = load_data(sort_path + "main06.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    grid = pyplot.GridSpec(2, 4)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[0, :3])
    labels = {"b": "baseline method",
              "i": r"[ $\mathcal{L}_c + \mathcal{C}$ ] - method",
              "c": r"[ $\mathcal{L}_i + \mathcal{C}$ ] - method",
              "a": r"$\mathcal{C}$ - method"}
    colors = pyplot.get_cmap("binary")(linspace(0.0, 0.8, 4))
    for (agent_name, values), bias, color in zip(task_data["a"].items(), [-3, -1, 1, 3], colors):
        pyplot.bar(arange(20, 151, 10) + bias, values, width=2, fc=color, ec="k", lw=0.75, label=labels[agent_name])
    pyplot.legend(loc="upper left", framealpha=1, fontsize=7)
    pyplot.plot([95, 95, 105, 105], [85, 88, 88, 85], lw=0.75, color="k")
    pyplot.vlines(100, 88, 91, lw=0.75, color="k")
    pyplot.text(100, 97, "used for (c) - (f)", va="center", ha="center", fontsize=7)
    pyplot.xlabel("maximum generation", fontsize=8)
    pyplot.ylabel("qualified agent number", fontsize=8)
    pyplot.xticks(arange(20, 151, 10), arange(20, 151, 10), fontsize=7)
    pyplot.yticks(arange(0, 101, 20), arange(0, 101, 20), fontsize=7)
    pyplot.xlim(14, 156)
    pyplot.ylim(0, 100)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[0, 3])
    cases = {"S": [200, 200, 199, 197, 192], "F1": [198, 198, 196, 192, 184],
             "F2": [192, 191, 188, 181, 170], "F3": [175, 177, 181, 187, 177]}
    colors = ["#98BAC6", "#DABCBB", "#B2859B", "#A39EC0"]
    pyplot.hlines(195, 0, 5, lw=0.75, ls="--", color="k", zorder=0)
    pyplot.vlines(3.5, 158, 200, lw=0.75, ls="--", color="k", zorder=0)
    for zorder, (label, values) in enumerate(cases.items()):
        pyplot.plot(arange(5) + 0.5, values, color=colors[zorder],
                    lw=1.5, zorder=zorder + 1, marker="o", markersize=4, label=label)
    pyplot.text(3.5, 203, "training noise level = 30%", va="top", ha="center", fontsize=7)
    pyplot.legend(loc="lower left", fontsize=7, title="performance type", title_fontsize=7, ncol=2)
    pyplot.xlabel("evaluating noise level", fontsize=8)
    pyplot.ylabel("evaluating performance", fontsize=8)
    pyplot.xticks(arange(5) + 0.5, ["0%", "10%", "20%", "30%", "40%"], fontsize=7)
    pyplot.yticks([195], ["pass"], fontsize=7)
    pyplot.xlim(0, 5)
    pyplot.ylim(163, 203)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    for index, panel_index in enumerate(["c", "d", "e", "f"]):
        short_label, s, f1, f2, f3 = task_data[panel_index]

        # noinspection PyTypeChecker
        pyplot.subplot(grid[1, index])
        pyplot.title(labels[short_label], fontsize=8)

        used_ratios, used_types, used_colors = [], [], []
        for ratio, pie_type, color in zip([s, f1, f2, f3], ["S", "F1", "F2", "F3"], colors):
            if ratio > 0:
                used_ratios.append(ratio)
                used_types.append(pie_type + " = " + str(ratio) + "%")
                used_colors.append(color)
        pyplot.pie(used_ratios, labels=used_types, colors=used_colors,
                   textprops={"fontsize": 7, "color": "k"}, wedgeprops={"lw": 0.75, "ec": "k"})

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.743, 0.99, "b", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.47, "c", va="center", ha="center", fontsize=12)
    figure.text(0.268, 0.47, "d", va="center", ha="center", fontsize=12)
    figure.text(0.515, 0.47, "e", va="center", ha="center", fontsize=12)
    figure.text(0.743, 0.47, "f", va="center", ha="center", fontsize=12)

    pyplot.savefig(save_path + "main06.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    main_01()
    # main_02()
    # main_03()
    # main_04()
    # main_05()
    # main_06()
