from matplotlib import pyplot, markers, patches, lines, rcParams
from numpy import arange, zeros, linspace, sum

from works import acyclic_motifs, draw_info, load_data, adjust_format


def supp01():
    pyplot.figure(figsize=(10, 9))
    rcParams["font.family"] = "Times New Roman"
    pyplot.subplots_adjust(wspace=0, hspace=0)
    grid = pyplot.GridSpec(5, 5)
    for type_index, (motif_type, motifs) in enumerate(acyclic_motifs.items()):
        info = draw_info[motif_type]
        for motif_index, motif in enumerate(motifs):
            pyplot.subplot(grid[motif_index, type_index])
            if motif_index == 0:
                pyplot.title(motif_type.replace("-", " "), fontsize=10)
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

            pyplot.text(x=info[1][0], y=info[2][0] - 0.06, s="x", fontsize=10, va="top", ha="center")
            if motif_type != "fork":
                pyplot.text(x=info[1][1], y=info[2][1] - 0.06, s="y", fontsize=10, va="top", ha="center")
            else:
                pyplot.text(x=info[1][1], y=info[2][1] + 0.06, s="y", fontsize=10, va="bottom", ha="center")
            if motif_type != "chain":
                pyplot.text(x=info[1][2], y=info[2][2] + 0.06, s="z", fontsize=10, va="bottom", ha="center")
            else:
                pyplot.text(x=info[1][2], y=info[2][2] - 0.06, s="z", fontsize=10, va="top", ha="center")

            pyplot.fill_between([0, 1], [0], [1], color=info[0], zorder=1)
            pyplot.xlim(0, 1)
            pyplot.ylim(0, 0.9)
            pyplot.xticks([])
            pyplot.yticks([])

    pyplot.subplot(grid[4, :])
    pyplot.scatter([0.2], [0.2], color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.4], marker=markers.MarkerStyle("o", fillstyle="right"),
                   color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.4], marker=markers.MarkerStyle("o", fillstyle="left"),
                   color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.6], color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.8], color="black", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.annotate(s="", xy=(1.8, 0.65), xytext=(1.3, 0.65),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    shrinkA=0, shrinkB=0, lw=1.5), zorder=2)
    pyplot.annotate(s="", xy=(1.8, 0.25), xytext=(1.3, 0.25),
                    arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                    shrinkA=0, shrinkB=0, lw=1.5), zorder=2)
    pyplot.text(0.3, 0.8, "output node", ha="left", va="center", fontsize=10)
    pyplot.text(0.3, 0.6, "hidden node", ha="left", va="center", fontsize=10)
    pyplot.text(0.3, 0.4, "hidden / input node", ha="left", va="center", fontsize=10)
    pyplot.text(0.3, 0.2, "input node", ha="left", va="center", fontsize=10)
    pyplot.text(1.52, 0.72, "positive effect", ha="center", va="center", fontsize=10)
    pyplot.text(1.52, 0.32, "negative effect", ha="center", va="center", fontsize=10)
    pyplot.xlim(0, 2)
    pyplot.ylim(0, 1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.axis("off")

    pyplot.savefig("../data/figures/supp01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp02():
    landscapes = load_data("../data/results/task02/intuition landscapes.npy")
    figure = pyplot.figure(figsize=(10, 11))
    pyplot.subplots_adjust(left=0.05, bottom=0.05, right=0.92, wspace=0.5, hspace=0.7)
    rcParams["font.family"] = "Times New Roman"
    for index, landscape in enumerate(landscapes):
        compressed_landscape = zeros(shape=(20, 20))
        for index_1, location_1 in enumerate(linspace(0, 40, 20, dtype=int)):
            for index_2, location_2 in enumerate(linspace(0, 40, 20, dtype=int)):
                compressed_landscape[index_1, index_2] = landscape[location_1, location_2]
        pyplot.subplot(12, 12, index + 1)
        pyplot.title(str(index + 1), fontsize=10)
        pyplot.pcolormesh(arange(20), arange(20), compressed_landscape, vmin=-1, vmax=+1,
                          shading="gouraud", cmap="rainbow")
        pyplot.xlim(0, 19)
        pyplot.ylim(0, 19)
        pyplot.xticks([])
        pyplot.yticks([])

    # noinspection PyTypeChecker
    bar = pyplot.colorbar(cax=pyplot.axes(tuple([0.95, 0.15, 0.02, 0.7])), ticks=[-1, 0, +1])
    bar.set_label("output z", fontsize=10)
    bar.ax.tick_params(labelsize=10)
    bar.ax.set_yticklabels(["\N{MINUS SIGN}1", "0", "+1"])

    figure.add_artist(lines.Line2D([0.04, 0.04], [0.04, 0.88], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.04, 0.92], [0.04, 0.04], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.04, 0.04], [0.035, 0.04], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.48, 0.48], [0.035, 0.04], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.92, 0.92], [0.035, 0.04], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.035, 0.04], [0.04, 0.04], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.035, 0.04], [0.46, 0.46], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.035, 0.04], [0.88, 0.88], lw=0.75, color="black"))
    figure.text(0.48, 0.005, "input x", va="center", ha="center", fontsize=10)
    figure.text(0.005, 0.46, "input y", va="center", ha="center", rotation="vertical", fontsize=10)
    figure.text(0.04, 0.02, "\N{MINUS SIGN}1", va="center", ha="center", fontsize=10)
    figure.text(0.48, 0.02, "0", va="center", ha="center", fontsize=10)
    figure.text(0.92, 0.02, "+1", va="center", ha="center", fontsize=10)
    figure.text(0.02, 0.04, "\N{MINUS SIGN}1", va="center", ha="center", fontsize=10)
    figure.text(0.02, 0.46, "0", va="center", ha="center", fontsize=10)
    figure.text(0.02, 0.88, "+1", va="center", ha="center", fontsize=10)

    pyplot.savefig("../data/figures/supp02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp03():
    figure = pyplot.figure(figsize=(10, 3.5), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"
    grid = pyplot.GridSpec(2, 4)

    cases = load_data("../data/results/task02/terminal cases.pkl")

    for plot_index, case_info in zip([0, 2], [cases["min"][1], cases["max"][1]]):
        motif, landscape = case_info
        pyplot.subplot(grid[0, plot_index])
        info = draw_info[motif.t]
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
        for former, latter, weight in [(1, 2, motif.w[0].value()), (1, 3, motif.w[1].value()),
                                       (2, 3, motif.w[2].value())]:
            if weight > 0:
                pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                                arrowprops=dict(arrowstyle="-|>", color="black",
                                                shrinkA=6, shrinkB=6, lw=1), zorder=2)
            elif weight < 0:
                pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                                arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                                shrinkA=6, shrinkB=6, lw=1), zorder=2)
        pyplot.text(x=info[1][0], y=info[2][0] - 0.06, s="x", fontsize=9, va="top", ha="center")
        pyplot.text(x=info[1][1], y=info[2][1] - 0.06, s="y", fontsize=9, va="top", ha="center")
        pyplot.text(x=info[1][2], y=info[2][2] + 0.06, s="z", fontsize=9, va="bottom", ha="center")
        pyplot.text(x=(info[1][0] + info[1][1]) / 2.0, y=info[1][0] - 0.06,
                    s=adjust_format("%.1e" % motif.w[0].value()), fontsize=8, va="top", ha="center")
        pyplot.text(x=(info[1][0] + info[1][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                    s=adjust_format("%.1e" % motif.w[1].value()), fontsize=8, va="bottom", ha="right")
        pyplot.text(x=(info[1][1] + info[1][2]) / 2.0 + 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                    s=adjust_format("%.1e" % motif.w[2].value()), fontsize=8, va="bottom", ha="left")
        pyplot.text(x=info[1][1] + 0.06, y=info[2][1] - 0.01,
                    s=adjust_format("%.1e" % motif.b[0].value()), fontsize=8, va="center", ha="left")
        pyplot.text(x=info[1][2] + 0.06, y=info[2][2] - 0.01,
                    s=adjust_format("%.1e" % motif.b[1].value()), fontsize=8, va="center", ha="left")
        pyplot.text(x=info[1][1], y=info[2][1] + 0.1, s=motif.a[0] + " / " + motif.g[0], fontsize=9,
                    va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
        pyplot.text(x=info[1][2], y=info[2][2] + 0.2, s=motif.a[1] + " / " + motif.g[1], fontsize=9,
                    va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
        pyplot.xlim(0.0, 1.0)
        pyplot.ylim(0.1, 1.0)
        pyplot.axis("off")

        pyplot.subplot(grid[1, plot_index])
        mesh = pyplot.pcolormesh(arange(41), arange(41), landscape, vmin=-1, vmax=+1, cmap="rainbow")
        bar = pyplot.colorbar(mesh, ticks=[-1, 0, +1])
        bar.set_label("output z", fontsize=10)
        bar.ax.tick_params(labelsize=10)
        bar.ax.set_yticklabels(["\N{MINUS SIGN}1", "0", "+1"])
        pyplot.xlabel("input x", fontsize=10)
        pyplot.ylabel("input y", fontsize=10)
        pyplot.xlim(0, 40)
        pyplot.ylim(0, 40)
        pyplot.xticks([0, 20, 40], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
        pyplot.yticks([0, 20, 40], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)

    for plot_index, case_info in zip([1, 3], [cases["min"][2], cases["max"][2]]):
        motif, landscape = case_info
        pyplot.subplot(grid[0, plot_index])
        info = draw_info[motif.t]
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
        for former, latter, weight in [(1, 3, motif.w[0].value()), (2, 3, motif.w[1].value())]:
            if weight > 0:
                pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                                arrowprops=dict(arrowstyle="-|>", color="black",
                                                shrinkA=6, shrinkB=6, lw=1), zorder=2)
            elif weight < 0:
                pyplot.annotate(s="", xy=(x[latter - 1], y[latter - 1]), xytext=(x[former - 1], y[former - 1]),
                                arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                                shrinkA=6, shrinkB=6, lw=1), zorder=2)
        pyplot.text(x=info[1][0], y=info[2][0] - 0.06, s="x", fontsize=9, va="top", ha="center")
        pyplot.text(x=info[1][1], y=info[2][1] - 0.06, s="y", fontsize=9, va="top", ha="center")
        pyplot.text(x=info[1][2], y=info[2][2] + 0.06, s="z", fontsize=9, va="bottom", ha="center")
        pyplot.text(x=(info[1][0] + info[1][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                    s=adjust_format("%.1e" % motif.w[0].value()), fontsize=8, va="bottom", ha="right")
        pyplot.text(x=(info[1][1] + info[1][2]) / 2.0 + 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                    s=adjust_format("%.1e" % motif.w[1].value()), fontsize=8, va="bottom", ha="left")
        pyplot.text(x=info[1][2] + 0.06, y=info[2][2] - 0.01,
                    s=adjust_format("%.1e" % motif.b[0].value()), fontsize=8, va="center", ha="left")
        pyplot.text(x=info[1][2], y=info[2][2] + 0.2, s=motif.a[0] + " / " + motif.g[0], fontsize=9,
                    va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
        pyplot.xlim(0.0, 1.0)
        pyplot.ylim(0.1, 1.0)
        pyplot.axis("off")

        pyplot.subplot(grid[1, plot_index])
        mesh = pyplot.pcolormesh(arange(41), arange(41), landscape, vmin=-1, vmax=+1, cmap="rainbow")
        bar = pyplot.colorbar(mesh, ticks=[-1, 0, +1])
        bar.set_label("output z", fontsize=10)
        bar.ax.tick_params(labelsize=10)
        bar.ax.set_yticklabels(["\N{MINUS SIGN}1", "0", "+1"])
        pyplot.xlabel("input x", fontsize=10)
        pyplot.ylabel("input y", fontsize=10)
        pyplot.xlim(0, 40)
        pyplot.ylim(0, 40)
        pyplot.xticks([0, 20, 40], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
        pyplot.yticks([0, 20, 40], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)

    figure.text(0.025, 0.99, "a", va="center", ha="center", fontsize=12)
    figure.text(0.515, 0.99, "b", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/supp03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp04():
    landscapes = load_data("../data/results/task03/landscapes.npy")
    figure = pyplot.figure(figsize=(10, 11))
    pyplot.subplots_adjust(left=0.05, bottom=0.05, right=0.92, wspace=0.5, hspace=0.7)
    rcParams["font.family"] = "Times New Roman"
    for index, landscape in enumerate(landscapes):
        compressed_landscape = zeros(shape=(20, 20))
        for index_1, location_1 in enumerate(linspace(0, 40, 20, dtype=int)):
            for index_2, location_2 in enumerate(linspace(0, 40, 20, dtype=int)):
                compressed_landscape[index_1, index_2] = landscape[location_1, location_2]
        pyplot.subplot(12, 12, index + 1)
        pyplot.title(str(index + 1), fontsize=10)
        pyplot.pcolormesh(arange(20), arange(20), compressed_landscape, vmin=-1, vmax=+1,
                          shading="gouraud", cmap="rainbow")
        pyplot.xlim(0, 19)
        pyplot.ylim(0, 19)
        pyplot.xticks([])
        pyplot.yticks([])

    # noinspection PyTypeChecker
    bar = pyplot.colorbar(cax=pyplot.axes(tuple([0.95, 0.15, 0.02, 0.7])), ticks=[-1, 0, +1])
    bar.set_label("output z", fontsize=10)
    bar.ax.tick_params(labelsize=10)
    bar.ax.set_yticklabels(["\N{MINUS SIGN}1", "0", "+1"])

    figure.add_artist(lines.Line2D([0.04, 0.04], [0.04, 0.88], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.04, 0.94], [0.04, 0.04], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.04, 0.04], [0.035, 0.04], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.49, 0.49], [0.035, 0.04], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.94, 0.94], [0.035, 0.04], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.035, 0.04], [0.04, 0.04], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.035, 0.04], [0.46, 0.46], lw=0.75, color="black"))
    figure.add_artist(lines.Line2D([0.035, 0.04], [0.88, 0.88], lw=0.75, color="black"))
    figure.text(0.49, 0.005, "input x", va="center", ha="center", fontsize=10)
    figure.text(0.005, 0.46, "input y", va="center", ha="center", rotation="vertical", fontsize=10)
    figure.text(0.04, 0.02, "\N{MINUS SIGN}1", va="center", ha="center", fontsize=10)
    figure.text(0.49, 0.02, "0", va="center", ha="center", fontsize=10)
    figure.text(0.94, 0.02, "+1", va="center", ha="center", fontsize=10)
    figure.text(0.02, 0.04, "\N{MINUS SIGN}1", va="center", ha="center", fontsize=10)
    figure.text(0.02, 0.46, "0", va="center", ha="center", fontsize=10)
    figure.text(0.02, 0.88, "+1", va="center", ha="center", fontsize=10)

    pyplot.savefig("../data/figures/supp04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp05():
    params = load_data(load_path="../data/results/task03/max-min params.npy")

    pyplot.figure(figsize=(10, 3.5), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"

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

    pyplot.savefig("../data/figures/supp05.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp06():
    evolved_motifs, index = load_data(load_path="../data/results/task04/evolved motifs.npy"), 0
    names = ["geometry", "baseline", "novelty"]

    pyplot.figure(figsize=(10, 6), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"
    for method_index, method_data in enumerate(evolved_motifs):
        for noise_index, noise_data in enumerate(method_data):
            pyplot.subplot(3, 4, index + 1)
            if noise_index > 0:
                pyplot.title(names[method_index] + " (under " + str(noise_index * 10) + "% noise)", fontsize=10)
            else:
                pyplot.title(names[method_index] + " (noise-free)", fontsize=10)

            if sum(noise_data[0]) > 0:
                pyplot.plot(arange(20) + 1, noise_data[0], color="#FA897B", label="incoherent loop", zorder=1)
            else:
                pyplot.text(10, 0.5, "The evolution process does not\nproduce incoherent loops",
                            va="bottom", ha="center", fontsize=8)
            pyplot.plot(arange(20) + 1, noise_data[1], color="#86E3CE", label="collider", zorder=0)
            pyplot.legend(loc="upper left", fontsize=8)
            pyplot.xlabel("generation", fontsize=10)
            pyplot.ylabel("average motif number", fontsize=10)
            pyplot.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20], fontsize=10)
            pyplot.yticks([0, 10, 20, 30], [0, 10, 20, 30], fontsize=10)
            pyplot.xlim(-0.5, 20.5)
            pyplot.ylim(0, 30)
            index += 1

    pyplot.savefig("../data/figures/supp06.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp07():
    generations, index = load_data("../data/results/task04/generations.npy"), 0
    names, colors = ["geometry", "baseline", "novelty"], ["#845EC2", "#4FFBDF", "#00C2A8", "#008B74"]

    pyplot.figure(figsize=(10, 6), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"
    for method_index, method_data in enumerate(generations):
        for noise_index, (values, color) in enumerate(zip(method_data, colors)):
            pyplot.subplot(3, 4, index + 1)
            if noise_index > 0:
                pyplot.title(names[method_index] + " (under " + str(noise_index * 10) + "% noise)", fontsize=10)
            else:
                pyplot.title(names[method_index] + " (noise-free)", fontsize=10)
            pyplot.plot(arange(20) + 1, values, color="black", linewidth=1)
            pyplot.fill_between(arange(20) + 1, 0, values, color=color)
            for generation in [5, 10, 15]:
                pyplot.vlines(generation, 0, 100, linewidth=0.75, color="black", linestyle="--")
            for proportion in [25, 50, 75]:
                pyplot.hlines(proportion, 0, 20, linewidth=0.75, color="black", linestyle="--")
            pyplot.xlabel("completed generation", fontsize=10)
            pyplot.ylabel("proportion", fontsize=10)
            pyplot.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20], fontsize=10)
            pyplot.yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=10)
            pyplot.xlim(0, 20)
            pyplot.ylim(0, 100)
            index += 1

    pyplot.savefig("../data/figures/supp07.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp08():
    names = ["geometry", "baseline", "novelty"]

    figure = pyplot.figure(figsize=(10, 9), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"

    normal_results = load_data(load_path="../data/results/task04/train results.npy")

    for index in range(3):
        pyplot.subplot(3, 3, 1 + index)
        pyplot.title(names[index] + " (normal setup)", fontsize=10)
        pyplot.boxplot(normal_results[index].tolist(), positions=[0.5, 1.5, 2.5, 3.5], showfliers=False,
                       boxprops=dict(color="black", facecolor="linen", linewidth=1),
                       medianprops=dict(color="orangered", linewidth=2), patch_artist=True)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.ylabel("train performance", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.yticks([40, 80, 120, 160, 200], [40, 80, 120, 160, 200], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylim(30, 210)

    changed_results = load_data(load_path="../data/results/task04/changed performances.pkl")

    for index in range(3):
        pyplot.subplot(3, 3, 4 + index)
        pyplot.title(names[index] + " (increase the maximum generation)", fontsize=10)
        pyplot.boxplot(changed_results["20 to 100"][0][index].tolist(), positions=[0.5, 1.5, 2.5, 3.5],
                       boxprops=dict(color="black", facecolor="lightgreen", linewidth=1),
                       medianprops=dict(color="green", linewidth=2), patch_artist=True, showfliers=False)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.ylabel("train performance", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.yticks([40, 80, 120, 160, 200], [40, 80, 120, 160, 200], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylim(30, 210)

    for index in range(3):
        pyplot.subplot(3, 3, 7 + index)
        pyplot.title(names[index] + " (allow more propagation equations)", fontsize=10)
        pyplot.boxplot(changed_results["combine"][0][index].tolist(), positions=[0.5, 1.5, 2.5, 3.5], showfliers=False,
                       boxprops=dict(color="black", facecolor="lightgreen", linewidth=1),
                       medianprops=dict(color="green", linewidth=2), patch_artist=True)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.ylabel("train performance", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.yticks([40, 80, 120, 160, 200], [40, 80, 120, 160, 200], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylim(30, 210)

    figure.align_labels()

    pyplot.savefig("../data/figures/supp08.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp09():
    names = ["geometry", "baseline", "novelty"]

    figure = pyplot.figure(figsize=(10, 9), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"

    performances = load_data(load_path="../data/results/task04/performances.npy")

    for index in range(3):
        pyplot.subplot(3, 3, 1 + index)
        pyplot.title(names[index] + " (normal setup)", fontsize=10)
        pyplot.pcolormesh(arange(5), arange(5), performances[index].T, vmin=100, vmax=200, cmap="spring")
        for i in range(4):
            for j in range(4):
                pyplot.text(i + 0.5, j + 0.5, "%.2f" % performances[index, i, j], va="center", ha="center", fontsize=10)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.ylabel("evaluating error scale", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.yticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylim(0, 4)

    changed_results = load_data(load_path="../data/results/task04/changed performances.pkl")

    for index in range(3):
        pyplot.subplot(3, 3, 4 + index)
        pyplot.title(names[index] + " (increase the maximum generation)", fontsize=10)
        pyplot.pcolormesh(arange(5), arange(5), changed_results["20 to 100"][1][index].T,
                          vmin=100, vmax=200, cmap="spring")
        for i in range(4):
            for j in range(4):
                pyplot.text(i + 0.5, j + 0.5, "%.2f" % changed_results["20 to 100"][1][index, i, j],
                            va="center", ha="center", fontsize=10)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.ylabel("evaluating error scale", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.yticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylim(0, 4)

    for index in range(3):
        pyplot.subplot(3, 3, 7 + index)
        pyplot.title(names[index] + " (allow more propagation equations)", fontsize=10)
        pyplot.pcolormesh(arange(5), arange(5), changed_results["combine"][1][index].T,
                          vmin=100, vmax=200, cmap="spring")
        for i in range(4):
            for j in range(4):
                pyplot.text(i + 0.5, j + 0.5, "%.2f" % changed_results["combine"][1][index, i, j],
                            va="center", ha="center", fontsize=10)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.ylabel("evaluating error scale", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.yticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylim(0, 4)

    figure.align_labels()

    pyplot.savefig("../data/figures/supp09.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp10():
    names = ["geometry", "baseline", "novelty"]

    figure = pyplot.figure(figsize=(10, 9), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"

    counts = load_data(load_path="../data/results/task04/final motifs.npy")

    for index in range(3):
        pyplot.subplot(3, 3, 1 + index)
        pyplot.title(names[index] + " (normal setup)", fontsize=10)
        pyplot.bar(arange(4) + 0.3, counts[index][:, 0], width=0.4,
                   linewidth=0.75, edgecolor="black", color="#FA897B", label="incoherent loop")
        pyplot.bar(arange(4) + 0.7, counts[index][:, 1], width=0.4,
                   linewidth=0.75, edgecolor="black", color="#86E3CE", label="collider")
        for position, (value_1, value_2) in enumerate(zip(counts[index][:, 0], counts[index][:, 1])):
            pyplot.text(position + 0.3, value_1 + 0.2, "%.1f" % value_1, va="bottom", ha="center", fontsize=8)
            pyplot.text(position + 0.7, value_2 + 0.2, "%.1f" % value_2, va="bottom", ha="center", fontsize=8)
        pyplot.legend(loc="upper left", fontsize=8)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylabel("average motif number", fontsize=10)
        pyplot.yticks([0, 80, 160, 240], [0, 80, 160, 240], fontsize=10)
        pyplot.ylim(0, 240)

    changed_counts = load_data(load_path="../data/results/task04/changed motifs.pkl")

    for index in range(3):
        pyplot.subplot(3, 3, 4 + index)
        pyplot.title(names[index] + " (increase the maximum generation)", fontsize=10)
        values = [[], []]
        for error_scale in range(4):
            values[0].append(sum(changed_counts["20 to 100"][(names[index][0], error_scale)][-4:]))
            values[1].append(sum(changed_counts["20 to 100"][(names[index][0], error_scale)][:4]))
        pyplot.bar(arange(4) + 0.3, values[0], width=0.4,
                   linewidth=0.75, edgecolor="black", color="#FA897B", label="incoherent loop")
        pyplot.bar(arange(4) + 0.7, values[1], width=0.4,
                   linewidth=0.75, edgecolor="black", color="#86E3CE", label="collider")
        for position, (value_1, value_2) in enumerate(zip(values[0], values[1])):
            pyplot.text(position + 0.3, value_1 + 0.2, "%.1f" % value_1, va="bottom", ha="center", fontsize=8)
            pyplot.text(position + 0.7, value_2 + 0.2, "%.1f" % value_2, va="bottom", ha="center", fontsize=8)
        pyplot.legend(loc="upper left", fontsize=8)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylabel("average motif number", fontsize=10)
        pyplot.yticks([0, 80, 160, 240], [0, 80, 160, 240], fontsize=10)
        pyplot.ylim(0, 240)

    for index in range(3):
        pyplot.subplot(3, 3, 7 + index)
        pyplot.title(names[index] + " (allow more propagation equations)", fontsize=10)
        values = [[], []]
        for error_scale in range(4):
            values[0].append(sum(changed_counts["combine"][(names[index][0], error_scale)][-4:]))
            values[1].append(sum(changed_counts["combine"][(names[index][0], error_scale)][:4]))
        pyplot.bar(arange(4) + 0.3, values[0], width=0.4,
                   linewidth=0.75, edgecolor="black", color="#FA897B", label="incoherent loop")
        pyplot.bar(arange(4) + 0.7, values[1], width=0.4,
                   linewidth=0.75, edgecolor="black", color="#86E3CE", label="collider")
        for position, (value_1, value_2) in enumerate(zip(values[0], values[1])):
            pyplot.text(position + 0.3, value_1 + 0.2, "%.1f" % value_1, va="bottom", ha="center", fontsize=8)
            pyplot.text(position + 0.7, value_2 + 0.2, "%.1f" % value_2, va="bottom", ha="center", fontsize=8)
        pyplot.legend(loc="upper left", fontsize=8)
        pyplot.xlabel("training error scale", fontsize=10)
        pyplot.xticks(arange(4) + 0.5, ["0%", "10%", "20%", "30%"], fontsize=10)
        pyplot.xlim(0, 4)
        pyplot.ylabel("average motif number", fontsize=10)
        pyplot.yticks([0, 80, 160, 240], [0, 80, 160, 240], fontsize=10)
        pyplot.ylim(0, 240)

    figure.align_labels()

    pyplot.savefig("../data/figures/supp10.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    supp01()
    supp02()
    supp03()
    supp04()
    supp05()
    supp06()
    supp07()
    supp08()
    supp09()
    supp10()
