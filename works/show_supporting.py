from matplotlib import pyplot, markers, rcParams
from numpy import arange, abs, mean

from works import acyclic_motifs, draw_info, load_data, adjust_format


def supporting01():
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

    pyplot.savefig("../data/figures/supporting01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supporting02():
    pass


def supporting03():
    figure = pyplot.figure(figsize=(10, 7), tight_layout=True)
    rcParams["font.family"] = "Times New Roman"
    grid = pyplot.GridSpec(4, 4)

    locations = load_data("../data/results/task02/locations.npy")
    cases = load_data("../data/results/task02/terminal cases.pkl")

    pyplot.subplot(grid[:2, :])
    pyplot.scatter(locations[arange(0, 288, 2), 0], locations[arange(0, 288, 2), 1],
                   s=20, color="#86E3CE", label="loop")
    pyplot.scatter(locations[arange(1, 288, 2), 0], locations[arange(1, 288, 2), 1],
                   s=20, color="#FA897B", label="collider")
    difference = mean(abs(cases["min"][1][2] - cases["min"][1][3]))
    pyplot.text(x=locations[cases["min"][0] * 2, 0], y=locations[cases["min"][0] * 2, 1] + 0.8,
                s="b [" + adjust_format("%.1E" % difference) + "]", fontsize=10, va="bottom", ha="center")
    difference = mean(abs(cases["max"][1][2] - cases["max"][1][3]))
    pyplot.text(x=locations[cases["max"][0] * 2, 0], y=locations[cases["max"][0] * 2, 1] + 0.8,
                s="c [" + adjust_format("%.1E" % difference) + "]", fontsize=10, va="bottom", ha="center")
    pyplot.legend(loc="upper right", fontsize=10)
    pyplot.xlabel("tSNE of output landscape difference", fontsize=10)
    pyplot.ylabel("tSNE of output landscape difference", fontsize=10, labelpad=20)
    pyplot.xlim(-22, 22)
    pyplot.ylim(-22, 20)
    pyplot.xticks([])
    pyplot.yticks([])

    for plot_index, case_pair in zip([0, 2], [cases["min"][1], cases["max"][1]]):
        motif, _, landscape, _ = case_pair
        pyplot.subplot(grid[2, plot_index])
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
        pyplot.text(x=(info[1][0] + info[1][1]) / 2.0, y=info[1][0] - 0.06,
                    s=adjust_format("%.1e" % motif.w[0].value()), fontsize=9, va="top", ha="center")
        pyplot.text(x=(info[1][0] + info[1][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                    s=adjust_format("%.1e" % motif.w[1].value()), fontsize=9, va="bottom", ha="right")
        pyplot.text(x=(info[1][1] + info[1][2]) / 2.0 + 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                    s=adjust_format("%.1e" % motif.w[2].value()), fontsize=9, va="bottom", ha="left")
        pyplot.text(x=info[1][1], y=info[2][1] + 0.1, s=motif.a[0] + " / " + motif.g[0], fontsize=9,
                    va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
        pyplot.text(x=info[1][2], y=info[2][2] + 0.1, s=motif.a[1] + " / " + motif.g[1], fontsize=9,
                    va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
        pyplot.xlim(0.0, 1.0)
        pyplot.ylim(0.1, 0.9)
        pyplot.axis("off")
        
        pyplot.subplot(grid[3, plot_index])
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

    for plot_index, case_pair in zip([1, 3], [cases["min"][1], cases["max"][1]]):
        _, motif, _, landscape = case_pair
        pyplot.subplot(grid[2, plot_index])
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
        pyplot.text(x=(info[1][0] + info[1][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                    s=adjust_format("%.1e" % motif.w[0].value()), fontsize=9, va="bottom", ha="right")
        pyplot.text(x=(info[1][1] + info[1][2]) / 2.0 + 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                    s=adjust_format("%.1e" % motif.w[1].value()), fontsize=9, va="bottom", ha="left")
        pyplot.text(x=info[1][2], y=info[2][2] + 0.1, s=motif.a[0] + " / " + motif.g[0], fontsize=9,
                    va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
        pyplot.xlim(0.0, 1.0)
        pyplot.ylim(0.1, 0.9)
        pyplot.axis("off")

        pyplot.subplot(grid[3, plot_index])
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
    figure.text(0.025, 0.50, "b", va="center", ha="center", fontsize=12)
    figure.text(0.515, 0.50, "c", va="center", ha="center", fontsize=12)

    pyplot.savefig("../data/figures/supporting03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    supporting01()
    # supporting02()
    supporting03()
