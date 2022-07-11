from matplotlib import pyplot, markers, rcParams
from numpy import arange, linspace, log10, argsort, min, max, where

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
    figure = pyplot.figure(figsize=(10, 6))

    lipschitz_constants = load_data("../data/results/task01/lipschitz loop.npy")
    usages = where(lipschitz_constants > 0)
    locations = load_data("../data/results/task01/location loop.npy")[usages]
    lipschitz_constants = lipschitz_constants[usages]
    order = argsort(lipschitz_constants)

    ax1 = pyplot.subplot(1, 2, 1)
    pyplot.scatter(locations[order, 0], locations[order, 1], c=log10(lipschitz_constants[order]),
                   cmap="RdYlGn_r", vmin=-1, vmax=2)
    pyplot.text(110, 110, "loop", fontsize=10, va="top", ha="right")
    pyplot.vlines(0, -120, 120, color="black", linewidth=0.75, linestyle="--", zorder=2)
    pyplot.hlines(0, -120, 120, color="black", linewidth=0.75, linestyle="--", zorder=2)
    pyplot.xlabel("t-SNE of output landscape difference", fontsize=10)
    pyplot.ylabel("t-SNE of output landscape difference", fontsize=10)
    pyplot.xlim(-120, 120)
    pyplot.ylim(-120, 120)
    pyplot.xticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.yticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)

    lipschitz_constants = load_data("../data/results/task01/lipschitz collider.npy")
    usages = where(lipschitz_constants > 0)
    locations = load_data("../data/results/task01/location collider.npy")[usages]
    lipschitz_constants = lipschitz_constants[usages]

    ax2 = pyplot.subplot(1, 2, 2)
    order = argsort(lipschitz_constants)
    mesh = pyplot.scatter(locations[order, 0], locations[order, 1], c=log10(lipschitz_constants[order]),
                          cmap="RdYlGn_r", vmin=-1, vmax=2)
    pyplot.text(110, 110, "collider", fontsize=10, va="top", ha="right")
    pyplot.vlines(0, -120, 120, color="black", linewidth=0.75, linestyle="--", zorder=2)
    pyplot.hlines(0, -120, 120, color="black", linewidth=0.75, linestyle="--", zorder=2)
    pyplot.xlabel("t-SNE of output landscape difference", fontsize=10)
    pyplot.ylabel("t-SNE of output landscape difference", fontsize=10)
    pyplot.xlim(-120, 120)
    pyplot.ylim(-120, 120)
    pyplot.xticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)
    pyplot.yticks([-120, 0, 120], ["\N{MINUS SIGN}1", "0", "+1"], fontsize=10)

    bar = figure.colorbar(mesh, ax=[ax1, ax2], shrink=0.6, ticks=[-1, 0, 1, 2], location="bottom")
    bar.set_label("Lipschitz constant", fontsize=10)
    bar.ax.set_yticklabels(["1E\N{MINUS SIGN}1", "1E+0", "1E+1", "1E+2"])
    bar.ax.tick_params(labelsize=10)
    pyplot.savefig("../data/figures/supporting02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supporting03():
    landscapes = load_data("../data/results/task02/intuition landscapes.npy")

    pyplot.figure(figsize=(10, 12), tight_layout=True)
    for index, landscape in enumerate(landscapes):
        pyplot.subplot(12, 12, index + 1)
        pyplot.title(str(index + 1), fontsize=10)
        pyplot.pcolormesh(arange(41), arange(41), landscape, vmin=-1, vmax=+1, cmap="rainbow")
        pyplot.xlabel("x", fontsize=10)
        pyplot.ylabel("y", fontsize=10)
        pyplot.xlim(0, 40)
        pyplot.ylim(0, 40)
        pyplot.xticks([])
        pyplot.yticks([])

    pyplot.savefig("../data/figures/supporting03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supporting04():
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
        pyplot.text(x=(info[1][0] + info[1][2]) / 2.0 - 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                    s=adjust_format("%.1e" % motif.w[0].value()), fontsize=9, va="bottom", ha="right")
        pyplot.text(x=(info[1][1] + info[1][2]) / 2.0 + 0.03, y=(info[1][0] + info[1][2]) / 2.0 + 0.08,
                    s=adjust_format("%.1e" % motif.w[1].value()), fontsize=9, va="bottom", ha="left")
        pyplot.text(x=info[1][2], y=info[2][2] + 0.1, s=motif.a[0] + " / " + motif.g[0], fontsize=9,
                    va="bottom", ha="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
        pyplot.xlim(0.0, 1.0)
        pyplot.ylim(0.1, 0.9)
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

    pyplot.savefig("../data/figures/supporting04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supporting05():
    landscapes = load_data("../data/results/task03/landscapes.npy")

    pyplot.figure(figsize=(10, 12), tight_layout=True)
    for index, landscape in enumerate(landscapes):
        pyplot.subplot(12, 12, index + 1)
        pyplot.title(str(index + 1), fontsize=10)
        pyplot.pcolormesh(arange(41), arange(41), landscape, vmin=-1, vmax=+1, cmap="rainbow")
        pyplot.xlabel("x", fontsize=10)
        pyplot.ylabel("y", fontsize=10)
        pyplot.xlim(0, 40)
        pyplot.ylim(0, 40)
        pyplot.xticks([])
        pyplot.yticks([])

    pyplot.savefig("../data/figures/supporting05.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supporting06():
    pass
    # losses = load_data(load_path="../data/results/task03/max-min losses.npy")
    # constants = load_data(load_path="../data/results/task03/lipschitz constants.npy")
    # changes = constants[:, 1] - constants[:, 0]
    # right_indices, wrong_indices = where(changes > 0)[0], where(changes <= 0)[0]
    #
    # pyplot.figure(figsize=(10, 8))
    # rcParams["font.family"] = "Times New Roman"
    #
    # pyplot.subplot(2, 1, 1)
    # pyplot.scatter(changes[right_indices], losses[right_indices], color="silver", edgecolor="black", label="positive")
    # pyplot.scatter(changes[wrong_indices], losses[wrong_indices], color="black", label="negative")
    # pyplot.legend(loc="upper right", fontsize=8)
    #
    # pyplot.subplot(2, 1, 2)
    # pyplot.show()


def supporting07():
    lipschitz_paths = load_data(load_path="../data/results/task03/lipschitz paths.pkl")
    rugosity_paths = load_data(load_path="../data/results/task03/rugosity paths.pkl")

    pyplot.figure(figsize=(10, 12), tight_layout=True)
    for index, (lipschitz_path, rugosity_path) in enumerate(zip(lipschitz_paths, rugosity_paths)):
        pyplot.subplot(12, 12, index + 1)
        lipschitz_path -= min(lipschitz_path)
        lipschitz_path /= max(lipschitz_path)
        rugosity_path -= min(rugosity_path)
        rugosity_path /= max(rugosity_path)
        pyplot.title(str(index + 1), fontsize=10)
        pyplot.plot(linspace(0, 1, len(lipschitz_path)), lipschitz_path, color="blue")
        pyplot.plot(linspace(0, 1, len(rugosity_path)), rugosity_path, color="orange")
        pyplot.xticks([])
        pyplot.yticks([])

    pyplot.savefig("../data/figures/supporting07.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    # supporting01()
    # supporting02()
    # supporting03()
    # supporting04()
    # supporting05()
    supporting07()
