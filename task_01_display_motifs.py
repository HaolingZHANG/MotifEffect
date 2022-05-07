from matplotlib import pyplot
from matplotlib.markers import MarkerStyle

from hypothesis import acyclic_motifs


if __name__ == "__main__":
    draw_info = {
        "collider":
            ("#86E3CE", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1, 2], [3], [], r"$\mathcal{C}"),
        "fork":
            ("#D0E6A5", [0.5, 0.2, 0.8], [0.20, 0.70, 0.70], [1], [2, 3], [], r"$\mathcal{F}"),
        "chain":
            ("#FFDD94", [0.2, 0.5, 0.8], [0.20, 0.45, 0.70], [1], [3], [], r"$\mathcal{A}"),
        "coherent-loop":
            ("#CCABD8", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2], r"$\mathcal{L}"),
        "incoherent-loop":
            ("#FA897B", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2], r"$\mathcal{L}^\prime")
    }

    pyplot.figure(figsize=(10, 8))
    pyplot.subplots_adjust(wspace=0, hspace=0)
    grid = pyplot.GridSpec(4, 5)
    for type_index, (motif_type, motifs) in enumerate(acyclic_motifs.items()):
        info = draw_info[motif_type]
        for motif_index, motif in enumerate(motifs):
            pyplot.subplot(grid[motif_index, type_index])
            if motif_index == 0:
                pyplot.title(motif_type)
            pyplot.text(0.05, 0.95, info[6] + "_" + str(motif_index + 1) + "$", va="top", ha="left", fontsize=14)
            for index, (px, py) in enumerate(zip(info[1], info[2])):
                if index + 1 in info[3]:
                    pyplot.scatter(px, py, color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
                elif index + 1 in info[4]:
                    pyplot.scatter(px, py, color="black", edgecolor="black", lw=1.5, s=120, zorder=2)
                elif index + 1 in info[5]:
                    pyplot.scatter(px, py, marker=MarkerStyle("o", fillstyle="right"),
                                   color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
                    pyplot.scatter(px, py, marker=MarkerStyle("o", fillstyle="left"),
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

            pyplot.fill_between([0, 1], [0], [1], color=info[0], zorder=1)
            pyplot.xlim(0, 1)
            pyplot.ylim(0, 1)
            pyplot.xticks([])
            pyplot.yticks([])

    pyplot.subplot(grid[3, 0:2])
    pyplot.scatter([0.2], [0.2], color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.4], marker=MarkerStyle("o", fillstyle="right"),
                   color="white", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.4], marker=MarkerStyle("o", fillstyle="left"),
                   color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.6], color="gray", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.scatter([0.2], [0.8], color="black", edgecolor="black", lw=1.5, s=120, zorder=2)
    pyplot.annotate(s="", xy=(1.8, 0.7), xytext=(1.3, 0.7),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    shrinkA=0, shrinkB=0, lw=1.5), zorder=2)
    pyplot.annotate(s="", xy=(1.8, 0.3), xytext=(1.3, 0.3),
                    arrowprops=dict(arrowstyle="-|>", color="black", linestyle="dotted",
                                    shrinkA=0, shrinkB=0, lw=1.5), zorder=2)
    pyplot.text(0.3, 0.8, "output node", ha="left", va="center")
    pyplot.text(0.3, 0.6, "hidden node", ha="left", va="center")
    pyplot.text(0.3, 0.4, "hidden / input node", ha="left", va="center")
    pyplot.text(0.3, 0.2, "input node", ha="left", va="center")
    pyplot.text(1.52, 0.7, "positive\n\neffect", ha="center", va="center")
    pyplot.text(1.52, 0.3, "negative\n\neffect", ha="center", va="center")
    pyplot.xlim(0, 2)
    pyplot.ylim(0, 1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.axis("off")

    pyplot.savefig("./results/figures/[01] acyclic neural motifs.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()
