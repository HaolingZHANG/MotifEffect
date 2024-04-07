"""
@Author      : Haoling Zhang
@Description : Generate videos of the escape process for this work.
"""
from cv2 import imread, VideoWriter, resize
from gc import collect
from logging import getLogger, CRITICAL
from matplotlib import pyplot, rcParams
from numpy import linspace
from os import path, remove, mkdir
from typing import Tuple
from warnings import filterwarnings

from effect import NeuralMotif, calculate_landscape
from works import load_data

filterwarnings("ignore")

getLogger("matplotlib").setLevel(CRITICAL)

rcParams["font.family"] = "Arial"
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Linux Libertine"
rcParams["mathtext.cal"] = "Lucida Calligraphy"
rcParams["mathtext.it"] = "Linux Libertine:italic"
rcParams["mathtext.bf"] = "Linux Libertine:bold"

motif_types, motif_indices = ["incoherent-loop", "coherent-loop", "collider"], [1, 2, 3, 4]
raw_path, temp_path, fps, figure_size = "./raw/", "./temp/", 24, (1800, 1080)


def output_info_frame(info: str,
                      color: str,
                      size: int,
                      frame_index: int) \
        -> Tuple[str, bool]:
    """
    Plot the information frame.

    :param info: information of this frame.
    :type info: str

    :param color: background color of this frame.
    :type color: str

    :param size: font size.
    :type size: int

    :param frame_index: frame index in the video.
    :type frame_index: int

    :return: image path and boolean indicating information layout.
    :rtype: str, bool
    """
    if not path.exists(path=temp_path + str(frame_index).zfill(5) + ".png"):
        pyplot.figure(figsize=(10, 6), tight_layout=True)
        pyplot.fill_between([0, 1], 0, 1, color=color, zorder=0)
        pyplot.text(0.5, 0.5, info, va="center", ha="center", fontsize=size, zorder=1)
        pyplot.xlim(0, 1)
        pyplot.ylim(0, 1)
        pyplot.axis("off")
        pyplot.savefig(temp_path + str(frame_index).zfill(5) + ".png", bbox_inches="tight", dpi=300)
        pyplot.close()

        collect()

    return temp_path + str(frame_index).zfill(5) + ".png", True


def output_data_frame(source_motif: NeuralMotif,
                      target_motif: NeuralMotif,
                      loss: float,
                      iteration: int,
                      frame_index: int) \
        -> Tuple[str, bool]:
    """
    Plot the data frame.

    :param source_motif: source motif object, i.e., loops in this work.
    :type source_motif: effect.networks.NeuralMotif

    :param target_motif: target motif object, i.e., colliders in this work.
    :type target_motif: effect.networks.NeuralMotif

    :param loss: L2 loss value between landscapes of source and target motif.
    :type loss: float

    :param iteration: iteration number in the maximum-minimum loss search.
    :type iteration: int

    :param frame_index: frame index in the video.
    :type frame_index: int

    :return: image path and boolean indicating information layout.
    :rtype: str, bool
    """
    if not path.exists(path=temp_path + str(frame_index).zfill(5) + ".png"):
        value_range, points = (-1, +1), 41

        pyplot.figure(figsize=(10, 6), tight_layout=True)
        pyplot.text(0.020, 2.37, "iteration: " + str(iteration), va="center", ha="left", fontsize=20, zorder=2)
        pyplot.text(2.700, 2.37, "loss: %.5f" % loss, va="center", ha="right", fontsize=20, zorder=2)
        pyplot.text(0.925, 2.10, "motif information", va="center", ha="center", fontsize=12)
        pyplot.fill_between([0.05, 1.80], 1.98, 2.02, color="#DDDDDD", zorder=0)
        for matrix_index in range(2):
            pyplot.scatter([0.05], [matrix_index + 0.20], fc="w", ec="k", lw=1.5, s=40, zorder=2)
            pyplot.scatter([0.45], [matrix_index + 0.20], fc="silver", ec="k", lw=1.5, s=40, zorder=2)
            pyplot.scatter([0.25], [matrix_index + 0.80], fc="k", ec="k", lw=1.5, s=40, zorder=2)
            pyplot.text(0.05, matrix_index + 0.15, "$x$", va="top", ha="center", fontsize=12)
            pyplot.text(0.45, matrix_index + 0.15, "$y$", va="top", ha="center", fontsize=12)
            pyplot.text(0.25, matrix_index + 0.85, "$z$", va="bottom", ha="center", fontsize=12)
        pyplot.annotate("", xy=(0.05, 0.20), xytext=(0.25, 0.80),
                        arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="black",
                                        shrinkA=6, shrinkB=6, lw=1.5,
                                        ls=("-" if target_motif.w[0].value() > 0 else ":")))
        pyplot.annotate("", xy=(0.45, 0.20), xytext=(0.25, 0.80),
                        arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="black",
                                        shrinkA=6, shrinkB=6, lw=1.5,
                                        ls=("-" if target_motif.w[1].value() > 0 else ":")))
        pyplot.annotate("", xy=(0.05, 1.20), xytext=(0.45, 1.20),
                        arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="black",
                                        shrinkA=6, shrinkB=6, lw=1.5,
                                        ls=("-" if source_motif.w[0].value() > 0 else ":")))
        pyplot.annotate("", xy=(0.05, 1.20), xytext=(0.25, 1.80),
                        arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="black",
                                        shrinkA=6, shrinkB=6, lw=1.5,
                                        ls=("-" if source_motif.w[1].value() > 0 else ":")))
        pyplot.annotate("", xy=(0.45, 1.20), xytext=(0.25, 1.80),
                        arrowprops=dict(arrowstyle="<|-, head_length=0.2, head_width=0.15", color="black",
                                        shrinkA=6, shrinkB=6, lw=1.5,
                                        ls=("-" if source_motif.w[2].value() > 0 else ":")))
        for location, info_1, info_2, value in zip([0.65, 0.50, 0.35],
                                                   ["weight", "weight", "bias"],
                                                   [r"$x \rightarrow z$", r"$y \rightarrow z$",
                                                    r"$x,y \rightarrow z$"],
                                                   [target_motif.w[0].value(), target_motif.w[1].value(),
                                                    target_motif.b[0].value()]):
            pyplot.text(0.55, location, info_1, va="center", ha="left", fontsize=10)
            pyplot.text(0.75, location, info_2, va="center", ha="left", fontsize=12)
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
                                                   [source_motif.w[0].value(), source_motif.w[1].value(),
                                                    source_motif.w[2].value(), source_motif.b[0].value(),
                                                    source_motif.b[1].value()]):
            pyplot.text(0.55, location, info_1, va="center", ha="left", fontsize=10)
            pyplot.text(0.75, location, info_2, va="center", ha="left", fontsize=12)
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
        for matrix_index, matrix in enumerate([calculate_landscape(value_range, points, target_motif),
                                               calculate_landscape(value_range, points, source_motif)]):
            pyplot.pcolormesh(linspace(2.10, 2.50, 41), linspace(matrix_index + 0.2, matrix_index + 0.8, 41), matrix,
                              cmap="PRGn", vmin=-1, vmax=1, zorder=1)
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

        pyplot.xlim(0.00, 2.80)
        pyplot.ylim(0.00, 2.50)
        pyplot.axis("off")
        pyplot.savefig(temp_path + str(frame_index).zfill(5) + ".png", bbox_inches="tight", dpi=300)
        pyplot.close()

        collect()

    return temp_path + str(frame_index).zfill(5) + ".png", False


def generate_video():
    """
    Generate video of the escape process.
    """
    if not path.exists(raw_path + "videos/"):
        mkdir(raw_path + "videos/")

    for motif_index in motif_indices:
        for motif_type in motif_types[:2]:
            total_path = raw_path + "videos/" + motif_type + "." + str(motif_index) + ".avi"

            if path.exists(total_path):
                continue

            load_path = raw_path + "particular/" + motif_type + "." + str(motif_index) + ".escape-process.pkl"
            record, frame_links = load_data(load_path=load_path), []

            if motif_type == "incoherent-loop":
                info = "train incoherent loop " + str(motif_index) + "\n\nto escape from colliders"
                link = output_info_frame(info, "#FCB1AB", 40, len(frame_links) + 1)
            elif motif_type == "coherent-loop":
                info = "train coherent loop " + str(motif_index) + "\n\nto escape from colliders"
                link = output_info_frame(info, "#FCE0AB", 40, len(frame_links) + 1)
            else:
                raise ValueError("No such motif type!")
            frame_links.append(link)

            for sample_index, sample in enumerate(record):
                link = output_info_frame("case " + str(sample_index + 1).zfill(3) + " / " + str(len(record)),
                                         "w", 30, len(frame_links) + 1)
                frame_links.append(link)
                for index, ((motif_1, motif_2), loss) in enumerate(zip(sample[0], sample[1])):
                    link = output_data_frame(motif_1, motif_2, loss, index + 1, len(frame_links) + 1)
                    frame_links.append(link)

            writer = VideoWriter(total_path, VideoWriter.fourcc(*"DIVX"), fps, figure_size)
            for frame_path, flag in frame_links:
                frame = imread(frame_path)

                if frame.shape != (figure_size[1], figure_size[0], 3):
                    frame = resize(frame, dsize=figure_size)

                if flag:
                    for _ in range(fps):
                        writer.write(frame)
                else:
                    writer.write(frame)
            writer.release()

            for link in frame_links:
                remove(link[0])


if __name__ == "__main__":
    generate_video()
