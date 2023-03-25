from datetime import datetime
from numpy import zeros
from numpy import load as n_load
from numpy import save as n_save
from pickle import load as p_load
from pickle import dump as p_save

from practice import acyclic_motifs


draw_info = {
    "collider": ("#88CCF8", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2]),
    "fork": ("#FFFFFF", [0.5, 0.2, 0.8], [0.20, 0.70, 0.70], [1], [3], [2]),
    "chain": ("#FFFFFF", [0.2, 0.5, 0.8], [0.20, 0.45, 0.70], [1], [3], []),
    "coherent-loop": ("#FCE0AB", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2]),
    "incoherent-loop": ("#FCB1AB", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2])
}


def adjust_format(value):
    if value[0] != "-":
        value = "+" + value
    value = value.replace("e", "E").replace("-", "\N{MINUS SIGN}")
    value = value.replace("00", "0").replace("01", "1").replace("02", "2").replace("03", "3")
    value = value.replace("04", "4").replace("05", "5").replace("06", "6").replace("07", "7")
    return value


class Monitor(object):

    def __init__(self):
        self.last_time = None

    def __call__(self, current_state, total_state, extra=None):
        if self.last_time is None:
            self.last_time = datetime.now()

        if current_state == 0:
            return

        position = int(current_state / total_state * 100)

        string = "|"

        for index in range(0, 100, 5):
            if position >= index:
                string += "█"
            else:
                string += " "

        string += "|"

        pass_time = (datetime.now() - self.last_time).total_seconds()
        wait_time = int(pass_time * (total_state - current_state) / current_state)

        string += " " * (3 - len(str(position))) + str(position) + "% ("

        string += " " * (len(str(total_state)) - len(str(current_state))) + str(current_state) + "/" + str(total_state)

        if current_state < total_state:
            minute, second = divmod(wait_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") wait " + "%04d:%02d:%02d" % (hour, minute, second)
        else:
            minute, second = divmod(pass_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") used " + "%04d:%02d:%02d" % (hour, minute, second)

        if extra is not None:
            string += " " + str(extra).replace("\'", "").replace("{", "(").replace("}", ")") + "."
        else:
            string += "."

        print("\r" + string, end="", flush=True)

        if current_state >= total_state:
            self.last_time = None
            print()


def get_reference_motifs():
    references = []
    for key, motifs in acyclic_motifs.items():
        for motif in motifs:
            matrix = zeros(shape=(3, 3))
            for former, latter in motif.edges:
                matrix[former - 1, latter - 1] = motif.get_edge_data(former, latter)["weight"]
            references.append(matrix)
    return references


def save_data(save_path, information):
    if ".pkl" in save_path:
        with open(save_path, "wb") as file:
            p_save(obj=information, file=file)
    elif ".npy" in save_path:
        n_save(file=save_path, arr=information)
    else:
        raise ValueError("No such type of file path.")


def load_data(load_path):
    if ".pkl" in load_path:
        with open(load_path, "rb") as file:
            return p_load(file=file)
    elif ".npy" in load_path:
        return n_load(load_path)
    else:
        raise ValueError("No such type of file path.")
