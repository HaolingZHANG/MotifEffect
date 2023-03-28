from numpy import load as n_load
from numpy import save as n_save
from pickle import load as p_load
from pickle import dump as p_save


draw_info = {
    "collider": ("#88CCF8", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2]),
    "fork": ("#FFFFFF", [0.5, 0.2, 0.8], [0.20, 0.70, 0.70], [1], [3], [2]),
    "chain": ("#FFFFFF", [0.2, 0.5, 0.8], [0.20, 0.45, 0.70], [1], [3], []),
    "coherent-loop": ("#FCE0AB", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2]),
    "incoherent-loop": ("#FCB1AB", [0.2, 0.8, 0.5], [0.20, 0.20, 0.70], [1], [3], [2])
}


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
