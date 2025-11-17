import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

def l2_error(y_num, y_true):
    return np.linalg.norm(y_num - y_true) / np.sqrt(y_num.size)

def linf_error(y_num, y_true):
    return np.max(np.abs(y_num - y_true))

def save_figure(fig, fname, tight=True):
    os.makedirs("figures", exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=200)

def print_table(headers, rows):
    print(tabulate(rows, headers=headers, tablefmt="github"))
