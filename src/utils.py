"""
Módulo de utilidades para el proyecto.

Este módulo contiene funciones auxiliares para cálculo de errores, guardado de
figuras y visualización de tablas en formato legible.
"""

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

def l2_error(y_num, y_true):
    """
    Calcula el error L2 (norma euclidiana normalizada) entre solución numérica y analítica.
    
    Args:
        y_num: Solución numérica (array)
        y_true: Solución verdadera/analítica (array)
    
    Returns:
        Error L2: ||y_num - y_true|| / sqrt(n)
    """
    return np.linalg.norm(y_num - y_true) / np.sqrt(y_num.size)

def linf_error(y_num, y_true):
    """
    Calcula el error L∞ (norma del máximo) entre solución numérica y analítica.
    
    Args:
        y_num: Solución numérica (array)
        y_true: Solución verdadera/analítica (array)
    
    Returns:
        Error L∞: max(|y_num - y_true|)
    """
    return np.max(np.abs(y_num - y_true))

def save_figure(fig, fname, tight=True):
    """
    Guarda una figura de matplotlib en el directorio figures/.
    
    Args:
        fig: Figura de matplotlib a guardar
        fname: Nombre del archivo (se guardará en figures/fname)
        tight: Si es True, aplica tight_layout antes de guardar (default: True)
    """
    os.makedirs("figures", exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=200)

def print_table(headers, rows):
    """
    Imprime una tabla formateada usando tabulate.
    
    Args:
        headers: Lista de nombres de columnas
        rows: Lista de filas (cada fila es una lista de valores)
    """
    print(tabulate(rows, headers=headers, tablefmt="github"))
