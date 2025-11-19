"""
Módulo de métodos numéricos para resolver ecuaciones diferenciales ordinarias.

Este módulo contiene las implementaciones de los métodos numéricos utilizados
para resolver EDOs: Euler (orden 1), Runge-Kutta de cuarto orden (RK4, orden 4),
y Adams-Bashforth de dos pasos (AB2, orden 2).
"""

import numpy as np

def euler_step(f, t, y, h):
    """
    Realiza un paso del método de Euler (orden 1).
    
    Args:
        f: Función f(t, y) que define la EDO
        t: Tiempo actual
        y: Estado actual
        h: Tamaño del paso
    
    Returns:
        Estado siguiente: y_{n+1} = y_n + h * f(t_n, y_n)
    """
    return y + h * f(t, y)

def rk4_step(f, t, y, h):
    """
    Realiza un paso del método de Runge-Kutta de cuarto orden (RK4).
    
    Args:
        f: Función f(t, y) que define la EDO
        t: Tiempo actual
        y: Estado actual
        h: Tamaño del paso
    
    Returns:
        Estado siguiente calculado con el método RK4
    """
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def ab2_step(f_prev, f_curr, y_curr, h):
    """
    Realiza un paso del método de Adams-Bashforth de dos pasos (AB2).
    
    Args:
        f_prev: Valor de f(t_{n-1}, y_{n-1})
        f_curr: Valor de f(t_n, y_n)
        y_curr: Estado actual y_n
        h: Tamaño del paso
    
    Returns:
        Estado siguiente: y_{n+1} = y_n + h*(3/2 f_n - 1/2 f_{n-1})
    """
    return y_curr + h*(1.5*f_curr - 0.5*f_prev)
