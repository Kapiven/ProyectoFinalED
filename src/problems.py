"""
Módulo de definición de problemas de ecuaciones diferenciales.

Este módulo contiene las definiciones de diversos problemas de EDOs con sus
soluciones analíticas cuando están disponibles. Incluye problemas de primer orden
(exponencial, logística), segundo orden (oscilador armónico, oscilador amortiguado),
sistemas lineales (rotación, sistemas estables), y el modelo no lineal
depredador-presa de Rosenzweig-MacArthur.
"""

import numpy as np
from scipy.linalg import expm
from math import cos, sin, sqrt, exp

# First order (analytical)
def expo_problem(r=1.0, y0=1.0):
    """
    Define el problema de crecimiento exponencial: dy/dt = r*y.
    
    Args:
        r: Tasa de crecimiento (default: 1.0)
        y0: Condición inicial (default: 1.0)
    
    Returns:
        f: Función f(t, y) = r*y
        y0: Array con condición inicial [y0]
        analytic: Función solución analítica y(t) = y0 * exp(r*t)
    """
    def f(t, y):
        return r * y
    def analytic(t):
        return y0 * np.exp(r * t)
    return f, np.array([y0]), analytic

def logistic_problem(r=1.0, K=10.0, y0=1.0):
    """
    Define el problema de crecimiento logístico: dy/dt = r*y*(1 - y/K).
    
    Args:
        r: Tasa de crecimiento intrínseca (default: 1.0)
        K: Capacidad de carga (default: 10.0)
        y0: Condición inicial (default: 1.0)
    
    Returns:
        f: Función f(t, y) = r*y*(1 - y/K)
        y0: Array con condición inicial [y0]
        analytic: Función solución analítica y(t) = K / (1 + C*exp(-r*t))
    """
    def f(t, y):
        return r * y * (1 - y / K)
    def analytic(t):
        # analytic solution for logistic with y0
        C = (K - y0) / y0
        return K / (1 + C * np.exp(-r * t))
    return f, np.array([y0]), analytic

# Second order -> system 1st order 
def harmonic_oscillator(omega=1.0, x0=1.0, v0=0.0):
    """
    Define el problema del oscilador armónico: d²x/dt² = -ω²x.
    Convertido a sistema de primer orden: [x, v] donde v = dx/dt.
    
    Args:
        omega: Frecuencia angular (default: 1.0)
        x0: Posición inicial (default: 1.0)
        v0: Velocidad inicial (default: 0.0)
    
    Returns:
        f: Función f(t, [x, v]) = [v, -ω²x]
        y0: Array con condiciones iniciales [x0, v0]
        analytic: Función solución analítica [x(t), v(t)]
    """
    def f(t, y):
        # y = [x, v]
        x, v = y
        dxdt = v
        dvdt = -omega**2 * x
        return np.array([dxdt, dvdt])
    def analytic(t):
        x = x0 * cos(omega * t) + (v0/omega) * sin(omega * t)
        v = -x0 * omega * sin(omega * t) + v0 * cos(omega * t)
        return np.array([x, v])
    return f, np.array([x0, v0]), analytic

def damped_oscillator(omega=1.0, zeta=0.1, x0=1.0, v0=0.0):
    """
    Define el problema del oscilador amortiguado: d²x/dt² + 2ζω(dx/dt) + ω²x = 0.
    
    Args:
        omega: Frecuencia angular (default: 1.0)
        zeta: Coeficiente de amortiguamiento (default: 0.1)
        x0: Posición inicial (default: 1.0)
        v0: Velocidad inicial (default: 0.0)
    
    Returns:
        f: Función f(t, [x, v]) que define el sistema
        y0: Array con condiciones iniciales [x0, v0]
        analytic: Función solución analítica [x(t), v(t)]
    """
    def f(t, y):
        x, v = y
        dxdt = v
        dvdt = -2*zeta*omega*v - omega**2 * x
        return np.array([dxdt, dvdt])
    def analytic(t):
        # analytic depends on zeta: implement underdamped case
        wd = omega*sqrt(max(0.0, 1 - zeta**2))
        A = x0
        B = (v0 + zeta*omega*x0)/wd
        x = np.exp(-zeta*omega*t)*(A*cos(wd*t) + B*sin(wd*t))
        v = np.exp(-zeta*omega*t)*(-A*wd*sin(wd*t) + B*wd*cos(wd*t)) - zeta*omega*x
        return np.array([x, v])
    return f, np.array([x0, v0]), analytic

# Linear 2x2
def rotation_system(omega=1.0, x0=1.0, y0=0.0):
    """
    Define un sistema lineal de rotación: d/dt [x, y] = A * [x, y]
    donde A = [[0, -ω], [ω, 0]].
    
    Args:
        omega: Velocidad angular de rotación (default: 1.0)
        x0: Condición inicial para x (default: 1.0)
        y0: Condición inicial para y (default: 0.0)
    
    Returns:
        f: Función f(t, [x, y]) = A * [x, y]
        y0: Array con condiciones iniciales [x0, y0]
        analytic: Función solución analítica usando exponencial de matriz
    """
    A = np.array([[0.0, -omega],[omega, 0.0]])
    def f(t, y):
        return A.dot(y)
    def analytic(t):
        M = expm(A*t)
        return M.dot(np.array([x0, y0]))
    return f, np.array([x0, y0]), analytic

def linear_stable(a=1.0, b=0.2, c=0.1, d=1.0, x0=1.0, y0=1.0):
    """
    Define un sistema lineal estable 2x2: d/dt [x, y] = A * [x, y]
    donde A = [[-a, b], [c, -d]].
    
    Args:
        a, b, c, d: Parámetros de la matriz A
        x0: Condición inicial para x (default: 1.0)
        y0: Condición inicial para y (default: 1.0)
    
    Returns:
        f: Función f(t, [x, y]) = A * [x, y]
        y0: Array con condiciones iniciales [x0, y0]
        analytic: Función solución analítica usando exponencial de matriz
    """
    A = np.array([[-a, b],[c, -d]])
    def f(t, y):
        return A.dot(y)
    def analytic(t):
        M = expm(A*t)
        return M.dot(np.array([x0, y0]))
    return f, np.array([x0, y0]), analytic

# Rosenzweig-MacArthur (nonlinear 2x2)
def rosenzweig_macarthur(r=1.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.2, x0=5.0, y0=1.0):
    """
    Define el modelo depredador-presa de Rosenzweig-MacArthur.
    
    Sistema de ecuaciones:
        dx/dt = r*x*(1 - x/K) - (a*x*y) / (1 + a*h*x)
        dy/dt = e*(a*x*y) / (1 + a*h*x) - m*y
    
    Args:
        r: Tasa de crecimiento intrínseca de presas (default: 1.0)
        K: Capacidad de carga para presas (default: 10.0)
        a: Tasa de encuentro/depredación (default: 1.0)
        h: Tiempo de manejo (handling time) (default: 0.1)
        e: Eficiencia de conversión (default: 0.5)
        m: Tasa de mortalidad de predadores (default: 0.2)
        x0: Población inicial de presas (default: 5.0)
        y0: Población inicial de predadores (default: 1.0)
    
    Returns:
        f: Función f(t, [x, y]) que define el sistema
        y0: Array con condiciones iniciales [x0, y0]
        (No tiene solución analítica conocida)
    """
    def f(t, y):
        x, yv = y
        functional = (a*x* yv) / (1 + a*h*x)
        dx = r*x*(1 - x/K) - functional
        dy = e*functional - m*yv
        return np.array([dx, dy])
    return f, np.array([x0, y0])

# Optional extended version
def rosenzweig_extended(r=1.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.2, c=0.01, x0=5.0, y0=1.0):
    """
    Versión extendida del modelo Rosenzweig-MacArthur con competencia entre predadores.
    
    Similar a rosenzweig_macarthur pero con término adicional -c*y² en la ecuación
    de los predadores para modelar competencia intraspecífica.
    
    Args:
        r, K, a, h, e, m, x0, y0: Parámetros iguales a rosenzweig_macarthur
        c: Coeficiente de competencia entre predadores (default: 0.01)
    
    Returns:
        f: Función f(t, [x, y]) que define el sistema extendido
        y0: Array con condiciones iniciales [x0, y0]
    """
    def f(t, y):
        x, yv = y
        functional = (a*x* yv) / (1 + a*h*x)
        dx = r*x*(1 - x/K) - functional
        dy = e*functional - m*yv - c*yv**2
        return np.array([dx, dy])
    return f, np.array([x0, y0])
