import numpy as np
from scipy.linalg import expm
from math import cos, sin, sqrt, exp

# First order (analytical)
def expo_problem(r=1.0, y0=1.0):
    def f(t, y):
        return r * y
    def analytic(t):
        return y0 * np.exp(r * t)
    return f, np.array([y0]), analytic

def logistic_problem(r=1.0, K=10.0, y0=1.0):
    def f(t, y):
        return r * y * (1 - y / K)
    def analytic(t):
        # analytic solution for logistic with y0
        C = (K - y0) / y0
        return K / (1 + C * np.exp(-r * t))
    return f, np.array([y0]), analytic

# Second order -> system 1st order 
def harmonic_oscillator(omega=1.0, x0=1.0, v0=0.0):
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
    A = np.array([[0.0, -omega],[omega, 0.0]])
    def f(t, y):
        return A.dot(y)
    def analytic(t):
        M = expm(A*t)
        return M.dot(np.array([x0, y0]))
    return f, np.array([x0, y0]), analytic

def linear_stable(a=1.0, b=0.2, c=0.1, d=1.0, x0=1.0, y0=1.0):
    A = np.array([[-a, b],[c, -d]])
    def f(t, y):
        return A.dot(y)
    def analytic(t):
        M = expm(A*t)
        return M.dot(np.array([x0, y0]))
    return f, np.array([x0, y0]), analytic

# Rosenzweig-MacArthur (nonlinear 2x2)
def rosenzweig_macarthur(r=1.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.2, x0=5.0, y0=1.0):
    def f(t, y):
        x, yv = y
        functional = (a*x* yv) / (1 + a*h*x)
        dx = r*x*(1 - x/K) - functional
        dy = e*functional - m*yv
        return np.array([dx, dy])
    return f, np.array([x0, y0])

# Optional extended version
def rosenzweig_extended(r=1.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.2, c=0.01, x0=5.0, y0=1.0):
    def f(t, y):
        x, yv = y
        functional = (a*x* yv) / (1 + a*h*x)
        dx = r*x*(1 - x/K) - functional
        dy = e*functional - m*yv - c*yv**2
        return np.array([dx, dy])
    return f, np.array([x0, y0])
