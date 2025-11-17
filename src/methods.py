import numpy as np

def euler_step(f, t, y, h):
    return y + h * f(t, y)

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def ab2_step(f_prev, f_curr, y_curr, h):
    # AB2: y_{n+1} = y_n + h*(3/2 f_n - 1/2 f_{n-1})
    return y_curr + h*(1.5*f_curr - 0.5*f_prev)
