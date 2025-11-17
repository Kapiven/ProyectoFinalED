import numpy as np
from .methods import euler_step, rk4_step, ab2_step

def solve(f, y0, t_span, h, method='rk4'):
    """
    f: function f(t, y) -> array_like
    y0: ndarray initial state
    t_span: (t0, tf)
    h: step size
    method: 'euler', 'rk4', 'ab2'
    returns: t_vals (N+1,), y_vals (N+1, dim)
    """
    t0, tf = t_span
    t_vals = np.arange(t0, tf + 1e-12, h)
    n_steps = len(t_vals)
    y0 = np.asarray(y0)
    dim = y0.shape[0] if y0.ndim>0 else 1
    y_vals = np.zeros((n_steps, dim))
    y_vals[0] = y0
    # helper to call f and ensure ndarray
    def F(tt, yy):
        res = f(tt, yy)
        return np.asarray(res)
    if method == 'euler':
        for i in range(n_steps-1):
            t = t_vals[i]
            y = y_vals[i]
            y_vals[i+1] = euler_step(F, t, y, h)
        return t_vals, y_vals
    elif method == 'rk4':
        for i in range(n_steps-1):
            t = t_vals[i]
            y = y_vals[i]
            y_vals[i+1] = rk4_step(F, t, y, h)
        return t_vals, y_vals
    elif method == 'ab2':
        # need one starter step (use RK4)
        if n_steps < 2:
            return t_vals, y_vals
        # first step with RK4
        y_vals[0] = y0
        y_vals[1] = rk4_step(F, t_vals[0], y_vals[0], h)
        f_prev = F(t_vals[0], y_vals[0])
        f_curr = F(t_vals[1], y_vals[1])
        for i in range(1, n_steps-1):
            y_next = ab2_step(f_prev, f_curr, y_vals[i], h)
            y_vals[i+1] = y_next
            # rotate
            f_prev = f_curr
            f_curr = F(t_vals[i+1], y_vals[i+1])
        return t_vals, y_vals
    else:
        raise ValueError("Unknown method")
