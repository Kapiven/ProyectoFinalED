import numpy as np
import matplotlib.pyplot as plt
from .integrator import solve
from .problems import expo_problem, logistic_problem, harmonic_oscillator, rotation_system, rosenzweig_macarthur
from .utils import l2_error, linf_error, save_figure, print_table
import os

def convergence_test(problem_factory, analytic_available=True, methods=['rk4','ab2'], hs=[0.5,0.1,0.05,0.01], T=5.0):
    f, y0, analytic = problem_factory()
    rows = []
    for method in methods:
        errors = []
        for h in hs:
            t, y = solve(f, y0, (0.0, T), h, method=method)
            yT_num = y[-1]
            if analytic_available:
                yT_true = analytic(T)
                e = l2_error(yT_num, yT_true)
            else:
                # if no analytic, use very fine RK4 as reference
                tr, yr = solve(f, y0, (0.0, T), 1e-4, method='rk4')
                yT_true = yr[-1]
                e = l2_error(yT_num, yT_true)
            rows.append([method, h, float(e)])
            errors.append(e)
        # plot error vs h
    return rows

def plot_solution(problem_factory, methods=['rk4','ab2','euler'], h=0.01, T=20.0, name_prefix="sol"):
    f, y0, analytic = problem_factory()
    t_ref, y_ref = solve(f, y0, (0.0, T), 1e-4, method='rk4')
    os.makedirs("figures", exist_ok=True)
    plt.figure()
    for method in methods:
        t, y = solve(f, y0, (0.0, T), h, method=method)
        if y.shape[1] == 1:
            plt.plot(t, y[:,0], label=method)
            plt.plot(t_ref, y_ref[:,0], '--', label='ref')
        else:
            plt.plot(t, y[:,0], label=f"{method} x")
            plt.plot(t, y[:,1], label=f"{method} y", alpha=0.6)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("solution")
    plt.title(f"Solution comparison")
    save_figure(plt.gcf(), f"{name_prefix}.png")
    plt.close()

def run_all():
    # Convergence tests examples
    os.makedirs("figures", exist_ok=True)
    hs = [0.5, 0.1, 0.05, 0.01]
    print("Convergence - Exponential")
    rows = convergence_test(expo_problem, analytic_available=True, methods=['rk4','ab2'], hs=hs, T=2.0)
    print_table(["method","h","L2_error"], rows)
    print("Convergence - Logistic")
    rows = convergence_test(logistic_problem, analytic_available=True, methods=['rk4','ab2'], hs=hs, T=5.0)
    print_table(["method","h","L2_error"], rows)

    # Harmonic oscillator plot
    plot_solution(harmonic_oscillator, methods=['rk4','ab2','euler'], h=0.01, T=20.0, name_prefix="harmonic")

    # Rosenzweig-MacArthur simulation (no analytic)
    f, y0 = rosenzweig_macarthur()
    t, y = solve(f, y0, (0.0, 100.0), 0.05, method='rk4')
    plt.figure()
    plt.plot(t, y[:,0], label='prey (x)')
    plt.plot(t, y[:,1], label='predator (y)')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("population")
    plt.title("Rosenzweig–MacArthur (RK4)")
    save_figure(plt.gcf(), "rosenzweig_time_series.png")
    plt.close()

    # Phase plot
    plt.figure()
    plt.plot(y[:,0], y[:,1])
    plt.xlabel("x (prey)")
    plt.ylabel("y (predator)")
    plt.title("Phase portrait (Rosenzweig–MacArthur)")
    save_figure(plt.gcf(), "rosenzweig_phase.png")
    plt.close()

if __name__ == "__main__":
    run_all()
    print("Experiments completed. Figures saved in ./figures/")
