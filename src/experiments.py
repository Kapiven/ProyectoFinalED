"""
Módulo de experimentos y simulaciones del proyecto.

Este módulo contiene las funciones principales para ejecutar:
1. Pruebas de validación y convergencia de métodos numéricos
2. Análisis de escenarios del modelo depredador-presa Rosenzweig-MacArthur

"""

import numpy as np
import matplotlib.pyplot as plt
from .integrator import solve
from .problems import expo_problem, logistic_problem, harmonic_oscillator, rosenzweig_macarthur
from .utils import l2_error, save_figure, print_table
from .scenarios import get_all_scenarios
import os

# -----------------------------------------------------------------------------
# PARTE 1: VALIDACIÓN DE MÉTODOS (Req. Rúbrica)
# -----------------------------------------------------------------------------

def run_validation_tests():
    """
    Ejecuta pruebas de convergencia para los modelos con solución analítica.
    Esto cumple con los requisitos de la rúbrica[cite: 205, 207, 218].
    """
    print("=================================================")
    print(" INICIANDO PRUEBAS DE VALIDACIÓN Y CONVERGENCIA ")
    print("=================================================\n")
    
    hs = [0.5, 0.1, 0.05, 0.01] # Tamaños de paso para el estudio de convergencia
    T = 5.0 # Tiempo final de simulación
    methods = ['rk4', 'ab2'] # Métodos a comparar
    
    # --- Prueba 1: ED 1er Orden (Exponencial) ---
    print("Convergence - Exponential")
    f, y0, analytic = expo_problem()
    rows, errors_dict = convergence_test(f, y0, analytic, methods, hs, T)
    print_table(["method", "h", "L2_error"], rows)
    plot_solution(f, y0, analytic, methods, h=0.1, T=T, name_prefix="convergence_exponential")
    plot_convergence(errors_dict, "convergence_exponential")
    print("\n")
    
    # --- Prueba 2: ED 1er Orden (Logística) ---
    print("Convergence - Logistic")
    f, y0, analytic = logistic_problem()
    rows, errors_dict = convergence_test(f, y0, analytic, methods, hs, T)
    print_table(["method", "h", "L2_error"], rows)
    plot_solution(f, y0, analytic, methods, h=0.1, T=T, name_prefix="convergence_logistic")
    plot_convergence(errors_dict, "convergence_logistic")
    print("\n")
    
    # --- Prueba 3: Oscilador Armónico (comparación de métodos) ---
    print("Harmonic Oscillator - Method Comparison")
    f, y0, analytic = harmonic_oscillator()
    # Para esta gráfica, usamos todos los métodos incluyendo Euler
    plot_solution_comparison(f, y0, methods=['rk4', 'ab2', 'euler'], h=0.01, T=20.0, name_prefix="harmonic")
    print("\n")


def convergence_test(f, y0, analytic, methods, hs, T):
    """
    Función auxiliar para calcular errores de convergencia.
    
    Returns:
        rows: Lista de [method, h, error] para la tabla
        errors_dict: Diccionario con errores por método para gráficas
    """
    rows = []
    errors_dict = {}
    for method in methods:
        errors = []
        for h in hs:
            t, y = solve(f, y0, (0.0, T), h, method=method)
            yT_num = y[-1]
            yT_true = analytic(T)
            e = l2_error(yT_num, yT_true)
            rows.append([method, h, f"{e:.2e}"])
            errors.append((h, float(e)))
        errors_dict[method] = errors
    return rows, errors_dict

def plot_convergence(errors_dict, name_prefix):
    """
    Genera gráfica de error vs tamaño de paso (h) para visualizar la convergencia.
    
    Args:
        errors_dict: Diccionario con errores por método {method: [(h, error), ...]}
        name_prefix: Prefijo para el nombre del archivo
    """
    plt.figure(figsize=(10, 6))
    
    for method, errors in errors_dict.items():
        hs = [h for h, _ in errors]
        errs = [e for _, e in errors]
        plt.loglog(hs, errs, 'o-', label=f'{method}', linewidth=2, markersize=8)
    
    plt.xlabel("Tamaño de paso (h)", fontsize=12)
    plt.ylabel("Error L2", fontsize=12)
    plt.title(f"Convergencia: Error vs Tamaño de Paso - {name_prefix}", fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6, which='both')
    save_figure(plt.gcf(), f"{name_prefix}_convergence.png")
    plt.close()


def plot_solution(f, y0, analytic, methods, h, T, name_prefix):
    """
    Función auxiliar para graficar la solución numérica vs. la analítica.
    """
    plt.figure(figsize=(10, 6))
    
    # Graficar solución analítica de referencia
    t_ref = np.linspace(0, T, 200)
    y_ref = np.array([analytic(t) for t in t_ref])
    
    # Asegurar que y_ref sea 2D
    if y_ref.ndim == 1:
        y_ref = y_ref.reshape(-1, 1)
    
    if y_ref.shape[1] == 1: # ED 1D
        plt.plot(t_ref, y_ref[:, 0], 'k--', label='Analítica', linewidth=2)
    else: # Sistemas
        plt.plot(t_ref, y_ref[:, 0], 'k--', label='Analítica (x)', linewidth=2)
        plt.plot(t_ref, y_ref[:, 1], 'k:', label='Analítica (y)', linewidth=2, alpha=0.7)

    # Graficar soluciones numéricas
    for method in methods:
        t, y = solve(f, y0, (0.0, T), h, method=method)
        if y.shape[1] == 1:
            plt.plot(t, y[:, 0], '.-', label=f'{method} (h={h})', alpha=0.8)
        else:
            plt.plot(t, y[:, 0], '.-', label=f'{method} x (h={h})', alpha=0.8)
            plt.plot(t, y[:, 1], '.--', label=f'{method} y (h={h})', alpha=0.8)

    plt.legend()
    plt.xlabel("Tiempo (t)")
    plt.ylabel("Solución y(t)")
    plt.title(f"Validación de Métodos: {name_prefix}")
    plt.grid(True, linestyle='--', alpha=0.6)
    save_figure(plt.gcf(), f"{name_prefix}_solucion.png")
    plt.close()


def plot_solution_comparison(f, y0, methods=['rk4','ab2','euler'], h=0.01, T=20.0, name_prefix="sol"):
    """
    Genera gráfica de comparación de métodos numéricos (como la original del oscilador armónico).
    Muestra las componentes x e y de cada método sin solución analítica de referencia.
    """
    plt.figure(figsize=(10, 6))
    
    # Graficar soluciones numéricas para cada método
    for method in methods:
        t, y = solve(f, y0, (0.0, T), h, method=method)
        if y.shape[1] == 1:
            plt.plot(t, y[:, 0], label=method, linewidth=1.5)
        else:
            plt.plot(t, y[:, 0], label=f"{method} x", linewidth=1.5)
            plt.plot(t, y[:, 1], label=f"{method} y", linewidth=1.5, alpha=0.8)
    
    plt.legend(fontsize=10)
    plt.xlabel("t", fontsize=11)
    plt.ylabel("solution", fontsize=11)
    plt.title("Solution comparison", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_figure(plt.gcf(), f"{name_prefix}.png")
    plt.close()


# -----------------------------------------------------------------------------
# PARTE 2: ANÁLISIS NO LINEAL (Req. Rúbrica y Usuario)
# -----------------------------------------------------------------------------

def analyze_scenario(scenario, T=100.0, h=0.05, method='rk4'):
    """
    Analiza un escenario específico y retorna métricas y datos.
    
    Returns:
        dict: Diccionario con 't', 'y', 'metrics' (valores finales, máximos, mínimos, etc.)
    """
    f, y0 = scenario.get_problem()
    t, y = solve(f, y0, (0.0, T), h, method=method)
    
    # Calcular métricas
    metrics = {
        'x_final': float(y[-1, 0]),
        'y_final': float(y[-1, 1]),
        'x_max': float(np.max(y[:, 0])),
        'y_max': float(np.max(y[:, 1])),
        'x_min': float(np.min(y[:, 0])),
        'y_min': float(np.min(y[:, 1])),
        'x_mean': float(np.mean(y[:, 0])),
        'y_mean': float(np.mean(y[:, 1])),
        'x_std': float(np.std(y[:, 0])),
        'y_std': float(np.std(y[:, 1])),
    }
    
    return {'t': t, 'y': y, 'metrics': metrics}


def run_nonlinear_scenarios():
    """
    Ejecuta simulaciones para el sistema no lineal (Rosenzweig-MacArthur)
    probando diferentes escenarios con distintos parámetros y condiciones iniciales.
    Genera tablas comparativas y gráficas para cada escenario.
    """
    print("=================================================")
    print(" INICIANDO ANÁLISIS DE ESCENARIOS NO LINEALES ")
    print("=================================================\n")
    
    scenarios = get_all_scenarios()
    T = 100.0  # Tiempo de simulación
    h = 0.05   # Paso de tiempo
    
    # Almacenar resultados para tabla comparativa
    comparison_rows = []
    all_results = []
    
    print(f"Total de escenarios a analizar: {len(scenarios)}\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{'='*60}")
        print(f"Escenario {i}: {scenario.name}")
        print(f"Descripción: {scenario.description}")
        print(f"Parámetros: r={scenario.r}, K={scenario.K}, a={scenario.a}, "
              f"h={scenario.h}, e={scenario.e}, m={scenario.m}")
        print(f"Condiciones iniciales: x0={scenario.x0}, y0={scenario.y0}")
        print(f"{'='*60}\n")
        
        # Analizar escenario
        result = analyze_scenario(scenario, T=T, h=h, method='rk4')
        all_results.append((scenario, result))
        
        # Agregar a tabla comparativa
        m = result['metrics']
        comparison_rows.append([
            scenario.name,
            f"{scenario.x0:.2f}",
            f"{scenario.y0:.2f}",
            f"{m['x_final']:.4f}",
            f"{m['y_final']:.4f}",
            f"{m['x_max']:.4f}",
            f"{m['y_max']:.4f}",
            f"{m['x_mean']:.4f}",
            f"{m['y_mean']:.4f}"
        ])
        
        # Generar gráficas para este escenario
        t = result['t']
        y = result['y']
        
        # 1. Series temporales
        plt.figure(figsize=(12, 6))
        plt.plot(t, y[:, 0], 'b-', label='Presas (x)', linewidth=2)
        plt.plot(t, y[:, 1], 'r-', label='Depredadores (y)', linewidth=2)
        plt.legend(fontsize=11)
        plt.xlabel("Tiempo (t)", fontsize=12)
        plt.ylabel("Población", fontsize=12)
        plt.title(f"Series Temporales - {scenario.name}", fontsize=13, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        save_figure(plt.gcf(), f"escenario_{i:02d}_series_temporales.png")
        plt.close()
        
        # 2. Diagrama de fase
        plt.figure(figsize=(8, 8))
        plt.plot(y[:, 0], y[:, 1], 'g-', linewidth=2, label='Trayectoria')
        plt.plot(y[0, 0], y[0, 1], 'ko', markersize=12, label='Inicio', zorder=5)
        plt.plot(y[-1, 0], y[-1, 1], 'rX', markersize=12, label='Final', zorder=5)
        plt.xlabel("Población de Presas (x)", fontsize=12)
        plt.ylabel("Población de Depredadores (y)", fontsize=12)
        plt.title(f"Retrato de Fase - {scenario.name}", fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        save_figure(plt.gcf(), f"escenario_{i:02d}_retrato_fase.png")
        plt.close()
        
        # 3. Gráfica combinada (series + fase)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Series temporales
        ax1.plot(t, y[:, 0], 'b-', label='Presas (x)', linewidth=2)
        ax1.plot(t, y[:, 1], 'r-', label='Depredadores (y)', linewidth=2)
        ax1.set_xlabel("Tiempo (t)", fontsize=11)
        ax1.set_ylabel("Población", fontsize=11)
        ax1.set_title("Series Temporales", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Retrato de fase
        ax2.plot(y[:, 0], y[:, 1], 'g-', linewidth=2, label='Trayectoria')
        ax2.plot(y[0, 0], y[0, 1], 'ko', markersize=10, label='Inicio', zorder=5)
        ax2.plot(y[-1, 0], y[-1, 1], 'rX', markersize=10, label='Final', zorder=5)
        ax2.set_xlabel("Población de Presas (x)", fontsize=11)
        ax2.set_ylabel("Población de Depredadores (y)", fontsize=11)
        ax2.set_title("Retrato de Fase", fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.suptitle(f"{scenario.name}", fontsize=14, fontweight='bold', y=1.02)
        save_figure(fig, f"escenario_{i:02d}_completo.png")
        plt.close()
        
        print(f"[OK] Graficas generadas para Escenario {i}\n")
    
    # Imprimir tabla comparativa
    print("\n" + "="*100)
    print("TABLA COMPARATIVA DE ESCENARIOS")
    print("="*100)
    print_table([
        "Escenario",
        "x0 inicial",
        "y0 inicial",
        "x final",
        "y final",
        "x máximo",
        "y máximo",
        "x promedio",
        "y promedio"
    ], comparison_rows)
    
    # Generar gráfica comparativa de todos los escenarios
    plot_comparison_all_scenarios(all_results)
    
    print("\n" + "="*100)
    print("Análisis de escenarios completado.")
    print(f"Total de gráficas generadas: {len(scenarios) * 3}")
    print("Gráficos guardados en ./figures/")
    print("="*100)


def plot_comparison_all_scenarios(all_results):
    """
    Genera gráficas comparativas de todos los escenarios.
    """
    n_scenarios = len(all_results)
    
    # Gráfica 1: Comparación de series temporales (presas)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    for i, (scenario, result) in enumerate(all_results):
        t = result['t']
        y = result['y']
        axes[0].plot(t, y[:, 0], label=f"Esc. {i+1}: {scenario.name[:30]}", linewidth=1.5, alpha=0.8)
        axes[1].plot(t, y[:, 1], label=f"Esc. {i+1}: {scenario.name[:30]}", linewidth=1.5, alpha=0.8)
    
    axes[0].set_xlabel("Tiempo (t)", fontsize=11)
    axes[0].set_ylabel("Población de Presas (x)", fontsize=11)
    axes[0].set_title("Comparación de Presas - Todos los Escenarios", fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8, ncol=2, loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].set_xlabel("Tiempo (t)", fontsize=11)
    axes[1].set_ylabel("Población de Depredadores (y)", fontsize=11)
    axes[1].set_title("Comparación de Depredadores - Todos los Escenarios", fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=8, ncol=2, loc='best')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_figure(fig, "comparacion_todos_escenarios_series.png")
    plt.close()
    
    # Gráfica 2: Comparación de retratos de fase
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, n_scenarios))
    
    for i, (scenario, result) in enumerate(all_results):
        y = result['y']
        plt.plot(y[:, 0], y[:, 1], '-', color=colors[i], linewidth=2, 
                label=f"Esc. {i+1}: {scenario.name[:25]}", alpha=0.8)
        plt.plot(y[0, 0], y[0, 1], 'o', color=colors[i], markersize=8, zorder=5)
    
    plt.xlabel("Población de Presas (x)", fontsize=12)
    plt.ylabel("Población de Depredadores (y)", fontsize=12)
    plt.title("Comparación de Retratos de Fase - Todos los Escenarios", fontsize=13, fontweight='bold')
    plt.legend(fontsize=8, ncol=2, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_figure(plt.gcf(), "comparacion_todos_escenarios_fase.png")
    plt.close()
    
    print("\n[OK] Graficas comparativas generadas")


# -----------------------------------------------------------------------------
# EJECUTOR PRINCIPAL
# -----------------------------------------------------------------------------

def run_all():
    """
    Función principal que ejecuta todas las simulaciones:
    1. Pruebas de validación y convergencia (opcional)
    2. Análisis de escenarios del modelo depredador-presa
    """
    os.makedirs("figures", exist_ok=True)
    
    # 1. Correr la validación (tablas de convergencia)
    run_validation_tests()
    
    # 2. Correr los escenarios del proyecto
    run_nonlinear_scenarios()
    
    print("\n" + "="*100)
    print("¡TODAS LAS SIMULACIONES COMPLETADAS!")
    print("Resultados y gráficos guardados en ./figures/")
    print("="*100)

if __name__ == "__main__":
    # Esto asume que estás en la carpeta raíz (ProyectoFinalED)
    # y ejecutas como: python -m src.experiments
    # Si ejecutas 'python src/experiments.py' directamente,
    # cambia las importaciones al inicio a:
    # from integrator import solve
    # from problems import ...
    # from utils import ...
    
    run_all()
