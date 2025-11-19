# Proyecto Final — Ecuaciones Diferenciales  
## Modelación y Simulación del Sistema Depredador–Presa (Lodka Volterra/Rosenzweig–MacArthur)

Este proyecto implementa y analiza numéricamente el modelo depredador–presa de **Lodka Volterra/Rosenzweig–MacArthur**, utilizando los métodos numéricos **Runge–Kutta de cuarto orden (RK4)** y **Adams–Bashforth de dos pasos (AB2)**.  
Además, se validan los métodos usando EDOs con solución analítica y se generan visualizaciones para interpretar la dinámica ecológica del sistema.

---

## 📂 Estructura del repositorio

ProyectoFinalED/
- src/
  - integrator.py # Implementación de RK4, AB2 y Euler
  - problems.py # EDOs: exponencial, logística, armónico, depredador–presa
  - scenarios.py # Definición de escenarios con diferentes parámetros y condiciones iniciales
  - experiments.py # Pruebas de convergencia y simulaciones finales
  - utils.py # Funciones auxiliares (errores, tablas, guardado de figuras)
    
- figures/ # Gráficas generadas automáticamente
- report/ # Informe en LaTeX
- README.md

---

## Instalación y ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/Kapiven/ProyectoFinalED.git
cd ProyectoFinalED
```

### 2. Instalar dependencias
```bash
pip install numpy matplotlib scipy tabulate
```

### 3. Ejecutar simulaciones
```bash
python -m src.experiments
```

Los gráficos se generan automáticamente en figures/

## Análisis de Escenarios

El proyecto incluye un sistema completo de análisis de escenarios que permite estudiar el comportamiento del modelo depredador-presa bajo diferentes condiciones:

### Escenarios Implementados

1. **Equilibrio Base**: Condiciones estándar que tienden al equilibrio
2. **Presas Abundantes**: Población inicial de presas alta, pocos predadores
3. **Predadores Abundantes**: Población inicial de presas baja, muchos predadores
4. **Alta Capacidad de Carga**: Ambiente con mayor capacidad de carga (K alto)
5. **Alta Depredación**: Predadores muy eficientes en encontrar y capturar presas
6. **Baja Eficiencia**: Baja eficiencia de conversión de presas en predadores
7. **Alta Mortalidad**: Alta tasa de mortalidad de los predadores
8. **Crecimiento Rápido**: Presas con alta tasa de crecimiento intrínseca
9. **Poblaciones Mínimas**: Poblaciones iniciales muy bajas, riesgo de extinción
10. **Oscilaciones**: Parámetros que favorecen oscilaciones sostenidas

### Resultados Generados

Para cada escenario se generan:
- **Series temporales**: Evolución de las poblaciones de presas y predadores en el tiempo
- **Retrato de fase**: Trayectoria en el espacio de fases (presas vs. predadores)
- **Gráfica combinada**: Visualización completa con ambos análisis

Además, se generan gráficas comparativas que muestran todos los escenarios simultáneamente y una **tabla comparativa** con métricas clave de cada escenario (valores finales, máximos, mínimos, promedios).

### Personalización de Escenarios

Los escenarios se pueden modificar o agregar nuevos en `src/scenarios.py`. Cada escenario permite ajustar:
- Parámetros del modelo: `r`, `K`, `a`, `h`, `e`, `m`
- Condiciones iniciales: `x0` (presas), `y0` (predadores)

## Métodos numéricos implementados

- Runge–Kutta de 4to orden (RK4): Método explícito, Orden 4, Alta estabilidad y precisión, Método principal para las simulaciones.
- Adams–Bashforth de dos pasos (AB2): Método predictor, Orden 2, Útil para comparar convergencia contra RK4.
- Euler: Orden 1, Menor precisión, Incluido solo con fines educativos

## Explicación de cada archivo
- integrator.py → contiene los métodos numéricos (RK4, AB2, Euler)
- problems.py → define las EDOs (exponencial, logística, armónico, depredador-presa)
- scenarios.py → define los diferentes escenarios con parámetros y condiciones iniciales
- experiments.py → ejecuta pruebas de convergencia y análisis de escenarios
- utils.py → funciones auxiliares (errores, tablas, guardado de figuras)

## Lista de dependencias
- numpy
- matplotlib
- scipy
- tabulate

## Nota sobre estructura esperada
Ejecutar desde la raíz del proyecto
- Python 3.12 recomendable

## Autores

- Karen Pineda
- Paula Daniela de León
- Daniella Cordero
- Alejandro Andrews


