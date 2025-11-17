# Proyecto Final — Ecuaciones Diferenciales  
## Modelación y Simulación del Sistema Depredador–Presa (Lodka Volterra/Rosenzweig–MacArthur)

Este proyecto implementa y analiza numéricamente el modelo depredador–presa de **Lodka Volterra/Rosenzweig–MacArthur**, utilizando los métodos numéricos **Runge–Kutta de cuarto orden (RK4)** y **Adams–Bashforth de dos pasos (AB2)**.  
Además, se validan los métodos usando EDOs con solución analítica y se generan visualizaciones para interpretar la dinámica ecológica del sistema.

---

## 📂 Estructura del repositorio

ProyectoFinalED/
│
├── src/
│ ├── integrator.py # Implementación de RK4, AB2 y Euler
│ ├── problems.py # EDOs: exponencial, logística, armónico, depredador–presa
│ ├── experiments.py # Pruebas de convergencia y simulaciones finales
│ ├── utils.py # Funciones auxiliares (errores, tablas, guardado de figuras)
│ └── init.py
│
├── figures/ # Gráficas generadas automáticamente
│
├── report/ # Informe en LaTeX (opcional)
│
└── README.md

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

## Métodos numéricos implementados

- Runge–Kutta de 4to orden (RK4): Método explícito, Orden 4, Alta estabilidad y precisión, Método principal para las simulaciones.
- Adams–Bashforth de dos pasos (AB2): Método predictor, Orden 2, Útil para comparar convergencia contra RK4.
- Euler: Orden 1, Menor precisión, Incluido solo con fines educativos

## Explicación de cada archivo
- integrator.py → contiene los métodos
- problems.py → define las EDOs
- experiments.py → ejecuta pruebas y simulaciones
- utils.py → funciones auxiliares

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


