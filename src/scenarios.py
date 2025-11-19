"""
Definición de escenarios para el modelo depredador-presa Rosenzweig-MacArthur.
Cada escenario representa diferentes condiciones ecológicas y parámetros del modelo.
"""

from .problems import rosenzweig_macarthur

class Scenario:
    """Clase para definir un escenario del modelo depredador-presa."""
    
    def __init__(self, name, description, r=1.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.2, x0=5.0, y0=1.0):
        """
        Parámetros del modelo Rosenzweig-MacArthur:
        - r: tasa de crecimiento intrínseca de las presas
        - K: capacidad de carga del ambiente para las presas
        - a: tasa de encuentro/depredación
        - h: tiempo de manejo (handling time)
        - e: eficiencia de conversión (presas -> predadores)
        - m: tasa de mortalidad de los predadores
        - x0: población inicial de presas
        - y0: población inicial de predadores
        """
        self.name = name
        self.description = description
        self.r = r
        self.K = K
        self.a = a
        self.h = h
        self.e = e
        self.m = m
        self.x0 = x0
        self.y0 = y0
    
    def get_problem(self):
        """Retorna la función del problema con los parámetros del escenario."""
        return rosenzweig_macarthur(
            r=self.r, K=self.K, a=self.a, h=self.h, 
            e=self.e, m=self.m, x0=self.x0, y0=self.y0
        )
    
    def get_params_dict(self):
        """Retorna un diccionario con todos los parámetros del escenario."""
        return {
            'r': self.r,
            'K': self.K,
            'a': self.a,
            'h': self.h,
            'e': self.e,
            'm': self.m,
            'x0': self.x0,
            'y0': self.y0
        }


def get_all_scenarios():
    """
    Retorna una lista de todos los escenarios definidos.
    Cada escenario representa diferentes condiciones ecológicas.
    """
    scenarios = []
    
    # Escenario 1: Condiciones base (equilibrio)
    scenarios.append(Scenario(
        name="Escenario 1: Equilibrio Base",
        description="Condiciones iniciales y parámetros estándar que tienden al equilibrio",
        r=1.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.2,
        x0=5.0, y0=1.0
    ))
    
    # Escenario 2: Muchas presas, pocos predadores
    scenarios.append(Scenario(
        name="Escenario 2: Presas Abundantes",
        description="Población inicial de presas alta, pocos predadores",
        r=1.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.2,
        x0=8.0, y0=0.5
    ))
    
    # Escenario 3: Pocas presas, muchos predadores
    scenarios.append(Scenario(
        name="Escenario 3: Predadores Abundantes",
        description="Población inicial de presas baja, muchos predadores",
        r=1.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.2,
        x0=2.0, y0=3.0
    ))
    
    # Escenario 4: Alta capacidad de carga
    scenarios.append(Scenario(
        name="Escenario 4: Alta Capacidad de Carga",
        description="Ambiente con mayor capacidad de carga (K alto)",
        r=1.0, K=20.0, a=1.0, h=0.1, e=0.5, m=0.2,
        x0=10.0, y0=1.0
    ))
    
    # Escenario 5: Alta tasa de depredación
    scenarios.append(Scenario(
        name="Escenario 5: Alta Depredación",
        description="Predadores muy eficientes en encontrar y capturar presas",
        r=1.0, K=10.0, a=2.0, h=0.1, e=0.5, m=0.2,
        x0=5.0, y0=1.0
    ))
    
    # Escenario 6: Baja eficiencia de conversión
    scenarios.append(Scenario(
        name="Escenario 6: Baja Eficiencia",
        description="Baja eficiencia de conversión de presas en predadores",
        r=1.0, K=10.0, a=1.0, h=0.1, e=0.2, m=0.2,
        x0=5.0, y0=1.0
    ))
    
    # Escenario 7: Alta mortalidad de predadores
    scenarios.append(Scenario(
        name="Escenario 7: Alta Mortalidad",
        description="Alta tasa de mortalidad de los predadores",
        r=1.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.5,
        x0=5.0, y0=1.0
    ))
    
    # Escenario 8: Alta tasa de crecimiento de presas
    scenarios.append(Scenario(
        name="Escenario 8: Crecimiento Rápido",
        description="Presas con alta tasa de crecimiento intrínseca",
        r=2.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.2,
        x0=5.0, y0=1.0
    ))
    
    # Escenario 9: Condiciones extremas - casi extinción
    scenarios.append(Scenario(
        name="Escenario 9: Poblaciones Mínimas",
        description="Poblaciones iniciales muy bajas, riesgo de extinción",
        r=1.0, K=10.0, a=1.0, h=0.1, e=0.5, m=0.2,
        x0=0.5, y0=0.1
    ))
    
    # Escenario 10: Sistema con oscilaciones sostenidas
    scenarios.append(Scenario(
        name="Escenario 10: Oscilaciones",
        description="Parámetros que favorecen oscilaciones sostenidas",
        r=1.5, K=15.0, a=1.2, h=0.15, e=0.6, m=0.15,
        x0=3.0, y0=2.0
    ))
    
    return scenarios


def get_scenario_by_name(name):
    """Retorna un escenario específico por su nombre."""
    all_scenarios = get_all_scenarios()
    for scenario in all_scenarios:
        if scenario.name == name:
            return scenario
    return None

