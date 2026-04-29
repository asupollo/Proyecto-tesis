import numpy as np
import sympy as sp
from models.ito_dynamics import ItoProcess
from estimators.classical_fourier import ClassicalFourier
from estimators.fourier_malliavin import FourierMalliavin
from visualization.plotter import ItoPlotter

def run_full_validation():
    t = sp.Symbol('t')
    mi_mu = 0
    # Usamos una función interesante para notar el desempeño (ej. sen(t) + 1.5)
    mi_sigma = t*(2*sp.pi-t)
    
    # =======================================================
    # 1. PARÁMETROS DEL MODELO
    # =======================================================
    K_banda = 10     # Frecuencias para la reconstrucción de Fejér [-K, K]
    n_nivel = 20     # Nivel para la partición asintótica
    
    # N_n ahora crece a una potencia de 1.5
    N_n = int(10 * (n_nivel**1.5)) 
    m_n = 1000 * (n_nivel**2)
    
    print("\n--- INICIANDO VALIDACIÓN EMPÍRICA COMPLETA ---")
    print(f"Proceso de volatilidad: sigma_s = {mi_sigma}")
    print(f"Banda Fejér (K)       : {K_banda}")
    print(f"Banda Malliavin (N_n) : {N_n}")
    print(f"Muestras (m_n)        : {m_n}")
    
    # =======================================================
    # 2. SIMULACIÓN
    # =======================================================
    print(f"\n1. Generando trayectoria del log-precio...")
    proceso = ItoProcess(x0=0.0, mu_expr=mi_mu, sigma_expr=mi_sigma)
    trayectoria = proceso.simulate(m=m_n)
    t_grid = trayectoria.t_grid
    
    # =======================================================
    # 3. CÁLCULOS DE VARIANZA
    # =======================================================
    print("2. Calculando Varianza Teórica Exacta...")
    varianza_real = np.array([proceso.sigma_func(val)**2 for val in t_grid])
    
    print("3. Calculando Reconstrucción Fejér Clásica...")
    varianza_clasica = ClassicalFourier.reconstruct_variance_classical(
        sigma_func=proceso.sigma_func, 
        K=K_banda, 
        t_eval=t_grid
    )
    
    print("4. Calculando Estimador de Fourier-Malliavin (Esto puede tomar un momento)...")
    estimador_fm = FourierMalliavin(N=N_n)
    varianza_fm = estimador_fm.reconstruct_variance(
        trajectory=trayectoria, 
        K=K_banda, 
        t_eval=t_grid
    )
    
    # =======================================================
    # 4. VISUALIZACIÓN
    # =======================================================
    print("\n5. Renderizando gráfica definitiva...")
    ItoPlotter.plot_full_comparison(
        trajectory=trayectoria,
        true_var=varianza_real,
        rec_clasica=varianza_clasica,
        rec_fm=varianza_fm,
        sde_string=proceso.get_latex_sde(),
        K=K_banda,
        N=N_n
    )

if __name__ == "__main__":
    run_full_validation()