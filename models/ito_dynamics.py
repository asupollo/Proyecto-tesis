import numpy as np
import sympy as sp
from dataclasses import dataclass

@dataclass
class ItoTrajectory:
    t_grid: np.ndarray
    x_values: np.ndarray
    dt: float
    m_samples: int

class ItoProcess:
    def __init__(self, x0: float = 0.0, mu_expr = None, sigma_expr = None):
        self.x0 = x0
        
        # 1. Definimos el símbolo del tiempo para que SymPy lo entienda
        self.t_sym = sp.Symbol('t')
        
        # 2. Asignamos las expresiones (o valores por defecto: 0 y 1)
        self.mu_expr = sp.sympify(mu_expr) if mu_expr is not None else sp.sympify(0)
        self.sigma_expr = sp.sympify(sigma_expr) if sigma_expr is not None else sp.sympify(1)
        
        # 3. EL TRUCO QUANT: "Lambdify". 
        # Esto convierte la ecuación simbólica lenta en una función de NumPy ultrarrápida.
        self.mu_func = sp.lambdify(self.t_sym, self.mu_expr, 'numpy')
        self.sigma_func = sp.lambdify(self.t_sym, self.sigma_expr, 'numpy')

    def get_latex_sde(self) -> str:
        """Construye automáticamente la EDE en formato LaTeX."""
        # Cambiamos la variable 't' por 's' para la integración formal
        s_sym = sp.Symbol('s')
        mu_s = self.mu_expr.subs(self.t_sym, s_sym)
        sigma_s = self.sigma_expr.subs(self.t_sym, s_sym)
        
        latex_str = r"X_t = "
        
        # Generar término del Drift si existe
        if self.mu_expr != 0:
            latex_str += rf"\int_0^t {sp.latex(mu_s)} \, ds "
            
        # Generar término de Difusión si existe
        if self.sigma_expr != 0:
            signo = "+ " if self.mu_expr != 0 else ""
            
            # Formatear el 1 para que no se imprima "1 dB_s", sino solo "dB_s"
            sigma_latex = sp.latex(sigma_s)
            if sigma_latex == "1":
                latex_str += rf"{signo}\int_0^t \, dB_s"
            else:
                latex_str += rf"{signo}\int_0^t {sigma_latex} \, dB_s"
                
        # Si ambos son cero
        if self.mu_expr == 0 and self.sigma_expr == 0:
            latex_str += "X_0"
            
        return latex_str

    def simulate(self, m: int) -> ItoTrajectory:
        t_end = 2 * np.pi
        dt = t_end / m
        
        t_grid = np.linspace(0, t_end, m + 1)
        x_path = np.zeros(m + 1)
        x_path[0] = self.x0
        
        dW = np.random.normal(0, np.sqrt(dt), m)
        
        for i in range(m):
            t_val = t_grid[i]
            # Usamos las funciones lambdificadas que son rapidísimas
            mu_val = self.mu_func(t_val)
            sigma_val = self.sigma_func(t_val)
            
            x_path[i+1] = x_path[i] + mu_val * dt + sigma_val * dW[i]
            
        return ItoTrajectory(t_grid=t_grid, x_values=x_path, dt=dt, m_samples=m)