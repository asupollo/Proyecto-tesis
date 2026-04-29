import numpy as np
from models.ito_dynamics import ItoTrajectory

class FourierMalliavin:
    def __init__(self, N: int):
        self.N = N
        self._cache = {} # Memoria para no repetir cálculos de c(n, nu)

    def get_c_n_nu(self, trajectory: ItoTrajectory, nu: int) -> complex:
        """
        Calcula el coeficiente c(n, nu). Si ya se calculó antes, lo devuelve de la caché.
        """
        if nu in self._cache:
            return self._cache[nu]
            
        t_i = trajectory.t_grid[:-1]
        dX_i = np.diff(trajectory.x_values)
        
        exponentes = np.exp(-1j * nu * t_i)
        suma = np.sum(exponentes * dX_i)
        
        c_val = suma / (2 * np.pi)
        
        self._cache[nu] = c_val
        return c_val

    def get_S_N_k(self, trajectory: ItoTrajectory, k: int) -> complex:
        """
        Construye la suma S(N, k).
        """
        suma_S = 0j
        
        for nu in range(-self.N, self.N + 1):
            c_nu = self.get_c_n_nu(trajectory, nu)
            c_k_menos_nu = self.get_c_n_nu(trajectory, k - nu)
            
            suma_S += c_nu * c_k_menos_nu
            
        return (2 * np.pi / (2 * self.N + 1)) * suma_S

    def reconstruct_variance(self, trajectory: ItoTrajectory, K: int, t_eval: np.ndarray) -> np.ndarray:
        """
        Reconstruye la trayectoria de la varianza (sigma^2) usando el núcleo de Fejér.
        Basado en la convergencia S(N, k) -> F(sigma^2)(k).
        """
        reconstruccion = np.zeros_like(t_eval, dtype=complex)
        
        # 1. Pre-calcular los coeficientes de la varianza S(N, nu)
        s_n_dict = {}
        for nu in range(-K, K + 1):
            # get_S_N_k ya incluye el factor (2pi / 2N+1) requerido por la Ec. (4)
            s_n_dict[nu] = self.get_S_N_k(trajectory, k=nu)
            
        # 2. Inversión de Fourier usando el núcleo de Fejér (Cesàro)
        for nu in range(-K, K + 1):
            peso_fejer = 1 - (abs(nu) / (K + 1))
            
            # Tomamos el coeficiente directo, sin dividir entre 2pi
            coef_espectral = s_n_dict[nu] 
            
            # Sumamos la onda armónica e^{i * nu * t} a toda la cuadrícula
            reconstruccion += peso_fejer * coef_espectral * np.exp(1j * nu * t_eval)
            
        return reconstruccion.real