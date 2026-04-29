import numpy as np
from scipy.integrate import quad
from typing import Callable

class ClassicalFourier:
    @staticmethod
    def get_true_volatility_coeff(sigma_func: Callable[[float], float], k: int) -> complex:
        """Calcula el coeficiente de Fourier k real de la varianza sigma^2."""
        def parte_real(t):
            return (sigma_func(t)**2) * np.cos(-k * t)
            
        def parte_imaginaria(t):
            return (sigma_func(t)**2) * np.sin(-k * t)
            
        integral_real, _ = quad(parte_real, 0, 2 * np.pi)
        integral_imag, _ = quad(parte_imaginaria, 0, 2 * np.pi)
        
        return (integral_real + 1j * integral_imag) / (2 * np.pi)

    @staticmethod
    def reconstruct_variance_classical(sigma_func: Callable[[float], float], K: int, t_eval: np.ndarray) -> np.ndarray:
        """
        Reconstruye la varianza teórica usando la serie de Fejér clásica.
        """
        reconstruccion = np.zeros_like(t_eval, dtype=complex)
        
        for k in range(-K, K + 1):
            # Coeficiente teórico
            c_k = ClassicalFourier.get_true_volatility_coeff(sigma_func, k)
            
            # Peso de Fejér (Núcleo de Cesàro)
            peso = 1 - abs(k) / (K + 1)
            
            # Inversión de la serie
            reconstruccion += peso * c_k * np.exp(1j * k * t_eval)
            
        return reconstruccion.real