import numpy as np
import matplotlib.pyplot as plt
from models.ito_dynamics import ItoTrajectory

class ItoPlotter:
    @staticmethod
    def setup_latex_style():
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.labelsize'] = 14

    @staticmethod
    def plot_full_comparison(trajectory: ItoTrajectory, true_var: np.ndarray, 
                             rec_clasica: np.ndarray, rec_fm: np.ndarray, 
                             sde_string: str, K: int, N: int):
        """
        Genera el panel definitivo comparando la Varianza Real, la Clásica y la de Malliavin.
        """
        ItoPlotter.setup_latex_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 1) PANEL SUPERIOR: Trayectoria del Log-Precio X_t
        ax1.plot(trajectory.t_grid, trajectory.x_values, color='#0047AB', linewidth=1.2)
        ax1.set_title(rf"${sde_string}$", pad=15)
        ax1.set_ylabel(r"$X_t$")
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2) PANEL INFERIOR: Comparación de Varianzas
        # Teórica (Negro punteado)
        ax2.plot(trajectory.t_grid, true_var, color='black', linestyle='--', linewidth=2, 
                 label=r'Varianza Real $\sigma^2_s$')
                 
        # Clásica (Naranja)
        ax2.plot(trajectory.t_grid, rec_clasica, color='#D95319', alpha=0.8, linewidth=2, 
                 label=rf'Fejér Clásica ($K={K}$)')
                 
        # Fourier-Malliavin (Magenta)
        ax2.plot(trajectory.t_grid, rec_fm, color='#FF00FF', alpha=0.8, linewidth=2, 
                 label=rf'Fourier-Malliavin ($K={K}, N={N}$)')
        
        ax2.set_xlabel(r"Tiempo $s$")
        ax2.set_ylabel(r"Varianza $\sigma^2_s$")
        ax2.legend(loc='upper right', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Formato del eje X en fracciones de pi
        ax2.set_xlim(0, 2*np.pi)
        pi_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        pi_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
        ax2.set_xticks(pi_ticks)
        ax2.set_xticklabels(pi_labels)
        
        plt.tight_layout()
        plt.show()