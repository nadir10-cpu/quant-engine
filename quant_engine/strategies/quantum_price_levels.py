"""
quant_engine/strategies/quantum_price_levels.py
Implementazione del Modello di Prezzo Quantistico (Quantum Finance Schrödinger Equation - QFSE)
del Dr. Raymond S.T. Lee (Springer, 2020).
Risolve l'oscillatore anarmonico quantistico (AHO) per identificare i Quantum Price Levels (QPL)
come stati risonanti stazionari non-lineari.
"""

import numpy as np
from typing import List, Tuple, Dict

class QuantumPriceModel:
    """
    Risolutore numerico per la Quantum Finance Schrödinger Equation:
    H_f psi = [ - (hbar_f^2 / (2 * m_f)) d^2/dx^2 + V(x) ] psi = E psi
    con potenziale anarmonico di 4° ordine:
    V(x) = 0.5 * mu * x^2 + lambda_param * x^4
    """

    def __init__(self, mu: float = 1.0, lambda_param: float = 0.15, hbar_f: float = 1.0, m_f: float = 1.0):
        self.mu = mu
        self.lambda_param = lambda_param
        self.hbar_f = hbar_f
        self.m_f = m_f

    def solve_hamiltonian(self, x_grid: np.ndarray, n_levels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretizza e diagonalizza l'Hamiltoniana finanziaria tramite Finite Difference Method (FDM).
        
        Args:
            x_grid: Array 1D di punti di discretizzazione del rendimento logaritmico normalizzato.
            n_levels: Numero di autofunzioni / livelli energetici da calcolare.
            
        Returns:
            Tuple (eigenvalues, eigenvectors):
            - eigenvalues: Array degli autovalori energetici QFEL (Quantum Finance Energy Levels).
            - eigenvectors: Matrice (N x n_levels) delle autofunzioni psi_n(x).
        """
        N = len(x_grid)
        dx = x_grid[1] - x_grid[0]
        
        # Potenziale anarmonico di quarto ordine
        V = 0.5 * self.mu * (x_grid**2) + self.lambda_param * (x_grid**4)
        
        # Termine cinetico: derivata seconda centrale (-d^2/dx^2)
        kinetic_factor = (self.hbar_f**2) / (2.0 * self.m_f * (dx**2))
        
        diag = 2.0 * kinetic_factor + V
        off_diag = -kinetic_factor * np.ones(N - 1)
        
        # Matrice Hamiltoniana tridiagonale simmetrica
        H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        
        # Diagonalizzazione spettrale hermitiana
        evals, evecs = np.linalg.eigh(H)
        
        # Normalizzazione in L^2: integrale |psi|^2 dx = 1
        for i in range(min(n_levels, len(evals))):
            norm = np.sqrt(np.sum(evecs[:, i]**2) * dx)
            if norm > 1e-12:
                evecs[:, i] /= norm
                
        return evals[:n_levels], evecs[:, :n_levels]

    def extract_quantum_price_levels(self, x_grid: np.ndarray, eigenvectors: np.ndarray) -> List[float]:
        """
        Estrae i Quantum Price Levels (QPL) identificando i picchi di massima densita
        di probabilita quantistica P(x) = |psi_n(x)|^2.
        
        Args:
            x_grid: Griglia dei valori di rendimento x.
            eigenvectors: Autofunzioni quantistiche psi_n(x).
            
        Returns:
            Lista ordinata dei livelli di prezzo quantistici (QPL) espressi in deviazioni standard/rendimento.
        """
        qpl_set = set()
        n_levels = eigenvectors.shape[1]
        
        for i in range(n_levels):
            prob = eigenvectors[:, i]**2
            # Identificazione picco globale e massimi locali per lo stato n
            peak_idx = np.argmax(prob)
            qpl_set.add(round(float(x_grid[peak_idx]), 4))
            
            # Cerca picchi secondari significativi per stati eccitati (n >= 1)
            for j in range(1, len(prob) - 1):
                if prob[j] > prob[j-1] and prob[j] > prob[j+1] and prob[j] > 0.3 * np.max(prob):
                    qpl_set.add(round(float(x_grid[j]), 4))
                    
        return sorted(list(qpl_set))

    def compute_quantum_support_resistance(self, current_price: float, historical_volatility: float, n_levels: int = 5) -> Dict[str, List[float]]:
        """
        Converte i QPL adimensionali in livelli di prezzo assoluti di supporto e resistenza per l'asset.
        """
        # Griglia centrata sul prezzo corrente
        x_grid = np.linspace(-3.5, 3.5, 250)
        evals, evecs = self.solve_hamiltonian(x_grid, n_levels=n_levels)
        qpl_normalized = self.extract_quantum_price_levels(x_grid, evecs)
        
        # Denormalizzazione al prezzo reale: S = S_0 * exp(qpl * sigma)
        real_levels = [round(current_price * np.exp(q * historical_volatility), 2) for q in qpl_normalized]
        
        supports = [lvl for lvl in real_levels if lvl < current_price]
        resistances = [lvl for lvl in real_levels if lvl > current_price]
        
        return {
            "current_price": current_price,
            "quantum_supports": sorted(supports, reverse=True),
            "quantum_resistances": sorted(resistances),
            "energy_eigenvalues_qfel": list(evals)
        }
