"""
quant_engine/tests/test_engine.py
Suite di Test Unitari per il Motore Quantitativo Istituzionale.
Verifica:
1. Stima corretta dell'hedge ratio di cointegrazione e half-life;
2. Risoluzione della Quantum Finance Schrödinger Equation (QFSE);
3. Circuit Breakers di Compliance e protezione Drawdown.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import date

from quant_engine.strategies.stat_arb_coint import CointegratedPairsStrategy
from quant_engine.strategies.quantum_price_levels import QuantumPriceModel
from quant_engine.risk.circuit_breakers import PropComplianceCircuitBreaker
from quant_engine.risk.sizing import InstitutionalPositionSizer

class TestQuantEngine(unittest.TestCase):

    def test_cointegration_hedge_ratio(self):
        """Verifica che la regressione identifichi correttamente il beta teorico di cointegrazione."""
        np.random.seed(42)
        n = 500
        x = np.cumsum(np.random.randn(n)) + 100.0
        # Y cointegrato con X con beta = 1.75
        noise = np.random.randn(n) * 0.5
        y = 1.75 * x + 5.0 + noise
        
        beta, alpha = CointegratedPairsStrategy.calculate_hedge_ratio(y, x)
        self.assertAlmostEqual(beta, 1.75, places=1)
        self.assertAlmostEqual(alpha, 5.0, delta=1.0)

    def test_half_life_estimation(self):
        """Verifica la stima corretta della half-life per un processo di Ornstein-Uhlenbeck simulato."""
        np.random.seed(42)
        n = 1000
        theta_true = 0.10
        expected_half_life = np.log(2.0) / theta_true # ~6.93 barre
        
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = spread[t-1] - theta_true * spread[t-1] + np.random.randn() * 0.2
            
        estimated_hl = CointegratedPairsStrategy.estimate_half_life(spread)
        self.assertGreater(estimated_hl, 4.0)
        self.assertLess(estimated_hl, 10.0)

    def test_qfse_solver(self):
        """Verifica che la QFSE produca autovalori reali crescenti ed estragga i livelli QPL."""
        q_model = QuantumPriceModel(mu=1.0, lambda_param=0.1)
        x_grid = np.linspace(-3, 3, 100)
        evals, evecs = q_model.solve_hamiltonian(x_grid, n_levels=5)
        
        # Gli autovalori devono essere strettamente crescenti E_0 < E_1 < E_2 ...
        self.assertEqual(len(evals), 5)
        for i in range(len(evals) - 1):
            self.assertLess(evals[i], evals[i+1])
            
        qpl = q_model.extract_quantum_price_levels(x_grid, evecs)
        self.assertGreater(len(qpl), 0)

    def test_circuit_breakers_hard_stop(self):
        """Verifica che il circuit breaker blocchi il trading in caso di superamento del Max Drawdown."""
        cb = PropComplianceCircuitBreaker(initial_capital=100000.0, max_daily_loss_pct=0.025, max_total_loss_pct=0.06)
        
        # Normale equity
        res1 = cb.update_and_check(current_equity=101000.0, trade_date=date(2026, 1, 1))
        self.assertTrue(res1["can_trade"])
        
        # Crollo improvviso al di sotto del 6% (equity 93,000 = -7%)
        res2 = cb.update_and_check(current_equity=93000.0, trade_date=date(2026, 1, 2))
        self.assertFalse(res2["can_trade"])
        self.assertEqual(res2["action"], "EMERGENCY_HALT")
        self.assertEqual(res2["reason"], "MAX_TOTAL_DRAWDOWN_EXCEEDED")

    def test_kelly_position_sizing(self):
        """Verifica che il calcolo del dimensionamento Kelly rispetti l'half-kelly e non allochi su edge negativo."""
        sizer = InstitutionalPositionSizer(target_annual_vol=0.12, half_kelly_mult=0.5)
        
        # Caso senza edge statistico: win rate 40% con win/loss 1:1 -> Kelly <= 0
        f_no_edge = sizer.compute_kelly_fraction(win_rate=0.40, win_loss_ratio=1.0)
        self.assertEqual(f_no_edge, 0.0)
        
        # Caso con edge: win rate 60% con win/loss 1.5:1 -> Kelly positivo
        f_edge = sizer.compute_kelly_fraction(win_rate=0.60, win_loss_ratio=1.5)
        self.assertGreater(f_edge, 0.0)
        self.assertLessEqual(f_edge, 0.20)

if __name__ == "__main__":
    unittest.main()
