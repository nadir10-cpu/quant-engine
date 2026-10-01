"""
quant_engine/strategies/stat_arb_coint.py
Strategia di Statistical Arbitrage e Pairs Trading basata su Cointegrazione,
Filtro di Kalman dinamico e calibrazione stocastica di Ornstein-Uhlenbeck.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from quant_engine.strategies.base_alpha import GenericAlphaUnit

class CointegratedPairsStrategy(GenericAlphaUnit):
    """
    Strategia di Pairs Trading avanzata:
    1. Filtro di Kalman adattivo online per stima di Beta (Hedge Ratio) e Alpha senza lag;
    2. Modello di Ornstein-Uhlenbeck per la stima della Half-Life di convergenza;
    3. Segnali Z-Score dinamici per ingresso, convergenza e stop-loss per divergenza.
    """

    def __init__(self, entry_zscore: float = 1.5, exit_zscore: float = 0.0, 
                 stop_loss_zscore: float = 3.0, lookback_window: int = 20, use_kalman: bool = True):
        super().__init__(name="StatArb_Kalman_Cointegration", params={
            "entry_zscore": entry_zscore,
            "exit_zscore": exit_zscore,
            "stop_loss_zscore": stop_loss_zscore,
            "lookback_window": lookback_window,
            "use_kalman": use_kalman
        })
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.lookback_window = lookback_window
        self.use_kalman = use_kalman

    @staticmethod
    def calculate_hedge_ratio(series_y: np.ndarray, series_x: np.ndarray) -> Tuple[float, float]:
        """Calcola l'hedge ratio beta e l'intercetta alpha tramite regressione OLS."""
        X = np.vstack([np.ones(len(series_x)), series_x]).T
        res = np.linalg.lstsq(X, series_y, rcond=None)
        alpha, beta = res[0]
        return float(beta), float(alpha)

    @staticmethod
    def kalman_filter_series(series_y: np.ndarray, series_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filtro di Kalman adattivo per l'aggiornamento online di Beta e Alpha."""
        n = len(series_y)
        beta = np.zeros(n)
        alpha = np.zeros(n)
        xhat = np.zeros((2, 1))
        P = np.eye(2) * 1.0
        R = 0.005
        Q = np.eye(2) * 1e-5
        
        for t in range(n):
            H = np.array([[1.0, series_x[t]]])
            xhat_pri = xhat
            P_pri = P + Q
            y_hat = H @ xhat_pri
            v = series_y[t] - y_hat[0, 0]
            S = H @ P_pri @ H.T + R
            K = P_pri @ H.T / S[0, 0]
            xhat = xhat_pri + K * v
            P = (np.eye(2) - K @ H) @ P_pri
            alpha[t] = xhat[0, 0]
            beta[t] = xhat[1, 0]
            
        return beta, alpha

    @staticmethod
    def estimate_half_life(spread: np.ndarray) -> float:
        """Stima la half-life stocastica di Ornstein-Uhlenbeck: ln(2) / theta."""
        if len(spread) < 10:
            return np.inf
        lagged = spread[:-1]
        delta = spread[1:] - lagged
        X = np.vstack([np.ones(len(lagged)), lagged]).T
        res = np.linalg.lstsq(X, delta, rcond=None)
        b = res[0][1]
        theta = -b
        if theta <= 0:
            return np.inf
        return float(np.log(2.0) / theta)

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """Calcola i segnali di trading con filtro Kalman e Z-score rolling."""
        if 'asset_y' not in market_data.columns or 'asset_x' not in market_data.columns:
            raise ValueError("market_data deve contenere le colonne 'asset_y' e 'asset_x'")
            
        y = market_data['asset_y'].values
        x = market_data['asset_x'].values
        n = len(y)
        
        signals = np.zeros(n)
        
        if self.use_kalman:
            betas, alphas = self.kalman_filter_series(y, x)
            spread = y - (betas * x + alphas)
        else:
            betas = np.zeros(n)
            spread = np.zeros(n)
            for t in range(self.lookback_window, n):
                b, a = self.calculate_hedge_ratio(y[t-self.lookback_window:t], x[t-self.lookback_window:t])
                betas[t] = b
                spread[t] = y[t] - (b * x[t] + a)
                
        current_pos = 0
        for t in range(self.lookback_window, n):
            win_sp = spread[t - self.lookback_window : t]
            mu = np.mean(win_sp)
            std = np.std(win_sp) + 1e-12
            z_score = (spread[t] - mu) / std
            
            if current_pos == 0:
                if z_score <= -self.entry_zscore:
                    current_pos = 1
                elif z_score >= self.entry_zscore:
                    current_pos = -1
            elif current_pos == 1:
                if z_score >= -self.exit_zscore or z_score <= -self.stop_loss_zscore:
                    current_pos = 0
            elif current_pos == -1:
                if z_score <= self.exit_zscore or z_score >= self.stop_loss_zscore:
                    current_pos = 0
                    
            signals[t] = float(current_pos)
            
        self.last_betas = pd.Series(betas, index=market_data.index)
        return pd.Series(signals, index=market_data.index)
