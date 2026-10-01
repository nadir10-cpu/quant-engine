"""
quant_engine/strategies/macro_swing.py
Strategia Macro Swing High-Conviction:
- Pochi trade (0.4 - 1/mese) ma con grandi profitti ($150-$350 a trade)
- Regime Filter (EMA 20/60) per operare SOLO nella direzione del trend
- Z-Score a 1.8 sigma su 20 giorni per entry su estensioni critiche
- Target: +4% del prezzo di entrata | Stop Loss: -1.5%
- Allocazione: 50% del conto per ogni trade
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from quant_engine.strategies.base_alpha import GenericAlphaUnit


class MacroSwingStrategy(GenericAlphaUnit):
    """
    Strategia Macro Swing direzionale a bassa frequenza e alto profitto per singolo trade.
    Lavora su singolo asset e genera segnali giornalieri con target fisso di profitto.
    """

    def __init__(self,
                 regime_fast: int = 20,
                 regime_slow: int = 60,
                 zscore_window: int = 20,
                 entry_zscore: float = 1.8,
                 profit_target_pct: float = 0.04,
                 stop_loss_pct: float = 0.015,
                 alloc_pct: float = 0.50):
        super().__init__(name="MacroSwing_HighConviction", params={
            "regime_fast": regime_fast,
            "regime_slow": regime_slow,
            "zscore_window": zscore_window,
            "entry_zscore": entry_zscore,
            "profit_target_pct": profit_target_pct,
            "stop_loss_pct": stop_loss_pct,
            "alloc_pct": alloc_pct
        })
        self.regime_fast = regime_fast
        self.regime_slow = regime_slow
        self.zscore_window = zscore_window
        self.entry_zscore = entry_zscore
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.alloc_pct = alloc_pct

    @staticmethod
    def compute_ema(prices: np.ndarray, span: int) -> np.ndarray:
        return pd.Series(prices).ewm(span=span, adjust=False).mean().values

    @staticmethod
    def compute_zscore(prices: np.ndarray, window: int) -> np.ndarray:
        zs = np.zeros(len(prices))
        for t in range(window, len(prices)):
            win = prices[t - window:t]
            zs[t] = (prices[t] - np.mean(win)) / (np.std(win) + 1e-12)
        return zs

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Genera i segnali di trade (+1 Long, -1 Short, 0 Flat).
        market_data deve contenere la colonna 'asset_y' (prezzo del singolo asset).
        """
        if 'asset_y' not in market_data.columns:
            raise ValueError("market_data deve contenere la colonna 'asset_y'")

        prices = market_data['asset_y'].values
        n = len(prices)

        ema_fast = self.compute_ema(prices, self.regime_fast)
        ema_slow = self.compute_ema(prices, self.regime_slow)
        regime = np.where(ema_fast > ema_slow, 1, -1)
        zs = self.compute_zscore(prices, self.zscore_window)

        signals = np.zeros(n)
        pos = 0
        entry_price = 0.0

        for t in range(self.regime_slow, n):
            p = prices[t]

            # Gestione chiusura posizione aperta (target/stop)
            if pos == 1:
                pnl_pct = (p - entry_price) / entry_price
                if pnl_pct >= self.profit_target_pct or pnl_pct <= -self.stop_loss_pct:
                    pos = 0
                    entry_price = 0.0
            elif pos == -1:
                pnl_pct = (entry_price - p) / entry_price
                if pnl_pct >= self.profit_target_pct or pnl_pct <= -self.stop_loss_pct:
                    pos = 0
                    entry_price = 0.0

            # Ricerca nuovo ingresso (solo se flat)
            if pos == 0:
                if regime[t] == 1 and zs[t] <= -self.entry_zscore:
                    pos = 1
                    entry_price = p
                elif regime[t] == -1 and zs[t] >= self.entry_zscore:
                    pos = -1
                    entry_price = p

            signals[t] = float(pos)

        self.last_betas = pd.Series(np.ones(n), index=market_data.index)
        return pd.Series(signals, index=market_data.index)
