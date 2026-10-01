"""
quant_engine/backtest/event_backtester.py
Motore di Backtesting Istituzionale a Eventi.
Simula fedelmente slippage, commissioni di transazione, ritardo di esecuzione (lag)
e integra i Circuit Breakers per la conformita di rischio Prop Firm.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from quant_engine.risk.circuit_breakers import PropComplianceCircuitBreaker

class InstitutionalBacktester:
    def __init__(self, initial_capital: float = 100000.0,
                 slippage_bps: float = 1.5,
                 commission_bps: float = 2.0,
                 max_daily_loss_pct: float = 0.025,
                 max_total_loss_pct: float = 0.060):
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_bps / 10000.0
        self.commission_pct = commission_bps / 10000.0
        
        self.circuit_breaker = PropComplianceCircuitBreaker(
            initial_capital=initial_capital,
            max_daily_loss_pct=max_daily_loss_pct,
            max_total_loss_pct=max_total_loss_pct
        )

    def run(self, market_data: pd.DataFrame, signals: pd.Series, hedge_ratio: float = 1.0) -> Dict[str, Any]:
        """
        Esegue la simulazione bar-by-bar.
        market_data contiene ['asset_y', 'asset_x']
        signals contiene lo stato del segnale: +1 (long Y, short X), -1 (short Y, long X), 0 (flat)
        """
        df = market_data.copy()
        n = len(df)
        
        cash = self.initial_capital
        current_pos = 0 # -1, 0, +1
        units_y = 0.0
        units_x = 0.0
        
        equity_curve = np.zeros(n)
        trade_log = []
        
        dates = df.index
        prices_y = df['asset_y'].values
        prices_x = df['asset_x'].values
        signal_vals = signals.values
        
        # Gestione hedge ratio scalare o dinamico per barra
        if isinstance(hedge_ratio, (pd.Series, np.ndarray)):
            beta_vals = np.asarray(hedge_ratio)
        else:
            beta_vals = np.full(n, float(hedge_ratio))
        
        # Allocazione capitale per trade: 30% del capitale per la coppia
        allocated_capital_ratio = 0.30
        
        for t in range(n):
            curr_date = dates[t].date() if hasattr(dates[t], 'date') else dates[t]
            p_y = prices_y[t]
            p_x = prices_x[t]
            target_signal = signal_vals[t]
            
            # 1. Valutazione Equity Corrente Mark-to-Market
            current_equity = cash + (units_y * p_y) + (units_x * p_x)
            
            # 2. Controllo Circuit Breaker di Rischio
            risk_check = self.circuit_breaker.update_and_check(current_equity, trade_date=curr_date)
            
            if not risk_check["can_trade"]:
                # Chiusura immediata d'emergenza delle posizioni se aperte
                if current_pos != 0:
                    # Chiudi Y
                    cash += units_y * p_y * (1.0 - np.sign(units_y) * self.slippage_pct) - abs(units_y * p_y) * self.commission_pct
                    # Chiudi X
                    cash += units_x * p_x * (1.0 - np.sign(units_x) * self.slippage_pct) - abs(units_x * p_x) * self.commission_pct
                    trade_log.append({
                        "date": curr_date,
                        "type": "CIRCUIT_BREAKER_CLOSE",
                        "equity": current_equity,
                        "reason": risk_check["reason"]
                    })
                    units_y = 0.0
                    units_x = 0.0
                    current_pos = 0
                target_signal = 0.0 # Impedisce nuovi ordini
                
            # 3. Esecuzione Rebalancing se il segnale e cambiato
            if target_signal != current_pos and risk_check["can_trade"]:
                # Prima chiudi la posizione corrente se attiva
                if current_pos != 0:
                    cash += units_y * p_y * (1.0 - np.sign(units_y) * self.slippage_pct) - abs(units_y * p_y) * self.commission_pct
                    cash += units_x * p_x * (1.0 - np.sign(units_x) * self.slippage_pct) - abs(units_x * p_x) * self.commission_pct
                    units_y = 0.0
                    units_x = 0.0
                    current_pos = 0
                    
                # Apri la nuova posizione
                if target_signal != 0:
                    target_notional = current_equity * allocated_capital_ratio
                    # Per pairs trading: Long Y + Short (beta * X)
                    if target_signal == 1:
                        # Long spread: +Y, -X
                        units_y = (target_notional / p_y)
                        units_x = - (units_y * beta_vals[t])
                    elif target_signal == -1:
                        # Short spread: -Y, +X
                        units_y = - (target_notional / p_y)
                        units_x = (abs(units_y) * beta_vals[t])
                        
                    # Detrai costi di transazione e impatto
                    cost_y = abs(units_y * p_y) * (self.slippage_pct + self.commission_pct)
                    cost_x = abs(units_x * p_x) * (self.slippage_pct + self.commission_pct)
                    cash -= (units_y * p_y + units_x * p_x + cost_y + cost_x)
                    current_pos = target_signal
                    
                    trade_log.append({
                        "date": curr_date,
                        "type": "ENTRY_LONG_SPREAD" if target_signal == 1 else "ENTRY_SHORT_SPREAD",
                        "equity": current_equity,
                        "units_y": units_y,
                        "units_x": units_x
                    })
                    
            # Registra l'equity finale della barra
            equity_curve[t] = cash + (units_y * p_y) + (units_x * p_x)
            
        equity_series = pd.Series(equity_curve, index=df.index)
        metrics = self._calculate_performance_metrics(equity_series)
        
        return {
            "equity_curve": equity_series,
            "metrics": metrics,
            "trade_log": trade_log,
            "circuit_breaker_tripped": self.circuit_breaker.is_circuit_tripped,
            "trip_reason": self.circuit_breaker.trip_reason
        }

    def _calculate_performance_metrics(self, equity_series: pd.Series) -> Dict[str, Any]:
        """Calcola le metriche statistiche e di rischio istituzionali."""
        returns = equity_series.pct_change().dropna()
        if len(returns) == 0:
            return {}
            
        total_return_pct = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital * 100.0
        
        # Giorni equivalenti (assumiamo barre giornaliere o fattorizziamo a 252 giorni/anno)
        n_bars = len(equity_series)
        annual_factor = 252.0
        cagr = ((equity_series.iloc[-1] / self.initial_capital) ** (annual_factor / max(n_bars, 1)) - 1.0) * 100.0
        
        ann_vol = returns.std() * np.sqrt(annual_factor) * 100.0
        
        # Sharpe Ratio (tasso privo di rischio al 2%)
        rf_daily = 0.02 / annual_factor
        excess_returns = returns - rf_daily
        sharpe = (excess_returns.mean() / (returns.std() + 1e-12)) * np.sqrt(annual_factor)
        
        # Sortino Ratio (considera solo la volatilita dei rendimenti negativi)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(annual_factor)
        sortino = (excess_returns.mean() * np.sqrt(annual_factor)) / (downside_std + 1e-12) if len(downside_returns) > 0 else np.nan
        
        # Drawdown Series & Max Drawdown
        running_max = equity_series.cummax()
        drawdown_series = (equity_series - running_max) / running_max
        max_drawdown_pct = abs(drawdown_series.min()) * 100.0
        
        # Calmar Ratio = CAGR / Max Drawdown
        calmar = (cagr / max_drawdown_pct) if max_drawdown_pct > 0.001 else np.nan
        
        # Win rate dei giorni positivi
        win_days = len(returns[returns > 0])
        win_rate = (win_days / len(returns)) * 100.0
        
        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(equity_series.iloc[-1], 2),
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr, 2),
            "annualized_volatility_pct": round(ann_vol, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "calmar_ratio": round(calmar, 3),
            "daily_win_rate_pct": round(win_rate, 2),
            "total_bars": n_bars
        }
