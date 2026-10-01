"""
quant_engine/risk/circuit_breakers.py
Modulo di Compliance Istituzionale e Circuit Breakers.
Garantisce che il sistema non violi MAI le regole di Max Daily Drawdown e Max Overall Drawdown
richieste da Prop Firms (es. FTMO, Topstep) e comitati di allocazione istituzionale.
"""

from datetime import date
from typing import Dict, Any

class PropComplianceCircuitBreaker:
    """
    Monitora in tempo reale l'equity e applica interruzioni d'emergenza (circuit breakers)
    se il drawdown supera le soglie di tolleranza prefissate.
    """

    def __init__(self, initial_capital: float = 100000.0, 
                 max_daily_loss_pct: float = 0.025, 
                 max_total_loss_pct: float = 0.060):
        self.initial_capital = initial_capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_total_loss_pct = max_total_loss_pct
        
        self.peak_equity = initial_capital
        self.day_start_equity = initial_capital
        self.current_date = None
        self.is_circuit_tripped = False
        self.trip_reason = ""

    def reset_day(self, current_equity: float, trade_date: date):
        """Reimposta l'equity di inizio giornata ad ogni cambio di sessione."""
        self.current_date = trade_date
        self.day_start_equity = current_equity
        if self.trip_reason == "DAILY_LOSS_LIMIT":
            # Resetta il blocco giornaliero alla nuova giornata
            self.is_circuit_tripped = False
            self.trip_reason = ""

    def update_and_check(self, current_equity: float, trade_date: date = None) -> Dict[str, Any]:
        """
        Valuta l'equity corrente e verifica se intervenire con chiusura forzata ordini.
        """
        if trade_date and trade_date != self.current_date:
            self.reset_day(current_equity, trade_date)

        # Aggiornamento picco storico
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calcolo Drawdown Totale dal picco e dal capitale iniziale
        total_drawdown_from_peak = (self.peak_equity - current_equity) / self.peak_equity
        total_drawdown_from_start = (self.initial_capital - current_equity) / self.initial_capital

        # Calcolo Perdita Giornaliera
        daily_loss_pct = (self.day_start_equity - current_equity) / self.day_start_equity

        # 1. Verifica Violazione Drawdown Totale (Hard Stop Assoluto)
        if total_drawdown_from_peak >= self.max_total_loss_pct or total_drawdown_from_start >= self.max_total_loss_pct:
            self.is_circuit_tripped = True
            self.trip_reason = "MAX_TOTAL_DRAWDOWN_EXCEEDED"
            return {
                "action": "EMERGENCY_HALT",
                "can_trade": False,
                "reason": self.trip_reason,
                "current_equity": current_equity,
                "total_drawdown_pct": round(total_drawdown_from_peak * 100, 2),
                "daily_loss_pct": round(daily_loss_pct * 100, 2)
            }

        # 2. Verifica Violazione Drawdown Giornaliero (Stop Trading per la sessione)
        if daily_loss_pct >= self.max_daily_loss_pct:
            self.is_circuit_tripped = True
            self.trip_reason = "DAILY_LOSS_LIMIT"
            return {
                "action": "CLOSE_AND_HALT_TODAY",
                "can_trade": False,
                "reason": self.trip_reason,
                "current_equity": current_equity,
                "total_drawdown_pct": round(total_drawdown_from_peak * 100, 2),
                "daily_loss_pct": round(daily_loss_pct * 100, 2)
            }

        return {
            "action": "CONTINUE",
            "can_trade": True,
            "reason": "OK",
            "current_equity": current_equity,
            "total_drawdown_pct": round(total_drawdown_from_peak * 100, 2),
            "daily_loss_pct": round(daily_loss_pct * 100, 2)
        }
