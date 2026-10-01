"""
quant_engine/risk/sizing.py
Motore Istituzionale di Dimensionamento della Posizione (Position Sizing).
Combina Volatility Targeting (Risk Parity) e Criterio di Kelly Frazionario (Half-Kelly).
"""

import numpy as np

class InstitutionalPositionSizer:
    """
    Gestisce la dimensione ottima dei trade proteggendo il capitale dal variance drag
    e calibrando il rischio a un livello costante di volatilita di portafoglio.
    """

    def __init__(self, target_annual_vol: float = 0.12, half_kelly_mult: float = 0.5, max_leverage: float = 2.0):
        self.target_annual_vol = target_annual_vol
        self.half_kelly_mult = half_kelly_mult
        self.max_leverage = max_leverage

    def compute_volatility_scalar(self, current_annualized_vol: float) -> float:
        """
        Calcola il moltiplicatore di posizione inversamente alla volatilita dell'asset (Volatility Targeting).
        Vol scalar = Target Vol / Realized Vol
        """
        if current_annualized_vol <= 0.001:
            return 1.0
        scalar = self.target_annual_vol / current_annualized_vol
        return min(scalar, self.max_leverage)

    def compute_kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calcola la frazione ottimale di capitale da allocare secondo il criterio di Kelly:
        f* = (b*p - q) / b = (p*(b + 1) - 1) / b
        Applicando la regola prudenziale dell'Half-Kelly: f_safe = 0.5 * f*
        """
        if win_loss_ratio <= 0.01:
            return 0.0
            
        b = win_loss_ratio
        p = win_rate
        q = 1.0 - p
        
        kelly_full = (b * p - q) / b
        
        # Se non c'e edge statistico (kelly <= 0), non allocare
        if kelly_full <= 0:
            return 0.0
            
        # Half-Kelly rule
        kelly_safe = kelly_full * self.half_kelly_mult
        
        # Cap massimo per singolo trade: max 20% del capitale totale
        return min(kelly_safe, 0.20)

    def get_order_size(self, equity: float, asset_price: float, asset_vol: float, 
                       win_rate: float = 0.55, win_loss_ratio: float = 1.5,
                       stop_loss_pct: float = 0.02) -> dict:
        """
        Calcola l'esatto numero di unita e il valore nozionale per l'ordine.
        
        Args:
            equity: Capitale totale disponibile sul conto (USD)
            asset_price: Prezzo corrente dell'asset
            asset_vol: Volatilita annualizzata stimata dell'asset (es. 0.20 per 20%)
            win_rate: Percentuale di successo storica della strategia
            win_loss_ratio: Rapporto guadagno medio / perdita media
            stop_loss_pct: Distanza percentuale dello stop loss (es. 0.02 per 2%)
            
        Returns:
            Dizionario con 'units', 'notional_usd', 'capital_at_risk_usd', 'kelly_fraction'
        """
        kelly_f = self.compute_kelly_fraction(win_rate, win_loss_ratio)
        vol_scalar = self.compute_volatility_scalar(asset_vol)
        
        # Capitale a rischio desiderato basato su Kelly (max 2% dell'equity)
        max_risk_dollars = equity * min(kelly_f, 0.02)
        
        # Calcolo dimensione in base alla distanza dello stop loss
        risk_per_unit = asset_price * stop_loss_pct
        if risk_per_unit <= 1e-6:
            units = 0.0
        else:
            units = (max_risk_dollars / risk_per_unit) * vol_scalar
            
        notional = units * asset_price
        
        # Controllo che il nozionale non ecceda la leva massima consentita
        max_notional = equity * self.max_leverage
        if notional > max_notional:
            notional = max_notional
            units = notional / asset_price

        return {
            "units": round(units, 4),
            "notional_usd": round(notional, 2),
            "capital_at_risk_usd": round(units * risk_per_unit, 2),
            "kelly_fraction": round(kelly_f, 4),
            "volatility_scalar": round(vol_scalar, 3)
        }
