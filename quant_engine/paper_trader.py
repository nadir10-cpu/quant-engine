"""
quant_engine/paper_trader.py
Multi-Asset Long-Only Dip Buy Paper Trader
Backtest: $5,000 -> $6,497 (+29.94%) su 33 mesi | ~1.3 trade/settimana
Logica: Compra solo sui dip estremi (Z <= -1.0) in regime rialzista (EMA20 > EMA60)
Target: +5% | Stop: -2.5% | 4 asset da $1,250 ciascuno
"""

import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.insert(0, current_dir)
if parent_dir  not in sys.path: sys.path.insert(0, parent_dir)


ASSETS = [
    {"ticker": "NVDA", "name": "NVIDIA Corp."},
    {"ticker": "QQQ",  "name": "Nasdaq 100 ETF"},
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "GLD",  "name": "Oro ETF"},
]

ENTRY_Z     = 1.0    # z-score threshold (dip entry)
ZSCORE_WIN  = 8      # finestra z-score (giorni)
REGIME_FAST = 20     # EMA veloce per regime filter
REGIME_SLOW = 60     # EMA lenta per regime filter
PROFIT_PCT  = 0.05   # +5% target per chiudere in profitto
STOP_PCT    = 0.025  # -2.5% stop loss
ALLOC_PCT   = 0.25   # 25% del conto per ogni asset


class MultiAssetDipBuyTrader:
    """
    Paper trader Long-Only Dip Buy su portafoglio di 4 asset.
    Ogni asset opera in modo indipendente con il proprio budget ($1,250).
    """

    def __init__(self,
                 initial_capital: float = 5000.0,
                 reports_dir:     str   = None,
                 state_file:      str   = None):

        self.initial_capital = initial_capital
        self.reports_dir = reports_dir or os.path.join(current_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.trade_log_file = os.path.join(self.reports_dir, "paper_trades.csv")
        self.state_file     = state_file or os.path.join(current_dir, "state.json")

        # Stato per ogni asset: capital, units, entry_price, position (0/1)
        self.positions = {
            a["ticker"]: {
                "capital":    initial_capital * ALLOC_PCT,
                "units":      0.0,
                "entry_price":0.0,
                "position":   0,    # 0=flat, 1=long
                "entry_time": None
            }
            for a in ASSETS
        }

        if not os.path.exists(self.trade_log_file):
            pd.DataFrame(columns=[
                "timestamp","ticker","action","price","units",
                "pnl_usd","total_equity"
            ]).to_csv(self.trade_log_file, index=False)

        self.load_state()

    # ─── PERSISTENZA ──────────────────────────────────────────────────────────

    def save_state(self):
        state = {
            "initial_capital": self.initial_capital,
            "last_updated":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "positions": {}
        }
        for tk, pos in self.positions.items():
            state["positions"][tk] = {
                "capital":     pos["capital"],
                "units":       pos["units"],
                "entry_price": pos["entry_price"],
                "position":    pos["position"],
                "entry_time":  pos["entry_time"].strftime("%Y-%m-%d %H:%M:%S") if pos["entry_time"] else None
            }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)

    def load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            if "positions" not in state:
                return
            self.initial_capital = state.get("initial_capital", self.initial_capital)
            for tk, saved in state["positions"].items():
                if tk in self.positions:
                    self.positions[tk]["capital"]     = saved.get("capital",     self.initial_capital * ALLOC_PCT)
                    self.positions[tk]["units"]       = saved.get("units",       0.0)
                    self.positions[tk]["entry_price"] = saved.get("entry_price", 0.0)
                    self.positions[tk]["position"]    = saved.get("position",    0)
                    et = saved.get("entry_time")
                    self.positions[tk]["entry_time"]  = datetime.strptime(et, "%Y-%m-%d %H:%M:%S") if et else None
            print(f"[STATE LOADED] Aggiornato: {state.get('last_updated','?')}")
        except Exception as e:
            print(f"[STATE LOAD WARNING] {e}")

    # ─── LOG ──────────────────────────────────────────────────────────────────

    def log_trade(self, ticker: str, action: str, price: float, units: float,
                  pnl: float, total_equity: float):
        rec = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker":       ticker,
            "action":       action,
            "price":        round(price,  4),
            "units":        round(units,  6),
            "pnl_usd":      round(pnl,    2),
            "total_equity": round(total_equity, 2)
        }
        pd.DataFrame([rec]).to_csv(self.trade_log_file, mode="a", header=False, index=False)

    # ─── FETCH + INDICATORI ───────────────────────────────────────────────────

    def fetch(self, ticker: str) -> tuple:
        """Restituisce (prezzo_corrente, bullish: bool, zscore: float)"""
        df = yf.download(ticker, period="120d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        prices = df["Close"].dropna()
        p = prices.values
        curr_price = float(p[-1])

        ema_f = pd.Series(p).ewm(span=REGIME_FAST, adjust=False).mean().values
        ema_s = pd.Series(p).ewm(span=REGIME_SLOW, adjust=False).mean().values
        bullish = bool(ema_f[-1] > ema_s[-1])

        win = p[-ZSCORE_WIN:]
        zs  = (p[-1] - np.mean(win)) / (np.std(win) + 1e-12)
        return curr_price, bullish, float(zs)

    # ─── CICLO PRINCIPALE ─────────────────────────────────────────────────────

    def execute_cycle(self):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "=" * 70)
        print(f"  MULTI-ASSET DIP-BUY PAPER TRADER  |  {now_str}")
        print(f"  Asset: NVDA | QQQ | AAPL | GLD  |  Target +5% | Stop -2.5%")
        print("=" * 70)

        total_equity = 0.0
        trade_events = []

        for asset in ASSETS:
            ticker = asset["ticker"]
            name   = asset["name"]
            pos    = self.positions[ticker]

            curr_price, bullish, zs = self.fetch(ticker)
            mark_to_market = pos["capital"] + pos["units"] * curr_price
            total_equity  += mark_to_market

            print(f"\n  [{ticker}] {name}")
            print(f"   Prezzo: ${curr_price:,.2f}  |  Regime: {'BULLISH' if bullish else 'BEARISH'}  |  Z-Score: {zs:+.2f}")

            # ── Chiusura posizione ──
            if pos["position"] == 1:
                ret = (curr_price - pos["entry_price"]) / pos["entry_price"]
                unrealized = pos["units"] * (curr_price - pos["entry_price"])
                print(f"   LONG aperto @ ${pos['entry_price']:,.2f}  |  PnL: ${unrealized:+.2f} ({ret*100:+.2f}%)")

                if ret >= PROFIT_PCT:
                    pnl = pos["units"] * (curr_price - pos["entry_price"]) - abs(pos["units"]*curr_price)*0.0003
                    pos["capital"] += pos["units"] * curr_price - abs(pos["units"]*curr_price)*0.0003
                    self.log_trade(ticker, "CLOSE_TARGET", curr_price, pos["units"], pnl, pos["capital"])
                    trade_events.append(f"   *** TARGET +5% su {ticker}: +${pnl:.2f} USD ***")
                    pos["units"]=0.0; pos["position"]=0; pos["entry_price"]=0.0; pos["entry_time"]=None
                elif ret <= -STOP_PCT:
                    pnl = pos["units"] * (curr_price - pos["entry_price"]) - abs(pos["units"]*curr_price)*0.0003
                    pos["capital"] += pos["units"] * curr_price - abs(pos["units"]*curr_price)*0.0003
                    self.log_trade(ticker, "CLOSE_STOP", curr_price, pos["units"], pnl, pos["capital"])
                    trade_events.append(f"   [STOP -2.5%] {ticker}: ${pnl:.2f} USD")
                    pos["units"]=0.0; pos["position"]=0; pos["entry_price"]=0.0; pos["entry_time"]=None

            # ── Apertura nuova posizione ──
            if pos["position"] == 0:
                if bullish and zs <= -ENTRY_Z:
                    units = pos["capital"] / curr_price
                    pos["capital"]     -= units * curr_price * (1 + 0.0003)
                    pos["units"]        = units
                    pos["entry_price"]  = curr_price
                    pos["position"]     = 1
                    pos["entry_time"]   = datetime.now()
                    target_p = curr_price * (1 + PROFIT_PCT)
                    stop_p   = curr_price * (1 - STOP_PCT)
                    self.log_trade(ticker, "OPEN_LONG", curr_price, units, 0.0, pos["capital"])
                    trade_events.append(
                        f"   >>> LONG {ticker} @ ${curr_price:,.2f} | Target ${target_p:,.2f} (+${units*(target_p-curr_price):.2f}) | Stop ${stop_p:,.2f}"
                    )
                    print(f"   >>> LONG APERTO | Target ${target_p:,.2f} (+${units*(target_p-curr_price):.2f} USD) | Stop ${stop_p:,.2f}")
                else:
                    cond = "Regime BEARISH" if not bullish else f"Z={zs:+.2f} (soglia -{ENTRY_Z})"
                    print(f"   Flat - Nessun segnale ({cond})")

        pnl_tot = total_equity - self.initial_capital
        print("\n" + "-" * 70)
        if trade_events:
            for ev in trade_events: print(ev)
        print(f"\n  Equity Totale:  ${total_equity:,.2f}  |  PnL dal 1 Ott: ${pnl_tot:+.2f} ({pnl_tot/self.initial_capital*100:+.2f}%)")
        print("=" * 70)

        self.save_state()
        return total_equity
