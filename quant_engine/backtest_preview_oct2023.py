"""
quant_engine/backtest_preview_oct2023.py
Esegue la simulazione di backtest storica sul periodo 1 Ottobre 2023 - 1 Novembre 2023
per la coppia ETH-USD / BTC-USD con il capitale Prop Firm di $5,000 USD.
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from quant_engine.strategies.stat_arb_coint import CointegratedPairsStrategy
from quant_engine.backtest.event_backtester import InstitutionalBacktester
from quant_engine.backtest.tear_sheet import InstitutionalTearSheetGenerator

def run_october_2023_backtest():
    print("=" * 70)
    print("   BACKTEST PREVIEW: OTT-NOV 2023 ($5,000 PROP FIRM CHALLENGE)")
    print("=" * 70)
    
    ticker_y = "ETH-USD"
    ticker_x = "BTC-USD"
    start_date = "2023-08-15" # Include periodo di burn-in per la cointegrazione
    end_date = "2023-11-02"
    
    print(f"\n[1/4] Scaricamento dati storici Yahoo Finance: {ticker_y} vs {ticker_x} ({start_date} -> {end_date})...")
    df_y = yf.download(ticker_y, start=start_date, end=end_date, interval="1d", progress=False)
    df_x = yf.download(ticker_x, start=start_date, end=end_date, interval="1d", progress=False)
    
    if isinstance(df_y.columns, pd.MultiIndex):
        df_y.columns = df_y.columns.get_level_values(0)
    if isinstance(df_x.columns, pd.MultiIndex):
        df_x.columns = df_x.columns.get_level_values(0)
        
    close_y = df_y["Close"].rename("asset_y")
    close_x = df_x["Close"].rename("asset_x")
    pair_data = pd.concat([close_y, close_x], axis=1).dropna()
    
    print(f"      Dati acquisiti: {len(pair_data)} giornate borsistiche.")

    print("\n[2/4] Calcolo Cointegrazione e Generazione Segnali Z-Score...")
    strategy = CointegratedPairsStrategy(entry_zscore=1.4, exit_zscore=0.2, stop_loss_zscore=3.2, lookback_window=30)
    signals = strategy.generate_signals(pair_data)
    rolling_betas = strategy.last_betas
    
    # Filtra il periodo da ottobre 2023 in poi
    oct_mask = pair_data.index >= "2023-10-01"
    pair_oct = pair_data[oct_mask]
    signals_oct = signals[oct_mask]
    betas_oct = rolling_betas[oct_mask] if isinstance(rolling_betas, pd.Series) else rolling_betas

    print("\n[3/4] Esecuzione Event-Driven Backtest su Conto $5,000 USD...")
    backtester = InstitutionalBacktester(
        initial_capital=5000.0,
        slippage_bps=1.5,
        commission_bps=2.0,
        max_daily_loss_pct=0.08, # Prop firm limits
        max_total_loss_pct=0.20
    )
    
    results = backtester.run(pair_oct, signals_oct, hedge_ratio=betas_oct)
    metrics = results["metrics"]

    print("\n[4/4] Generazione Tear Sheet Istituzionale HTML...")
    reports_dir = os.path.join(current_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, "backtest_preview_oct2023.html")
    
    report_gen = InstitutionalTearSheetGenerator()
    report_gen.generate_html_report(results, strategy_name=f"Preview StatArb ETH/BTC ($5k) - Ottobre 2023", output_path=report_file)

    print("\n" + "=" * 70)
    print("            RISULTATI SIMULAZIONE STORICA (OTTOBRE 2023)")
    print("=" * 70)
    print(f"  Capitale Iniziale:           ${metrics.get('initial_capital', 5000):,.2f}")
    print(f"  Capitale Finale:             ${metrics.get('final_equity', 5000):,.2f}")
    print(f"  Rendimento Totale:           {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"  Volatilita Annualizzata:     {metrics.get('annualized_volatility_pct', 0):.2f}%")
    print(f"  SHARPE RATIO:                {metrics.get('sharpe_ratio', 0)}")
    print(f"  MAX DRAWDOWN:                -{metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Conformita Regole Prop Firm: {'SUPERATO (COMPLIANT)' if metrics.get('max_drawdown_pct', 100) < 8.0 else 'VIOLAZIONE'}")
    print(f"  File Report Salvato:         {report_file}")
    print("=" * 70)

if __name__ == "__main__":
    run_october_2023_backtest()
