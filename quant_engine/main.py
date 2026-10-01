"""
quant_engine/main.py
Entrypoint Principale del Motore Quantitativo Istituzionale.
Esegue la pipeline completa:
1. Data Ingestion (Coppia cointegrata ad alta liquidita);
2. Stima StatArb & Half-life di Ornstein-Uhlenbeck;
3. Risoluzione della Quantum Finance Schrödinger Equation (QFSE) per i livelli QPL;
4. Backtest ad eventi con costi realistici (slippage + commissioni);
5. Compliance Prop Firm e generazione Tear Sheet Istituzionale (HTML).
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd

# Assicura importazione pacchetti interni
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from quant_engine.data.ingestion import MarketDataIngestor
from quant_engine.strategies.stat_arb_coint import CointegratedPairsStrategy
from quant_engine.strategies.quantum_price_levels import QuantumPriceModel
from quant_engine.risk.sizing import InstitutionalPositionSizer
from quant_engine.backtest.event_backtester import InstitutionalBacktester
from quant_engine.backtest.tear_sheet import InstitutionalTearSheetGenerator

def load_config() -> dict:
    config_path = os.path.join(current_dir, "config", "settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_pipeline(ticker_y: str = "EWA", ticker_x: str = "EWC", start_date: str = "2021-01-01"):
    print("=" * 70)
    print("   INSTITUTIONAL QUANTITATIVE TRADING & CAPITAL RAISING ENGINE")
    print("=" * 70)

    config = load_config()
    initial_cap = config["engine"]["initial_capital"]
    risk_cfg = config["risk_management"]
    exec_cfg = config["execution"]
    strat_cfg = config["strategy_parameters"]["stat_arb"]

    # 1. DATA INGESTION
    print(f"\n[1/5] Ingestion dati di mercato per la coppia: {ticker_y} vs {ticker_x}...")
    ingestor = MarketDataIngestor()
    pair_data = ingestor.fetch_pair(ticker_y=ticker_y, ticker_x=ticker_x, start_date=start_date)
    print(f"      Dati acquisiti: {len(pair_data)} barre storiche giornaliere.")

    # 2. STATISTICAL ARBITRAGE & STOCHASTIC CALIBRATION
    print("\n[2/5] Calibrazione Statistica & Half-Life di Ornstein-Uhlenbeck...")
    strategy = CointegratedPairsStrategy(
        entry_zscore=strat_cfg["entry_zscore"],
        exit_zscore=strat_cfg["exit_zscore"],
        stop_loss_zscore=strat_cfg["stop_loss_zscore"],
        lookback_window=strat_cfg["lookback_window"]
    )
    
    signals = strategy.generate_signals(pair_data)
    rolling_betas = strategy.last_betas
    beta_mean = float(rolling_betas[rolling_betas > 0].mean())
    spread = pair_data['asset_y'].values - beta_mean * pair_data['asset_x'].values
    half_life = strategy.estimate_half_life(spread)
    
    print(f"      Hedge Ratio medio (Beta): {beta_mean:.4f}")
    print(f"      Half-Life di ritorno alla media: {half_life:.1f} giorni")

    # 3. QUANTUM FINANCE SCHRÖDINGER EQUATION (QFSE)
    print("\n[3/5] Risoluzione Equazione di Schrödinger Finanziaria (QFSE)...")
    q_model = QuantumPriceModel(
        mu=config["strategy_parameters"]["quantum_price_levels"]["mu"],
        lambda_param=config["strategy_parameters"]["quantum_price_levels"]["lambda_param"]
    )
    curr_price_y = float(pair_data['asset_y'].iloc[-1])
    ret_y = pair_data['asset_y'].pct_change().dropna()
    vol_y = float(ret_y.std() * np.sqrt(252))
    
    q_results = q_model.compute_quantum_support_resistance(
        current_price=curr_price_y,
        historical_volatility=vol_y,
        n_levels=5
    )
    print(f"      Prezzo Corrente {ticker_y}: ${curr_price_y:.2f}")
    print(f"      Supporti Quantistici (QPL): {q_results['quantum_supports']}")
    print(f"      Resistenze Quantistiche (QPL): {q_results['quantum_resistances']}")

    # 4. EVENT-DRIVEN BACKTEST CON COSTI E COMPLIANCE PROP FIRM
    print("\n[4/5] Esecuzione Backtest Istituzionale con Slippage e Commissioni...")
    backtester = InstitutionalBacktester(
        initial_capital=initial_cap,
        slippage_bps=exec_cfg["slippage_bps"],
        commission_bps=exec_cfg["commission_bps"],
        max_daily_loss_pct=risk_cfg["max_daily_drawdown"],
        max_total_loss_pct=risk_cfg["max_total_drawdown"]
    )
    
    results = backtester.run(pair_data, signals, hedge_ratio=rolling_betas)
    metrics = results["metrics"]

    # 5. GENERAZIONE TEAR SHEET ISTITUZIONALE
    print("\n[5/5] Generazione Tear Sheet Istituzionale HTML per Investitori...")
    report_gen = InstitutionalTearSheetGenerator()
    report_file = report_gen.generate_html_report(results, strategy_name=f"Pairs Trading Cointegrato ({ticker_y}/{ticker_x}) + QFSE")
    print(f"      Report generato con successo: {report_file}")

    print("\n" + "=" * 70)
    print("                     METRICHE CHIAVE DI PERFORMANCE")
    print("=" * 70)
    print(f"  Capitale Iniziale:           ${metrics.get('initial_capital', 0):,.2f}")
    print(f"  Capitale Finale:             ${metrics.get('final_equity', 0):,.2f}")
    print(f"  Rendimento Totale:           +{metrics.get('total_return_pct', 0)}%")
    print(f"  CAGR (Rendimento Annuo):     +{metrics.get('cagr_pct', 0)}%")
    print(f"  Volatilita Annualizzata:     {metrics.get('annualized_volatility_pct', 0)}%")
    print(f"  SHARPE RATIO:                {metrics.get('sharpe_ratio', 0)}")
    print(f"  SORTINO RATIO:               {metrics.get('sortino_ratio', 0)}")
    print(f"  MAX DRAWDOWN:                -{metrics.get('max_drawdown_pct', 0)}%")
    print(f"  CALMAR RATIO:                {metrics.get('calmar_ratio', 0)}")
    print(f"  Trade Totali Eseguiti:       {len(results.get('trade_log', []))}")
    print(f"  Conformita Regole Prop Firm: {'SUPERATO (COMPLIANT)' if metrics.get('max_drawdown_pct', 100) < 6.0 else 'VIOLAZIONE'}")
    print("=" * 70)

    return results

if __name__ == "__main__":
    run_pipeline()
