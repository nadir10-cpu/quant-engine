"""
quant_engine/multi_pair_october_comparison.py
Confronta le performance del modello Kalman StatArb su diverse coppie non-crypto (ETF, Metalli, Energia, Azioni)
nel periodo 1 Ottobre 2023 - 1 Novembre 2023 con capitale di $5,000 USD.
Genera un report di classifica (Leaderboard) HTML.
"""

import os
import sys
import json
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

PAIRS_TO_TEST = [
    {"y": "EWA", "x": "EWC", "name": "EWA/EWC (Australia vs Canada ETF)"},
    {"y": "GLD", "x": "SLV", "name": "GLD/SLV (Oro vs Argento)"},
    {"y": "USO", "x": "BNO", "name": "USO/BNO (WTI vs Brent Petrolio)"},
    {"y": "XLF", "x": "XLI", "name": "XLF/XLI (Finanza vs Industria US)"},
    {"y": "AAPL", "x": "MSFT", "name": "AAPL/MSFT (Apple vs Microsoft)"},
    {"y": "SPY", "x": "QQQ", "name": "SPY/QQQ (S&P500 vs Nasdaq)"},
    {"y": "ETH-USD", "x": "BTC-USD", "name": "ETH/BTC (Crypto Benchmark)"}
]

def run_multi_pair_backtest():
    print("=" * 75)
    print("   SIMULAZIONE COMPARATIVA MULTI-PAIR: OTTOBRE 2023 ($5,000 PROP FIRM)")
    print("=" * 75)
    
    start_date = "2023-08-15"
    end_date = "2023-11-02"
    
    results_summary = []
    
    for pair in PAIRS_TO_TEST:
        t_y = pair["y"]
        t_x = pair["x"]
        p_name = pair["name"]
        
        print(f"\nScanning {p_name} ({t_y} vs {t_x})...")
        try:
            df_y = yf.download(t_y, start=start_date, end=end_date, interval="1d", progress=False)
            df_x = yf.download(t_x, start=start_date, end=end_date, interval="1d", progress=False)
            
            if isinstance(df_y.columns, pd.MultiIndex):
                df_y.columns = df_y.columns.get_level_values(0)
            if isinstance(df_x.columns, pd.MultiIndex):
                df_x.columns = df_x.columns.get_level_values(0)
                
            close_y = df_y["Close"].rename("asset_y")
            close_x = df_x["Close"].rename("asset_x")
            pair_data = pd.concat([close_y, close_x], axis=1).dropna()
            
            strategy = CointegratedPairsStrategy(entry_zscore=1.5, exit_zscore=0.0, stop_loss_zscore=3.0, lookback_window=20, use_kalman=True)
            signals = strategy.generate_signals(pair_data)
            betas = strategy.last_betas
            
            oct_mask = pair_data.index >= "2023-10-01"
            pair_oct = pair_data[oct_mask]
            signals_oct = signals[oct_mask]
            betas_oct = betas[oct_mask] if isinstance(betas, pd.Series) else betas
            
            backtester = InstitutionalBacktester(
                initial_capital=5000.0,
                slippage_bps=1.5,
                commission_bps=2.0,
                max_daily_loss_pct=0.08,
                max_total_loss_pct=0.20
            )
            
            res = backtester.run(pair_oct, signals_oct, hedge_ratio=betas_oct)
            m = res["metrics"]
            
            pnl_val = m.get("final_equity", 5000.0) - 5000.0
            pnl_pct = m.get("total_return_pct", 0.0)
            sharpe = m.get("sharpe_ratio", 0.0)
            max_dd = m.get("max_drawdown_pct", 0.0)
            n_trades = len(res.get("trade_log", []))
            
            results_summary.append({
                "pair_name": p_name,
                "ticker_y": t_y,
                "ticker_x": t_x,
                "final_equity": m.get("final_equity", 5000.0),
                "pnl_usd": pnl_val,
                "total_return_pct": pnl_pct,
                "sharpe_ratio": sharpe,
                "max_drawdown_pct": max_dd,
                "trades": n_trades,
                "compliant": "SUPERATO" if max_dd < 8.0 else "VIOLAZIONE"
            })
            
            print(f"   --> Profitto/Perdita: {pnl_val:+.2f} USD ({pnl_pct:+.2f}%) | Sharpe: {sharpe} | Max DD: -{max_dd}% | Trade: {n_trades}")
        except Exception as e:
            print(f"   --> Errore durante il test di {p_name}: {e}")
            
    # Ordina per Rendimento % decrescente
    df_res = pd.DataFrame(results_summary).sort_values(by="total_return_pct", ascending=False)
    
    # Genera report HTML
    generate_comparison_html(df_res)

def generate_comparison_html(df_res: pd.DataFrame):
    reports_dir = os.path.join(current_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    html_file = os.path.join(reports_dir, "multi_pair_comparison_oct2023.html")
    
    rows = df_res.to_dict(orient="records")
    best_pair = rows[0] if rows else {}
    
    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>Classifica Multi-Pair Quant Engine - Ottobre 2023</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 35px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        h1 {{ color: #38bdf8; margin-top: 0; font-size: 26px; border-bottom: 2px solid #334155; padding-bottom: 12px; }}
        .winner-box {{ background: linear-gradient(135deg, #065f46, #047857); border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid #10b981; }}
        .winner-title {{ font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #a7f3d0; margin-bottom: 5px; }}
        .winner-name {{ font-size: 24px; font-weight: bold; color: #ffffff; }}
        .winner-stats {{ display: flex; gap: 30px; margin-top: 15px; font-size: 16px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 25px; background: #0f172a; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 14px 18px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ background: #1e293b; color: #94a3b8; font-weight: 600; text-transform: uppercase; }}
        tr:hover {{ background: #1e293b; }}
        .rank-1 {{ background: rgba(16, 185, 129, 0.15); font-weight: bold; }}
        .positive {{ color: #10b981; font-weight: bold; }}
        .negative {{ color: #ef4444; font-weight: bold; }}
        .badge {{ padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 12px; }}
        .badge-win {{ background: #065f46; color: #34d399; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 CLASSIFICA STRATEGIE MULTI-PAIR (OTTOBRE 2023)</h1>
        <div style="color:#94a3b8; margin-bottom:20px;">Simulazione su Conto $5,000 USD | Periodo: 1 Ottobre - 1 Novembre 2023 | Modello: Kalman StatArb</div>

        <div class="winner-box">
            <div class="winner-title">🥇 MIGLIORE COPPIA IN ASSOLUTO</div>
            <div class="winner-name">{best_pair.get("pair_name", "")}</div>
            <div class="winner-stats">
                <div>Profitto Totale: <strong style="color:#34d399">${best_pair.get("pnl_usd", 0):+.2f} ({best_pair.get("total_return_pct", 0):+.2f}%)</strong></div>
                <div>Sharpe Ratio: <strong>{best_pair.get("sharpe_ratio", 0)}</strong></div>
                <div>Max Drawdown: <strong>-{best_pair.get("max_drawdown_pct", 0)}%</strong></div>
            </div>
        </div>

        <h3>Tabella Comparativa Risultati ($5,000 USD Base)</h3>
        <table>
            <thead>
                <tr>
                    <th>Posizione</th>
                    <th>Coppia di Asset</th>
                    <th>Capitale Finale</th>
                    <th>Profitto / Perdita</th>
                    <th>Rendimento %</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Prop Firm Compliance</th>
                </tr>
            </thead>
            <tbody>
                {"".join([f'''<tr class="{'rank-1' if idx == 0 else ''}">
                    <td>#{idx+1}</td>
                    <td><strong>{r.get("pair_name","")}</strong></td>
                    <td>${r.get("final_equity",0):,.2f}</td>
                    <td class="{'positive' if r.get('pnl_usd',0) >= 0 else 'negative'}">${r.get("pnl_usd",0):+.2f}</td>
                    <td class="{'positive' if r.get('total_return_pct',0) >= 0 else 'negative'}">{r.get("total_return_pct",0):+.2f}%</td>
                    <td>{r.get("sharpe_ratio",0)}</td>
                    <td>-{r.get("max_drawdown_pct",0)}%</td>
                    <td><span class="badge badge-win">{r.get("compliant","")}</span></td>
                </tr>''' for idx, r in enumerate(rows)])}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"\n[REPORT GENERATO] Classifica comparativa creata in: {html_file}")

if __name__ == "__main__":
    run_multi_pair_backtest()
