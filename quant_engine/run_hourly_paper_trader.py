"""
quant_engine/run_hourly_paper_trader.py
Wrapper esecuzione oraria per MultiAssetDipBuyTrader.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.insert(0, current_dir)
if parent_dir  not in sys.path: sys.path.insert(0, parent_dir)

from quant_engine.paper_trader import MultiAssetDipBuyTrader, ASSETS, ALLOC_PCT, PROFIT_PCT, STOP_PCT, ENTRY_Z


def _trade_rows(trades: list) -> str:
    if not trades:
        return '<tr><td colspan="6" style="text-align:center;color:#64748b;padding:30px;">Nessun trade ancora — In attesa del prossimo dip...</td></tr>'
    rows = []
    for t in reversed(trades):
        action = t.get("action", "")
        acolor = "#10b981" if "TARGET" in action else "#ef4444" if "STOP" in action else "#3b82f6"
        pnl    = t.get("pnl_usd", 0)
        pc     = "#10b981" if pnl > 0 else "#ef4444" if pnl < 0 else "#94a3b8"
        rows.append(
            f'<tr>'
            f'<td>{t.get("timestamp","")}</td>'
            f'<td><b style="color:{acolor}">{action}</b></td>'
            f'<td style="font-weight:600">{t.get("ticker","")}</td>'
            f'<td>${t.get("price",0):,.2f}</td>'
            f'<td style="color:{pc}">${pnl:+.2f}</td>'
            f'<td>${t.get("total_equity",0):,.2f}</td>'
            f'</tr>'
        )
    return "".join(rows)


def generate_dashboard(trader: MultiAssetDipBuyTrader, prices: dict):
    reports_dir = os.path.join(current_dir, "reports")
    dash_file   = os.path.join(reports_dir, "dashboard.html")

    # Leggi trade log
    trades = []
    if os.path.exists(trader.trade_log_file):
        try:
            trades = pd.read_csv(trader.trade_log_file).to_dict(orient="records")
        except Exception:
            pass

    # Equity curve
    eq_labels = [t["timestamp"] for t in trades] if trades else [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    eq_values = [t["total_equity"] for t in trades] if trades else [trader.initial_capital]

    # Totali
    total_eq = sum(
        pos["capital"] + pos["units"] * prices.get(tk, pos["entry_price"])
        for tk, pos in trader.positions.items()
    )
    total_pnl     = total_eq - trader.initial_capital
    total_pnl_pct = total_pnl / trader.initial_capital * 100
    all_wins  = [t for t in trades if t.get("pnl_usd", 0) > 0]
    all_loss  = [t for t in trades if t.get("pnl_usd", 0) < 0]
    wr        = len(all_wins) / len(trades) * 100 if trades else 0
    avg_win   = sum(t["pnl_usd"] for t in all_wins)  / len(all_wins)  if all_wins  else 0
    avg_loss  = sum(t["pnl_usd"] for t in all_loss)  / len(all_loss)  if all_loss  else 0

    # Asset cards
    asset_cards_html = ""
    for asset in ASSETS:
        tk  = asset["ticker"]
        pos = trader.positions[tk]
        cp  = prices.get(tk, pos["entry_price"] or 0)
        eq  = pos["capital"] + pos["units"] * cp
        ret = (eq - trader.initial_capital * ALLOC_PCT) / (trader.initial_capital * ALLOC_PCT) * 100
        if pos["position"] == 1:
            unr    = pos["units"] * (cp - pos["entry_price"])
            unr_p  = (cp - pos["entry_price"]) / pos["entry_price"] * 100
            tp     = pos["entry_price"] * (1 + PROFIT_PCT)
            sl     = pos["entry_price"] * (1 - STOP_PCT)
            status_html = f"""
                <div style="margin-top:10px;padding:10px;background:#0f2a1f;border-radius:8px;font-size:12px;">
                    <div style="color:#a7f3d0;font-weight:600">LONG APERTO</div>
                    <div style="color:#94a3b8;margin-top:4px">Entry: <b style="color:#f8fafc">${pos['entry_price']:,.2f}</b></div>
                    <div style="color:#94a3b8">Target: <b style="color:#10b981">${tp:,.2f}</b> &nbsp; Stop: <b style="color:#ef4444">${sl:,.2f}</b></div>
                    <div style="color:{'#10b981' if unr>=0 else '#ef4444'};font-weight:700;margin-top:4px">PnL: ${unr:+.2f} ({unr_p:+.2f}%)</div>
                </div>"""
        else:
            status_html = '<div style="margin-top:10px;color:#64748b;font-size:12px;">FLAT — In attesa di segnale dip</div>'

        color = "#10b981" if ret >= 0 else "#ef4444"
        asset_cards_html += f"""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="font-weight:700;font-size:1.1rem">{tk}</div>
                <div style="font-size:.8rem;color:{color};font-weight:700">{ret:+.1f}%</div>
            </div>
            <div style="color:#94a3b8;font-size:.8rem;margin-top:2px">{asset['name']}</div>
            <div style="margin-top:8px;font-size:.9rem">
                <span style="color:#94a3b8">Equity: </span><b style="color:#f8fafc">${eq:,.2f}</b>
                &nbsp;&nbsp;<span style="color:#94a3b8">Prezzo: </span><b style="color:#38bdf8">${cp:,.2f}</b>
            </div>
            {status_html}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="3600">
<title>Dip-Buy Paper Trading | Multi-Asset</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
:root{{--bg:#0f172a;--card:#1e293b;--border:#334155;--text:#f8fafc;--muted:#94a3b8;--green:#10b981;--red:#ef4444;--blue:#3b82f6;--purple:#8b5cf6;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:28px}}
.container{{max-width:1200px;margin:0 auto}}
header{{margin-bottom:28px}}
h1{{font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#3b82f6,#8b5cf6,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.sub{{color:var(--muted);font-size:.9rem;margin-top:5px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:24px}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px}}
.kpi-label{{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.kpi-val{{font-size:1.5rem;font-weight:700}}
.green{{color:var(--green)}}.red{{color:var(--red)}}.blue{{color:var(--blue)}}.purple{{color:var(--purple)}}
.asset-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-bottom:24px}}
.chart-box{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:24px}}
.chart-title{{color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:14px}}
.table-box{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:11px 14px;text-align:left;border-bottom:1px solid var(--border);font-size:.85rem}}
th{{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.3px}}
tr:hover{{background:#1a2744}}
.footer{{text-align:center;color:var(--muted);font-size:.75rem;margin-top:24px}}
</style>
</head>
<body>
<div class="container">
<header>
<h1>Multi-Asset Dip-Buy Paper Trader</h1>
<div class="sub">NVDA &middot; QQQ &middot; AAPL &middot; GLD &nbsp;|&nbsp; Long-Only &middot; Target +{PROFIT_PCT*100:.0f}% &middot; Stop -{STOP_PCT*100:.1f}% &nbsp;|&nbsp; $5,000 Prop Firm</div>
</header>

<div class="kpi-grid">
<div class="kpi"><div class="kpi-label">Capitale Iniziale</div><div class="kpi-val blue">$5,000.00</div></div>
<div class="kpi"><div class="kpi-label">Equity Totale</div><div class="kpi-val {'green' if total_eq>=trader.initial_capital else 'red'}">${total_eq:,.2f}</div></div>
<div class="kpi"><div class="kpi-label">PnL dal 1 Ott</div><div class="kpi-val {'green' if total_pnl>=0 else 'red'}">{total_pnl:+.2f} ({total_pnl_pct:+.2f}%)</div></div>
<div class="kpi"><div class="kpi-label">Trade Totali</div><div class="kpi-val purple">{len(trades)}</div></div>
<div class="kpi"><div class="kpi-label">Win Rate</div><div class="kpi-val {'green' if wr>=50 else 'red'}">{wr:.1f}%</div></div>
<div class="kpi"><div class="kpi-label">Avg Win / Loss</div><div class="kpi-val">${avg_win:+.0f} / ${avg_loss:.0f}</div></div>
</div>

<div class="asset-grid">{asset_cards_html}</div>

<div class="chart-box">
<div class="chart-title">Equity Curve</div>
<canvas id="eqChart" height="80"></canvas>
</div>

<div class="table-box">
<div class="chart-title" style="margin-bottom:14px">Registro Operazioni</div>
<div style="overflow-x:auto">
<table>
<thead><tr><th>Ora</th><th>Azione</th><th>Asset</th><th>Prezzo</th><th>PnL</th><th>Equity</th></tr></thead>
<tbody>{_trade_rows(trades)}</tbody>
</table>
</div>
</div>

<div class="footer">Aggiornato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; Long-Only Dip Buy Engine v3.0</div>
</div>
<script>
new Chart(document.getElementById('eqChart'),{{
    type:'line',
    data:{{
        labels:{json.dumps(eq_labels)},
        datasets:[{{
            label:'Equity ($)',
            data:{json.dumps(eq_values)},
            borderColor:'#3b82f6',
            backgroundColor:'rgba(59,130,246,0.07)',
            fill:true,tension:0.4,borderWidth:2,pointRadius:3,pointBackgroundColor:'#3b82f6'
        }},{{
            label:'Capitale Iniziale',
            data:Array({len(eq_labels)}).fill(5000),
            borderColor:'rgba(100,116,139,0.5)',
            borderDash:[6,4],borderWidth:1.5,pointRadius:0,fill:false
        }}]
    }},
    options:{{
        responsive:true,
        plugins:{{legend:{{labels:{{color:'#94a3b8',font:{{size:12}}}}}}}},
        scales:{{
            x:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',maxTicksLimit:10}}}},
            y:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',callback:v=>'$'+v.toLocaleString()}}}}
        }}
    }}
}});
</script>
</body>
</html>"""

    with open(dash_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DASHBOARD] {dash_file}")


def run_hourly():
    print(f"\n{'='*70}")
    print(f"  ESECUZIONE ORARIA  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    trader = MultiAssetDipBuyTrader(initial_capital=5000.0)
    trader.execute_cycle()

    # Recupera prezzi aggiornati per la dashboard
    import yfinance as yf
    prices = {}
    for asset in ASSETS:
        tk = asset["ticker"]
        df = yf.download(tk, period="2d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        prices[tk] = float(df["Close"].iloc[-1])

    generate_dashboard(trader, prices)


if __name__ == "__main__":
    run_hourly()
