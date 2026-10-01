@echo off
REM Batch Runner per Quant Engine Paper Trader Orario
cd /d "%~dp0.."
python "quant_engine\run_hourly_paper_trader.py" >> "quant_engine\reports\execution.log" 2>&1
