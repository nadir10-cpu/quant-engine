"""
quant_engine/data/ingestion.py
Modulo di Data Ingestion da API finanziarie (yfinance) con caching integrato.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional
from quant_engine.data.storage import MarketDataStore

class MarketDataIngestor:
    def __init__(self, data_store: Optional[MarketDataStore] = None):
        self.store = data_store or MarketDataStore()

    def fetch_ohlcv(self, ticker: str, start_date: str = "2020-01-01", 
                    end_date: str = None, interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
        """
        Recupera le barre OHLCV da cache o tramite API yfinance.
        """
        if not force_refresh:
            cached_df = self.store.load(ticker, timeframe=interval)
            if cached_df is not None and len(cached_df) > 0:
                return cached_df

        # Download da Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if df is None or df.empty:
            raise ValueError(f"Dati non trovati per il ticker: {ticker}")

        # Se il dataframe ha MultiIndex sulle colonne, appiattiscilo
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Rinomina colonne in minuscolo
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        })

        # Salva in cache Parquet
        self.store.save(ticker, df, timeframe=interval)
        return df

    def fetch_pair(self, ticker_y: str, ticker_x: str, start_date: str = "2020-01-01",
                   interval: str = "1d") -> pd.DataFrame:
        """
        Recupera e allinea i prezzi di chiusura di una coppia per il Pairs Trading.
        Ritorna un DataFrame con colonne ['asset_y', 'asset_x'] perfettamente allineate.
        """
        df_y = self.fetch_ohlcv(ticker_y, start_date=start_date, interval=interval)
        df_x = self.fetch_ohlcv(ticker_x, start_date=start_date, interval=interval)

        close_y = df_y['close'].rename('asset_y')
        close_x = df_x['close'].rename('asset_x')

        pair_df = pd.concat([close_y, close_x], axis=1).dropna()
        return pair_df
