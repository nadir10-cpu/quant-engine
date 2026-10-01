"""
quant_engine/data/storage.py
Modulo di persistenza e serializzazione veloce dei dati di mercato in formato Parquet / Pickle.
Elimina l'overhead di parsing e massimizza la velocita di caricamento dei dati storici.
"""

import os
import pandas as pd
from typing import Optional

class MarketDataStore:
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.storage_dir = os.path.join(base_dir, "cache")
        else:
            self.storage_dir = storage_dir
            
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_path(self, ticker: str, timeframe: str) -> str:
        clean_ticker = ticker.replace("^", "").replace("=", "_").replace("/", "_")
        return os.path.join(self.storage_dir, f"{clean_ticker}_{timeframe}.parquet")

    def save(self, ticker: str, df: pd.DataFrame, timeframe: str = "1d"):
        """Salva il DataFrame storico in formato Parquet compresso."""
        if df is None or df.empty:
            return
        path = self._get_path(ticker, timeframe)
        try:
            df.to_parquet(path, compression="snappy")
        except Exception:
            # Fallback a pickle se pyarrow non e installato
            pickle_path = path.replace(".parquet", ".pkl")
            df.to_pickle(pickle_path)

    def load(self, ticker: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """Carica il DataFrame storico dalla cache locale ad alta velocita."""
        path = self._get_path(ticker, timeframe)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
                
        pickle_path = path.replace(".parquet", ".pkl")
        if os.path.exists(pickle_path):
            return pd.read_pickle(pickle_path)
            
        return None
