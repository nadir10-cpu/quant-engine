"""
quant_engine/strategies/base_alpha.py
Classe base astratta per la generazione modulare di segnali Alpha (ispirata all'architettura QT101).
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class GenericAlphaUnit(ABC):
    """
    Interfaccia standardizzata per tutte le unità generatrici di Alpha.
    Separa rigorosamente la logica di calcolo del segnale (feature extraction)
    dalla gestione degli ordini, del rischio e dell'esecuzione.
    """
    
    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}
        
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Calcola i segnali di trading normalizzati.
        
        Args:
            market_data: pd.DataFrame contenente serie storiche (Open, High, Low, Close, Volume)
            
        Returns:
            pd.Series contenente lo score di segnale compreso tra -1.0 (forte short) e +1.0 (forte long).
            0.0 indica neutralità / assenza di segnale.
        """
        pass

    def __repr__(self) -> str:
        return f"<AlphaUnit: {self.name} | Params: {self.params}>"
