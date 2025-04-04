import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies import rsi_trading_strategy, calculate_rsi

class TestRSIStrategy(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'Close': np.random.normal(100, 5, 100)
        }, index=dates)
    
    def test_rsi_calculation(self):
        rsi = calculate_rsi(self.df['Close'])
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))
    
    def test_rsi_trading_strategy(self):
        signals = rsi_trading_strategy(self.df)
        self.assertIn('signal', signals.columns)
        self.assertIn('positions', signals.columns)

if __name__ == '__main__':
    unittest.main()