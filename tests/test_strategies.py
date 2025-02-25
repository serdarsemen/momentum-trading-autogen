# tests/test_strategies.py
import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies import momentum_trading_strategy, compute_returns

class TestMomentumStrategy(unittest.TestCase):
    
    def setUp(self):
        # Create dummy data for testing
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'Open': np.random.normal(100, 5, 100),
            'High': np.random.normal(105, 5, 100),
            'Low': np.random.normal(95, 5, 100),
            'Close': np.random.normal(100, 5, 100),
            'Volume': np.random.normal(1000000, 200000, 100)
        }, index=dates)
    
    def test_momentum_trading_strategy(self):
        # Test with valid parameters
        signals = momentum_trading_strategy(self.df, 5, 20)
        
        # Check if required columns are present
        self.assertIn('price', signals.columns)
        self.assertIn('short_mavg', signals.columns)
        self.assertIn('long_mavg', signals.columns)
        self.assertIn('signal', signals.columns)
        self.assertIn('positions', signals.columns)
        
        # Check shapes
        self.assertEqual(len(signals), len(self.df))
    
    def test_compute_returns(self):
        # First generate signals
        signals = momentum_trading_strategy(self.df, 5, 20)
        
        # Then compute returns
        final_return, cum_returns = compute_returns(signals)
        
        # Check types
        self.assertIsInstance(final_return, float)
        self.assertIsInstance(cum_returns, pd.Series)
        
        # Check lengths
        self.assertEqual(len(cum_returns), len(signals))

if __name__ == '__main__':
    unittest.main()