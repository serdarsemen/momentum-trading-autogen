import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies import macd_trading_strategy, calculate_macd

class TestMACDStrategy(unittest.TestCase):
    
    def setUp(self):
        # Create dummy data for testing
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'Close': np.random.normal(100, 5, 100)
        }, index=dates)
    
    def test_calculate_macd(self):
        # Test MACD calculation
        macd_line, signal_line, histogram = calculate_macd(self.df['Close'])
        
        # Check if all components are calculated
        self.assertEqual(len(macd_line), len(self.df))
        self.assertEqual(len(signal_line), len(self.df))
        self.assertEqual(len(histogram), len(self.df))
        
        # Check if histogram equals macd_line - signal_line
        np.testing.assert_array_almost_equal(
            histogram,
            macd_line - signal_line
        )
    
    def test_macd_trading_strategy(self):
        # Test strategy generation
        signals = macd_trading_strategy(self.df)
        
        # Check if required columns are present
        required_columns = ['price', 'macd_line', 'signal_line', 
                          'histogram', 'signal', 'positions']
        for col in required_columns:
            self.assertIn(col, signals.columns)
        
        # Check shapes
        self.assertEqual(len(signals), len(self.df))
        
        # Check if positions are valid (-1, 0, or 1)
        positions = signals['positions'].dropna()
        self.assertTrue(all(p in [-1.0, 0.0, 1.0] for p in positions))

if __name__ == '__main__':
    unittest.main()