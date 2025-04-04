"""
Unit tests for the Bollinger Bands trading strategy implementation.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategies.bollinger_bands_strategy import (
    calculate_bollinger_bands,
    bollinger_bands_strategy,
    calculate_atr,
    run_bollinger_bands_analysis
)

class TestBollingerBandsStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across multiple tests."""
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        cls.test_data = pd.DataFrame({
            'Open': np.random.normal(100, 5, 100),
            'High': np.random.normal(105, 5, 100),
            'Low': np.random.normal(95, 5, 100),
            'Close': np.random.normal(100, 5, 100),
            'Volume': np.random.normal(1000000, 200000, 100)
        }, index=dates)

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        middle, upper, lower = calculate_bollinger_bands(
            self.test_data['Close'],
            window=20,
            num_std=2.0
        )
        
        self.assertIsInstance(middle, pd.Series)
        self.assertIsInstance(upper, pd.Series)
        self.assertIsInstance(lower, pd.Series)
        
        # Check bands relationship
        self.assertTrue(all(upper >= middle))
        self.assertTrue(all(lower <= middle))

    def test_bollinger_bands_strategy(self):
        """Test trading strategy signal generation."""
        signals = bollinger_bands_strategy(
            self.test_data,
            window=20,
            num_std=2.0
        )
        
        required_columns = [
            'price', 'middle_band', 'upper_band', 'lower_band',
            'bandwidth', 'percent_b', 'signal', 'positions'
        ]
        
        for col in required_columns:
            self.assertIn(col, signals.columns)
            
        # Check signal values
        self.assertTrue(all(signals['signal'].isin([0.0, 1.0])))

    def test_calculate_atr(self):
        """Test ATR calculation."""
        atr = calculate_atr(self.test_data, period=14)
        
        self.assertIsInstance(atr, pd.Series)
        self.assertTrue(all(atr >= 0))  # ATR should always be positive

    def test_run_bollinger_bands_analysis(self):
        """Test the complete strategy analysis function."""
        results = run_bollinger_bands_analysis(
            self.test_data,
            window=20,
            num_std=2.0,
            use_atr=True,
            save_results=False
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('signals', results)
        self.assertIn('final_return', results)
        self.assertIn('cumulative_returns', results)
        self.assertIn('parameters', results)

    def test_edge_cases(self):
        """Test strategy behavior with edge cases."""
        # Test with very short data
        short_data = self.test_data.head(3)
        signals = bollinger_bands_strategy(short_data)
        self.assertTrue(signals['positions'].isna().all())
        
        # Test with constant prices
        constant_data = pd.DataFrame({
            'Close': [100] * 30,
            'High': [105] * 30,
            'Low': [95] * 30
        }, index=pd.date_range(start='2024-01-01', periods=30))
        
        signals = bollinger_bands_strategy(constant_data)
        self.assertEqual(signals['bandwidth'].iloc[-1], 0.0)

def run_tests():
    unittest.main()

if __name__ == '__main__':
    run_tests()