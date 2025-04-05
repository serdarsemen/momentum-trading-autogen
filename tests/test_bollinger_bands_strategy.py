"""
Unit tests for the Bollinger Bands trading strategy implementation.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Tuple

from src.strategies.bollinger_bands_strategy import (
    calculate_bollinger_bands,
    bollinger_bands_strategy,
    calculate_atr,
    run_bollinger_bands_analysis,
    plot_bollinger_bands_strategy
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

        # Create specific test case with known values
        cls.known_data = pd.DataFrame({
            'Close': [100] * 10 + [110] * 10 + [90] * 10
        }, index=pd.date_range(start='2024-01-01', periods=30))

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        # Test with default parameters
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
        
        # Test with known values
        middle, upper, lower = calculate_bollinger_bands(
            self.known_data['Close'],
            window=10,
            num_std=2.0
        )
        
        # For constant price periods, bands should converge
        self.assertAlmostEqual(
            upper[9] - lower[9],
            0.0,
            places=6
        )
        
        # Test different window sizes
        for window in [10, 20, 50]:
            middle, upper, lower = calculate_bollinger_bands(
                self.test_data['Close'],
                window=window
            )
            self.assertEqual(len(middle), len(self.test_data))

    def test_calculate_atr(self):
        """Test ATR calculation."""
        atr = calculate_atr(self.test_data, period=14)
        
        self.assertIsInstance(atr, pd.Series)
        self.assertTrue(all(atr >= 0))  # ATR should always be positive
        
        # Test different periods
        for period in [7, 14, 21]:
            atr = calculate_atr(self.test_data, period=period)
            self.assertEqual(len(atr), len(self.test_data))

    def test_bollinger_bands_strategy(self):
        """Test trading strategy signal generation."""
        signals = bollinger_bands_strategy(
            self.test_data,
            window=20,
            num_std=2.0,
            use_atr=True
        )
        
        required_columns = [
            'price', 'middle_band', 'upper_band', 'lower_band',
            'bandwidth', 'percent_b', 'signal', 'positions'
        ]
        
        # Check if required columns are present
        for col in required_columns:
            self.assertIn(col, signals.columns)
            
        # Check signal values
        self.assertTrue(all(signals['signal'].isin([0.0, 1.0])))
        
        # Test with ATR
        signals_with_atr = bollinger_bands_strategy(
            self.test_data,
            use_atr=True,
            atr_period=14
        )
        self.assertIn('atr', signals_with_atr.columns)

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
        
        required_keys = [
            'signals',
            'final_return',
            'cumulative_returns',
            'parameters'
        ]
        
        # Check if results contain required keys
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check parameters
        self.assertEqual(results['parameters']['window'], 20)
        self.assertEqual(results['parameters']['num_std'], 2.0)
        self.assertEqual(results['parameters']['use_atr'], True)

    def test_plot_bollinger_bands_strategy(self):
        """Test strategy visualization."""
        signals = bollinger_bands_strategy(self.test_data)
        
        # Test plot generation without saving
        fig, axes = plot_bollinger_bands_strategy(
            signals,
            title='Test Plot'
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(axes)
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)

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
        
        # Test with missing values
        data_with_nan = self.test_data.copy()
        data_with_nan.loc[data_with_nan.index[5], 'Close'] = np.nan
        signals = bollinger_bands_strategy(data_with_nan)
        self.assertTrue(signals['signal'].notna().any())

    def test_parameter_validation(self):
        """Test strategy behavior with different parameters."""
        # Test different window sizes
        for window in [10, 20, 50]:
            signals = bollinger_bands_strategy(
                self.test_data,
                window=window
            )
            self.assertIsInstance(signals, pd.DataFrame)
        
        # Test different standard deviations
        for num_std in [1.5, 2.0, 2.5]:
            signals = bollinger_bands_strategy(
                self.test_data,
                num_std=num_std
            )
            self.assertIsInstance(signals, pd.DataFrame)
        
        # Test with and without ATR
        signals_no_atr = bollinger_bands_strategy(
            self.test_data,
            use_atr=False
        )
        self.assertNotIn('atr', signals_no_atr.columns)
        
        signals_with_atr = bollinger_bands_strategy(
            self.test_data,
            use_atr=True
        )
        self.assertIn('atr', signals_with_atr.columns)

def run_tests():
    unittest.main()

if __name__ == '__main__':
    run_tests()
