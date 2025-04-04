"""
Unit tests for the Stochastic Oscillator trading strategy.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.stochastic_strategy import (
    calculate_stochastic,
    stochastic_trading_strategy,
    run_stochastic_analysis
)

class TestStochasticStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across multiple tests."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        cls.test_data = pd.DataFrame({
            'High': [100 + i + np.random.random() for i in range(100)],
            'Low': [100 + i - np.random.random() for i in range(100)],
            'Close': [100 + i + np.random.random() - 0.5 for i in range(100)]
        }, index=dates)
        
        # Create specific test case for known values
        cls.known_data = pd.DataFrame({
            'High':  [110, 112, 108, 109, 110, 112, 111, 110, 109, 108],
            'Low':   [108, 109, 106, 107, 108, 109, 108, 107, 106, 105],
            'Close': [109, 110, 107, 108, 109, 110, 109, 108, 107, 106]
        })

    def test_calculate_stochastic(self):
        """Test Stochastic Oscillator calculation."""
        # Test with default parameters
        k, d = calculate_stochastic(self.test_data)
        
        self.assertIsInstance(k, pd.Series)
        self.assertIsInstance(d, pd.Series)
        
        # Check values are within valid range (0-100)
        self.assertTrue((k.dropna() >= 0).all() and (k.dropna() <= 100).all())
        self.assertTrue((d.dropna() >= 0).all() and (d.dropna() <= 100).all())
        
        # Test with known values
        k, d = calculate_stochastic(
            self.known_data,
            k_period=5,
            d_period=3,
            smooth_k=1
        )
        
        # Calculate expected %K for the 5th point manually
        # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        expected_k = (109 - 106) / (112 - 106) * 100  # For 5th point
        self.assertAlmostEqual(k[4], expected_k, places=2)

    def test_stochastic_trading_strategy(self):
        """Test trading strategy signal generation."""
        signals = stochastic_trading_strategy(
            self.test_data,
            k_period=14,
            d_period=3,
            smooth_k=3,
            oversold=20,
            overbought=80
        )
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertTrue('signal' in signals.columns)
        self.assertTrue('positions' in signals.columns)
        
        # Check signal values are valid (0 or 1)
        unique_signals = signals['signal'].unique()
        self.assertTrue(all(signal in [0, 1] for signal in unique_signals))
        
        # Check positions are valid (-1, 0, or 1)
        unique_positions = signals['positions'].dropna().unique()
        self.assertTrue(all(pos in [-1, 0, 1] for pos in unique_positions))

    def test_run_stochastic_analysis(self):
        """Test the complete analysis function."""
        results = run_stochastic_analysis(
            self.test_data,
            k_period=14,
            d_period=3,
            smooth_k=3,
            oversold=20,
            overbought=80,
            symbol='TEST',
            save_results=False
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertTrue('signals' in results)
        self.assertTrue('final_return' in results)
        self.assertTrue('cumulative_returns' in results)
        self.assertTrue('parameters' in results)
        
        # Check parameters are correctly stored
        params = results['parameters']
        self.assertEqual(params['k_period'], 14)
        self.assertEqual(params['d_period'], 3)
        self.assertEqual(params['smooth_k'], 3)
        self.assertEqual(params['oversold'], 20)
        self.assertEqual(params['overbought'], 80)

    def test_edge_cases(self):
        """Test strategy behavior with edge cases."""
        # Test with very short data
        short_data = self.test_data.head(5)
        signals = stochastic_trading_strategy(short_data)
        self.assertTrue(signals['percent_k'].isna().all())  # Should be all NaN
        
        # Test with constant prices
        constant_data = pd.DataFrame({
            'High': [100] * 20,
            'Low': [100] * 20,
            'Close': [100] * 20
        })
        k, d = calculate_stochastic(constant_data)
        # When high and low are equal, %K should be 100
        self.assertTrue(np.isnan(k.iloc[0]))  # First points should be NaN
        
        # Test with missing data
        data_with_nan = self.test_data.copy()
        data_with_nan.loc[data_with_nan.index[5], 'Close'] = np.nan
        signals = stochastic_trading_strategy(data_with_nan)
        self.assertTrue(np.isnan(signals.loc[signals.index[5], 'percent_k']))

    def test_parameter_validation(self):
        """Test strategy behavior with different parameters."""
        # Test with different lookback periods
        for k_period in [5, 9, 14, 21]:
            signals = stochastic_trading_strategy(
                self.test_data,
                k_period=k_period
            )
            self.assertIsInstance(signals, pd.DataFrame)
        
        # Test with different smoothing values
        for smooth_k in [1, 2, 3, 5]:
            signals = stochastic_trading_strategy(
                self.test_data,
                smooth_k=smooth_k
            )
            self.assertIsInstance(signals, pd.DataFrame)
        
        # Test with different oversold/overbought levels
        signals = stochastic_trading_strategy(
            self.test_data,
            oversold=30,
            overbought=70
        )
        self.assertIsInstance(signals, pd.DataFrame)

def run_tests():
    unittest.main()

if __name__ == '__main__':
    run_tests()