"""
Unit tests for the momentum trading strategy implementation.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategies.momentum_trading_strategy import (
    calculate_moving_average,
    momentum_trading_strategy,
    compute_returns,
    run_strategy_analysis
)

class TestMomentumStrategy(unittest.TestCase):
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
            'Close': [100, 102, 104, 103, 105, 107, 108, 107, 106, 105]
        }, index=pd.date_range(start='2024-01-01', periods=10))

    def test_calculate_moving_average(self):
        """Test moving average calculation."""
        # Test with known values
        data = pd.Series([1, 2, 3, 4, 5])
        ma = calculate_moving_average(data, window=3)
        
        # First two values should be rolling means
        self.assertEqual(ma[0], 1)  # Only first value available
        self.assertEqual(ma[1], 1.5)  # Mean of first two values
        self.assertEqual(ma[2], 2)  # Mean of first three values
        
        # Test with actual price data
        ma = calculate_moving_average(self.known_data['Close'], window=5)
        self.assertEqual(len(ma), len(self.known_data))
        self.assertTrue(all(isinstance(x, (int, float)) for x in ma))

    def test_momentum_trading_strategy(self):
        """Test trading strategy signal generation."""
        # Test with default parameters
        signals = momentum_trading_strategy(
            self.test_data,
            short_window=5,
            long_window=20
        )
        
        # Check required columns exist
        required_columns = ['price', 'short_mavg', 'long_mavg', 'signal', 'positions']
        for col in required_columns:
            self.assertIn(col, signals.columns)
        
        # Check signal values are valid (0 or 1)
        unique_signals = signals['signal'].unique()
        self.assertTrue(all(signal in [0.0, 1.0] for signal in unique_signals))
        
        # Check positions are valid (-1, 0, or 1)
        positions = signals['positions'].dropna()
        self.assertTrue(all(pos in [-1.0, 0.0, 1.0] for pos in positions))
        
        # Test with known data
        signals = momentum_trading_strategy(
            self.known_data,
            short_window=3,
            long_window=5
        )
        self.assertEqual(len(signals), len(self.known_data))

    def test_compute_returns(self):
        """Test return calculation functionality."""
        # Generate signals first
        signals = momentum_trading_strategy(self.test_data)
        
        # Calculate returns
        final_return, cum_returns = compute_returns(signals)
        
        # Check types
        self.assertIsInstance(final_return, float)
        self.assertIsInstance(cum_returns, pd.Series)
        
        # Check lengths
        self.assertEqual(len(cum_returns), len(signals))
        
        # Check cumulative returns start at 1.0
        self.assertAlmostEqual(cum_returns.iloc[0], 1.0)

    def test_run_strategy_analysis(self):
        """Test the complete strategy analysis function."""
        results = run_strategy_analysis(
            self.test_data,
            short_window=5,
            long_window=20,
            symbol='TEST',
            save_results=False
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('signals', results)
        self.assertIn('final_return', results)
        self.assertIn('cumulative_returns', results)
        self.assertIn('parameters', results)
        
        # Check parameters
        params = results['parameters']
        self.assertEqual(params['short_window'], 5)
        self.assertEqual(params['long_window'], 20)
        self.assertEqual(params['symbol'], 'TEST')

    def test_edge_cases(self):
        """Test strategy behavior with edge cases."""
        # Test with very short data
        short_data = self.test_data.head(3)
        signals = momentum_trading_strategy(short_data)
        self.assertTrue(signals['positions'].isna().all())  # Should be all NaN
        
        # Test with constant prices
        constant_data = pd.DataFrame({
            'Close': [100] * 20
        }, index=pd.date_range(start='2024-01-01', periods=20))
        
        signals = momentum_trading_strategy(constant_data)
        self.assertTrue((signals['short_mavg'] == signals['long_mavg']).all())
        
        # Test with missing data
        data_with_nan = self.test_data.copy()
        data_with_nan.loc[data_with_nan.index[5], 'Close'] = np.nan
        signals = momentum_trading_strategy(data_with_nan)
        self.assertTrue(np.isnan(signals.loc[signals.index[5], 'price']))

    def test_parameter_validation(self):
        """Test strategy behavior with different parameters."""
        # Test with different window sizes
        window_pairs = [(5, 20), (10, 30), (20, 50)]
        
        for short_window, long_window in window_pairs:
            signals = momentum_trading_strategy(
                self.test_data,
                short_window=short_window,
                long_window=long_window
            )
            self.assertIsInstance(signals, pd.DataFrame)
            self.assertEqual(len(signals), len(self.test_data))

def run_tests():
    unittest.main()

if __name__ == '__main__':
    run_tests()