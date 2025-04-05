"""
Unit tests for the Fair Value Gap (FVG) trading strategy implementation.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.strategies.fvg_strategy import (
    FVG,
    identify_fvg,
    check_fvg_fill,
    fvg_trading_strategy,
    run_fvg_analysis
)

class TestFVGStrategy(unittest.TestCase):
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
        
        # Create specific test case with known FVG patterns
        cls.known_data = pd.DataFrame({
            'Open':  [100, 101, 105, 104, 103],
            'High':  [102, 103, 106, 105, 104],
            'Low':   [99,  100, 104, 103, 102],
            'Close': [101, 102, 105, 104, 103]
        }, index=pd.date_range(start='2024-01-01', periods=5))

    def test_fvg_dataclass(self):
        """Test FVG dataclass creation and attributes."""
        timestamp = datetime.now()
        fvg = FVG(
            timestamp=timestamp,
            high=100.0,
            low=95.0,
            type='bullish'
        )
        
        self.assertEqual(fvg.timestamp, timestamp)
        self.assertEqual(fvg.high, 100.0)
        self.assertEqual(fvg.low, 95.0)
        self.assertEqual(fvg.type, 'bullish')
        self.assertFalse(fvg.filled)
        self.assertIsNone(fvg.fill_time)

    def test_identify_fvg(self):
        """Test FVG identification logic."""
        bullish_fvgs, bearish_fvgs = identify_fvg(
            self.known_data,
            min_gap_size=0.001
        )
        
        # Check if FVGs are identified correctly
        self.assertTrue(len(bullish_fvgs) > 0 or len(bearish_fvgs) > 0)
        
        # Check FVG attributes
        if bullish_fvgs:
            fvg = bullish_fvgs[0]
            self.assertIsInstance(fvg, FVG)
            self.assertEqual(fvg.type, 'bullish')
            self.assertFalse(fvg.filled)
        
        if bearish_fvgs:
            fvg = bearish_fvgs[0]
            self.assertIsInstance(fvg, FVG)
            self.assertEqual(fvg.type, 'bearish')
            self.assertFalse(fvg.filled)

    def test_check_fvg_fill(self):
        """Test FVG fill detection."""
        # Test bullish FVG
        bullish_fvg = FVG(
            timestamp=datetime.now(),
            high=100.0,
            low=95.0,
            type='bullish'
        )
        
        # Should not fill
        self.assertFalse(check_fvg_fill(bullish_fvg, 93.0, 94.0))
        # Should fill
        self.assertTrue(check_fvg_fill(bullish_fvg, 94.0, 101.0))
        
        # Test bearish FVG
        bearish_fvg = FVG(
            timestamp=datetime.now(),
            high=105.0,
            low=100.0,
            type='bearish'
        )
        
        # Should not fill
        self.assertFalse(check_fvg_fill(bearish_fvg, 106.0, 107.0))
        # Should fill
        self.assertTrue(check_fvg_fill(bearish_fvg, 99.0, 106.0))

    def test_fvg_trading_strategy(self):
        """Test FVG trading strategy signal generation."""
        signals = fvg_trading_strategy(
            self.test_data,
            timeframe='4h',
            min_gap_size=0.001,
            max_active_fvgs=3
        )
        
        required_columns = [
            'price',
            'signal',
            'stop_loss',
            'take_profit',
            'active_fvgs'
        ]
        
        # Check if required columns are present
        for col in required_columns:
            self.assertIn(col, signals.columns)
        
        # Check signal values
        self.assertTrue(all(signals['signal'].isin([0.0, 1.0, -1.0])))
        
        # Check active FVGs count
        self.assertTrue(all(signals['active_fvgs'] >= 0))
        self.assertTrue(all(signals['active_fvgs'] <= 3))

    def test_run_fvg_analysis(self):
        """Test complete FVG analysis run."""
        results = run_fvg_analysis(
            self.test_data,
            timeframe='4h',
            min_gap_size=0.001,
            max_active_fvgs=3,
            symbol='TEST',
            save_results=False
        )
        
        # Check if results contain required keys
        required_keys = [
            'signals',
            'final_return',
            'cumulative_returns',
            'bullish_fvgs',
            'bearish_fvgs',
            'parameters'
        ]
        
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check parameters
        self.assertEqual(results['parameters']['timeframe'], '4h')
        self.assertEqual(results['parameters']['min_gap_size'], 0.001)
        self.assertEqual(results['parameters']['max_active_fvgs'], 3)
        self.assertEqual(results['parameters']['symbol'], 'TEST')

    def test_timeframe_parameter_adjustments(self):
        """Test parameter adjustments for different timeframes."""
        # Test 15m timeframe
        signals_15m = fvg_trading_strategy(
            self.test_data,
            timeframe='15m',
            min_gap_size=0.001,
            max_active_fvgs=3,
            stop_loss_pct=0.02,
            take_profit_pct=0.03
        )
        
        # Test 4h timeframe
        signals_4h = fvg_trading_strategy(
            self.test_data,
            timeframe='4h',
            min_gap_size=0.001,
            max_active_fvgs=3,
            stop_loss_pct=0.02,
            take_profit_pct=0.03
        )
        
        # 15m should have more active FVGs on average
        avg_active_15m = signals_15m['active_fvgs'].mean()
        avg_active_4h = signals_4h['active_fvgs'].mean()
        self.assertGreaterEqual(avg_active_15m, avg_active_4h)

if __name__ == '__main__':
    unittest.main()