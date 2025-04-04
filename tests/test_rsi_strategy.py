import unittest
import pandas as pd
import numpy as np
from src.strategies import rsi_trading_strategy, calculate_rsi

class TestRSIStrategy(unittest.TestCase):
    def setUp(self):
        # Setup test data
        self.test_data = pd.DataFrame({
            'Close': [10, 12, 11, 13, 14, 13, 12, 11, 13, 15],
            'Date': pd.date_range(start='2024-01-01', periods=10)
        }).set_index('Date')

    def test_calculate_rsi(self):
        rsi = calculate_rsi(self.test_data['Close'], period=14)
        self.assertIsInstance(rsi, pd.Series)
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))

    def test_rsi_trading_strategy(self):
        results = rsi_trading_strategy(self.test_data, period=14)
        self.assertIn('signals', results)
        self.assertIn('returns', results)

if __name__ == '__main__':
    unittest.main()
