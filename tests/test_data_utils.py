import unittest
from datetime import datetime
from src.utils import (
    get_current_date,
    download_stock_data,
    calculate_returns_statistics
)

class TestDataUtils(unittest.TestCase):
    def test_get_current_date(self):
        current_date = get_current_date()
        self.assertIsInstance(current_date, datetime)

    def test_download_stock_data(self):
        df = download_stock_data("AAPL", "2024-01-01", "2024-01-31")
        self.assertIsNotNone(df)
        self.assertIn('Close', df.columns)

    def test_calculate_returns_statistics(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        stats = calculate_returns_statistics(returns)
        self.assertIn('sharpe_ratio', stats)
        self.assertIn('max_drawdown', stats)