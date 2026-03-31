"""
Tests for data utilities.
"""
import pytest
import pandas as pd
import os
from src.utils.data_loader import (
    clean_data,
    get_data_summary,
)


class TestDataLoader:
    """Test cases for data loading utilities."""

    def test_clean_data_removes_customer_id(self, sample_dataframe):
        """Test that CustomerID column is removed during cleaning."""
        df = sample_dataframe.copy()
        df["CustomerID"] = [1, 2, 3, 4, 5]
        cleaned = clean_data(df)
        assert "CustomerID" not in cleaned.columns

    def test_clean_data_handles_missing_values(self):
        """Test that missing values are handled."""
        df = pd.DataFrame(
            {
                "Age": [25, None, 45],
                "Gender": ["Male", "Female", "Male"],
                "Tenure": [6, 24, 12],
                "Usage Frequency": [25, 18, 10],
                "Support Calls": [1, 2, 5],
                "Payment Delay": [3, 8, 15],
                "Subscription Type": ["Basic", "Premium", "Standard"],
                "Contract Length": ["Monthly", "Annual", "Monthly"],
                "Total Spend": [300, 900, 450],
                "Last Interaction": [3, 10, 20],
                "Churn": [1, 0, 1],
            }
        )
        cleaned = clean_data(df)
        assert cleaned.isnull().sum().sum() == 0

    def test_get_data_summary_returns_correct_structure(self, sample_dataframe):
        """Test that data summary has all required fields."""
        summary = get_data_summary(sample_dataframe)
        assert "total_rows" in summary
        assert "churn_count" in summary
        assert "churn_rate" in summary
        assert "columns" in summary
        assert summary["total_rows"] == 5

    def test_get_data_summary_calculates_churn_rate(self, sample_dataframe):
        """Test that churn rate is calculated correctly."""
        summary = get_data_summary(sample_dataframe)
        expected_rate = (3 / 5) * 100
        assert summary["churn_rate"] == expected_rate

    def test_get_data_summary_counts_churn(self, sample_dataframe):
        """Test that churn count is correct."""
        summary = get_data_summary(sample_dataframe)
        assert summary["churn_count"] == 3
