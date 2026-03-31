"""
Pytest configuration and fixtures.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing."""
    return {
        "Age": 35,
        "Gender": "Male",
        "Tenure": 24,
        "Usage Frequency": 20,
        "Support Calls": 2,
        "Payment Delay": 5,
        "Subscription Type": "Premium",
        "Contract Length": "Annual",
        "Total Spend": 800,
        "Last Interaction": 7,
    }


@pytest.fixture
def sample_customer_churn():
    """Sample customer data with churn indicator."""
    return {
        "Age": 45,
        "Gender": "Female",
        "Tenure": 12,
        "Usage Frequency": 8,
        "Support Calls": 6,
        "Payment Delay": 20,
        "Subscription Type": "Basic",
        "Contract Length": "Monthly",
        "Total Spend": 400,
        "Last Interaction": 25,
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    data = {
        "Age": [25, 35, 45, 55, 65],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Tenure": [6, 24, 12, 36, 48],
        "Usage Frequency": [25, 18, 10, 15, 20],
        "Support Calls": [1, 2, 5, 3, 1],
        "Payment Delay": [3, 8, 15, 12, 5],
        "Subscription Type": ["Basic", "Premium", "Standard", "Basic", "Premium"],
        "Contract Length": ["Monthly", "Annual", "Monthly", "Quarterly", "Annual"],
        "Total Spend": [300, 900, 450, 600, 950],
        "Last Interaction": [3, 10, 20, 15, 5],
        "Churn": [1, 0, 1, 1, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_path(tmp_path):
    """Create a temporary data path for testing."""
    return tmp_path / "test_data.csv"
