"""
Tests for signal detection agent.
"""
import pytest
import pandas as pd
from src.agents.signal_agent import SignalAgent
from src.config import THRESHOLDS


class TestSignalAgent:
    """Test cases for SignalAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = SignalAgent()

    def test_no_signals_for_healthy_customer(self, sample_customer_data):
        """Test that healthy customer has no critical signals."""
        row = pd.Series(sample_customer_data)
        signals = self.agent.run(row)
        assert len(signals) == 0

    def test_detects_high_support_calls(self):
        """Test detection of high support call count."""
        data = {
            "Age": 35,
            "Gender": "Male",
            "Tenure": 24,
            "Usage Frequency": 20,
            "Support Calls": 5,
            "Payment Delay": 5,
            "Subscription Type": "Premium",
            "Contract Length": "Annual",
            "Total Spend": 800,
            "Last Interaction": 7,
        }
        row = pd.Series(data)
        signals = self.agent.run(row)
        signal_names = [s.name for s in signals]
        assert "support_calls" in signal_names

    def test_detects_payment_delay(self):
        """Test detection of payment delay."""
        data = {
            "Age": 35,
            "Gender": "Male",
            "Tenure": 24,
            "Usage Frequency": 20,
            "Support Calls": 2,
            "Payment Delay": 20,
            "Subscription Type": "Premium",
            "Contract Length": "Annual",
            "Total Spend": 800,
            "Last Interaction": 7,
        }
        row = pd.Series(data)
        signals = self.agent.run(row)
        signal_names = [s.name for s in signals]
        assert "payment_delay" in signal_names

    def test_detects_low_spend(self):
        """Test detection of low total spend."""
        data = {
            "Age": 35,
            "Gender": "Male",
            "Tenure": 24,
            "Usage Frequency": 20,
            "Support Calls": 2,
            "Payment Delay": 5,
            "Subscription Type": "Basic",
            "Contract Length": "Monthly",
            "Total Spend": 250,
            "Last Interaction": 7,
        }
        row = pd.Series(data)
        signals = self.agent.run(row)
        signal_names = [s.name for s in signals]
        assert "total_spend" in signal_names

    def test_detects_low_interaction(self):
        """Test detection of long time since last interaction."""
        data = {
            "Age": 35,
            "Gender": "Male",
            "Tenure": 24,
            "Usage Frequency": 20,
            "Support Calls": 2,
            "Payment Delay": 5,
            "Subscription Type": "Premium",
            "Contract Length": "Annual",
            "Total Spend": 800,
            "Last Interaction": 28,
        }
        row = pd.Series(data)
        signals = self.agent.run(row)
        signal_names = [s.name for s in signals]
        assert "last_interaction" in signal_names

    def test_detects_multiple_signals(self, sample_customer_churn):
        """Test detection of multiple signals for at-risk customer."""
        row = pd.Series(sample_customer_churn)
        signals = self.agent.run(row)
        assert len(signals) >= 3

    def test_signal_severity_levels(self):
        """Test that signal severity is correctly assigned."""
        data = {
            "Age": 35,
            "Gender": "Male",
            "Tenure": 24,
            "Usage Frequency": 5,
            "Support Calls": 8,
            "Payment Delay": 30,
            "Subscription Type": "Premium",
            "Contract Length": "Monthly",
            "Total Spend": 200,
            "Last Interaction": 28,
        }
        row = pd.Series(data)
        signals = self.agent.run(row)
        critical_signals = [s for s in signals if s.severity == "critical"]
        high_signals = [s for s in signals if s.severity == "high"]
        assert len(critical_signals) > 0

    def test_signal_descriptions(self):
        """Test that signal descriptions are properly formatted."""
        data = {
            "Age": 35,
            "Gender": "Male",
            "Tenure": 24,
            "Usage Frequency": 5,
            "Support Calls": 8,
            "Payment Delay": 30,
            "Subscription Type": "Premium",
            "Contract Length": "Monthly",
            "Total Spend": 200,
            "Last Interaction": 28,
        }
        row = pd.Series(data)
        signals = self.agent.run(row)
        for signal in signals:
            assert signal.description is not None
            assert len(signal.description) > 0
