"""
Tests for risk assessment agent.
"""
import pytest
from src.agents.risk_agent import RiskAgent
from src.schemas import ChurnSignal


class TestRiskAgent:
    """Test cases for RiskAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = RiskAgent()

    def test_no_risk_with_no_signals(self):
        """Test that no signals means no risk."""
        score, level = self.agent.calculate_risk_score([])
        assert score == 0.0
        assert level == "none"

    def test_low_risk_with_single_signal(self):
        """Test low risk with single signal."""
        signals = [
            ChurnSignal(
                name="support_calls",
                value=5.0,
                threshold=4.0,
                severity="high",
                description="High support usage",
            )
        ]
        score, level = self.agent.calculate_risk_score(signals)
        assert level == "low"

    def test_medium_risk_with_two_signals(self):
        """Test medium risk with two signals."""
        signals = [
            ChurnSignal(
                name="support_calls",
                value=5.0,
                threshold=4.0,
                severity="high",
                description="High support usage",
            ),
            ChurnSignal(
                name="payment_delay",
                value=20.0,
                threshold=17.0,
                severity="high",
                description="Payment delayed",
            ),
        ]
        score, level = self.agent.calculate_risk_score(signals)
        assert level == "medium"

    def test_high_risk_with_multiple_signals(self):
        """Test high risk with multiple signals."""
        signals = [
            ChurnSignal(
                name="support_calls",
                value=5.0,
                threshold=4.0,
                severity="high",
                description="High support usage",
            ),
            ChurnSignal(
                name="payment_delay",
                value=20.0,
                threshold=17.0,
                severity="high",
                description="Payment delayed",
            ),
            ChurnSignal(
                name="total_spend",
                value=200.0,
                threshold=300.0,
                severity="critical",
                description="Very low spend",
            ),
        ]
        score, level = self.agent.calculate_risk_score(signals)
        assert level == "high"

    def test_critical_risk_with_many_signals(self):
        """Test critical risk with many critical signals."""
        signals = [
            ChurnSignal(
                name="support_calls",
                value=8.0,
                threshold=4.0,
                severity="critical",
                description="Critical support usage",
            ),
            ChurnSignal(
                name="payment_delay",
                value=30.0,
                threshold=17.0,
                severity="critical",
                description="Critical payment delay",
            ),
            ChurnSignal(
                name="total_spend",
                value=100.0,
                threshold=300.0,
                severity="critical",
                description="Critical low spend",
            ),
            ChurnSignal(
                name="last_interaction",
                value=28.0,
                threshold=15.0,
                severity="critical",
                description="Critical disengagement",
            ),
        ]
        score, level = self.agent.calculate_risk_score(signals)
        assert level == "critical"
        assert score >= 75

    def test_assessment_includes_contract_adjustment(self):
        """Test that annual contract reduces churn probability."""
        signals = [
            ChurnSignal(
                name="support_calls",
                value=5.0,
                threshold=4.0,
                severity="high",
                description="High support usage",
            ),
        ]
        customer_data = {"Contract Length": "Annual"}
        assessment = self.agent.run(123, signals, customer_data)
        assert assessment.churn_probability < 0.15

    def test_assessment_returns_correct_structure(self):
        """Test that assessment has all required fields."""
        signals = [
            ChurnSignal(
                name="support_calls",
                value=5.0,
                threshold=4.0,
                severity="high",
                description="High support usage",
            ),
        ]
        assessment = self.agent.run(123, signals)
        assert assessment.customer_id == 123
        assert assessment.risk_level is not None
        assert assessment.risk_score > 0
        assert assessment.signal_count == 1
        assert len(assessment.signals) == 1
