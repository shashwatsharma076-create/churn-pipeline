"""
Tests for action agent.
"""
import pytest
from unittest.mock import patch
from src.agents.action_agent import ActionAgent
from src.schemas import ChurnSignal, RiskAssessment


class TestActionAgent:
    """Test cases for ActionAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ActionAgent()

    def test_critical_risk_gets_urgent_actions(self):
        """Test that critical risk gets urgent actions."""
        signals = [
            ChurnSignal(
                name="support_calls",
                value=8.0,
                threshold=4.0,
                severity="critical",
                description="Critical support usage",
            ),
        ]
        assessment = RiskAssessment(
            customer_id=123,
            risk_level="critical",
            risk_score=85,
            signal_count=3,
            signals=signals,
            churn_probability=0.85,
        )
        with patch.object(self.agent, '_enhance_with_llm', return_value="Test description"):
            actions = self.agent.run(assessment)
        assert len(actions) >= 2
        priorities = [a.priority for a in actions]
        assert "critical" in priorities or "high" in priorities

    def test_high_risk_gets_urgent_actions(self):
        """Test that high risk gets urgent actions."""
        signals = [
            ChurnSignal(
                name="payment_delay",
                value=20.0,
                threshold=17.0,
                severity="high",
                description="Payment delayed",
            ),
        ]
        assessment = RiskAssessment(
            customer_id=123,
            risk_level="high",
            risk_score=60,
            signal_count=2,
            signals=signals,
            churn_probability=0.6,
        )
        with patch.object(self.agent, '_enhance_with_llm', return_value="Test description"):
            actions = self.agent.run(assessment)
        assert len(actions) >= 2

    def test_medium_risk_gets_preventive_actions(self):
        """Test that medium risk gets preventive actions."""
        signals = [
            ChurnSignal(
                name="total_spend",
                value=550.0,
                threshold=600.0,
                severity="high",
                description="Low spend",
            ),
        ]
        assessment = RiskAssessment(
            customer_id=123,
            risk_level="medium",
            risk_score=35,
            signal_count=1,
            signals=signals,
            churn_probability=0.35,
        )
        with patch.object(self.agent, '_enhance_with_llm', return_value="Test description"):
            actions = self.agent.run(assessment)
        action_types = [a.action_type for a in actions]
        assert "check_in" in action_types or "feature_highlight" in action_types

    def test_low_risk_gets_loyalty_actions(self):
        """Test that low risk gets loyalty actions."""
        assessment = RiskAssessment(
            customer_id=123,
            risk_level="low",
            risk_score=10,
            signal_count=0,
            signals=[],
            churn_probability=0.1,
        )
        with patch.object(self.agent, '_enhance_with_llm', return_value="Test description"):
            actions = self.agent.run(assessment)
        action_types = [a.action_type for a in actions]
        assert "appreciation" in action_types or "referral" in action_types

    def test_payment_delay_triggers_payment_plan_action(self):
        """Test that payment delay triggers payment plan action."""
        signals = [
            ChurnSignal(
                name="payment_delay",
                value=20.0,
                threshold=17.0,
                severity="high",
                description="Payment delayed",
            ),
        ]
        assessment = RiskAssessment(
            customer_id=123,
            risk_level="high",
            risk_score=60,
            signal_count=1,
            signals=signals,
            churn_probability=0.6,
        )
        with patch.object(self.agent, '_enhance_with_llm', return_value="Test description"):
            actions = self.agent.run(assessment)
        action_types = [a.action_type for a in actions]
        assert "payment_plan" in action_types

    def test_support_calls_triggers_proactive_support_action(self):
        """Test that high support calls trigger proactive support action."""
        signals = [
            ChurnSignal(
                name="support_calls",
                value=6.0,
                threshold=4.0,
                severity="high",
                description="High support usage",
            ),
        ]
        assessment = RiskAssessment(
            customer_id=123,
            risk_level="high",
            risk_score=60,
            signal_count=1,
            signals=signals,
            churn_probability=0.6,
        )
        with patch.object(self.agent, '_enhance_with_llm', return_value="Test description"):
            actions = self.agent.run(assessment)
        action_types = [a.action_type for a in actions]
        assert "proactive_support" in action_types

    def test_action_has_required_fields(self):
        """Test that actions have all required fields."""
        assessment = RiskAssessment(
            customer_id=123,
            risk_level="critical",
            risk_score=85,
            signal_count=3,
            signals=[],
            churn_probability=0.85,
        )
        with patch.object(self.agent, '_enhance_with_llm', return_value="Test description"):
            actions = self.agent.run(assessment)
        for action in actions:
            assert action.action_type is not None
            assert action.priority is not None
            assert action.description is not None
            assert action.target_customer_segment is not None
            assert action.expected_impact is not None
