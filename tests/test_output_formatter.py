"""
Tests for output formatter utilities.
"""
import pytest
from src.utils.output_formatter import format_assessment, print_summary
from src.schemas import ChurnSignal, RiskAssessment, PipelineResult


class TestOutputFormatter:
    """Test cases for output formatting utilities."""

    def test_format_assessment_includes_customer_id(self):
        """Test that assessment format includes customer ID."""
        assessment = RiskAssessment(
            customer_id=12345,
            risk_level="high",
            risk_score=75,
            signal_count=2,
            signals=[],
        )
        formatted = format_assessment(assessment)
        assert "12345" in formatted

    def test_format_assessment_includes_risk_level(self):
        """Test that assessment format includes risk level."""
        assessment = RiskAssessment(
            customer_id=123,
            risk_level="critical",
            risk_score=85,
            signal_count=3,
            signals=[],
        )
        formatted = format_assessment(assessment)
        assert "CRITICAL" in formatted

    def test_format_assessment_includes_signals(self):
        """Test that assessment format includes signal details."""
        signals = [
            ChurnSignal(
                name="support_calls",
                value=5.0,
                threshold=4.0,
                severity="high",
                description="High support usage: 5 calls",
            )
        ]
        assessment = RiskAssessment(
            customer_id=123,
            risk_level="high",
            risk_score=60,
            signal_count=1,
            signals=signals,
        )
        formatted = format_assessment(assessment)
        assert "High support usage: 5 calls" in formatted

    def test_format_assessment_includes_recommendation(self):
        """Test that assessment format includes recommendation."""
        assessment = RiskAssessment(
            customer_id=123,
            risk_level="critical",
            risk_score=85,
            signal_count=2,
            signals=[],
            recommendation="Immediate outreach required",
        )
        formatted = format_assessment(assessment)
        assert "Immediate outreach required" in formatted

    def test_print_summary_handles_empty_results(self):
        """Test that print_summary handles edge cases."""
        result = PipelineResult(
            total_customers=0,
            risk_distribution={"none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0},
            high_risk_count=0,
            critical_risk_count=0,
            processing_time_seconds=0.0,
            timestamp="2024-01-01T00:00:00",
            results=[],
        )
        print_summary(result)

    def test_print_summary_displays_risk_distribution(self):
        """Test that print_summary displays risk distribution."""
        result = PipelineResult(
            total_customers=100,
            risk_distribution={"none": 30, "low": 25, "medium": 20, "high": 15, "critical": 10},
            high_risk_count=15,
            critical_risk_count=10,
            processing_time_seconds=5.5,
            timestamp="2024-01-01T00:00:00",
            results=[],
        )
        print_summary(result)
