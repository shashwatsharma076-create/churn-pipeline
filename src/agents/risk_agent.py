"""
Risk Assessment Agent - Evaluates overall churn risk level.
"""
from typing import List
import pandas as pd
from .base import BaseAgent
from src.schemas import ChurnSignal, RiskAssessment


SYSTEM_PROMPT = """You are a customer churn risk analyst. 
Given customer signals and risk factors, assess the churn probability 
and provide actionable insights."""


class RiskAgent(BaseAgent):
    """Agent responsible for assessing customer churn risk."""

    def calculate_risk_score(self, signals: List[ChurnSignal]) -> tuple:
        """Calculate risk score and level from signals."""
        if not signals:
            return 0.0, "none"

        score = 0.0
        for signal in signals:
            if signal.severity == "critical":
                score += 25
            elif signal.severity == "high":
                score += 15
            else:
                score += 5

        if score >= 75:
            level = "critical"
        elif score >= 50:
            level = "high"
        elif score >= 25:
            level = "medium"
        elif score >= 10:
            level = "low"
        else:
            level = "none"

        return min(score, 100), level

    def run(self, customer_id: int, signals: List[ChurnSignal], customer_data: dict = None) -> RiskAssessment:
        """Assess risk for a single customer."""
        risk_score, risk_level = self.calculate_risk_score(signals)

        churn_prob = risk_score / 100.0
        if customer_data and customer_data.get("Contract Length") == "Annual":
            churn_prob *= 0.7
        elif customer_data and customer_data.get("Contract Length") == "Quarterly":
            churn_prob *= 0.85

        return RiskAssessment(
            customer_id=customer_id,
            risk_level=risk_level,
            risk_score=risk_score,
            signal_count=len(signals),
            signals=signals,
            churn_probability=round(churn_prob, 3),
        )

    def get_risk_explanation(self, assessment: RiskAssessment) -> str:
        """Generate human-readable risk explanation using LLM."""
        signal_summary = "\n".join([
            f"- {s.description} ({s.severity})" for s in assessment.signals
        ]) if assessment.signals else "No significant signals detected."

        prompt = f"""Customer ID: {assessment.customer_id}
        Risk Level: {assessment.risk_level.upper()} (Score: {assessment.risk_score})
        Signals Detected:
        {signal_summary}

        Provide a brief explanation of this customer's churn risk in 2-3 sentences.
        Focus on the most critical factors and overall likelihood to churn."""
        return self.call_llm(prompt, SYSTEM_PROMPT)

    def batch_assess(self, df: pd.DataFrame, signal_agents_results: List[List[ChurnSignal]]) -> List[RiskAssessment]:
        """Process multiple customers at once."""
        assessments = []
        for idx, signals in enumerate(signal_agents_results):
            cust_id = df.iloc[idx].get("CustomerID", idx)
            customer_data = df.iloc[idx].to_dict()
            assessment = self.run(cust_id, signals, customer_data)
            assessments.append(assessment)
        return assessments
