"""
Orchestrator Agent - Coordinates all other agents.
"""
from typing import List, Optional
import pandas as pd
import time
from datetime import datetime

from .base import BaseAgent
from .signal_agent import SignalAgent
from .risk_agent import RiskAgent
from .action_agent import ActionAgent
from src.schemas import RiskAssessment, PipelineResult


class OrchestratorAgent(BaseAgent):
    """Main orchestrator that coordinates the entire pipeline."""

    def __init__(self):
        super().__init__()
        self.signal_agent = SignalAgent()
        self.risk_agent = RiskAgent()
        self.action_agent = ActionAgent()

    def run(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None,
        use_llm_enhancement: bool = False,
        use_ml: bool = False,
        ml_model=None
    ) -> PipelineResult:
        """Run the full churn analysis pipeline."""
        start_time = time.time()

        if sample_size:
            df = df.head(sample_size)

        ml_probs = None
        if use_ml and ml_model is not None:
            print("Loading ML predictions...")
            ml_probs = ml_model.predict_batch(df)["churn_probability"].tolist()

        print(f"Processing {len(df)} customers...")
        assessments = []
        risk_distribution = {"none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}

        for idx, row in df.iterrows():
            cust_id = row.get("CustomerID", idx)

            signals = self.signal_agent.run(row)
            assessment = self.risk_agent.run(cust_id, signals, row.to_dict())

            if use_ml and ml_probs is not None:
                ml_prob = ml_probs[idx] if idx < len(ml_probs) else 0.5
                agent_prob = assessment.churn_probability or 0.0
                combined = (float(ml_prob) + float(agent_prob)) / 2
                assessment.churn_probability = round(float(combined), 3)

            if use_llm_enhancement and assessment.signal_count > 0:
                try:
                    assessment.recommendation = self.risk_agent.get_risk_explanation(assessment)
                except Exception:
                    pass

            assessments.append(assessment)
            risk_distribution[assessment.risk_level] += 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} customers...")

        processing_time = time.time() - start_time

        return PipelineResult(
            total_customers=len(df),
            risk_distribution=risk_distribution,
            high_risk_count=risk_distribution["high"],
            critical_risk_count=risk_distribution["critical"],
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat(),
            results=assessments
        )

    def run_single_customer(self, customer_data: dict, use_llm: bool = True) -> dict:
        """Process a single customer through the pipeline."""
        signals = self.signal_agent.run(pd.Series(customer_data))
        assessment = self.risk_agent.run(
            customer_data.get("CustomerID", 0),
            signals,
            customer_data
        )

        if use_llm:
            assessment.recommendation = self.risk_agent.get_risk_explanation(assessment)

        actions = self.action_agent.run(assessment)

        return {
            "assessment": assessment,
            "signals": signals,
            "actions": actions
        }

    def generate_summary_report(self, result: PipelineResult) -> str:
        """Generate a summary report of the pipeline results."""
        prompt = f"""Generate a summary report for customer churn analysis:

        Total Customers Analyzed: {result.total_customers}
        Risk Distribution:
        - Critical: {result.risk_distribution['critical']}
        - High: {result.risk_distribution['high']}
        - Medium: {result.risk_distribution['medium']}
        - Low: {result.risk_distribution['low']}
        - None: {result.risk_distribution['none']}

        Processing Time: {result.processing_time_seconds:.2f} seconds

        Provide key insights and recommended next steps."""

        return self.call_llm(prompt)
