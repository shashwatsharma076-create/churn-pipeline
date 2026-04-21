"""
Unified Churn Pipeline - Combines ML predictions with AI agent analysis.
"""
import time
from datetime import datetime
from typing import List, Optional
from pathlib import Path

import pandas as pd

from src.agents import SignalAgent, RiskAgent, ActionAgent, OrchestratorAgent
from src.models import ChurnPredictor
from src.schemas import RiskAssessment, RetentionAction
from src.utils import load_or_create_clean_data, save_results, print_summary


class UnifiedChurnPipeline:
    """Combines ML model predictions with AI agent analysis for comprehensive churn insights."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.predictor = None
        self.signal_agent = SignalAgent()
        self.risk_agent = RiskAgent()
        self.action_agent = ActionAgent()
        self.orchestrator = OrchestratorAgent()

    def load_model(self, model_path: Optional[Path] = None) -> ChurnPredictor:
        """Load the trained ML model."""
        path = model_path or self.model_path
        if path and path.exists():
            self.predictor = ChurnPredictor(path)
            print(f"Loaded ML model from: {path}")
        else:
            print("Warning: No ML model found. Running in AI-agent-only mode.")
        return self.predictor

    def predict_churn_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get ML-based churn probabilities for all customers."""
        if self.predictor is None:
            print("No ML model loaded. Skipping ML predictions.")
            return df

        results = df.copy()
        probabilities = []

        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_results = self.predictor.predict_batch(batch)
            probs = batch_results["churn_probability"]
            if hasattr(probs, 'tolist'):
                probs = probs.tolist()
            probabilities.extend(probs)

        results["ml_churn_probability"] = probabilities
        return results

    def analyze_high_risk_customers(
        self,
        df: pd.DataFrame,
        use_llm: bool = False,
        risk_threshold: float = 0.5
    ) -> List[dict]:
        """Use AI agents to analyze high-risk customers in depth."""
        high_risk = df[df["ml_churn_probability"] >= risk_threshold].copy()
        print(f"\nAnalyzing {len(high_risk)} high-risk customers with AI agents...")

        analyses = []
        for idx, row in high_risk.iterrows():
            customer_analysis = self._analyze_single_customer(row, use_llm)
            analyses.append(customer_analysis)

            if len(analyses) % 50 == 0:
                print(f"  Analyzed {len(analyses)} customers...")

        return analyses

    def _analyze_single_customer(self, row: pd.Series, use_llm: bool) -> dict:
        """Analyze a single customer using AI agents."""
        customer_data = row.to_dict()
        customer_id = customer_data.get("CustomerID", 0)

        signals = self.signal_agent.run(row)
        risk_assessment = self.risk_agent.run(customer_id, signals, customer_data)

        if use_llm and risk_assessment.signal_count > 0:
            try:
                risk_assessment.recommendation = self.risk_agent.get_risk_explanation(risk_assessment)
            except Exception:
                pass

        actions = self.action_agent.run(risk_assessment)

        return {
            "customer_id": customer_id,
            "ml_churn_probability": customer_data.get("ml_churn_probability", 0),
            "agent_risk_level": risk_assessment.risk_level,
            "agent_risk_score": risk_assessment.risk_score,
            "signals": [
                {"name": s.name, "severity": s.severity, "description": s.description}
                for s in signals
            ],
            "signal_count": len(signals),
            "recommendation": risk_assessment.recommendation,
            "retention_actions": [
                {
                    "type": a.action_type,
                    "priority": a.priority,
                    "description": a.description,
                    "expected_impact": a.expected_impact,
                }
                for a in actions
            ],
        }

    def run(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None,
        use_llm: bool = False,
        risk_threshold: float = 0.5
    ) -> dict:
        """Run the complete unified pipeline."""
        start_time = time.time()

        if sample_size:
            df = df.head(sample_size).copy()

        print("=" * 60)
        print("UNIFIED CHURN ANALYSIS PIPELINE")
        print("=" * 60)

        df_with_predictions = self.predict_churn_probability(df)

        print(f"\nTotal customers: {len(df_with_predictions)}")
        print(f"ML model predictions: {'Yes' if self.predictor else 'No'}")
        print(f"Risk threshold: {risk_threshold:.0%}")

        ml_risk_dist = {
            "high_risk": (df_with_predictions["ml_churn_probability"] >= risk_threshold).sum(),
            "medium_risk": ((df_with_predictions["ml_churn_probability"] >= 0.3) & 
                           (df_with_predictions["ml_churn_probability"] < risk_threshold)).sum(),
            "low_risk": (df_with_predictions["ml_churn_probability"] < 0.3).sum(),
        }

        print(f"\nML Risk Distribution:")
        print(f"  High Risk (>= {risk_threshold:.0%}):   {ml_risk_dist['high_risk']}")
        print(f"  Medium Risk (30-50%):    {ml_risk_dist['medium_risk']}")
        print(f"  Low Risk (< 30%):        {ml_risk_dist['low_risk']}")

        if self.predictor:
            high_risk_analyses = self.analyze_high_risk_customers(
                df_with_predictions, use_llm, risk_threshold
            )
        else:
            high_risk_analyses = []

        processing_time = time.time() - start_time

        result = {
            "total_customers": len(df_with_predictions),
            "ml_risk_distribution": ml_risk_dist,
            "high_risk_analyses": high_risk_analyses,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.now().isoformat(),
            "has_ml_predictions": self.predictor is not None,
        }

        self._print_summary(result)

        return result

    def _print_summary(self, result: dict) -> None:
        """Print a summary of the unified pipeline results."""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total Customers Analyzed: {result['total_customers']}")
        print(f"High-Risk Customers (AI Analysis): {len(result['high_risk_analyses'])}")
        print(f"Processing Time: {result['processing_time_seconds']:.2f}s")

        if result['high_risk_analyses']:
            print("\nTop 5 Highest Risk Customers:")
            sorted_customers = sorted(
                result['high_risk_analyses'],
                key=lambda x: (x['agent_risk_score'], x['ml_churn_probability']),
                reverse=True
            )[:5]

            for i, customer in enumerate(sorted_customers, 1):
                print(f"\n{i}. Customer #{customer['customer_id']}")
                print(f"   ML Probability: {customer['ml_churn_probability']:.1%}")
                print(f"   Agent Risk: {customer['agent_risk_level'].upper()} ({customer['agent_risk_score']}/100)")
                print(f"   Signals: {customer['signal_count']} - {[s['name'] for s in customer['signals']]}")

        print("=" * 60)

    def save_unified_results(self, result: dict, output_path: Path) -> None:
        """Save unified pipeline results to JSON."""
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")


def main():
    """Run the unified pipeline."""
    import argparse
    from src.config import PATHS

    parser = argparse.ArgumentParser(description="Unified Churn Analysis Pipeline")
    parser.add_argument("--sample", type=int, default=None, help="Number of customers to process")
    parser.add_argument("--llm", action="store_true", help="Enable LLM enhancement")
    parser.add_argument("--threshold", type=float, default=0.5, help="Risk threshold (0-1)")
    parser.add_argument("--model", type=str, default=None, help="Path to ML model")

    args = parser.parse_args()

    df = load_or_create_clean_data()
    pipeline = UnifiedChurnPipeline()

    if args.model:
        pipeline.load_model(Path(args.model))
    else:
        default_model = PATHS["models"] / "best_model.pkl"
        if default_model.exists():
            pipeline.load_model(default_model)

    result = pipeline.run(
        df,
        sample_size=args.sample,
        use_llm=args.llm,
        risk_threshold=args.threshold
    )

    output_path = PATHS["outputs"] / f"unified_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    pipeline.save_unified_results(result, output_path)


if __name__ == "__main__":
    main()
