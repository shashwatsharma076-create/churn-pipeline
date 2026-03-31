"""
Output formatting utilities.
"""
import json
from typing import List
from pathlib import Path
from src.schemas import RiskAssessment, PipelineResult
from src.config import PATHS


def format_assessment(assessment: RiskAssessment) -> str:
    """Format a single assessment for display."""
    output = []
    output.append(f"Customer ID: {assessment.customer_id}")
    output.append(f"Risk Level: {assessment.risk_level.upper()}")
    output.append(f"Risk Score: {assessment.risk_score}/100")
    output.append(f"Churn Probability: {assessment.churn_probability or 'N/A'}")
    output.append(f"Signals: {assessment.signal_count}")

    for signal in assessment.signals:
        output.append(f"  - {signal.description}")

    if assessment.recommendation:
        output.append(f"Recommendation: {assessment.recommendation}")

    return "\n".join(output)


def save_results(result: PipelineResult, output_path: Path = None) -> None:
    """Save pipeline results to JSON file."""
    output_path = output_path or PATHS["outputs"] / f"churn_analysis_{result.timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": {
            "total_customers": result.total_customers,
            "risk_distribution": result.risk_distribution,
            "high_risk_count": result.high_risk_count,
            "critical_risk_count": result.critical_risk_count,
            "processing_time_seconds": result.processing_time_seconds,
            "timestamp": result.timestamp,
        },
        "assessments": [
            {
                "customer_id": a.customer_id,
                "risk_level": a.risk_level,
                "risk_score": a.risk_score,
                "churn_probability": a.churn_probability,
                "signal_count": a.signal_count,
                "signals": [
                    {
                        "name": s.name,
                        "value": s.value,
                        "severity": s.severity,
                        "description": s.description,
                    }
                    for s in a.signals
                ],
                "recommendation": a.recommendation,
            }
            for a in result.results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_path}")


def print_summary(result: PipelineResult) -> None:
    """Print a summary of the pipeline results."""
    print("\n" + "=" * 50)
    print("CHURN ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total Customers: {result.total_customers}")
    print(f"Processing Time: {result.processing_time_seconds:.2f}s")
    print(f"\nRisk Distribution:")
    for level, count in result.risk_distribution.items():
        pct = (count / result.total_customers * 100) if result.total_customers > 0 else 0
        print(f"  {level.upper():>10}: {count:>6} ({pct:>5.1f}%)")
    print(f"\nAt-Risk Customers:")
    print(f"  High Risk:     {result.high_risk_count}")
    print(f"  Critical Risk: {result.critical_risk_count}")
    print("=" * 50 + "\n")


def print_top_risk_customers(result: PipelineResult, n: int = 10) -> None:
    """Print the top N highest risk customers."""
    sorted_results = sorted(
        result.results,
        key=lambda x: (x.risk_score, x.signal_count),
        reverse=True
    )[:n]

    print(f"\nTop {n} Highest Risk Customers:")
    print("-" * 50)
    for i, assessment in enumerate(sorted_results, 1):
        print(f"{i}. Customer #{assessment.customer_id}")
        print(f"   Level: {assessment.risk_level.upper()} | Score: {assessment.risk_score}")
        print(f"   Signals: {', '.join([s.name for s in assessment.signals])}")
        print()
