#!/usr/bin/env python3
"""
Customer Churn Pipeline - Main Entry Point

A comprehensive CLI tool for customer churn analysis combining ML predictions
with AI agent-based risk assessment and retention recommendations.
"""
import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from src.config import PATHS, ensure_directories
from src.utils import load_or_create_clean_data, save_results, print_summary, print_top_risk_customers
from src.agents import OrchestratorAgent
from src.pipeline import UnifiedChurnPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Ensure all required directories exist."""
    ensure_directories()
    logger.info("Project directories initialized")


def run_agent_analysis(sample_size: int = None, use_llm: bool = False):
    """Run the AI agent-based churn analysis."""
    print("\n" + "=" * 60)
    print("CHURN SIGNAL DETECTION & RISK ASSESSMENT")
    print("=" * 60 + "\n")

    try:
        df = load_or_create_clean_data()
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.info("Please ensure data/raw/customer_churn.csv exists")
        return

    orchestrator = OrchestratorAgent()

    print(f"Loaded {len(df)} customers from dataset")
    if sample_size:
        print(f"Analyzing sample of {sample_size} customers...\n")
    else:
        print("Analyzing all customers...\n")

    result = orchestrator.run(df, sample_size=sample_size, use_llm_enhancement=use_llm)

    print_summary(result)

    output_file = PATHS["outputs"] / f"churn_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(result, output_file)

    print_top_risk_customers(result, n=10)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return result


def run_unified_analysis(sample_size: int = None, use_llm: bool = False, 
                         risk_threshold: float = 0.5, model_path: str = None):
    """Run the unified pipeline combining ML + AI agents."""
    print("\n" + "=" * 60)
    print("UNIFIED CHURN ANALYSIS (ML + AI AGENTS)")
    print("=" * 60 + "\n")

    try:
        df = load_or_create_clean_data()
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.info("Please ensure data/raw/customer_churn.csv exists")
        return

    pipeline = UnifiedChurnPipeline()

    if model_path:
        pipeline.load_model(Path(model_path))
    else:
        default_model = PATHS["models"] / "best_model.pkl"
        if default_model.exists():
            pipeline.load_model(default_model)

    result = pipeline.run(
        df,
        sample_size=sample_size,
        use_llm=use_llm,
        risk_threshold=risk_threshold
    )

    output_file = PATHS["outputs"] / f"unified_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    pipeline.save_unified_results(result, output_file)

    print("\n" + "=" * 60)
    print("Unified Analysis complete!")
    print("=" * 60)

    return result


def run_single_customer_analysis(customer_data: dict):
    """Analyze a single customer."""
    print("\n" + "=" * 60)
    print("SINGLE CUSTOMER ANALYSIS")
    print("=" * 60 + "\n")

    orchestrator = OrchestratorAgent()

    result = orchestrator.run_single_customer(customer_data, use_llm=True)

    print(f"\nRisk Level: {result['assessment'].risk_level.upper()}")
    print(f"Risk Score: {result['assessment'].risk_score}/100")
    print(f"Churn Probability: {result['assessment'].churn_probability:.1%}")
    print(f"Signals Detected: {result['assessment'].signal_count}")

    if result["signals"]:
        print("\nDetected Signals:")
        for signal in result["signals"]:
            print(f"  - [{signal.severity.upper()}] {signal.description}")

    if result["actions"]:
        print("\nRecommended Actions:")
        for action in result["actions"]:
            print(f"  - [{action.priority.upper()}] {action.description}")

    return result


def show_data_summary():
    """Display summary of the dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60 + "\n")

    try:
        df = load_or_create_clean_data()
        print(f"Total Customers: {len(df):,}")
        print(f"Total Features: {len(df.columns)}")
        print(f"\nChurn Statistics:")
        print(f"  Churn Rate: {df['Churn'].mean():.1%}")
        print(f"  Churned Customers: {df['Churn'].sum():,}")
        print(f"  Retained Customers: {(df['Churn'] == 0).sum():,}")

        print(f"\nFeatures: {', '.join(df.columns)}")

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Customer Churn Pipeline - ML + AI Agent Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py analyze                    Run AI agent analysis
  python main.py analyze --sample 100       Analyze 100 customers
  python main.py analyze --llm              Use LLM enhancement
  python main.py unified                    Run unified ML + AI pipeline
  python main.py unified --sample 1000       Unified with 1000 customers
  python main.py unified --threshold 0.3    Set risk threshold to 30%
  python main.py summary                    Show dataset summary
  python main.py single --customer '{"Age": 35, ...}'  Analyze single customer
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    analyze_parser = subparsers.add_parser("analyze", help="Run AI agent-based churn analysis")
    analyze_parser.add_argument(
        "--sample", type=int, default=None, help="Number of customers to analyze"
    )
    analyze_parser.add_argument(
        "--llm", action="store_true", help="Use LLM for enhanced analysis"
    )

    unified_parser = subparsers.add_parser("unified", help="Run unified ML + AI pipeline")
    unified_parser.add_argument(
        "--sample", type=int, default=None, help="Number of customers to analyze"
    )
    unified_parser.add_argument(
        "--llm", action="store_true", help="Use LLM for enhanced analysis"
    )
    unified_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Risk threshold (0-1)"
    )
    unified_parser.add_argument(
        "--model", type=str, default=None, help="Path to ML model"
    )

    summary_parser = subparsers.add_parser("summary", help="Show dataset summary")

    single_parser = subparsers.add_parser("single", help="Analyze a single customer")
    single_parser.add_argument(
        "--customer", type=str, required=True, help="Customer data as JSON string"
    )

    args = parser.parse_args()

    setup_directories()

    if args.command == "analyze":
        run_agent_analysis(sample_size=args.sample, use_llm=args.llm)
    elif args.command == "unified":
        run_unified_analysis(
            sample_size=args.sample,
            use_llm=args.llm,
            risk_threshold=args.threshold,
            model_path=args.model
        )
    elif args.command == "summary":
        show_data_summary()
    elif args.command == "single":
        try:
            customer_data = json.loads(args.customer)
            run_single_customer_analysis(customer_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
