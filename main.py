#!/usr/bin/env python3
"""
Customer Churn Pipeline - Main Entry Point

A comprehensive CLI tool for customer churn analysis, ML model training,
and risk assessment using AI agents.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PATHS, ensure_directories
from src.utils import load_or_create_clean_data, save_results, print_summary, print_top_risk_customers
from src.agents import OrchestratorAgent

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

    output_file = PATHS["outputs"] / f"churn_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(result, output_file)

    print_top_risk_customers(result, n=10)

    print("\n" + "=" * 60)
    print("Analysis complete!")
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
        print(f"  Churned Customers: {int(df['Churn'].sum()):,}")
        print(f"  Retained Customers: {int((df['Churn'] == 0).sum()):,}")

        print(f"\nFeatures: {', '.join(df.columns)}")
        print(f"\nData Types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Customer Churn Pipeline - Analyze and predict customer churn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py analyze                    Run full churn analysis
  python main.py analyze --sample 100       Analyze 100 customers
  python main.py analyze --llm              Use LLM enhancement
  python main.py summary                    Show dataset summary
  python main.py single --customer '{"Age": 35, ...}'  Analyze single customer
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    analyze_parser = subparsers.add_parser("analyze", help="Run churn signal analysis")
    analyze_parser.add_argument(
        "--sample", type=int, default=None, help="Number of customers to analyze"
    )
    analyze_parser.add_argument(
        "--llm", action="store_true", help="Use LLM for enhanced analysis"
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
    elif args.command == "summary":
        show_data_summary()
    elif args.command == "single":
        import json
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
