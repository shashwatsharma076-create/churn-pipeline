"""
Signal Detection Agent - Identifies churn signals from customer data.
"""
from typing import List, Dict
import pandas as pd
from .base import BaseAgent
from src.config import THRESHOLDS
from src.schemas import ChurnSignal


SYSTEM_PROMPT = """You are a customer churn signal detection expert. 
Analyze customer data and identify behavioral patterns that indicate potential churn.
Focus on actionable signals that can inform retention strategies."""


class SignalAgent(BaseAgent):
    """Agent responsible for detecting churn signals from customer data."""

    def run(self, customer_row: pd.Series) -> List[ChurnSignal]:
        """Analyze a single customer and return detected signals."""
        signals = []
        cust_id = customer_row.get("CustomerID", 0)

        if customer_row["Support Calls"] >= THRESHOLDS["support_calls"]["critical"]:
            signals.append(ChurnSignal(
                name="support_calls",
                value=float(customer_row["Support Calls"]),
                threshold=THRESHOLDS["support_calls"]["critical"],
                severity="critical",
                description=f"Critical: Customer has called support {int(customer_row['Support Calls'])} times"
            ))
        elif customer_row["Support Calls"] >= THRESHOLDS["support_calls"]["high"]:
            signals.append(ChurnSignal(
                name="support_calls",
                value=float(customer_row["Support Calls"]),
                threshold=THRESHOLDS["support_calls"]["high"],
                severity="high",
                description=f"High support usage: {int(customer_row['Support Calls'])} calls"
            ))

        if customer_row["Payment Delay"] >= THRESHOLDS["payment_delay"]["critical"]:
            signals.append(ChurnSignal(
                name="payment_delay",
                value=float(customer_row["Payment Delay"]),
                threshold=THRESHOLDS["payment_delay"]["critical"],
                severity="critical",
                description=f"Critical payment delay: {int(customer_row['Payment Delay'])} days"
            ))
        elif customer_row["Payment Delay"] >= THRESHOLDS["payment_delay"]["high"]:
            signals.append(ChurnSignal(
                name="payment_delay",
                value=float(customer_row["Payment Delay"]),
                threshold=THRESHOLDS["payment_delay"]["high"],
                severity="high",
                description=f"Payment delayed {int(customer_row['Payment Delay'])} days"
            ))

        if customer_row["Last Interaction"] >= THRESHOLDS["last_interaction"]["critical"]:
            signals.append(ChurnSignal(
                name="last_interaction",
                value=float(customer_row["Last Interaction"]),
                threshold=THRESHOLDS["last_interaction"]["critical"],
                severity="critical",
                description=f"No contact for {int(customer_row['Last Interaction'])} days"
            ))
        elif customer_row["Last Interaction"] >= THRESHOLDS["last_interaction"]["high"]:
            signals.append(ChurnSignal(
                name="last_interaction",
                value=float(customer_row["Last Interaction"]),
                threshold=THRESHOLDS["last_interaction"]["high"],
                severity="high",
                description=f"Last contacted {int(customer_row['Last Interaction'])} days ago"
            ))

        if customer_row["Total Spend"] <= THRESHOLDS["total_spend"]["very_low"]:
            signals.append(ChurnSignal(
                name="total_spend",
                value=float(customer_row["Total Spend"]),
                threshold=THRESHOLDS["total_spend"]["very_low"],
                severity="critical",
                description=f"Very low spend: ${customer_row['Total Spend']:.0f}"
            ))
        elif customer_row["Total Spend"] <= THRESHOLDS["total_spend"]["low"]:
            signals.append(ChurnSignal(
                name="total_spend",
                value=float(customer_row["Total Spend"]),
                threshold=THRESHOLDS["total_spend"]["low"],
                severity="high",
                description=f"Low spender: ${customer_row['Total Spend']:.0f}"
            ))

        if customer_row["Usage Frequency"] <= THRESHOLDS["usage_frequency"]["very_low"]:
            signals.append(ChurnSignal(
                name="usage_frequency",
                value=float(customer_row["Usage Frequency"]),
                threshold=THRESHOLDS["usage_frequency"]["very_low"],
                severity="critical",
                description=f"Near-zero usage: {int(customer_row['Usage Frequency'])} days/month"
            ))
        elif customer_row["Usage Frequency"] <= THRESHOLDS["usage_frequency"]["low"]:
            signals.append(ChurnSignal(
                name="usage_frequency",
                value=float(customer_row["Usage Frequency"]),
                threshold=THRESHOLDS["usage_frequency"]["low"],
                severity="high",
                description=f"Low usage: {int(customer_row['Usage Frequency'])} days/month"
            ))

        if customer_row["Tenure"] <= THRESHOLDS["tenure"]["very_low"]:
            signals.append(ChurnSignal(
                name="tenure",
                value=float(customer_row["Tenure"]),
                threshold=THRESHOLDS["tenure"]["very_low"],
                severity="high",
                description=f"New customer: only {int(customer_row['Tenure'])} months"
            ))

        return signals

    def analyze_with_llm(self, customer_data: Dict) -> str:
        """Use LLM for deeper semantic analysis of customer behavior."""
        prompt = f"""Analyze this customer for churn risk factors:
        - Support calls: {customer_data.get('Support Calls', 'N/A')}
        - Payment delay: {customer_data.get('Payment Delay', 'N/A')} days
        - Last interaction: {customer_data.get('Last Interaction', 'N/A')} days ago
        - Total spend: ${customer_data.get('Total Spend', 'N/A')}
        - Usage frequency: {customer_data.get('Usage Frequency', 'N/A')} days/month
        - Tenure: {customer_data.get('Tenure', 'N/A')} months
        - Subscription: {customer_data.get('Subscription Type', 'N/A')}
        - Contract: {customer_data.get('Contract Length', 'N/A')}

        Provide a brief analysis of their churn risk and key concerns."""
        return self.call_llm(prompt, SYSTEM_PROMPT)
