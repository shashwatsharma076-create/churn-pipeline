import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ── Signal thresholds from Phase 1 analysis ──────────────────
THRESHOLDS = {
    'high_support_calls':    4,
    'high_payment_delay':    17,
    'high_last_interaction': 15,
    'low_total_spend':       600,
    'low_usage_frequency':   14,
}

def get_risk_profile(row):
    """
    Takes one customer row and returns a dictionary
    describing their risk signals and overall risk level.
    
    This is the bridge between your structured data
    and the prompt we'll send to the AI.
    """
    signals = []
    
    # Check each threshold — same logic Agent 1 will use
    if row['Support Calls'] >= THRESHOLDS['high_support_calls']:
        signals.append({
            'name': 'high_support_calls',
            'value': int(row['Support Calls']),
            'description': f"called support {int(row['Support Calls'])} times"
        })
    
    if row['Payment Delay'] >= THRESHOLDS['high_payment_delay']:
        signals.append({
            'name': 'high_payment_delay', 
            'value': int(row['Payment Delay']),
            'description': f"payment delayed {int(row['Payment Delay'])} days"
        })
    
    if row['Last Interaction'] >= THRESHOLDS['high_last_interaction']:
        signals.append({
            'name': 'high_last_interaction',
            'value': int(row['Last Interaction']),
            'description': f"last contacted {int(row['Last Interaction'])} days ago"
        })
    
    if row['Total Spend'] <= THRESHOLDS['low_total_spend']:
        signals.append({
            'name': 'low_total_spend',
            'value': float(row['Total Spend']),
            'description': f"low spender at ${row['Total Spend']:.0f}"
        })
    
    if row['Usage Frequency'] <= THRESHOLDS['low_usage_frequency']:
        signals.append({
            'name': 'low_usage_frequency',
            'value': int(row['Usage Frequency']),
            'description': f"only uses service {int(row['Usage Frequency'])} days/month"
        })
    
    # Calculate risk level from signal count
    signal_count = len(signals)
    if signal_count >= 3:
        risk_level = "critical"
    elif signal_count == 2:
        risk_level = "high"
    elif signal_count == 1:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        'signals': signals,
        'signal_count': signal_count,
        'risk_level': risk_level,
        'churn_label': int(row['Churn']),
        'contract_length': row['Contract Length'],
        'subscription_type': row['Subscription Type'],
        'tenure': int(row['Tenure']),
    }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv('../data/processed/churn_clean.csv')
    
    # Test on first 3 rows
    for i in range(3):
        row = df.iloc[i]
        profile = get_risk_profile(row)
        print(f"\nCustomer {i+1}:")
        print(f"  Risk level:   {profile['risk_level']}")
        print(f"  Signal count: {profile['signal_count']}")
        print(f"  Signals:      {[s['name'] for s in profile['signals']]}")
        print(f"  Churn label:  {profile['churn_label']}")