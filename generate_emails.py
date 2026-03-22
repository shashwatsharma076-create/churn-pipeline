import pandas as pd
from config import PATHS
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

df = pd.read_csv(PATHS['clean_data'])
print(f"Loaded {len(df)} customers\n")

results = []

for i in range(5):
    row = df.iloc[i]

    signals = []
    if row['Support Calls'] >= 4:
        signals.append(f"called support {int(row['Support Calls'])} times")
    if row['Payment Delay'] >= 17:
        signals.append(f"payment delayed {int(row['Payment Delay'])} days")
    if row['Last Interaction'] >= 15:
        signals.append(f"not contacted in {int(row['Last Interaction'])} days")

    mood = "frustrated" if signals else "neutral or happy"

    prompt = f"""Write a short realistic email (3-4 sentences) from a customer 
who feels {mood}. Their situation: {', '.join(signals) if signals else 'no major issues'}.
Sound like a real person. No subject line. Just the email body."""

    response = client.chat.completions.create(
        model="anthropic/claude-3-haiku",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )

    email = response.choices[0].message.content.strip()

    result = {
        "customer_index": i,
        "churn_label":    int(row['Churn']),
        "signal_count":   len(signals),
        "signals":        signals,
        "email":          email,
    }

    results.append(result)

    print(f"Customer {i+1} | Churn: {int(row['Churn'])} | Signals: {len(signals)}")
    print(email)
    print("-" * 50)

# Save to file
output_path = os.path.join(PATHS['simulated_data'], 'emails.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} emails to {output_path}")