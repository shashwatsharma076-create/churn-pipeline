"""
One-File Runner for Customer Churn Pipeline
Run everything in one go: Prepare Data -> Train Model -> Run Analysis
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Create directories
from src.config import ensure_directories
ensure_directories()

print("=" * 70)
print("CUSTOMER CHURN PIPELINE - FULL RUN")
print("=" * 70)

# STEP 1: LOAD AND PREPARE DATA
print("\n[STEP 1] Loading and Preparing Data...")
print("-" * 50)

from src.utils import load_or_create_clean_data, get_data_summary

df = load_or_create_clean_data()
summary = get_data_summary(df)

print("Loaded:", summary['total_rows'], "customers")
print("Churn Rate:", summary['churn_rate'], "%")
print("Churned:", int(summary['churn_count']))
print("Retained:", summary['total_rows'] - int(summary['churn_count']))


# STEP 2: TRAIN ML MODEL
print("\n[STEP 2] Training ML Model...")
print("-" * 50)

df_sample = df.sample(10000, random_state=42)
sample_path = Path('data/processed/train_sample.csv')
df_sample.to_csv(sample_path, index=False)

from src.models import ChurnModelTrainer
from src.config import PATHS

trainer = ChurnModelTrainer(data_path=sample_path, models_dir=PATHS['models'])
trainer.preprocess()
trainer.train_models()
trainer.select_best_model()

model_path = trainer.save_model('best_model.pkl')
print("Model saved:", model_path)

os.remove(sample_path)


# STEP 3: RUN AI AGENT ANALYSIS
print("\n[STEP 3] Running AI Agent Analysis...")
print("-" * 50)

df_analysis = df.head(5000)

from src.agents import OrchestratorAgent

orchestrator = OrchestratorAgent()
result = orchestrator.run(df_analysis, sample_size=5000, use_llm_enhancement=False)

print("Analyzed:", result.total_customers, "customers")
print("Processing time:", round(result.processing_time_seconds, 2), "seconds")
print("\nRisk Distribution:")
for level, count in result.risk_distribution.items():
    pct = (count / result.total_customers) * 100
    print(f"  {level.upper():>10}: {count:>5} ({pct:.1f}%)")

print("\nAt-Risk Customers:")
print("  High Risk:", result.high_risk_count)
print("  Critical:", result.critical_risk_count)


# STEP 4: TOP HIGH-RISK CUSTOMERS
print("\n[TOP 10] HIGH-RISK CUSTOMERS:")
print("-" * 50)

top_risk = sorted(result.results, key=lambda x: (x.risk_score, x.signal_count), reverse=True)[:10]

for i, assessment in enumerate(top_risk, 1):
    print(f"\n{i}. Customer ID: {assessment.customer_id}")
    print(f"   Risk Level: {assessment.risk_level.upper()} (Score: {assessment.risk_score}/100)")
    print(f"   Churn Prob: {assessment.churn_probability}")
    print(f"   Signal Count: {assessment.signal_count}")
    if assessment.signals:
        signal_names = [s.name for s in assessment.signals[:3]]
        print(f"   Signal Types: {', '.join(signal_names)}")


# STEP 5: SINGLE CUSTOMER PREDICTION WITH ML
print("\n[STEP 5] Single Customer Prediction with ML Model...")
print("-" * 50)

from src.models import ChurnPredictor

# Load model
predictor = ChurnPredictor(PATHS['models'] / 'best_model.pkl')

# Test prediction
test_customer = {
    'Age': 35,
    'Gender': 'Male',
    'Tenure': 12,
    'Usage Frequency': 15,
    'Support Calls': 5,
    'Payment Delay': 20,
    'Subscription Type': 'Premium',
    'Contract Length': 'Annual',
    'Total Spend': 500,
    'Last Interaction': 10
}

prediction = predictor.predict(test_customer)
print("Test Customer Prediction:")
print(f"  Churn Prediction: {prediction['churn_prediction']}")
print(f"  Churn Probability: {prediction['churn_probability']:.2%}")
print(f"  Retention Probability: {prediction['retention_probability']:.2%}")


# DONE
print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print("Summary:")
print("  - Data:", summary['total_rows'], "customers")
print("  - Model:", trainer.best_model_name)
print("  - F1 Score:", round(trainer.results[trainer.best_model_name]['f1'], 4))
print("  - Analyzed:", result.total_customers, "customers")
print("  - High Risk:", result.high_risk_count + result.critical_risk_count)
print("\nRun: python run_pipeline.py")