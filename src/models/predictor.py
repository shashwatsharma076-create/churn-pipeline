"""
Churn Predictor - Load and use trained models for predictions.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Union


class ChurnPredictor:
    """Load trained model and make predictions on new data."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.optimal_threshold = 0.5
        self.load_model()

    def load_model(self) -> None:
        """Load the saved model and preprocessors."""
        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data["model"]
        self.scaler = self.model_data["scaler"]
        self.feature_columns = self.model_data["feature_columns"]
        self.optimal_threshold = self.model_data.get("optimal_threshold", 0.5)
        print(f"Loaded model: {self.model_data['model_name']} (threshold={self.optimal_threshold:.2f})")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features matching trainer's feature engineering."""
        df = df.copy()

        tenure_safe = df["Tenure"].replace(0, 1)

        df["spend_per_month"] = df["Total Spend"] / tenure_safe
        df["support_call_rate"] = df["Support Calls"] / tenure_safe
        df["payment_delay_ratio"] = df["Payment Delay"] / tenure_safe

        df["engagement_score"] = (
            df["Usage Frequency"] / 30.0
            + (30 - df["Last Interaction"]) / 30.0
            + (10 - df["Support Calls"]) / 10.0
        ) / 3.0

        df["spend_x_calls"] = df["Total Spend"] * df["Support Calls"]
        df["delay_x_calls"] = df["Payment Delay"] * df["Support Calls"]
        df["low_spend_high_calls"] = (
            (df["Total Spend"] < 500) & (df["Support Calls"] >= 5)
        ).astype(int)
        df["high_delay_low_usage"] = (
            (df["Payment Delay"] >= 15) & (df["Usage Frequency"] <= 10)
        ).astype(int)

        df["is_monthly_contract"] = (df["Contract Length"] == "Monthly").astype(int)
        df["is_premium"] = (df["Subscription Type"] == "Premium").astype(int)
        df["is_basic"] = (df["Subscription Type"] == "Basic").astype(int)

        df["age_group"] = pd.cut(
            df["Age"],
            bins=[0, 25, 35, 50, 100],
            labels=[0, 1, 2, 3],
        ).astype(int)

        df["tenure_group"] = pd.cut(
            df["Tenure"],
            bins=[0, 10, 20, 40, 100],
            labels=[0, 1, 2, 3],
        ).astype(int)

        df["is_high_engagement"] = (df["Usage Frequency"] > 20).astype(int)

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a DataFrame for prediction."""
        df = df.copy()

        df = df.drop(columns=["CustomerID", "Churn"], errors="ignore")

        df = self.engineer_features(df)

        categorical_cols = ["Gender", "Subscription Type", "Contract Length"]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_columns]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def predict(self, customer_data: Union[dict, pd.DataFrame]) -> dict:
        """Make prediction on customer data."""
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()

        X = self.preprocess(df)

        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]

        churn_prob = float(probability[1])
        tuned_prediction = 1 if churn_prob >= self.optimal_threshold else 0

        return {
            "churn_prediction": int(prediction),
            "churn_probability": churn_prob,
            "retention_probability": float(probability[0]),
            "churn_prediction_tuned": tuned_prediction,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on a batch of customers."""
        X = self.preprocess(df)

        results = df.copy()
        results["churn_prediction"] = self.model.predict(X)
        proba = self.model.predict_proba(X)
        results["churn_probability"] = proba[:, 1]
        results["churn_prediction_tuned"] = (
            proba[:, 1] >= self.optimal_threshold
        ).astype(int)

        return results
