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
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        self.load_model()

    def load_model(self) -> None:
        """Load the saved model and preprocessors."""
        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data["model"]
        self.label_encoders = self.model_data["label_encoders"]
        self.scaler = self.model_data["scaler"]
        self.feature_names = self.model_data["feature_names"]
        print(f"Loaded model: {self.model_data['model_name']}")

    def preprocess_single(self, customer_data: dict) -> np.ndarray:
        """Preprocess a single customer record."""
        df = pd.DataFrame([customer_data])

        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df[self.feature_names].values

    def predict(self, customer_data: Union[dict, pd.DataFrame]) -> dict:
        """Make prediction on customer data."""
        if isinstance(customer_data, dict):
            X = self.preprocess_single(customer_data)
        else:
            df = customer_data.copy()
            for col, le in self.label_encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            X = df[self.feature_names].values

        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]

        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability[1]),
            "retention_probability": float(probability[0]),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on a batch of customers."""
        results = df.copy()

        for col, le in self.label_encoders.items():
            if col in df.columns:
                results[col] = le.transform(df[col])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = self.scaler.transform(results[numeric_cols])
        X = pd.DataFrame(X, columns=numeric_cols)[self.feature_names]

        results["churn_prediction"] = self.model.predict(X)
        results["churn_probability"] = self.model.predict_proba(X)[:, 1]

        return results
