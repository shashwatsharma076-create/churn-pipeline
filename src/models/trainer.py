"""
ML Model Trainer for Customer Churn Prediction.
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


class ChurnModelTrainer:
    """Train and evaluate ML models for churn prediction."""

    def __init__(self, data_path: Path, models_dir: Path):
        self.data_path = data_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None

    def load_data(self) -> pd.DataFrame:
        """Load and return the dataset."""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded data: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def preprocess(self) -> tuple:
        """Preprocess data and return train/test splits."""
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        categorical_cols = ["Gender", "Subscription Type", "Contract Length"]
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Churn rate (train): {y_train.mean():.2%}")
        print(f"Churn rate (test): {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def train_models(self) -> dict:
        """Train multiple models and return results."""
        if self.X_train is None:
            self.preprocess()

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
        }

        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)

            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)

            results[name] = {
                "model": model,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
            }

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  AUC-ROC: {auc:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        self.results = results
        return results

    def select_best_model(self) -> tuple:
        """Select and return the best model based on F1 score."""
        if not hasattr(self, "results"):
            self.train_models()

        best_name = max(self.results, key=lambda x: self.results[x]["f1"])
        self.best_model_name = best_name
        self.best_model = self.results[best_name]["model"]

        print(f"\nBest Model: {best_name}")
        print(f"F1 Score: {self.results[best_name]['f1']:.4f}")
        print(f"AUC-ROC: {self.results[best_name]['auc']:.4f}")

        return self.best_model, self.best_model_name

    def save_model(self, filename: str = "best_model.pkl") -> Path:
        """Save the best model and preprocessors."""
        if self.best_model is None:
            self.select_best_model()

        model_path = self.models_dir / filename
        joblib.dump(
            {
                "model": self.best_model,
                "model_name": self.best_model_name,
                "label_encoders": self.label_encoders,
                "scaler": self.scaler,
                "feature_names": list(self.X_train.columns),
            },
            model_path,
        )
        print(f"Model saved to: {model_path}")
        return model_path

    def generate_classification_report(self) -> str:
        """Generate and return classification report for best model."""
        if self.best_model is None:
            self.select_best_model()

        y_pred = self.results[self.best_model_name]["y_pred"]
        report = classification_report(self.y_test, y_pred)
        print(f"\nClassification Report for {self.best_model_name}:")
        print(report)
        return report

    def plot_confusion_matrix(self, save_path: Path = None) -> None:
        """Plot confusion matrix for the best model."""
        if self.best_model is None:
            self.select_best_model()

        y_pred = self.results[self.best_model_name]["y_pred"]

        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {self.best_model_name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Confusion matrix saved to: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_roc_curve(self, save_path: Path = None) -> None:
        """Plot ROC curve for the best model."""
        if self.best_model is None:
            self.select_best_model()

        y_pred_proba = self.results[self.best_model_name]["y_pred_proba"]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc = self.results[self.best_model_name]["auc"]

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.best_model_name}")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"ROC curve saved to: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_feature_importance(self, save_path: Path = None) -> None:
        """Plot feature importance for tree-based models."""
        if self.best_model is None:
            self.select_best_model()

        if hasattr(self.best_model, "feature_importances_"):
            importance = self.best_model.feature_importances_
            features = list(self.X_train.columns)

            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame(
                {"feature": features, "importance": importance}
            ).sort_values("importance", ascending=True)

            plt.barh(importance_df["feature"], importance_df["importance"])
            plt.xlabel("Importance")
            plt.title(f"Feature Importance - {self.best_model_name}")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"Feature importance plot saved to: {save_path}")
            else:
                plt.show()
            plt.close()

    def run_full_pipeline(self) -> dict:
        """Run the complete training pipeline."""
        print("=" * 60)
        print("CHURN PREDICTION MODEL TRAINING PIPELINE")
        print("=" * 60)

        self.preprocess()
        self.train_models()
        self.select_best_model()
        self.generate_classification_report()

        outputs_dir = self.models_dir.parent / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        self.plot_confusion_matrix(outputs_dir / "confusion_matrix.png")
        self.plot_roc_curve(outputs_dir / "roc_curve.png")
        self.plot_feature_importance(outputs_dir / "feature_importance.png")

        model_path = self.save_model()

        return {
            "model_path": model_path,
            "best_model": self.best_model_name,
            "metrics": self.results[self.best_model_name],
        }
