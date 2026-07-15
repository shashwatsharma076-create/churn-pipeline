"""
ML Model Trainer for Customer Churn Prediction.
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
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

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


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
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.optimal_threshold = 0.5
        self.feature_columns = None

    def load_data(self) -> pd.DataFrame:
        """Load and return the dataset."""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded data: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features to boost model performance."""
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

    def preprocess(self) -> tuple:
        """Preprocess data and return train/test splits."""
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        original_rows = len(df)
        df = df.dropna()
        dropped_rows = original_rows - len(df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with missing values")

        df = df.drop(columns=["CustomerID"], errors="ignore")

        df = self.engineer_features(df)

        categorical_cols = ["Gender", "Subscription Type", "Contract Length"]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

        y = df["Churn"]
        X = df.drop(columns=["Churn"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_columns = list(X.columns)

        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Churn rate (train): {y_train.mean():.2%}")
        print(f"Churn rate (test): {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def train_models(self) -> dict:
        """Train multiple models and return results."""
        if self.X_train is None:
            self.preprocess()

        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_split=5,
                random_state=42,
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            ),
        }

        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                use_label_encoder=False,
            )

        if HAS_LIGHTGBM:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

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

    def optimize_threshold(self) -> float:
        """Find the optimal classification threshold that maximizes accuracy."""
        best_name = max(self.results, key=lambda x: self.results[x]["accuracy"])
        y_pred_proba = self.results[best_name]["y_pred_proba"]

        best_threshold = 0.5
        best_accuracy = 0.0

        for threshold in np.arange(0.3, 0.7, 0.01):
            y_pred = (y_pred_proba >= threshold).astype(int)
            acc = accuracy_score(self.y_test, y_pred)
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = threshold

        self.optimal_threshold = best_threshold
        print(f"\nOptimal threshold: {best_threshold:.2f} (accuracy: {best_accuracy:.4f})")

        for name in self.results:
            y_pred_proba = self.results[name]["y_pred_proba"]
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            self.results[name]["y_pred_tuned"] = y_pred
            self.results[name]["accuracy_tuned"] = accuracy_score(self.y_test, y_pred)
            self.results[name]["f1_tuned"] = f1_score(self.y_test, y_pred)
            print(f"  {name} - Tuned Accuracy: {self.results[name]['accuracy_tuned']:.4f}, "
                  f"Tuned F1: {self.results[name]['f1_tuned']:.4f}")

        return best_threshold

    def select_best_model(self) -> tuple:
        """Select and return the best model based on accuracy."""
        if not hasattr(self, "results"):
            self.train_models()

        self.optimize_threshold()

        best_name = max(self.results, key=lambda x: self.results[x]["accuracy_tuned"])
        self.best_model_name = best_name
        self.best_model = self.results[best_name]["model"]

        print(f"\nBest Model: {best_name}")
        print(f"Accuracy (tuned): {self.results[best_name]['accuracy_tuned']:.4f}")
        print(f"F1 Score (tuned): {self.results[best_name]['f1_tuned']:.4f}")
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
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "optimal_threshold": self.optimal_threshold,
            },
            model_path,
        )
        print(f"Model saved to: {model_path}")
        return model_path

    def generate_classification_report(self) -> str:
        """Generate and return classification report for best model."""
        if self.best_model is None:
            self.select_best_model()

        y_pred = self.results[self.best_model_name].get(
            "y_pred_tuned", self.results[self.best_model_name]["y_pred"]
        )
        report = classification_report(self.y_test, y_pred)
        print(f"\nClassification Report for {self.best_model_name} (threshold={self.optimal_threshold:.2f}):")
        print(report)
        return report

    def plot_confusion_matrix(self, save_path: Path = None) -> None:
        """Plot confusion matrix for the best model."""
        if self.best_model is None:
            self.select_best_model()

        y_pred = self.results[self.best_model_name].get(
            "y_pred_tuned", self.results[self.best_model_name]["y_pred"]
        )

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
