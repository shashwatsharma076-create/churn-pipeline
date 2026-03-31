"""
Tests for ML model trainer.
"""
import pytest
import pandas as pd
import numpy as np
from src.models.trainer import ChurnModelTrainer


def create_larger_sample_dataframe():
    """Create a larger sample dataframe for model training tests."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "Age": np.random.randint(18, 65, n_samples),
        "Gender": np.random.choice(["Male", "Female"], n_samples),
        "Tenure": np.random.randint(1, 60, n_samples),
        "Usage Frequency": np.random.randint(1, 30, n_samples),
        "Support Calls": np.random.randint(0, 10, n_samples),
        "Payment Delay": np.random.randint(0, 30, n_samples),
        "Subscription Type": np.random.choice(["Basic", "Premium", "Standard"], n_samples),
        "Contract Length": np.random.choice(["Monthly", "Quarterly", "Annual"], n_samples),
        "Total Spend": np.random.uniform(100, 1000, n_samples),
        "Last Interaction": np.random.randint(1, 30, n_samples),
        "Churn": np.random.choice([0, 1], n_samples),
    }
    return pd.DataFrame(data)


class TestChurnModelTrainer:
    """Test cases for ChurnModelTrainer."""

    def test_preprocessing_creates_train_test_split(self, tmp_path):
        """Test that preprocessing creates train/test splits."""
        df = create_larger_sample_dataframe()
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)
        models_dir = tmp_path / "models"

        trainer = ChurnModelTrainer(data_path, models_dir)
        trainer.load_data()
        trainer.preprocess()

        assert trainer.X_train is not None
        assert trainer.X_test is not None
        assert trainer.y_train is not None
        assert trainer.y_test is not None
        assert len(trainer.X_train) > len(trainer.X_test)

    def test_training_produces_results(self, tmp_path):
        """Test that training produces model results."""
        df = create_larger_sample_dataframe()
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)
        models_dir = tmp_path / "models"

        trainer = ChurnModelTrainer(data_path, models_dir)
        trainer.preprocess()
        results = trainer.train_models()

        assert "Logistic Regression" in results
        assert "Random Forest" in results
        assert "Gradient Boosting" in results

    def test_results_contain_metrics(self, tmp_path):
        """Test that results contain all required metrics."""
        df = create_larger_sample_dataframe()
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)
        models_dir = tmp_path / "models"

        trainer = ChurnModelTrainer(data_path, models_dir)
        trainer.preprocess()
        results = trainer.train_models()

        for model_name, model_results in results.items():
            assert "accuracy" in model_results
            assert "precision" in model_results
            assert "recall" in model_results
            assert "f1" in model_results
            assert "auc" in model_results

    def test_best_model_selection(self, tmp_path):
        """Test that best model is correctly selected."""
        df = create_larger_sample_dataframe()
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)
        models_dir = tmp_path / "models"

        trainer = ChurnModelTrainer(data_path, models_dir)
        trainer.preprocess()
        trainer.train_models()
        best_model, best_name = trainer.select_best_model()

        assert best_model is not None
        assert best_name is not None
        assert best_name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]

    def test_model_saving(self, tmp_path):
        """Test that model can be saved."""
        df = create_larger_sample_dataframe()
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)
        models_dir = tmp_path / "models"

        trainer = ChurnModelTrainer(data_path, models_dir)
        trainer.preprocess()
        trainer.train_models()
        trainer.select_best_model()
        model_path = trainer.save_model("test_model.pkl")

        assert model_path.exists()
