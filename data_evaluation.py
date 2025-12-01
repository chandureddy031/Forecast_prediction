import numpy as np
import json
import yaml
import mlflow
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from logger import get_logger

logger = get_logger("data_evaluation")

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_all_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics
    """
    # Focus on first feature (main prediction target)
    y_true_main = y_true[:, 0]
    y_pred_main = y_pred[:, 0]

    # Basic regression metrics
    mse = mean_squared_error(y_true_main, y_pred_main)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_main, y_pred_main)
    mape = mean_absolute_percentage_error(y_true_main, y_pred_main)
    r2 = r2_score(y_true_main, y_pred_main)

    # Additional metrics
    explained_var = 1 - (np.var(y_true_main - y_pred_main) / np.var(y_true_main))
    max_error = np.max(np.abs(y_true_main - y_pred_main))

    within_5pct = np.mean(np.abs(y_true_main - y_pred_main) / (np.abs(y_true_main) + 1e-10) < 0.05)
    within_10pct = np.mean(np.abs(y_true_main - y_pred_main) / (np.abs(y_true_main) + 1e-10) < 0.10)
    within_20pct = np.mean(np.abs(y_true_main - y_pred_main) / (np.abs(y_true_main) + 1e-10) < 0.20)

    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape * 100),
        'r2_score': float(r2),
        'explained_variance': float(explained_var),
        'max_error': float(max_error),
        'accuracy_within_5pct': float(within_5pct * 100),
        'accuracy_within_10pct': float(within_10pct * 100),
        'accuracy_within_20pct': float(within_20pct * 100)
    }

    return metrics

def evaluate_model():
    """
    Evaluate the trained model with comprehensive metrics
    """
    logger.info("Starting model evaluation...")

    # Load configuration
    config = load_config()

    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():

        # Load test data
        X_test = np.load("artifacts/X_test.npy")
        y_test = np.load("artifacts/y_test.npy")

        logger.info(f"Loaded test data - X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Load model (FIXED HERE)
        model = load_model("artifacts/model.keras")
        logger.info("Loaded trained model")

        # Make predictions
        y_pred = model.predict(X_test)
        logger.info("Generated predictions")

        # Calculate all metrics
        metrics = calculate_all_metrics(y_test, y_pred)

        logger.info("=" * 60)
        logger.info("COMPREHENSIVE EVALUATION METRICS:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")
            mlflow.log_metric(k, v)

        # Save metrics to JSON
        with open("artifacts/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("Saved metrics.json")

    logger.info("Model evaluation completed successfully!")

if __name__ == "__main__":
    evaluate_model()
