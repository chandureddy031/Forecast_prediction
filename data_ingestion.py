import pandas as pd
import numpy as np
import os
import yaml
from logger import get_logger

logger = get_logger("data_ingestion")

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def ingest_data():
    """
    Generate synthetic time series data for demonstration
    """
    logger.info("Starting data ingestion...")
    
    # Load config
    config = load_config()
    n_samples = config['data']['n_samples']
    
    # Create directories
    os.makedirs(config['paths']['data_raw'], exist_ok=True)
    
    # Generate synthetic time series data
    np.random.seed(42)
    
    # Create time series with trend and seasonality
    time = np.arange(n_samples)
    trend = 0.01 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 365)
    noise = np.random.normal(0, 2, n_samples)
    
    values = trend + seasonality + noise + 50
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='D'),
        'value': values,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random. randn(n_samples)
    })
    
    # Save to CSV
    output_path = config['data']['raw_path']
    df.to_csv(output_path, index=False)
    
    logger.info(f"Data ingestion completed.  Saved {len(df)} records to {output_path}")
    logger.info(f"Data shape: {df. shape}")
    
    return output_path

if __name__ == "__main__":
    ingest_data()