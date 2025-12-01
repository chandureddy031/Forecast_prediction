import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle
from logger import get_logger

logger = get_logger("data_preprocessing")

def preprocess_data():
    """
    Preprocess the raw data
    """
    logger.info("Starting data preprocessing...")
    
    # Create directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    # Load raw data
    df = pd.read_csv("data/raw/data.csv")
    logger.info(f"Loaded raw data with shape: {df.shape}")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Handle missing values
    df = df.fillna(method='ffill'). fillna(method='bfill')
    logger.info("Handled missing values")
    
    # Remove outliers using IQR method
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['value'] >= Q1 - 1.5 * IQR) & (df['value'] <= Q3 + 1.5 * IQR)]
    logger.info(f"Removed outliers.  New shape: {df.shape}")
    
    # Scale features
    scaler = StandardScaler()
    feature_cols = ['feature1', 'feature2']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save scaler
    scaler_path = "artifacts/scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save preprocessed data
    output_path = "data/processed/preprocessed_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessing completed. Saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    preprocess_data()