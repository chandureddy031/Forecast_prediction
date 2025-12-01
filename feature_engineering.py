import pandas as pd
import numpy as np
import pickle
import yaml
from logger import get_logger

logger = get_logger("feature_engineering")

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        config = yaml. safe_load(f)
    return config

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y. append(data[i + seq_length])
    return np.array(X), np.array(y)

def engineer_features():
    """
    Create features for LSTM model
    """
    logger.info("Starting feature engineering...")
    
    # Load config
    config = load_config()
    
    # Load preprocessed data
    df = pd.read_csv(config['data']['processed_path'])
    logger.info(f"Loaded preprocessed data with shape: {df.shape}")
    
    # Create lag features
    for lag in config['features']['lag_features']:
        df[f'value_lag_{lag}'] = df['value'].shift(lag)
    
    # Create rolling statistics
    for window in config['features']['rolling_windows']:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window). mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
    
    # Drop NaN values created by lag and rolling
    df = df.dropna()
    logger.info(f"Created engineered features.  New shape: {df.shape}")
    
    # Prepare features for LSTM
    feature_cols = ['value', 'feature1', 'feature2']
    for lag in config['features']['lag_features']:
        feature_cols.append(f'value_lag_{lag}')
    for window in config['features']['rolling_windows']:
        feature_cols. append(f'rolling_mean_{window}')
        feature_cols.append(f'rolling_std_{window}')
    
    data = df[feature_cols].values
    
    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    seq_length = config['features']['sequence_length']
    X, y = create_sequences(data_scaled, seq_length)
    logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
    
    # Split into train and test
    test_size = config['data']['test_size']
    train_size = int((1 - test_size) * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logger.info(f"Train set: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    # Save data and scaler
    artifacts_path = config['paths']['artifacts']
    np.save(f"{artifacts_path}/X_train.npy", X_train)
    np. save(f"{artifacts_path}/X_test.npy", X_test)
    np.save(f"{artifacts_path}/y_train.npy", y_train)
    np.save(f"{artifacts_path}/y_test.npy", y_test)
    
    with open(f"{artifacts_path}/feature_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns
    with open(f"{artifacts_path}/feature_columns. pkl", 'wb') as f:
        pickle.dump(feature_cols, f)
    
    logger.info("Feature engineering completed")
    
    return X_train. shape, X_test.shape

if __name__ == "__main__":
    engineer_features()