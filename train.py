import numpy as np
import pickle
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from logger import get_logger

logger = get_logger("train")

def build_lstm_model(input_shape):
    """
    Build LSTM model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(input_shape[1])  # Predict all features
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_model():
    """
    Train LSTM model
    """
    logger.info("Starting model training...")

    # Load training data
    X_train = np.load("artifacts/X_train.npy")
    y_train = np.load("artifacts/y_train.npy")
    X_test = np.load("artifacts/X_test.npy")
    y_test = np.load("artifacts/y_test.npy")
    
    logger.info(f"Loaded training data - X_train: {X_train.shape}, y_train: {y_train.shape}")
 
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    logger.info("Built LSTM model")
    model.summary(print_fn=lambda x: logger.info(x))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        "artifacts/best_model.keras",
        monitor='val_loss',
        save_best_only=True
    )
    
    # Train model
    logger.info("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # FIXED final model save (.keras instead of .h5)
    model.save("artifacts/model.keras")
    logger.info("Saved model to artifacts/model.keras")
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']]
    }
    
    with open("artifacts/training_history.json", 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    logger.info("Training completed successfully")
    
    return model


if __name__ == "__main__":
    train_model()
