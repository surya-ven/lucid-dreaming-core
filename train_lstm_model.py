import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def train_lrlr_lstm(data_path='lstm_training_data.npz', model_save_path='lrlr_lstm_model.keras'):
    """
    Trains an LSTM model to detect LRLR events.

    Args:
        data_path (str): Path to the .npz file containing training data (X, y).
        model_save_path (str): Path to save the trained Keras model.
    """
    print(f"Loading data from {data_path}...")
    try:
        data = np.load(data_path)
        X = data['X']
        y = data['y']
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please run lstm_data_extraction.py first.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if X.size == 0 or y.size == 0:
        print("Error: Loaded data is empty. Cannot train model.")
        return

    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    print(f"LRLR samples (1): {np.sum(y == 1)}, Non-LRLR samples (0): {np.sum(y == 0)}")

    # Reshape y to be (n_samples, 1) for Keras
    y = y.reshape(-1, 1)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Training set: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Validation set: X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Calculate class weights to handle imbalance
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train.flatten()), y=y_train.flatten())
    class_weights_dict = {i : weights[i] for i in range(len(weights))}
    print(f"Calculated class weights: {class_weights_dict}")


    # Define the LSTM model
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])

    model.summary()

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Callbacks
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)

    # Train the model
    print("\nStarting model training...")
    history = model.fit(X_train, y_train,
                        epochs=50,  # Adjust number of epochs as needed
                        batch_size=32, # Adjust batch size as needed
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint, early_stopping],
                        class_weight=class_weights_dict,
                        verbose=1)

    print("\nTraining finished.")

    # Evaluate the best model (restored by EarlyStopping)
    print("\nEvaluating model on validation set:")
    loss, accuracy, precision, recall = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Validation Loss: {loss:.4f}")
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print(f"  Validation Precision: {precision:.4f}")
    print(f"  Validation Recall: {recall:.4f}")

    print(f"\nBest model saved to {model_save_path}")

if __name__ == '__main__':
    # First, ensure data is generated
    # You might want to call create_lstm_data() from lstm_data_extraction.py here
    # or ensure it's run manually before this script.
    # For simplicity, this script assumes 'lstm_training_data.npz' exists.

    # Example:
    # from lstm_data_extraction import create_lstm_data
    # print("Ensuring LSTM training data exists...")
    # create_lstm_data() # This will run the data extraction if you uncomment
    # print("Data extraction/check complete.")
    
    train_lrlr_lstm()