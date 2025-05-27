#!/usr/bin/env python3
"""
1D Convolutional Neural Network for REM Sleep Stage Classification

This script trains a 1D CNN to classify 15-second EEG windows as REM or Non-REM sleep stages.
The model is designed specifically for multi-channel EEG time series data.

Architecture:
- Multiple 1D convolutional layers with increasing depth
- Batch normalization and dropout for regularization
- Global average pooling to reduce parameters
- Dense layers for final classification

Author: Benjamin Grayzel
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATA_FILE = "extracted_REM_windows.npz"
MODEL_SAVE_PATH = "rem_classifier_1d_cnn.h5"
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15  # Early stopping patience

def load_data(filepath):
    """Load and prepare the extracted REM data."""
    print("Loading data...")
    data = np.load(filepath)
    
    windows = data['windows']  # Shape: (n_windows, n_samples, n_channels)
    labels = data['labels']    # Shape: (n_windows,)
    
    print(f"Data shape: {windows.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"REM samples: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    
    return windows, labels

def create_1d_cnn_model(input_shape, num_classes=1):
    """
    Create a 1D CNN model for EEG classification.
    
    Architecture designed for EEG time series:
    - Multiple conv layers to capture temporal patterns at different scales
    - Batch normalization for training stability
    - Dropout for regularization
    - Global average pooling to reduce overfitting
    
    Args:
        input_shape: Tuple (time_steps, channels)
        num_classes: Number of output classes (1 for binary classification)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First conv block - capture high frequency patterns
        layers.Conv1D(filters=32, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Second conv block - capture medium frequency patterns
        layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Third conv block - capture lower frequency patterns
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Fourth conv block - deep feature extraction
        layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),  # Reduces overfitting vs Flatten
        layers.Dropout(0.4),
        
        # Dense layers for classification
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

def compile_model(model, learning_rate=LEARNING_RATE, class_weights=None):
    """Compile the model with appropriate loss and metrics."""
    
    # Use weighted loss if class weights provided
    if class_weights is not None:
        # Convert class weights to sample weights during training
        loss = 'binary_crossentropy'
    else:
        loss = 'binary_crossentropy'
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def create_callbacks(model_save_path, patience=PATIENCE):
    """Create training callbacks."""
    callbacks_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate reduction
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-7,
            verbose=1
        ),
        
        # CSV logger
        callbacks.CSVLogger('training_log.csv')
    ]
    
    return callbacks_list

def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Non-REM', 'REM'],
                              digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-REM', 'REM'],
                yticklabels=['Non-REM', 'REM'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC curve
    auc_score = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nAUC Score: {auc_score:.4f}")
    
    return y_pred, y_pred_proba, auc_score

def main():
    """Main training function."""
    print("="*60)
    print("1D CNN REM CLASSIFIER TRAINING")
    print("="*60)
    
    # Load data
    X, y = load_data(DATA_FILE)
    
    # Split data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT), 
        random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    print(f"\nClass distribution:")
    print(f"  Train - REM: {np.sum(y_train)} ({np.mean(y_train)*100:.1f}%)")
    print(f"  Val - REM: {np.sum(y_val)} ({np.mean(y_val)*100:.1f}%)")
    print(f"  Test - REM: {np.sum(y_test)} ({np.mean(y_test)*100:.1f}%)")
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print(f"\nClass weights: {class_weight_dict}")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, channels)
    print(f"\nInput shape: {input_shape}")
    
    model = create_1d_cnn_model(input_shape)
    model = compile_model(model, LEARNING_RATE)
    
    # Model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Create callbacks
    callback_list = create_callbacks(MODEL_SAVE_PATH, PATIENCE)
    
    # Train model
    print(f"\nStarting training for up to {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callback_list,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    print(f"\nLoading best model from {MODEL_SAVE_PATH}")
    best_model = keras.models.load_model(MODEL_SAVE_PATH)
    
    # Evaluate on test set
    y_pred, y_pred_proba, auc_score = evaluate_model(best_model, X_test, y_test)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✓ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✓ Training plots saved")
    print(f"✓ Final AUC Score: {auc_score:.4f}")
    
    # Save predictions for further analysis
    np.savez('test_predictions.npz',
             y_true=y_test,
             y_pred=y_pred,
             y_pred_proba=y_pred_proba)
    
    print(f"✓ Test predictions saved to: test_predictions.npz")
    
    return best_model, history, auc_score

if __name__ == "__main__":
    model, history, auc = main()



