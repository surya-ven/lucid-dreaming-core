#!/usr/bin/env python3
"""
Improved 1D CNN for REM Sleep Stage Classification with Class Imbalance Handling

This script trains a 1D CNN to classify 15-second EEG windows as REM or Non-REM sleep stages
with specific focus on handling extreme class imbalance (0.9% REM vs 99.1% Non-REM).

Key improvements:
- Focal Loss for class imbalance
- Optimal threshold finding
- Better class weighting
- SMOTE oversampling option
- Comprehensive evaluation metrics
- Periodic model checkpoints
- Training curve plotting
- Reduced epochs for faster testing

Author: Benjamin Grayzel
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, balanced_accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import tensorflow.keras.backend as K
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATA_FILE = "extracted_REM_windows.npz"
MODEL_SAVE_PATH = "rem_classifier_improved.keras"  # Use .keras format
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
BATCH_SIZE = 32  # Reduced batch size for stability
EPOCHS = 50  # Reduced epochs for faster testing
LEARNING_RATE = 0.0001  # Much lower learning rate
PATIENCE = 15  # Reduced patience
USE_SMOTE = True  # Enable SMOTE oversampling
USE_UNDERSAMPLING = True  # Enable undersampling of majority class
FOCAL_LOSS_GAMMA = 3.0  # Higher gamma for more focus on hard examples
FOCAL_LOSS_ALPHA = 0.95  # Strong weight for positive class (REM)
UNDERSAMPLE_RATIO = 0.1  # Keep only 10% of non-REM samples


class F1ScoreCallback(callbacks.Callback):
    """Custom callback to monitor F1 score during training."""
    
    def __init__(self, training_data, validation_data):
        super().__init__()
        self.training_data = training_data
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch, logs=None):
        # Validation F1
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        logs['val_f1_score'] = _val_f1
        
        # Training F1
        train_predict = (np.asarray(self.model.predict(self.training_data[0]))).round()
        train_targ = self.training_data[1]
        _train_f1 = f1_score(train_targ, train_predict)
        logs['f1_score'] = _train_f1
        
        print(f' — val_f1: {_val_f1:.4f} — train_f1: {_train_f1:.4f}')


class PlottingCallback(callbacks.Callback):
    """Custom callback to plot training curves periodically."""
    
    def __init__(self, plot_every=10):
        super().__init__()
        self.plot_every = plot_every
    
    def on_epoch_end(self, epoch, logs=None):
        # Plot every N epochs
        if (epoch + 1) % self.plot_every == 0:
            self.plot_current_progress(epoch + 1)
    
    def plot_current_progress(self, current_epoch):
        """Plot training progress up to current epoch."""
        try:
            # Read the CSV log file
            df = pd.read_csv('training_log_improved.csv')
            
            if len(df) < 2:  # Need at least 2 epochs to plot
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss
            axes[0, 0].plot(df['epoch'], df['loss'], label='Training Loss', color='blue')
            axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red')
            axes[0, 0].set_title(f'Model Loss (Epoch {current_epoch})')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # F1 Score (if available)
            if 'f1_score' in df.columns and 'val_f1_score' in df.columns:
                axes[0, 1].plot(df['epoch'], df['f1_score'], label='Training F1', color='blue')
                axes[0, 1].plot(df['epoch'], df['val_f1_score'], label='Validation F1', color='red')
                axes[0, 1].set_title(f'Model F1 Score (Epoch {current_epoch})')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('F1 Score')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            else:
                # Plot accuracy if F1 not available
                axes[0, 1].plot(df['epoch'], df['accuracy'], label='Training Accuracy', color='blue')
                axes[0, 1].plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', color='red')
                axes[0, 1].set_title(f'Model Accuracy (Epoch {current_epoch})')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Precision
            axes[1, 0].plot(df['epoch'], df['precision'], label='Training Precision', color='blue')
            axes[1, 0].plot(df['epoch'], df['val_precision'], label='Validation Precision', color='red')
            axes[1, 0].set_title(f'Model Precision (Epoch {current_epoch})')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Recall
            axes[1, 1].plot(df['epoch'], df['recall'], label='Training Recall', color='blue')
            axes[1, 1].plot(df['epoch'], df['val_recall'], label='Validation Recall', color='red')
            axes[1, 1].set_title(f'Model Recall (Epoch {current_epoch})')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(f'training_progress_epoch_{current_epoch:02d}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()  # Close to free memory
            
            print(f"Training progress plot saved: training_progress_epoch_{current_epoch:02d}.png")
            
        except Exception as e:
            print(f"Could not plot training progress: {e}")

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        gamma: Focusing parameter (higher gamma puts more focus on hard examples)
        alpha: Weighting factor for rare class
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    
    return focal_loss_fixed

def load_data(filepath):
    """Load and prepare the extracted REM data."""
    print("Loading data...")
    data = np.load(filepath)
    
    windows = data['windows']  # Shape: (n_windows, n_samples, n_channels)
    labels = data['labels']    # Shape: (n_windows,)
    
    print(f"Data shape: {windows.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"REM samples: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    print(f"Non-REM samples: {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
    
    return windows, labels

def undersample_majority_class(X, y, ratio=0.1, random_state=42):
    """
    Undersample the majority class (Non-REM) to reduce extreme imbalance.
    
    Args:
        X: Input data
        y: Labels
        ratio: Ratio of majority class samples to keep
        random_state: Random seed
    
    Returns:
        X_resampled, y_resampled: Balanced dataset
    """
    print(f"Applying undersampling with ratio {ratio}...")
    
    # Get indices for each class
    rem_indices = np.where(y == 1)[0]
    non_rem_indices = np.where(y == 0)[0]
    
    # Calculate how many non-REM samples to keep
    n_non_rem_keep = int(len(non_rem_indices) * ratio)
    
    # Randomly sample non-REM indices
    np.random.seed(random_state)
    non_rem_keep = np.random.choice(non_rem_indices, n_non_rem_keep, replace=False)
    
    # Combine indices
    keep_indices = np.concatenate([rem_indices, non_rem_keep])
    np.random.shuffle(keep_indices)
    
    X_resampled = X[keep_indices]
    y_resampled = y[keep_indices]
    
    print(f"After undersampling:")
    print(f"  Total samples: {len(y_resampled)}")
    print(f"  REM: {np.sum(y_resampled)} ({np.mean(y_resampled)*100:.1f}%)")
    print(f"  Non-REM: {np.sum(y_resampled == 0)} ({np.mean(y_resampled == 0)*100:.1f}%)")
    
    return X_resampled, y_resampled

def apply_smote(X, y, random_state=42, strategy='auto'):
    """Apply SMOTE oversampling to balance the dataset."""
    print("Applying SMOTE oversampling...")
    
    # Reshape for SMOTE (flatten time and channel dimensions)
    X_reshaped = X.reshape(X.shape[0], -1)
    
    # Apply SMOTE with k_neighbors=1 since we have very few positive samples
    n_rem_samples = np.sum(y)
    k_neighbors = min(3, n_rem_samples - 1) if n_rem_samples > 1 else 1
    
    # Use a more conservative sampling strategy
    if strategy == 'auto':
        # Don't oversample to 50-50, but to something more reasonable like 10-90
        n_majority = np.sum(y == 0)
        target_minority = int(n_majority * 0.15)  # Target 15% REM samples
        sampling_strategy = {1: target_minority}
    else:
        sampling_strategy = strategy
    
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors, 
                  sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
    
    # Reshape back to original shape
    X_resampled = X_resampled.reshape(-1, X.shape[1], X.shape[2])
    
    print(f"After SMOTE - Total samples: {len(y_resampled)}")
    print(f"REM samples: {np.sum(y_resampled)} ({np.mean(y_resampled)*100:.1f}%)")
    
    return X_resampled, y_resampled

def create_improved_1d_cnn_model(input_shape, num_classes=1):
    """
    Create a smaller, more regularized 1D CNN model for EEG classification.
    Reduced complexity to prevent overfitting on imbalanced data.
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First conv block - fewer filters, more regularization
        layers.Conv1D(filters=16, kernel_size=11, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),
        
        # Second conv block
        layers.Conv1D(filters=32, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),
        
        # Third conv block
        layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.5),
        
        # Global pooling instead of more conv layers
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.6),
        
        # Smaller dense layers
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

def f1_metric(y_true, y_pred):
    """F1 metric for monitoring during training."""
    y_pred = K.round(y_pred)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    return f1

def compile_model(model, learning_rate=LEARNING_RATE, use_focal_loss=True):
    """Compile the model with focal loss and appropriate metrics."""
    
    if use_focal_loss:
        loss = focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA)
        print(f"Using Focal Loss with gamma={FOCAL_LOSS_GAMMA}, alpha={FOCAL_LOSS_ALPHA}")
    else:
        loss = 'binary_crossentropy'
        print("Using Binary Cross-Entropy Loss")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy', 'precision', 'recall', f1_metric]
    )
    
    return model

def create_callbacks(model_save_path, patience=PATIENCE):
    """Create training callbacks with improved settings."""
    callbacks_list = [
        # Early stopping with F1 score monitoring
        callbacks.EarlyStopping(
            monitor='val_f1_score',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        
        # Model checkpoint - save best model
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_f1_score',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        
        # Periodic model checkpoint - save every 5 epochs
        callbacks.ModelCheckpoint(
            filepath='checkpoint_epoch_{epoch:02d}.keras',
            monitor='val_f1_score',
            save_best_only=False,
            save_freq=5,  # Save every 5 epochs (using save_freq instead of period)
            verbose=1
        ),
        
        # Learning rate reduction - more aggressive
        callbacks.ReduceLROnPlateau(
            monitor='val_f1_score',
            factor=0.3,  # More aggressive reduction
            patience=patience//3,  # Reduce faster
            min_lr=1e-8,
            verbose=1,
            mode='max'
        ),
        
        # CSV logger
        callbacks.CSVLogger('training_log_improved.csv'),
        
        # Custom callback to plot training curves periodically
        PlottingCallback()
    ]
    
    return callbacks_list

class F1ScoreCallback(callbacks.Callback):
    """Custom callback to monitor F1 score during training."""
    
    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        logs['val_f1_score'] = _val_f1
        
        train_predict = (np.asarray(self.model.predict(self.training_data[0]))).round()
        train_targ = self.training_data[1]
        _train_f1 = f1_score(train_targ, train_predict)
        logs['f1_score'] = _train_f1
        
        print(f' — val_f1: {_val_f1:.4f} — train_f1: {_train_f1:.4f}')

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal classification threshold using F1 score."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.3f} (F1 Score: {optimal_f1:.4f})")
    
    # Plot threshold vs F1 score
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal: {optimal_threshold:.3f}')
    plt.xlabel('Classification Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Classification Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_threshold

def plot_training_history(history):
    """Plot comprehensive training history."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
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
    
    # F1 Score
    if 'f1_score' in history.history:
        axes[0, 2].plot(history.history['f1_score'], label='Training F1')
        axes[0, 2].plot(history.history['val_f1_score'], label='Validation F1')
        axes[0, 2].set_title('Model F1 Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    
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
    
    # Learning rate
    if 'lr' in history.history:
        axes[1, 2].plot(history.history['lr'], label='Learning Rate')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model_comprehensive(model, X_test, y_test, optimal_threshold=0.5):
    """Comprehensive model evaluation with multiple metrics."""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred_default = (y_pred_proba > 0.5).astype(int)
    y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
    
    print(f"\nEvaluation with default threshold (0.5):")
    print("="*50)
    print(classification_report(y_test, y_pred_default, 
                              target_names=['Non-REM', 'REM'],
                              digits=4))
    
    print(f"\nEvaluation with optimal threshold ({optimal_threshold:.3f}):")
    print("="*50)
    print(classification_report(y_test, y_pred_optimal, 
                              target_names=['Non-REM', 'REM'],
                              digits=4))
    
    # Additional metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    f1_default = f1_score(y_test, y_pred_default)
    f1_optimal = f1_score(y_test, y_pred_optimal)
    bal_acc_default = balanced_accuracy_score(y_test, y_pred_default)
    bal_acc_optimal = balanced_accuracy_score(y_test, y_pred_optimal)
    
    print(f"\nAdditional Metrics:")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"F1 Score (default): {f1_default:.4f}")
    print(f"F1 Score (optimal): {f1_optimal:.4f}")
    print(f"Balanced Accuracy (default): {bal_acc_default:.4f}")
    print(f"Balanced Accuracy (optimal): {bal_acc_optimal:.4f}")
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Default threshold
    cm_default = confusion_matrix(y_test, y_pred_default)
    sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-REM', 'REM'],
                yticklabels=['Non-REM', 'REM'], ax=axes[0])
    axes[0].set_title('Confusion Matrix (Threshold=0.5)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Optimal threshold
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-REM', 'REM'],
                yticklabels=['Non-REM', 'REM'], ax=axes[1])
    axes[1].set_title(f'Confusion Matrix (Threshold={optimal_threshold:.3f})')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1].plot(recall, precision, linewidth=2)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('roc_pr_curves_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred_optimal, y_pred_proba, auc_score, f1_optimal

def main():
    """Main training function with improved class imbalance handling."""
    print("="*70)
    print("IMPROVED 1D CNN REM CLASSIFIER TRAINING")
    print("="*70)
    
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
    
    print(f"\nInitial data splits:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    print(f"\nInitial class distribution:")
    print(f"  Train - REM: {np.sum(y_train)} ({np.mean(y_train)*100:.1f}%)")
    print(f"  Val - REM: {np.sum(y_val)} ({np.mean(y_val)*100:.1f}%)")
    print(f"  Test - REM: {np.sum(y_test)} ({np.mean(y_test)*100:.1f}%)")
    
    # Apply undersampling first if enabled
    if USE_UNDERSAMPLING:
        X_train, y_train = undersample_majority_class(X_train, y_train, 
                                                     ratio=UNDERSAMPLE_RATIO)
        print(f"\nAfter undersampling - Training: {X_train.shape[0]} samples")
        print(f"  Train - REM: {np.sum(y_train)} ({np.mean(y_train)*100:.1f}%)")
    
    # Apply SMOTE if enabled
    if USE_SMOTE:
        X_train, y_train = apply_smote(X_train, y_train)
        print(f"\nAfter SMOTE - Training: {X_train.shape[0]} samples")
        print(f"  Train - REM: {np.sum(y_train)} ({np.mean(y_train)*100:.1f}%)")
    
    # Calculate class weights with more aggressive weighting
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Apply even more aggressive weighting if not using SMOTE extensively
    if not USE_SMOTE or USE_UNDERSAMPLING:
        # Triple the weight for REM class to combat extreme imbalance
        rem_weight = class_weights[1] * 3
        class_weight_dict = {0: class_weights[0], 1: rem_weight}
    else:
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print(f"\nClass weights: {class_weight_dict}")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, channels)
    print(f"\nInput shape: {input_shape}")
    
    model = create_improved_1d_cnn_model(input_shape)
    model = compile_model(model, LEARNING_RATE, use_focal_loss=True)
    
    # Model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Create callbacks including F1 score monitoring
    callback_list = create_callbacks(MODEL_SAVE_PATH, PATIENCE)
    f1_callback = F1ScoreCallback()
    f1_callback.training_data = (X_train, y_train)
    f1_callback.validation_data = (X_val, y_val)
    callback_list.append(f1_callback)
    
    # Train model
    print(f"\nStarting training for up to {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Using undersampling: {USE_UNDERSAMPLING}")
    print(f"Using SMOTE: {USE_SMOTE}")
    print(f"Using Focal Loss: True (gamma={FOCAL_LOSS_GAMMA}, alpha={FOCAL_LOSS_ALPHA})")
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict if not (USE_SMOTE and not USE_UNDERSAMPLING) else None,
        callbacks=callback_list,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    print(f"\nLoading best model from {MODEL_SAVE_PATH}")
    best_model = keras.models.load_model(MODEL_SAVE_PATH, compile=False)
    best_model = compile_model(best_model, LEARNING_RATE, use_focal_loss=True)
    
    # Find optimal threshold
    val_pred_proba = best_model.predict(X_val, verbose=0).flatten()
    optimal_threshold = find_optimal_threshold(y_val, val_pred_proba)
    
    # Evaluate on test set
    y_pred, y_pred_proba, auc_score, f1_score_final = evaluate_model_comprehensive(
        best_model, X_test, y_test, optimal_threshold
    )
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✓ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✓ Training plots saved")
    print(f"✓ Final AUC Score: {auc_score:.4f}")
    print(f"✓ Final F1 Score: {f1_score_final:.4f}")
    print(f"✓ Optimal Threshold: {optimal_threshold:.3f}")
    
    # Save predictions and optimal threshold
    np.savez('test_predictions_improved.npz',
             y_true=y_test,
             y_pred=y_pred,
             y_pred_proba=y_pred_proba,
             optimal_threshold=optimal_threshold)
    
    print(f"✓ Test predictions saved to: test_predictions_improved.npz")

if __name__ == "__main__":
    main()
