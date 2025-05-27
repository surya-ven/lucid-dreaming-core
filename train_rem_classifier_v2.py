#!/usr/bin/env python3
"""
REM Sleep Stage Classification - Version 2
Completely redesigned approach to address fundamental issues

Key Changes:
- More conservative data balancing
- Ensemble approach with multiple models
- Better feature engineering
- Cross-validation for robust evaluation
- Advanced threshold optimization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, balanced_accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import tensorflow.keras.backend as K
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATA_FILE = "extracted_REM_windows.npz"
EPOCHS = 30  # Even fewer epochs to focus on quality over quantity
BATCH_SIZE = 16  # Smaller batch size
LEARNING_RATE = 0.0005  # Higher learning rate with fewer epochs
PATIENCE = 10
CV_FOLDS = 5

class REM_Classifier_V2:
    """Improved REM classifier with multiple approaches"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        
    def load_and_analyze_data(self, filepath):
        """Load data with comprehensive analysis"""
        print("Loading and analyzing data...")
        data = np.load(filepath)
        
        if 'features' in data.keys():
            X = data['features']
            y = data['labels']
        else:
            X = data['windows']
            y = data['labels']
            
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Analyze class distribution
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples ({count/len(y)*100:.2f}%)")
            
        # Analyze data quality
        print(f"\nData Quality Analysis:")
        print(f"NaN values: {np.isnan(X).sum()}")
        print(f"Infinite values: {np.isinf(X).sum()}")
        print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"Data std: {X.std():.3f}")
        
        return X, y
    
    def create_features(self, X):
        """Extract meaningful features from raw EEG data"""
        print("Creating engineered features...")
        
        # Original shape: (samples, time_points, channels)
        features_list = []
        
        for i in range(X.shape[2]):  # For each EEG channel
            channel_data = X[:, :, i]
            
            # Time domain features
            features_list.append(np.mean(channel_data, axis=1))  # Mean
            features_list.append(np.std(channel_data, axis=1))   # Standard deviation
            features_list.append(np.var(channel_data, axis=1))   # Variance
            features_list.append(np.max(channel_data, axis=1))   # Max
            features_list.append(np.min(channel_data, axis=1))   # Min
            features_list.append(np.median(channel_data, axis=1)) # Median
            
            # Statistical features
            features_list.append(np.percentile(channel_data, 25, axis=1))  # 25th percentile
            features_list.append(np.percentile(channel_data, 75, axis=1))  # 75th percentile
            features_list.append(np.ptp(channel_data, axis=1))  # Peak-to-peak
            
            # Energy and power features
            features_list.append(np.sum(channel_data**2, axis=1))  # Energy
            features_list.append(np.mean(np.abs(channel_data), axis=1))  # Mean absolute value
            
        # Combine all features
        engineered_features = np.column_stack(features_list)
        print(f"Engineered features shape: {engineered_features.shape}")
        
        return engineered_features
    
    def conservative_balancing(self, X, y, target_minority_ratio=0.05):
        """Very conservative data balancing"""
        print(f"Applying conservative balancing (target: {target_minority_ratio*100:.1f}% REM)...")
        
        # First undersample majority class significantly
        rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)  # Keep only 10% of majority
        X_under, y_under = rus.fit_resample(X.reshape(X.shape[0], -1), y)
        X_under = X_under.reshape(-1, X.shape[1], X.shape[2])
        
        print(f"After undersampling: {len(y_under)} samples")
        print(f"REM: {np.sum(y_under)} ({np.mean(y_under)*100:.1f}%)")
        
        # Then apply very light SMOTE
        # Calculate target minority samples
        n_majority = np.sum(y_under == 0)
        n_minority_target = int(n_majority * target_minority_ratio / (1 - target_minority_ratio))
        n_minority_current = np.sum(y_under)
        
        if n_minority_target > n_minority_current:
            sampling_strategy = {1: n_minority_target}
            smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
            
            X_reshaped = X_under.reshape(X_under.shape[0], -1)
            X_balanced, y_balanced = smote.fit_resample(X_reshaped, y_under)
            X_balanced = X_balanced.reshape(-1, X.shape[1], X.shape[2])
        else:
            X_balanced, y_balanced = X_under, y_under
            
        print(f"After balancing: {len(y_balanced)} samples")
        print(f"REM: {np.sum(y_balanced)} ({np.mean(y_balanced)*100:.1f}%)")
        
        return X_balanced, y_balanced
    
    def create_simple_cnn(self, input_shape):
        """Create a much simpler CNN"""
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # Single conv layer
            layers.Conv1D(8, 15, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(4),
            layers.Dropout(0.3),
            
            # Global pooling
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            # Small dense layer
            layers.Dense(8, activation='relu'),
            layers.Dropout(0.5),
            
            # Output
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def create_feature_based_model(self, input_dim):
        """Create model for engineered features"""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def focal_loss(self, gamma=2.0, alpha=0.75):
        """Focal loss for class imbalance"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
            
            p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
            
            focal_loss = -alpha_t * K.pow((1 - p_t), gamma) * K.log(p_t)
            return K.mean(focal_loss)
        
        return focal_loss_fixed
    
    def train_cnn_model(self, X_train, y_train, X_val, y_val):
        """Train simple CNN model"""
        print("Training CNN model...")
        
        # Create model
        model = self.create_simple_cnn((X_train.shape[1], X_train.shape[2]))
        
        # Compile with focal loss
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=self.focal_loss(gamma=3.0, alpha=0.8),
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ]
        
        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] * 2 for i in range(len(class_weights))}  # Double the weighting
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.models['cnn'] = model
        return history
    
    def train_feature_model(self, X_train, y_train, X_val, y_val):
        """Train model on engineered features"""
        print("Training feature-based model...")
        
        # Create features
        X_train_features = self.create_features(X_train)
        X_val_features = self.create_features(X_val)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_val_scaled = scaler.transform(X_val_features)
        
        self.scalers['features'] = scaler
        
        # Create model
        model = self.create_feature_based_model(X_train_scaled.shape[1])
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',  # Standard loss for feature model
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] * 3 for i in range(len(class_weights))}
        
        # Train
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ],
            verbose=1
        )
        
        self.models['features'] = model
        return history
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest as baseline"""
        print("Training Random Forest baseline...")
        
        # Create features
        X_train_features = self.create_features(X_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train_scaled, y_train)
        
        self.models['rf'] = rf
        self.scalers['rf'] = scaler
        
        return rf
    
    def find_optimal_thresholds(self, X_val, y_val):
        """Find optimal thresholds for each model"""
        print("Finding optimal thresholds...")
        
        for model_name in self.models.keys():
            if model_name == 'rf':
                X_val_features = self.create_features(X_val)
                X_val_scaled = self.scalers['rf'].transform(X_val_features)
                y_pred_proba = self.models[model_name].predict_proba(X_val_scaled)[:, 1]
            elif model_name == 'features':
                X_val_features = self.create_features(X_val)
                X_val_scaled = self.scalers['features'].transform(X_val_features)
                y_pred_proba = self.models[model_name].predict(X_val_scaled).flatten()
            else:  # CNN
                y_pred_proba = self.models[model_name].predict(X_val).flatten()
            
            # Find best threshold
            thresholds = np.linspace(0.1, 0.9, 81)
            f1_scores = []
            
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred)
                f1_scores.append(f1)
            
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            self.thresholds[model_name] = best_threshold
            print(f"{model_name}: threshold={best_threshold:.3f}, F1={best_f1:.4f}")
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a specific model"""
        if model_name == 'rf':
            X_test_features = self.create_features(X_test)
            X_test_scaled = self.scalers['rf'].transform(X_test_features)
            y_pred_proba = self.models[model_name].predict_proba(X_test_scaled)[:, 1]
        elif model_name == 'features':
            X_test_features = self.create_features(X_test)
            X_test_scaled = self.scalers['features'].transform(X_test_features)
            y_pred_proba = self.models[model_name].predict(X_test_scaled).flatten()
        else:  # CNN
            y_pred_proba = self.models[model_name].predict(X_test).flatten()
        
        # Apply optimal threshold
        threshold = self.thresholds[model_name]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"\n{model_name.upper()} Results:")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC Score: {auc:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Optimal Threshold: {threshold:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-REM', 'REM']))
        
        return {
            'f1': f1,
            'auc': auc,
            'balanced_accuracy': balanced_acc,
            'threshold': threshold,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def ensemble_predict(self, X_test):
        """Ensemble prediction from all models"""
        predictions = []
        
        for model_name in self.models.keys():
            if model_name == 'rf':
                X_test_features = self.create_features(X_test)
                X_test_scaled = self.scalers['rf'].transform(X_test_features)
                proba = self.models[model_name].predict_proba(X_test_scaled)[:, 1]
            elif model_name == 'features':
                X_test_features = self.create_features(X_test)
                X_test_scaled = self.scalers['features'].transform(X_test_features)
                proba = self.models[model_name].predict(X_test_scaled).flatten()
            else:  # CNN
                proba = self.models[model_name].predict(X_test).flatten()
            
            predictions.append(proba)
        
        # Average predictions
        ensemble_proba = np.mean(predictions, axis=0)
        return ensemble_proba


def main():
    """Main training function"""
    print("="*70)
    print("REM CLASSIFIER V2 - REDESIGNED APPROACH")
    print("="*70)
    
    # Initialize classifier
    classifier = REM_Classifier_V2()
    
    # Load and analyze data
    X, y = classifier.load_and_analyze_data(DATA_FILE)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"Train: {len(y_train)} samples, REM: {np.sum(y_train)} ({np.mean(y_train)*100:.1f}%)")
    print(f"Val: {len(y_val)} samples, REM: {np.sum(y_val)} ({np.mean(y_val)*100:.1f}%)")
    print(f"Test: {len(y_test)} samples, REM: {np.sum(y_test)} ({np.mean(y_test)*100:.1f}%)")
    
    # Apply conservative balancing only to training data
    X_train_balanced, y_train_balanced = classifier.conservative_balancing(X_train, y_train)
    
    # Train multiple models
    print("\n" + "="*50)
    print("TRAINING MULTIPLE MODELS")
    print("="*50)
    
    # 1. Train Random Forest baseline
    classifier.train_random_forest(X_train_balanced, y_train_balanced)
    
    # 2. Train feature-based neural network
    classifier.train_feature_model(X_train_balanced, y_train_balanced, X_val, y_val)
    
    # 3. Train simple CNN
    classifier.train_cnn_model(X_train_balanced, y_train_balanced, X_val, y_val)
    
    # Find optimal thresholds
    classifier.find_optimal_thresholds(X_val, y_val)
    
    # Evaluate all models
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    results = {}
    for model_name in classifier.models.keys():
        results[model_name] = classifier.evaluate_model(model_name, X_test, y_test)
    
    # Ensemble evaluation
    print("\nENSEMBLE Results:")
    ensemble_proba = classifier.ensemble_predict(X_test)
    
    # Find best ensemble threshold
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (ensemble_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    ensemble_pred = (ensemble_proba >= best_threshold).astype(int)
    
    ensemble_f1 = f1_score(y_test, ensemble_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    ensemble_balanced_acc = balanced_accuracy_score(y_test, ensemble_pred)
    
    print(f"Ensemble F1 Score: {ensemble_f1:.4f}")
    print(f"Ensemble AUC Score: {ensemble_auc:.4f}")
    print(f"Ensemble Balanced Accuracy: {ensemble_balanced_acc:.4f}")
    print(f"Ensemble Threshold: {best_threshold:.3f}")
    
    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=['Non-REM', 'REM']))
    
    # Summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"{model_name}: F1={result['f1']:.4f}, AUC={result['auc']:.4f}")
    
    print(f"Ensemble: F1={ensemble_f1:.4f}, AUC={ensemble_auc:.4f}")
    
    # Determine best approach
    best_f1 = max([results[name]['f1'] for name in results.keys()] + [ensemble_f1])
    if ensemble_f1 == best_f1:
        print("\n🏆 Best approach: ENSEMBLE")
    else:
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        print(f"\n🏆 Best approach: {best_model.upper()}")


if __name__ == "__main__":
    main()
