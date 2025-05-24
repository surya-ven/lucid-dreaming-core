import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.utils import class_weight
from scipy import signal
from scipy.stats import skew, kurtosis
import joblib
import warnings
warnings.filterwarnings('ignore')


def extract_robust_features(X):
    """
    Extract rich, noise-robust features from EOG time series data.
    
    Args:
        X (np.ndarray): Input data of shape (n_samples, n_timesteps, n_channels)
        
    Returns:
        np.ndarray: Feature matrix of shape (n_samples, n_features)
    """
    n_samples, n_timesteps, n_channels = X.shape
    features_list = []
    
    print(f"Extracting features from {n_samples} samples...")
    
    for i in range(n_samples):
        sample_features = []
        
        for ch in range(n_channels):
            signal_data = X[i, :, ch]
            
            # --- Time Domain Features ---
            # Basic statistics
            sample_features.extend([
                np.mean(signal_data),
                np.std(signal_data),
                np.var(signal_data),
                np.median(signal_data),
                np.max(signal_data) - np.min(signal_data),  # Range
                np.percentile(signal_data, 25),
                np.percentile(signal_data, 75),
                np.percentile(signal_data, 90) - np.percentile(signal_data, 10),  # IQR-like
            ])
            
            # Higher order statistics
            sample_features.extend([
                skew(signal_data),
                kurtosis(signal_data),
                np.sqrt(np.mean(signal_data**2)),  # RMS
                np.mean(np.abs(signal_data)),  # Mean absolute value
            ])
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
            sample_features.append(zero_crossings / len(signal_data))
            
            # Activity and mobility (Hjorth parameters)
            diff1 = np.diff(signal_data)
            diff2 = np.diff(diff1)
            
            var_signal = np.var(signal_data)
            var_diff1 = np.var(diff1)
            var_diff2 = np.var(diff2)
            
            activity = var_signal
            mobility = np.sqrt(var_diff1 / var_signal) if var_signal > 0 else 0
            complexity = np.sqrt(var_diff2 / var_diff1) / mobility if var_diff1 > 0 and mobility > 0 else 0
            
            sample_features.extend([activity, mobility, complexity])
            
            # Envelope features
            analytic_signal = signal.hilbert(signal_data)
            envelope = np.abs(analytic_signal)
            sample_features.extend([
                np.mean(envelope),
                np.std(envelope),
                np.max(envelope),
            ])
            
            # --- Frequency Domain Features ---
            # Power spectral density
            freqs, psd = signal.welch(signal_data, fs=118, nperseg=min(256, len(signal_data)//2))
            
            # Frequency bands for EOG analysis
            delta_band = (0.5, 4)
            theta_band = (4, 8)
            alpha_band = (8, 13)
            beta_band = (13, 15)
            
            bands = [delta_band, theta_band, alpha_band, beta_band]
            band_powers = []
            
            for low, high in bands:
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                else:
                    band_power = 0
                band_powers.append(band_power)
            
            total_power = np.trapz(psd, freqs)
            
            # Relative band powers
            if total_power > 0:
                rel_band_powers = [bp / total_power for bp in band_powers]
            else:
                rel_band_powers = [0] * len(band_powers)
            
            sample_features.extend(band_powers + rel_band_powers)
            
            # Spectral features
            if total_power > 0:
                spectral_centroid = np.trapz(freqs * psd, freqs) / total_power
                spectral_bandwidth = np.sqrt(np.trapz((freqs - spectral_centroid)**2 * psd, freqs) / total_power)
            else:
                spectral_centroid = 0
                spectral_bandwidth = 0
            
            sample_features.extend([
                spectral_centroid,
                spectral_bandwidth,
                np.max(psd),  # Peak power
                freqs[np.argmax(psd)],  # Peak frequency
            ])
            
            # --- Wavelet Features ---
            # Simple wavelet energy using built-in signal processing
            try:
                # Approximate discrete wavelet transform using filters
                # Low-pass and high-pass decomposition
                b, a = signal.butter(4, 0.5, btype='lowpass', fs=118)
                low_freq = signal.filtfilt(b, a, signal_data)
                
                b, a = signal.butter(4, [0.5, 15], btype='bandpass', fs=118)
                band_freq = signal.filtfilt(b, a, signal_data)
                
                wavelet_energy_low = np.sum(low_freq**2)
                wavelet_energy_band = np.sum(band_freq**2)
                
                sample_features.extend([wavelet_energy_low, wavelet_energy_band])
            except:
                sample_features.extend([0, 0])
        
        # --- Cross-channel features ---
        if n_channels > 1:
            # Correlation between channels
            corr_coeff = np.corrcoef(X[i, :, 0], X[i, :, 1])[0, 1]
            if np.isnan(corr_coeff):
                corr_coeff = 0
            sample_features.append(corr_coeff)
            
            # Phase coupling (simplified)
            phase_diff = np.angle(signal.hilbert(X[i, :, 0])) - np.angle(signal.hilbert(X[i, :, 1]))
            phase_coupling = np.abs(np.mean(np.exp(1j * phase_diff)))
            sample_features.append(phase_coupling)
            
            # Difference signal features
            diff_signal = X[i, :, 0] - X[i, :, 1]
            sample_features.extend([
                np.std(diff_signal),
                np.max(diff_signal) - np.min(diff_signal),
                np.mean(np.abs(diff_signal))
            ])
        
        features_list.append(sample_features)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_samples} samples")
    
    feature_matrix = np.array(features_list)
    print(f"Feature extraction complete. Shape: {feature_matrix.shape}")
    
    return feature_matrix


def train_svm_lrlr(data_path='lstm_training_data.npz', model_save_path='lrlr_svm_model.pkl'):
    """
    Train an SVM model to detect LRLR events using rich feature extraction and cross-validation.
    
    Args:
        data_path (str): Path to the .npz file containing training data (X, y).
        model_save_path (str): Path to save the trained SVM model.
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
        print("Error: Empty data arrays")
        return

    # Clean NaNs and Infs from X data
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: Found NaN or Inf values in X data. Replacing with zeros.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    print(f"LRLR samples (1): {np.sum(y == 1)}, Non-LRLR samples (0): {np.sum(y == 0)}")

    # Extract features
    X_features = extract_robust_features(X)
    
    # Handle any remaining NaN/Inf in features
    if np.any(np.isnan(X_features)) or np.any(np.isinf(X_features)):
        print("Warning: Found NaN or Inf values in extracted features. Replacing with zeros.")
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Feature matrix shape: {X_features.shape}")
    
    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"Class weights: {class_weights_dict}")

    # Feature scaling
    scaler = StandardScaler()
    X_features_scaled = scaler.fit_transform(X_features)
    
    # Cross-validation setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # SVM hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    print("\nPerforming grid search with cross-validation...")
    
    # Grid search with cross-validation
    svm = SVC(class_weight='balanced', probability=True, random_state=42)
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=skf,
        scoring='f1',  # F1 score is good for imbalanced data
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_features_scaled, y)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_svm = grid_search.best_estimator_
    
    # Detailed cross-validation evaluation
    print("\n--- Detailed Cross-Validation Results ---")
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    fold_no = 1
    for train_index, test_index in skf.split(X_features_scaled, y):
        X_train_fold, X_test_fold = X_features_scaled[train_index], X_features_scaled[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        # Train model on fold
        fold_model = SVC(**grid_search.best_params_, class_weight='balanced', probability=True, random_state=42)
        fold_model.fit(X_train_fold, y_train_fold)
        
        # Predictions
        y_pred = fold_model.predict(X_test_fold)
        y_pred_proba = fold_model.predict_proba(X_test_fold)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test_fold, y_pred)
        precision = precision_score(y_test_fold, y_pred, zero_division=0)
        recall = recall_score(y_test_fold, y_pred, zero_division=0)
        f1 = f1_score(y_test_fold, y_pred, zero_division=0)
        auc = roc_auc_score(y_test_fold, y_pred_proba)
        
        cv_scores['accuracy'].append(accuracy)
        cv_scores['precision'].append(precision)
        cv_scores['recall'].append(recall)
        cv_scores['f1'].append(f1)
        cv_scores['auc'].append(auc)
        
        print(f"Fold {fold_no}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        fold_no += 1
    
    # Print average results
    print("\n--- Average Cross-Validation Results ---")
    for metric, scores in cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric.capitalize()}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    # Train final model on all data
    print("\n--- Training final model on all data ---")
    final_model = SVC(**grid_search.best_params_, class_weight='balanced', probability=True, random_state=42)
    final_model.fit(X_features_scaled, y)
    
    # Final evaluation on training data (not a true validation)
    y_pred_final = final_model.predict(X_features_scaled)
    y_pred_proba_final = final_model.predict_proba(X_features_scaled)[:, 1]
    
    print("\nFinal model evaluation on training data:")
    print(f"Accuracy: {accuracy_score(y, y_pred_final):.4f}")
    print(f"AUC: {roc_auc_score(y, y_pred_proba_final):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred_final))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred_final))
    
    # Save the model and scaler
    model_data = {
        'model': final_model,
        'scaler': scaler,
        'best_params': grid_search.best_params_,
        'cv_results': cv_scores,
        'feature_names': [f'feature_{i}' for i in range(X_features.shape[1])]
    }
    
    joblib.dump(model_data, model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    return final_model, scaler


if __name__ == '__main__':
    train_svm_lrlr()