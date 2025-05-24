# Author: Benjamin Grayzel
# Gradient-boosted trees (XGBoost/LightGBM) for LRLR detection with rich feature extraction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# For feature extraction
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
import pywt
from dtaidistance import dtw
import joblib
import os


class LRLRFeatureExtractor:
    """
    Feature extractor for LRLR detection using EOG signals.
    Extracts noise-robust features from time-series windows.
    """
    
    def __init__(self, srate=118):
        self.srate = srate
        # Create LRLR template for DTW distance computation
        self.lrlr_template = self._create_lrlr_template()
    
    def _create_lrlr_template(self):
        """Create an idealized LRLR template for DTW matching"""
        # Simple LRLR template: Left-Right-Left-Right eye movements
        # Each movement lasts ~200ms at 118Hz sampling rate
        movement_duration = int(0.2 * self.srate)  # ~24 samples
        
        template = np.zeros(4 * movement_duration)
        
        # Left movement (negative deflection)
        template[0:movement_duration] = -np.sin(np.linspace(0, np.pi, movement_duration))
        
        # Right movement (positive deflection)  
        template[movement_duration:2*movement_duration] = np.sin(np.linspace(0, np.pi, movement_duration))
        
        # Left movement
        template[2*movement_duration:3*movement_duration] = -np.sin(np.linspace(0, np.pi, movement_duration))
        
        # Right movement
        template[3*movement_duration:4*movement_duration] = np.sin(np.linspace(0, np.pi, movement_duration))
        
        return template
    
    def extract_features(self, X):
        """
        Extract comprehensive features from EOG signal windows.
        
        Args:
            X: numpy array of shape (n_samples, n_timepoints, n_channels)
               Expected to be (n_samples, 750, 2) for EOG data
        
        Returns:
            DataFrame with extracted features
        """
        n_samples = X.shape[0]
        features_list = []
        
        print(f"Extracting features from {n_samples} samples...")
        
        for i in range(n_samples):
            if i % 50 == 0:
                print(f"Processing sample {i}/{n_samples}")
            
            # Get the two EOG channels
            eog1 = X[i, :, 0]  # First EOG channel
            eog2 = X[i, :, 1]  # Second EOG channel
            eog_diff = eog1 - eog2  # Differential signal
            
            features = {}
            
            # === Basic Statistical Features ===
            for ch_name, signal_data in [('eog1', eog1), ('eog2', eog2), ('diff', eog_diff)]:
                features[f'{ch_name}_mean'] = np.mean(signal_data)
                features[f'{ch_name}_std'] = np.std(signal_data)
                features[f'{ch_name}_var'] = np.var(signal_data)
                features[f'{ch_name}_skewness'] = skew(signal_data)
                features[f'{ch_name}_kurtosis'] = kurtosis(signal_data)
                features[f'{ch_name}_min'] = np.min(signal_data)
                features[f'{ch_name}_max'] = np.max(signal_data)
                features[f'{ch_name}_range'] = np.max(signal_data) - np.min(signal_data)
                features[f'{ch_name}_median'] = np.median(signal_data)
                features[f'{ch_name}_mad'] = np.median(np.abs(signal_data - np.median(signal_data)))
                
                # Percentiles
                features[f'{ch_name}_p25'] = np.percentile(signal_data, 25)
                features[f'{ch_name}_p75'] = np.percentile(signal_data, 75)
                features[f'{ch_name}_iqr'] = features[f'{ch_name}_p75'] - features[f'{ch_name}_p25']
            
            # === Amplitude and Derivative Features ===
            for ch_name, signal_data in [('eog1', eog1), ('eog2', eog2), ('diff', eog_diff)]:
                # First and second derivatives
                first_deriv = np.diff(signal_data)
                second_deriv = np.diff(first_deriv)
                
                features[f'{ch_name}_deriv1_mean'] = np.mean(first_deriv)
                features[f'{ch_name}_deriv1_std'] = np.std(first_deriv)
                features[f'{ch_name}_deriv1_max'] = np.max(np.abs(first_deriv))
                
                features[f'{ch_name}_deriv2_mean'] = np.mean(second_deriv)
                features[f'{ch_name}_deriv2_std'] = np.std(second_deriv)
                features[f'{ch_name}_deriv2_max'] = np.max(np.abs(second_deriv))
                
                # Zero crossing rate
                features[f'{ch_name}_zcr'] = len(np.where(np.diff(np.signbit(signal_data)))[0])
                
                # MAD-scaled amplitude (robust to outliers)
                mad = np.median(np.abs(signal_data - np.median(signal_data)))
                if mad > 0:
                    features[f'{ch_name}_mad_scaled_max'] = np.max(np.abs(signal_data)) / mad
                else:
                    features[f'{ch_name}_mad_scaled_max'] = 0
            
            # === Frequency Domain Features ===
            for ch_name, signal_data in [('eog1', eog1), ('eog2', eog2), ('diff', eog_diff)]:
                # Power spectral density
                freqs, psd = signal.welch(signal_data, fs=self.srate, nperseg=min(256, len(signal_data)))
                
                # Frequency bands (typical for EOG)
                delta_band = (freqs >= 0.5) & (freqs < 4)
                theta_band = (freqs >= 4) & (freqs < 8)
                alpha_band = (freqs >= 8) & (freqs < 13)
                beta_band = (freqs >= 13) & (freqs < 30)
                
                features[f'{ch_name}_delta_power'] = np.sum(psd[delta_band])
                features[f'{ch_name}_theta_power'] = np.sum(psd[theta_band])
                features[f'{ch_name}_alpha_power'] = np.sum(psd[alpha_band])
                features[f'{ch_name}_beta_power'] = np.sum(psd[beta_band])
                features[f'{ch_name}_total_power'] = np.sum(psd)
                
                # Relative power
                if features[f'{ch_name}_total_power'] > 0:
                    features[f'{ch_name}_delta_rel'] = features[f'{ch_name}_delta_power'] / features[f'{ch_name}_total_power']
                    features[f'{ch_name}_theta_rel'] = features[f'{ch_name}_theta_power'] / features[f'{ch_name}_total_power']
                    features[f'{ch_name}_alpha_rel'] = features[f'{ch_name}_alpha_power'] / features[f'{ch_name}_total_power']
                    features[f'{ch_name}_beta_rel'] = features[f'{ch_name}_beta_power'] / features[f'{ch_name}_total_power']
                else:
                    features[f'{ch_name}_delta_rel'] = 0
                    features[f'{ch_name}_theta_rel'] = 0
                    features[f'{ch_name}_alpha_rel'] = 0
                    features[f'{ch_name}_beta_rel'] = 0
                
                # Spectral centroid and bandwidth
                if np.sum(psd) > 0:
                    features[f'{ch_name}_spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
                    features[f'{ch_name}_spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features[f'{ch_name}_spectral_centroid']) ** 2) * psd) / np.sum(psd))
                else:
                    features[f'{ch_name}_spectral_centroid'] = 0
                    features[f'{ch_name}_spectral_bandwidth'] = 0
            
            # === Wavelet Features ===
            for ch_name, signal_data in [('eog1', eog1), ('eog2', eog2), ('diff', eog_diff)]:
                # Discrete wavelet transform
                try:
                    coeffs = pywt.wavedec(signal_data, 'db4', level=4)
                    
                    for level, coeff in enumerate(coeffs):
                        features[f'{ch_name}_wavelet_energy_l{level}'] = np.sum(coeff ** 2)
                        features[f'{ch_name}_wavelet_mean_l{level}'] = np.mean(np.abs(coeff))
                        features[f'{ch_name}_wavelet_std_l{level}'] = np.std(coeff)
                except:
                    # If wavelet fails, set to zero
                    for level in range(5):
                        features[f'{ch_name}_wavelet_energy_l{level}'] = 0
                        features[f'{ch_name}_wavelet_mean_l{level}'] = 0
                        features[f'{ch_name}_wavelet_std_l{level}'] = 0
            
            # === DTW Distance to LRLR Template ===
            # Use differential signal for template matching
            try:
                # Downsample if window is much longer than template
                if len(eog_diff) > len(self.lrlr_template) * 2:
                    downsample_factor = len(eog_diff) // len(self.lrlr_template)
                    downsampled_diff = signal.decimate(eog_diff, downsample_factor, ftype='iir')
                else:
                    downsampled_diff = eog_diff
                
                dtw_distance = dtw.distance(downsampled_diff, self.lrlr_template)
                features['dtw_distance_to_lrlr'] = dtw_distance
            except:
                features['dtw_distance_to_lrlr'] = np.inf
            
            # === Peak Detection Features ===
            for ch_name, signal_data in [('eog1', eog1), ('eog2', eog2), ('diff', eog_diff)]:
                # Find peaks and troughs
                peaks, _ = signal.find_peaks(signal_data, height=np.std(signal_data))
                troughs, _ = signal.find_peaks(-signal_data, height=np.std(signal_data))
                
                features[f'{ch_name}_n_peaks'] = len(peaks)
                features[f'{ch_name}_n_troughs'] = len(troughs)
                features[f'{ch_name}_peak_trough_ratio'] = len(peaks) / max(len(troughs), 1)
                
                if len(peaks) > 0:
                    features[f'{ch_name}_peak_mean_amplitude'] = np.mean(signal_data[peaks])
                    features[f'{ch_name}_peak_std_amplitude'] = np.std(signal_data[peaks])
                else:
                    features[f'{ch_name}_peak_mean_amplitude'] = 0
                    features[f'{ch_name}_peak_std_amplitude'] = 0
                
                if len(troughs) > 0:
                    features[f'{ch_name}_trough_mean_amplitude'] = np.mean(-signal_data[troughs])
                    features[f'{ch_name}_trough_std_amplitude'] = np.std(-signal_data[troughs])
                else:
                    features[f'{ch_name}_trough_mean_amplitude'] = 0
                    features[f'{ch_name}_trough_std_amplitude'] = 0
            
            # === Cross-channel Features ===
            # Correlation between channels
            features['eog_channels_correlation'] = np.corrcoef(eog1, eog2)[0, 1]
            
            # Phase difference (using Hilbert transform)
            try:
                analytic1 = signal.hilbert(eog1)
                analytic2 = signal.hilbert(eog2)
                phase1 = np.angle(analytic1)
                phase2 = np.angle(analytic2)
                phase_diff = phase1 - phase2
                features['eog_phase_diff_mean'] = np.mean(phase_diff)
                features['eog_phase_diff_std'] = np.std(phase_diff)
            except:
                features['eog_phase_diff_mean'] = 0
                features['eog_phase_diff_std'] = 0
            
            # === Temporal Pattern Features ===
            # Auto-correlation at different lags
            for lag in [5, 10, 25, 50]:  # Different time lags
                if lag < len(eog_diff):
                    autocorr = np.corrcoef(eog_diff[:-lag], eog_diff[lag:])[0, 1]
                    features[f'diff_autocorr_lag_{lag}'] = autocorr
                else:
                    features[f'diff_autocorr_lag_{lag}'] = 0
            
            # Entropy measures
            for ch_name, signal_data in [('eog1', eog1), ('eog2', eog2), ('diff', eog_diff)]:
                # Approximate entropy
                try:
                    # Discretize signal for entropy calculation
                    discretized = np.digitize(signal_data, bins=np.linspace(np.min(signal_data), np.max(signal_data), 10))
                    features[f'{ch_name}_entropy'] = entropy(np.bincount(discretized))
                except:
                    features[f'{ch_name}_entropy'] = 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)


def train_gradient_boosted_models(data_path='lstm_training_data.npz', 
                                feature_save_path='lrlr_features.npz',
                                model_save_dir='models/'):
    """
    Train XGBoost and LightGBM models for LRLR detection with comprehensive feature extraction.
    
    Args:
        data_path: Path to the LSTM training data
        feature_save_path: Path to save extracted features
        model_save_dir: Directory to save trained models
    """
    
    # Create models directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    try:
        data = np.load(data_path)
        X_raw = data['X']  # Shape: (319, 750, 2)
        y = data['y']      # Shape: (319,)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    
    print(f"Raw data shape: {X_raw.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"LRLR samples (1): {np.sum(y == 1)}, Non-LRLR samples (0): {np.sum(y == 0)}")
    
    # Extract features
    feature_extractor = LRLRFeatureExtractor(srate=118)
    
    # Check if features already exist
    if os.path.exists(feature_save_path):
        print(f"Loading existing features from {feature_save_path}...")
        feature_data = np.load(feature_save_path, allow_pickle=True)
        X_features = pd.DataFrame(feature_data['features'].item())
        y_features = feature_data['labels']
    else:
        print("Extracting features...")
        X_features = feature_extractor.extract_features(X_raw)
        
        # Save features for future use
        np.savez(feature_save_path, features=X_features.to_dict(), labels=y)
        print(f"Features saved to {feature_save_path}")
    
    print(f"Extracted {X_features.shape[1]} features from {X_features.shape[0]} samples")
    print(f"Feature names: {list(X_features.columns[:10])}...")  # Show first 10 feature names
    
    # Handle any infinite or NaN values
    X_features = X_features.replace([np.inf, -np.inf], np.nan)
    X_features = X_features.fillna(0)
    
    # Prepare data for training
    X_array = X_features.values
    
    # Calculate class weights for imbalanced data
    class_weights = class_weight.compute_class_weight('balanced', 
                                                     classes=np.unique(y), 
                                                     y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class weights: {class_weight_dict}")
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # === Train XGBoost Model ===
    print("\n" + "="*50)
    print("Training XGBoost Model")
    print("="*50)
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': class_weights[1] / class_weights[0],  # Handle class imbalance
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    # Cross-validation for XGBoost
    xgb_cv_scores = cross_val_score(xgb_model, X_array, y, cv=cv, scoring='roc_auc')
    print(f"XGBoost CV AUC: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std() * 2:.4f})")
    
    # Train final XGBoost model
    xgb_model.fit(X_array, y)
    
    # Feature importance for XGBoost
    xgb_importance = pd.DataFrame({
        'feature': X_features.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 XGBoost features:")
    print(xgb_importance.head(10))
    
    # Save XGBoost model
    xgb_model_path = os.path.join(model_save_dir, 'lrlr_xgboost_model.pkl')
    joblib.dump(xgb_model, xgb_model_path)
    print(f"XGBoost model saved to {xgb_model_path}")
    
    # === Train LightGBM Model ===
    print("\n" + "="*50)
    print("Training LightGBM Model")
    print("="*50)
    
    # LightGBM parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'class_weight': 'balanced',
        'verbose': -1
    }
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    
    # Cross-validation for LightGBM
    lgb_cv_scores = cross_val_score(lgb_model, X_array, y, cv=cv, scoring='roc_auc')
    print(f"LightGBM CV AUC: {lgb_cv_scores.mean():.4f} (+/- {lgb_cv_scores.std() * 2:.4f})")
    
    # Train final LightGBM model
    lgb_model.fit(X_array, y)
    
    # Feature importance for LightGBM
    lgb_importance = pd.DataFrame({
        'feature': X_features.columns,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 LightGBM features:")
    print(lgb_importance.head(10))
    
    # Save LightGBM model
    lgb_model_path = os.path.join(model_save_dir, 'lrlr_lightgbm_model.pkl')
    joblib.dump(lgb_model, lgb_model_path)
    print(f"LightGBM model saved to {lgb_model_path}")
    
    # === Train with SMOTE ===
    print("\n" + "="*50)
    print("Training with SMOTE (Synthetic Minority Oversampling)")
    print("="*50)
    
    # Create SMOTE pipeline
    smote = SMOTE(random_state=42)
    
    # SMOTE + XGBoost
    smote_xgb_pipeline = ImbPipeline([
        ('smote', smote),
        ('scaler', RobustScaler()),
        ('classifier', xgb.XGBClassifier(**{k: v for k, v in xgb_params.items() if k != 'scale_pos_weight'}))
    ])
    
    smote_xgb_cv_scores = cross_val_score(smote_xgb_pipeline, X_array, y, cv=cv, scoring='roc_auc')
    print(f"SMOTE + XGBoost CV AUC: {smote_xgb_cv_scores.mean():.4f} (+/- {smote_xgb_cv_scores.std() * 2:.4f})")
    
    # Train final SMOTE + XGBoost model
    smote_xgb_pipeline.fit(X_array, y)
    
    # Save SMOTE + XGBoost model
    smote_xgb_path = os.path.join(model_save_dir, 'lrlr_smote_xgboost_model.pkl')
    joblib.dump(smote_xgb_pipeline, smote_xgb_path)
    print(f"SMOTE + XGBoost model saved to {smote_xgb_path}")
    
    # === Model Comparison ===
    print("\n" + "="*50)
    print("Model Comparison Summary")
    print("="*50)
    
    results_df = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM', 'SMOTE + XGBoost'],
        'CV_AUC_Mean': [xgb_cv_scores.mean(), lgb_cv_scores.mean(), smote_xgb_cv_scores.mean()],
        'CV_AUC_Std': [xgb_cv_scores.std(), lgb_cv_scores.std(), smote_xgb_cv_scores.std()]
    })
    
    print(results_df)
    
    # Save results
    results_path = os.path.join(model_save_dir, 'lrlr_model_comparison.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Save feature importance
    xgb_importance.to_csv(os.path.join(model_save_dir, 'lrlr_xgboost_feature_importance.csv'), index=False)
    lgb_importance.to_csv(os.path.join(model_save_dir, 'lrlr_lightgbm_feature_importance.csv'), index=False)
    
    return {
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'smote_xgb_model': smote_xgb_pipeline,
        'feature_extractor': feature_extractor,
        'features': X_features,
        'results': results_df
    }


if __name__ == '__main__':
    # Train all models
    results = train_gradient_boosted_models()
    print("\nTraining completed!")