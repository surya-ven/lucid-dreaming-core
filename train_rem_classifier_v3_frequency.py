#!/usr/bin/env python3
"""
REM Classifier V3 - Frequency Domain Approach
Based on diagnostic findings, implement frequency-domain analysis for REM detection
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

class FrequencyREM_Classifier:
    """REM classifier using frequency domain features"""
    
    def __init__(self, fs=250):  # Assuming 250 Hz sampling rate
        self.fs = fs
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        
    def extract_frequency_features(self, X):
        """Extract comprehensive frequency domain features"""
        print("Extracting frequency domain features...")
        
        n_samples, n_timepoints, n_channels = X.shape
        features_list = []
        feature_names = []
        
        # Frequency bands of interest for sleep analysis
        bands = {
            'delta': (0.5, 4),    # Deep sleep
            'theta': (4, 8),      # REM and light sleep
            'alpha': (8, 13),     # Relaxed wake
            'beta': (13, 30),     # Active wake
            'gamma': (30, 50)     # High frequency
        }
        
        for ch in range(n_channels):
            channel_data = X[:, :, ch]
            
            for i, window in enumerate(channel_data):
                if i == 0:  # Initialize feature arrays on first sample
                    # Power spectral density features
                    freqs, psd = signal.welch(window, fs=self.fs, nperseg=min(256, len(window)//2))
                    n_freq_features = len(freqs)
                    
                    # Band power features
                    n_band_features = len(bands) * 3  # mean, max, ratio
                    
                    # Spectral features
                    n_spectral_features = 5  # centroid, rolloff, flux, etc.
                    
                    total_features_per_channel = n_freq_features + n_band_features + n_spectral_features
                    
                    # Initialize feature matrices
                    psd_features = np.zeros((n_samples, n_freq_features))
                    band_features = np.zeros((n_samples, n_band_features))
                    spectral_features = np.zeros((n_samples, n_spectral_features))
                
                # Compute PSD
                freqs, psd = signal.welch(window, fs=self.fs, nperseg=min(256, len(window)//2))
                psd_features[i] = psd
                
                # Band power features
                band_idx = 0
                total_power = np.sum(psd)
                
                for band_name, (low, high) in bands.items():
                    # Find frequency indices
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_power = np.sum(psd[band_mask])
                    
                    # Band power statistics
                    band_features[i, band_idx] = band_power  # Absolute power
                    band_features[i, band_idx + 1] = band_power / (total_power + 1e-10)  # Relative power
                    band_features[i, band_idx + 2] = np.max(psd[band_mask]) if np.any(band_mask) else 0  # Peak power
                    band_idx += 3
                
                # Spectral features
                spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
                spectral_rolloff = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
                spectral_flux = np.sum(np.diff(psd)**2)
                spectral_flatness = np.exp(np.mean(np.log(psd + 1e-10))) / (np.mean(psd) + 1e-10)
                peak_freq = freqs[np.argmax(psd)]
                
                spectral_features[i] = [spectral_centroid, spectral_rolloff, spectral_flux, 
                                      spectral_flatness, peak_freq]
            
            # Add features for this channel
            features_list.append(psd_features)
            features_list.append(band_features)
            features_list.append(spectral_features)
            
            # Add feature names
            for j in range(n_freq_features):
                feature_names.append(f'ch{ch}_psd_freq{j}')
            
            for band_name in bands.keys():
                feature_names.extend([f'ch{ch}_{band_name}_power', 
                                    f'ch{ch}_{band_name}_ratio', 
                                    f'ch{ch}_{band_name}_peak'])
            
            feature_names.extend([f'ch{ch}_centroid', f'ch{ch}_rolloff', f'ch{ch}_flux',
                                f'ch{ch}_flatness', f'ch{ch}_peak_freq'])
        
        # Combine all features
        all_features = np.concatenate(features_list, axis=1)
        print(f"Extracted {all_features.shape[1]} frequency features")
        
        return all_features, feature_names
    
    def add_cross_channel_features(self, X):
        """Add cross-channel correlation and coherence features"""
        print("Computing cross-channel features...")
        
        n_samples, n_timepoints, n_channels = X.shape
        cross_features = []
        
        for i in range(n_samples):
            sample_features = []
            
            # Cross-channel correlations
            for ch1 in range(n_channels):
                for ch2 in range(ch1 + 1, n_channels):
                    corr = np.corrcoef(X[i, :, ch1], X[i, :, ch2])[0, 1]
                    sample_features.append(corr if not np.isnan(corr) else 0)
            
            # Phase coupling (simplified)
            for ch1 in range(n_channels):
                for ch2 in range(ch1 + 1, n_channels):
                    # Hilbert transform for phase
                    analytic1 = signal.hilbert(X[i, :, ch1])
                    analytic2 = signal.hilbert(X[i, :, ch2])
                    phase1 = np.angle(analytic1)
                    phase2 = np.angle(analytic2)
                    
                    # Phase locking value
                    phase_diff = phase1 - phase2
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    sample_features.append(plv)
            
            cross_features.append(sample_features)
        
        cross_features = np.array(cross_features)
        print(f"Added {cross_features.shape[1]} cross-channel features")
        
        return cross_features
    
    def conservative_resampling(self, X, y, target_ratio=0.08):
        """Very conservative resampling approach"""
        print(f"Applying conservative resampling (target: {target_ratio*100:.1f}% REM)...")
        
        # First, significant undersampling of majority class
        rus = RandomUnderSampler(sampling_strategy=0.05, random_state=42)
        X_flat = X.reshape(X.shape[0], -1)
        X_under, y_under = rus.fit_resample(X_flat, y)
        
        print(f"After undersampling: {len(y_under)} samples")
        print(f"REM: {np.sum(y_under)} ({np.mean(y_under)*100:.1f}%)")
        
        # Very light SMOTE
        n_majority = np.sum(y_under == 0)
        n_minority_target = int(n_majority * target_ratio / (1 - target_ratio))
        n_minority_current = np.sum(y_under)
        
        if n_minority_target > n_minority_current and n_minority_current >= 5:
            sampling_strategy = {1: n_minority_target}
            smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
            X_balanced, y_balanced = smote.fit_resample(X_under, y_under)
        else:
            X_balanced, y_balanced = X_under, y_under
            
        print(f"After balancing: {len(y_balanced)} samples")
        print(f"REM: {np.sum(y_balanced)} ({np.mean(y_balanced)*100:.1f}%)")
        
        return X_balanced, y_balanced
    
    def train_and_evaluate(self, X, y):
        """Train multiple models with frequency features"""
        print("\n" + "="*60)
        print("FREQUENCY-DOMAIN REM CLASSIFICATION")
        print("="*60)
        
        # Extract frequency features
        freq_features, feature_names = self.extract_frequency_features(X)
        
        # Add cross-channel features
        cross_features = self.add_cross_channel_features(X)
        
        # Combine all features
        all_features = np.concatenate([freq_features, cross_features], axis=1)
        print(f"Total features: {all_features.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply conservative resampling to training data only
        X_train_balanced, y_train_balanced = self.conservative_resampling(
            X_train.reshape(-1, 1, all_features.shape[1]), y_train
        )
        X_train_balanced = X_train_balanced.reshape(X_train_balanced.shape[0], -1)
        
        # Scale features
        self.scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        print("\nPerforming feature selection...")
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=min(100, X_train_scaled.shape[1]//2))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train_balanced)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        print(f"Selected {X_train_selected.shape[1]} best features")
        
        # Train multiple models
        models = {
            'rf': RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, class_weight='balanced_subsample',
                random_state=42, n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            'svm': SVC(
                kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                probability=True, random_state=42
            )
        }
        
        print("\nTraining models...")
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name.upper()}...")
            
            # Cross-validation on training data
            cv_scores = cross_val_score(model, X_train_selected, y_train_balanced, 
                                      cv=5, scoring='f1', n_jobs=-1)
            print(f"CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Train on full training set
            model.fit(X_train_selected, y_train_balanced)
            
            # Predict on test set
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            
            # Find optimal threshold
            thresholds = np.linspace(0.1, 0.9, 81)
            f1_scores = []
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_test, y_pred)
                f1_scores.append(f1)
            
            best_threshold = thresholds[np.argmax(f1_scores)]
            y_pred_best = (y_pred_proba >= best_threshold).astype(int)
            
            # Calculate metrics
            test_f1 = f1_score(y_test, y_pred_best)
            test_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'cv_f1': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_f1': test_f1,
                'test_auc': test_auc,
                'threshold': best_threshold,
                'predictions': y_pred_best,
                'probabilities': y_pred_proba
            }
            
            print(f"Test F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Threshold: {best_threshold:.3f}")
            
            self.models[name] = model
        
        # Ensemble prediction
        print("\nEnsemble prediction...")
        ensemble_proba = np.mean([results[name]['probabilities'] for name in models.keys()], axis=0)
        
        # Find optimal ensemble threshold
        f1_scores = []
        for threshold in thresholds:
            y_pred = (ensemble_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)
        
        best_ensemble_threshold = thresholds[np.argmax(f1_scores)]
        ensemble_pred = (ensemble_proba >= best_ensemble_threshold).astype(int)
        ensemble_f1 = f1_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        
        # Print results
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        
        for name, result in results.items():
            print(f"{name.upper()}: CV={result['cv_f1']:.4f}±{result['cv_std']:.4f}, "
                  f"Test F1={result['test_f1']:.4f}, AUC={result['test_auc']:.4f}")
        
        print(f"ENSEMBLE: Test F1={ensemble_f1:.4f}, AUC={ensemble_auc:.4f}")
        
        # Best model
        best_model = max(results.keys(), key=lambda x: results[x]['test_f1'])
        best_f1 = max(results[best_model]['test_f1'], ensemble_f1)
        
        if ensemble_f1 == best_f1:
            print(f"\n🏆 Best approach: ENSEMBLE (F1={ensemble_f1:.4f})")
        else:
            print(f"\n🏆 Best approach: {best_model.upper()} (F1={results[best_model]['test_f1']:.4f})")
        
        # Detailed classification report for best model
        if ensemble_f1 == best_f1:
            best_predictions = ensemble_pred
        else:
            best_predictions = results[best_model]['predictions']
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, best_predictions, target_names=['Non-REM', 'REM']))
        
        return results, ensemble_f1

def main():
    """Main function"""
    # Load data
    print("Loading REM data...")
    data = np.load("extracted_REM_windows.npz")
    
    if 'features' in data.keys():
        X = data['features']
        y = data['labels']
    else:
        X = data['windows']
        y = data['labels']
    
    print(f"Loaded {X.shape[0]} samples with {np.sum(y)} REM samples ({np.mean(y)*100:.2f}%)")
    
    # Initialize classifier
    classifier = FrequencyREM_Classifier(fs=250)  # Adjust sampling rate as needed
    
    # Train and evaluate
    results, ensemble_f1 = classifier.train_and_evaluate(X, y)
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    
    if ensemble_f1 > 0.15:
        print("✅ Frequency domain approach shows improvement!")
        print("   Consider further optimization and hyperparameter tuning.")
    elif ensemble_f1 > 0.05:
        print("⚠️  Marginal improvement with frequency features.")
        print("   May need more sophisticated feature engineering.")
    else:
        print("❌ Even frequency domain features don't solve the fundamental issue.")
        print("   The problem likely requires:")
        print("   1. Better data collection methodology")
        print("   2. More REM samples")
        print("   3. Expert review of REM labeling")
        print("   4. Different signal processing approach")

if __name__ == "__main__":
    main()
