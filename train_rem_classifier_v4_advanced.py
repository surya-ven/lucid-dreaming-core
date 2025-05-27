#!/usr/bin/env python3
"""
REM Classifier V4 - Advanced Feature Engineering
Implementation of next-phase recommendations
"""

import numpy as np
from scipy import signal
import pywt  # For wavelet transforms
from sklearn.ensemble import RandomForestClassifier

class AdvancedREM_Classifier:
    """Next-generation REM classifier with advanced features"""
    
    def __init__(self, fs=250):
        self.fs = fs
        
    def hjorth_parameters(self, signal):
        """Calculate Hjorth parameters (Activity, Mobility, Complexity)"""
        # Activity (variance)
        activity = np.var(signal)
        
        # Mobility (mean frequency)
        diff1 = np.diff(signal)
        mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
        
        # Complexity (frequency bandwidth)
        diff2 = np.diff(diff1)
        complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if np.var(diff1) > 0 and mobility > 0 else 0
        
        return activity, mobility, complexity
    
    def sample_entropy(self, signal, m=2, r=0.2):
        """Calculate sample entropy (signal complexity measure)"""
        N = len(signal)
        patterns = np.array([signal[i:i+m] for i in range(N-m+1)])
        
        # Calculate distances
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        phi = np.zeros(2)
        for i in [m, m+1]:
            patterns = np.array([signal[j:j+i] for j in range(N-i+1)])
            C = np.zeros(N-i+1)
            
            for j in range(N-i+1):
                template = patterns[j]
                matches = sum([1 for k in range(N-i+1) if k != j and _maxdist(template, patterns[k]) <= r])
                C[j] = matches / (N-i)
            
            phi[i-m] = np.mean(C)
        
        return -np.log(phi[1] / phi[0]) if phi[0] > 0 and phi[1] > 0 else 0
    
    def wavelet_features(self, signal):
        """Extract wavelet-based features"""
        # Discrete wavelet transform
        coeffs = pywt.wavedec(signal, 'db4', level=4)
        
        features = []
        for coeff in coeffs:
            features.extend([
                np.mean(coeff),
                np.std(coeff),
                np.var(coeff),
                np.sum(coeff**2)  # Energy
            ])
        
        return features
    
    def sleep_specific_features(self, signal):
        """Extract sleep-specific patterns"""
        features = []
        
        # Sleep spindle characteristics (11-15 Hz)
        # K-complex detection (slow wave components)
        # Delta power ratio
        # Theta/Alpha ratio
        
        # Placeholder for now - implement based on sleep literature
        freqs, psd = signal.welch(signal, fs=self.fs)
        
        # Sleep spindle band (11-15 Hz)
        spindle_mask = (freqs >= 11) & (freqs <= 15)
        spindle_power = np.sum(psd[spindle_mask])
        
        # Delta band (0.5-4 Hz) 
        delta_mask = (freqs >= 0.5) & (freqs <= 4)
        delta_power = np.sum(psd[delta_mask])
        
        # Theta band (4-8 Hz) - important for REM
        theta_mask = (freqs >= 4) & (freqs <= 8)
        theta_power = np.sum(psd[theta_mask])
        
        total_power = np.sum(psd)
        
        features.extend([
            spindle_power / (total_power + 1e-10),
            delta_power / (total_power + 1e-10),
            theta_power / (total_power + 1e-10),
            theta_power / (delta_power + 1e-10)  # Theta/Delta ratio
        ])
        
        return features
    
    def extract_all_features(self, X):
        """Extract comprehensive feature set"""
        n_samples, n_timepoints, n_channels = X.shape
        all_features = []
        
        for i in range(n_samples):
            sample_features = []
            
            for ch in range(n_channels):
                signal = X[i, :, ch]
                
                # Hjorth parameters
                activity, mobility, complexity = self.hjorth_parameters(signal)
                sample_features.extend([activity, mobility, complexity])
                
                # Sample entropy
                se = self.sample_entropy(signal)
                sample_features.append(se)
                
                # Wavelet features
                wavelet_feats = self.wavelet_features(signal)
                sample_features.extend(wavelet_feats)
                
                # Sleep-specific features
                sleep_feats = self.sleep_specific_features(signal)
                sample_features.extend(sleep_feats)
            
            all_features.append(sample_features)
        
        return np.array(all_features)

# Usage example:
# classifier = AdvancedREM_Classifier()
# features = classifier.extract_all_features(X)
# Train with these enhanced features
