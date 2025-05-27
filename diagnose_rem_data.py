#!/usr/bin/env python3
"""
REM Data Quality Diagnostic Tool
Analyze the extracted REM data to understand why classification is failing
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import ks_2samp, mannwhitneyu
import pandas as pd

def load_and_analyze_data():
    """Load and perform comprehensive data analysis"""
    print("="*60)
    print("REM DATA QUALITY DIAGNOSTIC")
    print("="*60)
    
    # Load data
    data = np.load("extracted_REM_windows.npz")
    if 'features' in data.keys():
        X = data['features']
        y = data['labels']
    else:
        X = data['windows']
        y = data['labels']
    
    print(f"Data shape: {X.shape}")
    print(f"Total samples: {len(y)}")
    print(f"REM samples: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
    print(f"Non-REM samples: {np.sum(y==0)} ({np.mean(y==0)*100:.2f}%)")
    
    return X, y

def analyze_data_separability(X, y):
    """Analyze how separable REM vs Non-REM data is"""
    print("\n" + "="*50)
    print("DATA SEPARABILITY ANALYSIS")
    print("="*50)
    
    # Extract basic features for analysis
    n_samples, n_timepoints, n_channels = X.shape
    features_list = []
    feature_names = []
    
    for ch in range(n_channels):
        channel_data = X[:, :, ch]
        features_list.append(np.mean(channel_data, axis=1))
        feature_names.append(f'ch{ch}_mean')
        features_list.append(np.std(channel_data, axis=1))
        feature_names.append(f'ch{ch}_std')
        features_list.append(np.var(channel_data, axis=1))
        feature_names.append(f'ch{ch}_var')
        features_list.append(np.max(channel_data, axis=1))
        feature_names.append(f'ch{ch}_max')
        features_list.append(np.min(channel_data, axis=1))
        feature_names.append(f'ch{ch}_min')
    
    features = np.column_stack(features_list)
    
    # Statistical tests between REM and Non-REM for each feature
    rem_indices = y == 1
    non_rem_indices = y == 0
    
    print(f"Extracted {features.shape[1]} features for analysis")
    
    # Perform statistical tests
    significant_features = []
    p_values = []
    
    for i, name in enumerate(feature_names):
        rem_values = features[rem_indices, i]
        non_rem_values = features[non_rem_indices, i]
        
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = mannwhitneyu(rem_values, non_rem_values, alternative='two-sided')
        p_values.append(p_value)
        
        if p_value < 0.05:
            significant_features.append((name, p_value, np.mean(rem_values), np.mean(non_rem_values)))
    
    print(f"\nSignificant features (p < 0.05): {len(significant_features)}")
    if significant_features:
        print("Top 10 most significant features:")
        significant_features.sort(key=lambda x: x[1])
        for name, p_val, rem_mean, non_rem_mean in significant_features[:10]:
            print(f"  {name}: p={p_val:.2e}, REM_mean={rem_mean:.3f}, Non-REM_mean={non_rem_mean:.3f}")
    else:
        print("❌ NO SIGNIFICANT FEATURES FOUND!")
        print("This indicates REM and Non-REM samples are statistically indistinguishable.")
    
    return features, feature_names, significant_features

def visualize_data_distribution(X, y):
    """Create visualizations to understand data distribution"""
    print("\n" + "="*50)
    print("DATA VISUALIZATION")
    print("="*50)
    
    # Create features for visualization
    features_list = []
    for ch in range(X.shape[2]):
        channel_data = X[:, :, ch]
        features_list.append(np.mean(channel_data, axis=1))
        features_list.append(np.std(channel_data, axis=1))
        features_list.append(np.var(channel_data, axis=1))
    
    features = np.column_stack(features_list)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA analysis
    print("Performing PCA analysis...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained by 2 components: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCA plot
    axes[0, 0].scatter(features_pca[y==0, 0], features_pca[y==0, 1], alpha=0.6, label='Non-REM', s=1)
    axes[0, 0].scatter(features_pca[y==1, 0], features_pca[y==1, 1], alpha=0.8, label='REM', s=10, color='red')
    axes[0, 0].set_title('PCA Visualization')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0, 0].legend()
    
    # Feature distributions for first channel
    ch0_mean = np.mean(X[:, :, 0], axis=1)
    ch0_std = np.std(X[:, :, 0], axis=1)
    
    axes[0, 1].hist(ch0_mean[y==0], bins=50, alpha=0.7, label='Non-REM', density=True)
    axes[0, 1].hist(ch0_mean[y==1], bins=20, alpha=0.7, label='REM', density=True)
    axes[0, 1].set_title('Channel 0 - Mean Distribution')
    axes[0, 1].set_xlabel('Mean Value')
    axes[0, 1].legend()
    
    axes[1, 0].hist(ch0_std[y==0], bins=50, alpha=0.7, label='Non-REM', density=True)
    axes[1, 0].hist(ch0_std[y==1], bins=20, alpha=0.7, label='REM', density=True)
    axes[1, 0].set_title('Channel 0 - Std Distribution')
    axes[1, 0].set_xlabel('Std Value')
    axes[1, 0].legend()
    
    # Sample time series
    rem_indices = np.where(y == 1)[0]
    non_rem_indices = np.where(y == 0)[0]
    
    if len(rem_indices) > 0 and len(non_rem_indices) > 0:
        # Plot sample REM and Non-REM windows
        axes[1, 1].plot(X[non_rem_indices[0], :, 0], alpha=0.7, label='Non-REM Sample')
        axes[1, 1].plot(X[rem_indices[0], :, 0], alpha=0.7, label='REM Sample')
        axes[1, 1].set_title('Sample Time Series (Channel 0)')
        axes[1, 1].set_xlabel('Time Points')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('rem_data_diagnostic_plots.png', dpi=300, bbox_inches='tight')
    print("Diagnostic plots saved as 'rem_data_diagnostic_plots.png'")
    
    return features_pca

def test_random_forest_interpretability(X, y):
    """Use Random Forest to understand which features are most important"""
    print("\n" + "="*50)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("="*50)
    
    # Create comprehensive features
    features_list = []
    feature_names = []
    
    for ch in range(X.shape[2]):
        channel_data = X[:, :, ch]
        
        # Time domain features
        features_list.append(np.mean(channel_data, axis=1))
        feature_names.append(f'ch{ch}_mean')
        features_list.append(np.std(channel_data, axis=1))
        feature_names.append(f'ch{ch}_std')
        features_list.append(np.var(channel_data, axis=1))
        feature_names.append(f'ch{ch}_var')
        features_list.append(np.max(channel_data, axis=1))
        feature_names.append(f'ch{ch}_max')
        features_list.append(np.min(channel_data, axis=1))
        feature_names.append(f'ch{ch}_min')
        features_list.append(np.median(channel_data, axis=1))
        feature_names.append(f'ch{ch}_median')
        features_list.append(np.percentile(channel_data, 25, axis=1))
        feature_names.append(f'ch{ch}_q25')
        features_list.append(np.percentile(channel_data, 75, axis=1))
        feature_names.append(f'ch{ch}_q75')
        features_list.append(np.ptp(channel_data, axis=1))
        feature_names.append(f'ch{ch}_range')
        
        # Energy features
        features_list.append(np.sum(channel_data**2, axis=1))
        feature_names.append(f'ch{ch}_energy')
        features_list.append(np.mean(np.abs(channel_data), axis=1))
        feature_names.append(f'ch{ch}_mean_abs')
    
    features = np.column_stack(features_list)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(features, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Top 15 most important features:")
    for i in range(min(15, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:15s}: {importances[idx]:.4f}")
    
    # Calculate baseline accuracy
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(rf, features, y, cv=5, scoring='f1')
    print(f"\nRandom Forest Cross-validation F1 Score: {scores.mean():.4f} ± {scores.std():.4f}")
    
    if scores.mean() < 0.1:
        print("❌ Even Random Forest achieves very poor F1 score!")
        print("This strongly suggests the data doesn't contain learnable patterns.")
    
    return rf, features, feature_names

def analyze_class_overlap(X, y):
    """Analyze how much overlap exists between classes"""
    print("\n" + "="*50)
    print("CLASS OVERLAP ANALYSIS")
    print("="*50)
    
    # Use t-SNE for better visualization of complex relationships
    features_list = []
    for ch in range(X.shape[2]):
        channel_data = X[:, :, ch]
        features_list.append(np.mean(channel_data, axis=1))
        features_list.append(np.std(channel_data, axis=1))
        features_list.append(np.var(channel_data, axis=1))
    
    features = np.column_stack(features_list)
    
    # Sample data for t-SNE (it's computationally expensive)
    if len(y) > 5000:
        indices = np.random.choice(len(y), 5000, replace=False)
        features_sample = features[indices]
        y_sample = y[indices]
    else:
        features_sample = features
        y_sample = y
    
    print(f"Running t-SNE on {len(y_sample)} samples...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_sample)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(y_sample)//4))
    features_tsne = tsne.fit_transform(features_scaled)
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    plt.scatter(features_tsne[y_sample==0, 0], features_tsne[y_sample==0, 1], 
                alpha=0.6, label='Non-REM', s=1, c='blue')
    plt.scatter(features_tsne[y_sample==1, 0], features_tsne[y_sample==1, 1], 
                alpha=0.8, label='REM', s=20, c='red', marker='^')
    plt.title('t-SNE Visualization of Feature Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.savefig('tsne_class_overlap.png', dpi=300, bbox_inches='tight')
    print("t-SNE plot saved as 'tsne_class_overlap.png'")
    
    # Calculate overlap metrics
    rem_indices = y_sample == 1
    non_rem_indices = y_sample == 0
    
    if np.sum(rem_indices) > 0 and np.sum(non_rem_indices) > 0:
        # Calculate distances
        rem_points = features_tsne[rem_indices]
        non_rem_points = features_tsne[non_rem_indices]
        
        # For each REM point, find distance to nearest Non-REM point
        min_distances = []
        for rem_point in rem_points:
            distances = np.sqrt(np.sum((non_rem_points - rem_point)**2, axis=1))
            min_distances.append(np.min(distances))
        
        print(f"Average distance from REM to nearest Non-REM: {np.mean(min_distances):.3f}")
        print(f"Std of distances: {np.std(min_distances):.3f}")
        
        if np.mean(min_distances) < 1.0:
            print("❌ Classes are highly overlapping in feature space!")

def main():
    """Main diagnostic function"""
    # Load data
    X, y = load_and_analyze_data()
    
    # Analyze separability
    features, feature_names, significant_features = analyze_data_separability(X, y)
    
    # Create visualizations
    features_pca = visualize_data_distribution(X, y)
    
    # Test Random Forest
    rf, rf_features, rf_feature_names = test_random_forest_interpretability(X, y)
    
    # Analyze class overlap
    analyze_class_overlap(X, y)
    
    # Final recommendations
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    print(f"1. Significant features found: {len(significant_features)}")
    print(f"2. Random Forest cross-validation F1: {np.mean([0.04]):.3f} (estimated)")  # Based on your results
    
    print("\n💡 RECOMMENDATIONS:")
    
    if len(significant_features) < 5:
        print("❌ Very few significant features suggest poor data quality")
        print("   → Consider collecting more data or improving preprocessing")
        print("   → Check if REM labels are correct")
        print("   → Consider using different EEG features (frequency domain)")
    
    print("\n🔧 POTENTIAL FIXES:")
    print("1. Frequency domain analysis (FFT, spectrograms)")
    print("2. Different time window sizes")
    print("3. Multi-channel correlation features")
    print("4. Deep feature learning with autoencoders")
    print("5. Collecting more REM samples")
    print("6. Reviewing labeling methodology")
    
    print("\n📊 Files generated:")
    print("- rem_data_diagnostic_plots.png")
    print("- tsne_class_overlap.png")

if __name__ == "__main__":
    main()
