import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import yasa
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib
matplotlib.use('Agg')

print("=== Comprehensive High-Confidence REM Analysis Across All 20 Nights ===")

# Initialize results storage
all_results = []
all_probabilities = []
all_ground_truth = []

# Process each night to get actual probabilities
for night in range(1, 21):
    print(f"\nProcessing Night {night}...")
    
    try:
        # Load data
        edf_file = f'provided_data/night_{night:02d}.edf'
        label_file = f'provided_data/night_{night:02d}_label.csv'
        
        # Load ground truth
        labels_df = pd.read_csv(label_file)
        
        # Load EEG and get YASA predictions
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        raw.rename_channels({'LF-FpZ': 'Fp1', 'RF-FpZ': 'Fp2'})
        sls = yasa.SleepStaging(raw, eeg_name='Fp2')
        yasa_predictions = sls.predict()
        yasa_proba = sls.predict_proba()
        
        # Align lengths
        min_length = min(len(labels_df) - 1, len(yasa_predictions))
        ground_truth = labels_df['Sleep stage'].iloc[1:min_length+1].values
        yasa_pred = yasa_predictions[:min_length]
        yasa_prob = yasa_proba.iloc[:min_length]
        
        # Create binary REM classification
        ground_truth_rem = (ground_truth == 'REM').astype(int)
        rem_probabilities = yasa_prob['R'].values
        
        # Store for global analysis
        all_probabilities.extend(rem_probabilities)
        all_ground_truth.extend(ground_truth_rem)
        
        # Calculate night-specific metrics
        total_rem_epochs = ground_truth_rem.sum()
        
        # Count REM episodes
        rem_epochs_idx = np.where(ground_truth_rem == 1)[0]
        rem_episodes = 0
        if len(rem_epochs_idx) > 0:
            episode_start = rem_epochs_idx[0]
            for i in range(1, len(rem_epochs_idx)):
                if rem_epochs_idx[i] != rem_epochs_idx[i-1] + 1:
                    rem_episodes += 1
                    episode_start = rem_epochs_idx[i]
            rem_episodes += 1  # Count the last episode
        
        # Calculate ROC
        if total_rem_epochs > 0:
            fpr, tpr, thresholds = roc_curve(ground_truth_rem, rem_probabilities)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = 0.5
        
        # Store results
        night_result = {
            'night': night,
            'total_epochs': min_length,
            'duration_hours': min_length / 120,
            'rem_epochs_total': total_rem_epochs,
            'rem_episodes_count': rem_episodes,
            'rem_percentage': (total_rem_epochs / min_length) * 100,
            'roc_auc': roc_auc,
            'mean_rem_probability': rem_probabilities.mean(),
            'rem_prob_when_true': rem_probabilities[ground_truth_rem == 1].mean() if total_rem_epochs > 0 else 0,
            'rem_prob_when_false': rem_probabilities[ground_truth_rem == 0].mean(),
            'probabilities': rem_probabilities,
            'ground_truth': ground_truth_rem
        }
        
        all_results.append(night_result)
        
    except Exception as e:
        print(f"Error processing night {night}: {e}")
        continue

print(f"\nSuccessfully processed {len(all_results)} nights")

# Convert to arrays for global analysis
all_probabilities = np.array(all_probabilities)
all_ground_truth = np.array(all_ground_truth)

print(f"Total epochs across all nights: {len(all_probabilities)}")
print(f"Total REM epochs: {all_ground_truth.sum()}")
print(f"Overall REM percentage: {all_ground_truth.mean() * 100:.1f}%")

# Calculate global ROC curve
fpr_global, tpr_global, thresholds_global = roc_curve(all_ground_truth, all_probabilities)
roc_auc_global = auc(fpr_global, tpr_global)

print(f"Global ROC AUC: {roc_auc_global:.3f}")

# Find thresholds that achieve target precisions
# For high confidence thresholds, we want high precision
precision_global, recall_global, pr_thresholds = precision_recall_curve(all_ground_truth, all_probabilities)

# Find thresholds for 90% and 95% precision
target_precisions = [0.90, 0.95]
optimal_thresholds = {}

for target_precision in target_precisions:
    # Find threshold that gives closest to target precision
    valid_indices = precision_global >= target_precision
    if np.any(valid_indices):
        # Among valid precisions, find the one with highest recall
        valid_recalls = recall_global[valid_indices]
        valid_thresholds = pr_thresholds[valid_indices[:-1]]  # pr_thresholds is one element shorter
        
        if len(valid_thresholds) > 0:
            best_idx = np.argmax(valid_recalls)
            threshold = valid_thresholds[best_idx]
            achieved_precision = precision_global[valid_indices][best_idx]
            achieved_recall = valid_recalls[best_idx]
        else:
            # Use highest threshold if no valid ones found
            threshold = pr_thresholds[-1]
            achieved_precision = precision_global[-2]
            achieved_recall = recall_global[-2]
    else:
        # Use highest threshold if target precision not achievable
        threshold = pr_thresholds[-1] if len(pr_thresholds) > 0 else 0.95
        achieved_precision = precision_global[-2] if len(precision_global) > 1 else 0
        achieved_recall = recall_global[-2] if len(recall_global) > 1 else 0
    
    optimal_thresholds[target_precision] = {
        'threshold': threshold,
        'precision': achieved_precision,
        'recall': achieved_recall
    }
    
    print(f"\nFor {target_precision*100:.0f}% target precision:")
    print(f"  Optimal threshold: {threshold:.4f}")
    print(f"  Achieved precision: {achieved_precision:.3f}")
    print(f"  Achieved recall: {achieved_recall:.3f}")

# Apply thresholds to each night and calculate performance
conf_90_threshold = optimal_thresholds[0.90]['threshold']
conf_95_threshold = optimal_thresholds[0.95]['threshold']

print(f"\nFixed thresholds to use:")
print(f"90% confidence threshold: {conf_90_threshold:.4f}")
print(f"95% confidence threshold: {conf_95_threshold:.4f}")

# Calculate performance for each night using fixed thresholds
results_df = pd.DataFrame()
all_90_preds = []
all_95_preds = []
all_gt = []

for result in all_results:
    night = result['night']
    probs = result['probabilities']
    gt = result['ground_truth']
    
    # Apply fixed thresholds
    pred_90 = (probs >= conf_90_threshold).astype(int)
    pred_95 = (probs >= conf_95_threshold).astype(int)
    
    # Store for global confusion matrix
    all_90_preds.extend(pred_90)
    all_95_preds.extend(pred_95)
    all_gt.extend(gt)
    
    # Calculate metrics for 90% threshold
    tp_90 = np.sum((gt == 1) & (pred_90 == 1))
    fp_90 = np.sum((gt == 0) & (pred_90 == 1))
    tn_90 = np.sum((gt == 0) & (pred_90 == 0))
    fn_90 = np.sum((gt == 1) & (pred_90 == 0))
    
    precision_90 = tp_90 / (tp_90 + fp_90) if (tp_90 + fp_90) > 0 else 0
    recall_90 = tp_90 / (tp_90 + fn_90) if (tp_90 + fn_90) > 0 else 0
    accuracy_90 = (tp_90 + tn_90) / len(gt)
    f1_90 = 2 * precision_90 * recall_90 / (precision_90 + recall_90) if (precision_90 + recall_90) > 0 else 0
    
    # Calculate metrics for 95% threshold
    tp_95 = np.sum((gt == 1) & (pred_95 == 1))
    fp_95 = np.sum((gt == 0) & (pred_95 == 1))
    tn_95 = np.sum((gt == 0) & (pred_95 == 0))
    fn_95 = np.sum((gt == 1) & (pred_95 == 0))
    
    precision_95 = tp_95 / (tp_95 + fp_95) if (tp_95 + fp_95) > 0 else 0
    recall_95 = tp_95 / (tp_95 + fn_95) if (tp_95 + fn_95) > 0 else 0
    accuracy_95 = (tp_95 + tn_95) / len(gt)
    f1_95 = 2 * precision_95 * recall_95 / (precision_95 + recall_95) if (precision_95 + recall_95) > 0 else 0
    
    # Store results
    night_metrics = {
        'night': night,
        'total_epochs': result['total_epochs'],
        'rem_epochs': result['rem_epochs_total'],
        'rem_percentage': result['rem_percentage'],
        'roc_auc': result['roc_auc'],
        
        # 90% confidence metrics
        'conf_90_tp': tp_90,
        'conf_90_fp': fp_90,
        'conf_90_tn': tn_90,
        'conf_90_fn': fn_90,
        'conf_90_precision': precision_90,
        'conf_90_recall': recall_90,
        'conf_90_accuracy': accuracy_90,
        'conf_90_f1': f1_90,
        'conf_90_detections': tp_90 + fp_90,
        
        # 95% confidence metrics
        'conf_95_tp': tp_95,
        'conf_95_fp': fp_95,
        'conf_95_tn': tn_95,
        'conf_95_fn': fn_95,
        'conf_95_precision': precision_95,
        'conf_95_recall': recall_95,
        'conf_95_accuracy': accuracy_95,
        'conf_95_f1': f1_95,
        'conf_95_detections': tp_95 + fp_95,
    }
    
    results_df = pd.concat([results_df, pd.DataFrame([night_metrics])], ignore_index=True)

# Convert to arrays for global metrics
all_90_preds = np.array(all_90_preds)
all_95_preds = np.array(all_95_preds)
all_gt = np.array(all_gt)

# Global confusion matrices
cm_90 = confusion_matrix(all_gt, all_90_preds)
cm_95 = confusion_matrix(all_gt, all_95_preds)

print(f"\n=== Global Performance Summary ===")
print(f"90% Confidence Threshold ({conf_90_threshold:.4f}):")
print(f"  Global Precision: {cm_90[1,1] / (cm_90[1,1] + cm_90[0,1]) if (cm_90[1,1] + cm_90[0,1]) > 0 else 0:.3f}")
print(f"  Global Recall: {cm_90[1,1] / (cm_90[1,0] + cm_90[1,1]):.3f}")
print(f"  Global Accuracy: {(cm_90[0,0] + cm_90[1,1]) / cm_90.sum():.3f}")
print(f"  Total detections: {cm_90[1,1] + cm_90[0,1]}")
print(f"  True positives: {cm_90[1,1]}")

print(f"\n95% Confidence Threshold ({conf_95_threshold:.4f}):")
print(f"  Global Precision: {cm_95[1,1] / (cm_95[1,1] + cm_95[0,1]) if (cm_95[1,1] + cm_95[0,1]) > 0 else 0:.3f}")
print(f"  Global Recall: {cm_95[1,1] / (cm_95[1,0] + cm_95[1,1]):.3f}")
print(f"  Global Accuracy: {(cm_95[0,0] + cm_95[1,1]) / cm_95.sum():.3f}")
print(f"  Total detections: {cm_95[1,1] + cm_95[0,1]}")
print(f"  True positives: {cm_95[1,1]}")

# Average per-night performance
print(f"\n=== Average Per-Night Performance ===")
print(f"90% Confidence:")
print(f"  Precision: {results_df['conf_90_precision'].mean():.3f} ± {results_df['conf_90_precision'].std():.3f}")
print(f"  Recall: {results_df['conf_90_recall'].mean():.3f} ± {results_df['conf_90_recall'].std():.3f}")
print(f"  F1-Score: {results_df['conf_90_f1'].mean():.3f} ± {results_df['conf_90_f1'].std():.3f}")
print(f"  Nights with detections: {(results_df['conf_90_detections'] > 0).sum()}/20")

print(f"\n95% Confidence:")
print(f"  Precision: {results_df['conf_95_precision'].mean():.3f} ± {results_df['conf_95_precision'].std():.3f}")
print(f"  Recall: {results_df['conf_95_recall'].mean():.3f} ± {results_df['conf_95_recall'].std():.3f}")
print(f"  F1-Score: {results_df['conf_95_f1'].mean():.3f} ± {results_df['conf_95_f1'].std():.3f}")
print(f"  Nights with detections: {(results_df['conf_95_detections'] > 0).sum()}/20")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('High-Confidence REM Detection Analysis: 90% vs 95% Thresholds', fontsize=16, fontweight='bold')

# 1. Global ROC Curve with threshold markers
axes[0, 0].plot(fpr_global, tpr_global, color='blue', lw=3, label=f'ROC Curve (AUC = {roc_auc_global:.3f})')
axes[0, 0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)

# Mark the 90% and 95% thresholds on ROC curve
for threshold, color, label in [(conf_90_threshold, 'orange', '90% Conf'), 
                                (conf_95_threshold, 'red', '95% Conf')]:
    # Find closest threshold in ROC data
    threshold_idx = np.argmin(np.abs(thresholds_global - threshold))
    if threshold_idx < len(fpr_global):
        axes[0, 0].scatter(fpr_global[threshold_idx], tpr_global[threshold_idx], 
                          color=color, s=100, label=f'{label} (t={threshold:.3f})', zorder=5)

axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('Global ROC Curve with Confidence Thresholds')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Confusion Matrix for 90% confidence
sns.heatmap(cm_90, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], 
            xticklabels=['Non-REM', 'REM'], yticklabels=['Non-REM', 'REM'])
axes[0, 1].set_title(f'90% Confidence Confusion Matrix\n(Threshold: {conf_90_threshold:.3f})')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# 3. Confusion Matrix for 95% confidence
sns.heatmap(cm_95, annot=True, fmt='d', cmap='Reds', ax=axes[0, 2],
            xticklabels=['Non-REM', 'REM'], yticklabels=['Non-REM', 'REM'])
axes[0, 2].set_title(f'95% Confidence Confusion Matrix\n(Threshold: {conf_95_threshold:.3f})')
axes[0, 2].set_xlabel('Predicted')
axes[0, 2].set_ylabel('Actual')

# 4. Per-night precision comparison
axes[1, 0].scatter(results_df['night'], results_df['conf_90_precision'], 
                  color='orange', s=60, alpha=0.8, label='90% Confidence')
axes[1, 0].scatter(results_df['night'], results_df['conf_95_precision'], 
                  color='red', s=60, alpha=0.8, label='95% Confidence')
axes[1, 0].axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% Target')
axes[1, 0].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Target')
axes[1, 0].set_xlabel('Night')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Per-Night Precision Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(range(1, 21, 2))

# 5. Per-night recall comparison
axes[1, 1].scatter(results_df['night'], results_df['conf_90_recall'], 
                  color='orange', s=60, alpha=0.8, label='90% Confidence')
axes[1, 1].scatter(results_df['night'], results_df['conf_95_recall'], 
                  color='red', s=60, alpha=0.8, label='95% Confidence')
axes[1, 1].set_xlabel('Night')
axes[1, 1].set_ylabel('Recall (Sensitivity)')
axes[1, 1].set_title('Per-Night Recall Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(range(1, 21, 2))

# 6. Detection counts by night
width = 0.35
nights = results_df['night']
x = np.arange(len(nights))

bars1 = axes[1, 2].bar(x - width/2, results_df['conf_90_detections'], width, 
                      label='90% Conf Detections', color='orange', alpha=0.8)
bars2 = axes[1, 2].bar(x + width/2, results_df['conf_95_detections'], width,
                      label='95% Conf Detections', color='red', alpha=0.8)

# Add REM ground truth line
axes[1, 2].plot(x, results_df['rem_epochs'], 'b-o', linewidth=2, markersize=4, 
               label='True REM Epochs', alpha=0.9)

axes[1, 2].set_xlabel('Night')
axes[1, 2].set_ylabel('Epoch Count')
axes[1, 2].set_title('Detection Counts vs Ground Truth')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_xticks(x[::2])
axes[1, 2].set_xticklabels(nights.iloc[::2])

plt.tight_layout()
plt.savefig('comprehensive_confidence_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nComprehensive analysis plot saved as comprehensive_confidence_analysis.png")
plt.close()

# Save detailed results
results_df.to_csv('confidence_threshold_analysis_results.csv', index=False)
print(f"Detailed results saved as confidence_threshold_analysis_results.csv")

# Create summary report
summary = {
    'analysis_type': 'High-Confidence REM Detection',
    'total_nights': len(all_results),
    'total_epochs': len(all_probabilities),
    'total_rem_epochs': all_ground_truth.sum(),
    'global_roc_auc': roc_auc_global,
    
    'conf_90_threshold': conf_90_threshold,
    'conf_90_global_precision': cm_90[1,1] / (cm_90[1,1] + cm_90[0,1]) if (cm_90[1,1] + cm_90[0,1]) > 0 else 0,
    'conf_90_global_recall': cm_90[1,1] / (cm_90[1,0] + cm_90[1,1]),
    'conf_90_avg_precision': results_df['conf_90_precision'].mean(),
    'conf_90_avg_recall': results_df['conf_90_recall'].mean(),
    'conf_90_nights_with_detections': (results_df['conf_90_detections'] > 0).sum(),
    
    'conf_95_threshold': conf_95_threshold,
    'conf_95_global_precision': cm_95[1,1] / (cm_95[1,1] + cm_95[0,1]) if (cm_95[1,1] + cm_95[0,1]) > 0 else 0,
    'conf_95_global_recall': cm_95[1,1] / (cm_95[1,0] + cm_95[1,1]),
    'conf_95_avg_precision': results_df['conf_95_precision'].mean(),
    'conf_95_avg_recall': results_df['conf_95_recall'].mean(),
    'conf_95_nights_with_detections': (results_df['conf_95_detections'] > 0).sum(),
}

# Save summary
summary_df = pd.DataFrame([summary])
summary_df.to_csv('confidence_analysis_summary.csv', index=False)

print(f"\n=== FINAL RECOMMENDATIONS ===")
print(f"For 90% confidence REM detection, use threshold: {conf_90_threshold:.4f}")
print(f"  - Expected precision: ~{summary['conf_90_global_precision']:.1%}")
print(f"  - Expected recall: ~{summary['conf_90_global_recall']:.1%}")
print(f"  - Will trigger on {summary['conf_90_nights_with_detections']}/20 nights")

print(f"\nFor 95% confidence REM detection, use threshold: {conf_95_threshold:.4f}")
print(f"  - Expected precision: ~{summary['conf_95_global_precision']:.1%}")
print(f"  - Expected recall: ~{summary['conf_95_global_recall']:.1%}")
print(f"  - Will trigger on {summary['conf_95_nights_with_detections']}/20 nights")

print(f"\nGlobal ROC AUC: {roc_auc_global:.3f} - indicates good discriminative ability")
print(f"Analysis complete! Check comprehensive_confidence_analysis.png for visualizations.")
