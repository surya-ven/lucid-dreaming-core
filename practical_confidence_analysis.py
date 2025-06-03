import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import yasa
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib
matplotlib.use('Agg')

print("=== Practical High-Confidence REM Analysis ===")

# Initialize results storage
all_results = []
all_probabilities = []
all_ground_truth = []

# Process first few nights to understand probability distributions
for night in range(1, 6):  # Just first 5 nights for quick analysis
    print(f"Processing Night {night}...")
    
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
        
        # Store for analysis
        all_probabilities.extend(rem_probabilities)
        all_ground_truth.extend(ground_truth_rem)
        
    except Exception as e:
        print(f"Error processing night {night}: {e}")
        continue

# Convert to arrays
all_probabilities = np.array(all_probabilities)
all_ground_truth = np.array(all_ground_truth)

print(f"\nProbability Distribution Analysis:")
print(f"Min REM probability: {all_probabilities.min():.4f}")
print(f"Max REM probability: {all_probabilities.max():.4f}")
print(f"Mean REM probability: {all_probabilities.mean():.4f}")

# Percentiles
percentiles = [50, 75, 80, 85, 90, 95, 99, 99.5, 99.9]
for p in percentiles:
    print(f"{p}th percentile: {np.percentile(all_probabilities, p):.4f}")

# Analyze probabilities when ground truth is REM
rem_true_probs = all_probabilities[all_ground_truth == 1]
rem_false_probs = all_probabilities[all_ground_truth == 0]

print(f"\nWhen ground truth IS REM:")
print(f"  Mean: {rem_true_probs.mean():.4f}")
print(f"  Max: {rem_true_probs.max():.4f}")
print(f"  95th percentile: {np.percentile(rem_true_probs, 95):.4f}")

print(f"\nWhen ground truth is NOT REM:")
print(f"  Mean: {rem_false_probs.mean():.4f}")
print(f"  Max: {rem_false_probs.max():.4f}")
print(f"  95th percentile: {np.percentile(rem_false_probs, 95):.4f}")

# Find realistic high-confidence thresholds
# Use top 10% and top 5% of actual probabilities as proxies for "high confidence"
top_10_pct_threshold = np.percentile(all_probabilities, 90)
top_5_pct_threshold = np.percentile(all_probabilities, 95)

print(f"\nProposed High-Confidence Thresholds:")
print(f"High confidence (top 10%): {top_10_pct_threshold:.4f}")
print(f"Very high confidence (top 5%): {top_5_pct_threshold:.4f}")

# Also try thresholds based on REM-specific probabilities
if len(rem_true_probs) > 0:
    rem_median = np.median(rem_true_probs)
    rem_75th = np.percentile(rem_true_probs, 75)
    rem_90th = np.percentile(rem_true_probs, 90)
    
    print(f"\nREM-specific thresholds:")
    print(f"REM median: {rem_median:.4f}")
    print(f"REM 75th percentile: {rem_75th:.4f}")
    print(f"REM 90th percentile: {rem_90th:.4f}")

# Calculate precision-recall curve to find best precision thresholds
precision, recall, pr_thresholds = precision_recall_curve(all_ground_truth, all_probabilities)

# Find thresholds that achieve reasonable precision levels
target_precisions = [0.5, 0.6, 0.7, 0.8]
achievable_thresholds = {}

for target_precision in target_precisions:
    valid_indices = precision >= target_precision
    if np.any(valid_indices):
        valid_recalls = recall[valid_indices]
        valid_thresholds = pr_thresholds[valid_indices[:-1]]
        
        if len(valid_thresholds) > 0:
            best_idx = np.argmax(valid_recalls)
            threshold = valid_thresholds[best_idx]
            achieved_precision = precision[valid_indices][best_idx]
            achieved_recall = valid_recalls[best_idx]
            
            achievable_thresholds[target_precision] = {
                'threshold': threshold,
                'precision': achieved_precision,
                'recall': achieved_recall
            }

print(f"\nAchievable Precision Thresholds:")
for target, results in achievable_thresholds.items():
    print(f"{target*100:.0f}% precision: threshold={results['threshold']:.4f}, "
          f"achieved_precision={results['precision']:.3f}, recall={results['recall']:.3f}")

# Choose practical thresholds
practical_70_threshold = achievable_thresholds.get(0.7, {}).get('threshold', top_10_pct_threshold)
practical_80_threshold = achievable_thresholds.get(0.8, {}).get('threshold', top_5_pct_threshold)

print(f"\nProposed Practical Thresholds:")
print(f"70% confidence threshold: {practical_70_threshold:.4f}")
print(f"80% confidence threshold: {practical_80_threshold:.4f}")

# Now run full analysis with practical thresholds
print(f"\n=== Running Full Analysis with Practical Thresholds ===")

# Process all 20 nights with practical thresholds
all_results = []
all_probabilities_full = []
all_ground_truth_full = []

for night in range(1, 21):
    print(f"Processing Night {night}...")
    
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
        all_probabilities_full.extend(rem_probabilities)
        all_ground_truth_full.extend(ground_truth_rem)
        
        # Store results
        night_result = {
            'night': night,
            'total_epochs': min_length,
            'rem_epochs_total': ground_truth_rem.sum(),
            'probabilities': rem_probabilities,
            'ground_truth': ground_truth_rem
        }
        all_results.append(night_result)
        
    except Exception as e:
        print(f"Error processing night {night}: {e}")
        continue

# Convert to arrays for global analysis
all_probabilities_full = np.array(all_probabilities_full)
all_ground_truth_full = np.array(all_ground_truth_full)

# Calculate performance for each night using practical thresholds
results_df = pd.DataFrame()
all_70_preds = []
all_80_preds = []
all_gt_full = []

for result in all_results:
    night = result['night']
    probs = result['probabilities']
    gt = result['ground_truth']
    
    # Apply practical thresholds
    pred_70 = (probs >= practical_70_threshold).astype(int)
    pred_80 = (probs >= practical_80_threshold).astype(int)
    
    # Store for global confusion matrix
    all_70_preds.extend(pred_70)
    all_80_preds.extend(pred_80)
    all_gt_full.extend(gt)
    
    # Calculate metrics for 70% threshold
    tp_70 = np.sum((gt == 1) & (pred_70 == 1))
    fp_70 = np.sum((gt == 0) & (pred_70 == 1))
    tn_70 = np.sum((gt == 0) & (pred_70 == 0))
    fn_70 = np.sum((gt == 1) & (pred_70 == 0))
    
    precision_70 = tp_70 / (tp_70 + fp_70) if (tp_70 + fp_70) > 0 else 0
    recall_70 = tp_70 / (tp_70 + fn_70) if (tp_70 + fn_70) > 0 else 0
    accuracy_70 = (tp_70 + tn_70) / len(gt)
    f1_70 = 2 * precision_70 * recall_70 / (precision_70 + recall_70) if (precision_70 + recall_70) > 0 else 0
    
    # Calculate metrics for 80% threshold
    tp_80 = np.sum((gt == 1) & (pred_80 == 1))
    fp_80 = np.sum((gt == 0) & (pred_80 == 1))
    tn_80 = np.sum((gt == 0) & (pred_80 == 0))
    fn_80 = np.sum((gt == 1) & (pred_80 == 0))
    
    precision_80 = tp_80 / (tp_80 + fp_80) if (tp_80 + fp_80) > 0 else 0
    recall_80 = tp_80 / (tp_80 + fn_80) if (tp_80 + fn_80) > 0 else 0
    accuracy_80 = (tp_80 + tn_80) / len(gt)
    f1_80 = 2 * precision_80 * recall_80 / (precision_80 + recall_80) if (precision_80 + recall_80) > 0 else 0
    
    # Store results
    night_metrics = {
        'night': night,
        'total_epochs': result['total_epochs'],
        'rem_epochs': result['rem_epochs_total'],
        
        # 70% confidence metrics
        'conf_70_tp': tp_70,
        'conf_70_fp': fp_70,
        'conf_70_precision': precision_70,
        'conf_70_recall': recall_70,
        'conf_70_accuracy': accuracy_70,
        'conf_70_f1': f1_70,
        'conf_70_detections': tp_70 + fp_70,
        
        # 80% confidence metrics
        'conf_80_tp': tp_80,
        'conf_80_fp': fp_80,
        'conf_80_precision': precision_80,
        'conf_80_recall': recall_80,
        'conf_80_accuracy': accuracy_80,
        'conf_80_f1': f1_80,
        'conf_80_detections': tp_80 + fp_80,
    }
    
    results_df = pd.concat([results_df, pd.DataFrame([night_metrics])], ignore_index=True)

# Convert to arrays for global metrics
all_70_preds = np.array(all_70_preds)
all_80_preds = np.array(all_80_preds)
all_gt_full = np.array(all_gt_full)

# Global confusion matrices
cm_70 = confusion_matrix(all_gt_full, all_70_preds)
cm_80 = confusion_matrix(all_gt_full, all_80_preds)

# Global ROC
fpr_global, tpr_global, thresholds_global = roc_curve(all_gt_full, all_probabilities_full)
roc_auc_global = auc(fpr_global, tpr_global)

print(f"\n=== Global Performance Summary ===")
print(f"70% Confidence Threshold ({practical_70_threshold:.4f}):")
print(f"  Global Precision: {cm_70[1,1] / (cm_70[1,1] + cm_70[0,1]) if (cm_70[1,1] + cm_70[0,1]) > 0 else 0:.3f}")
print(f"  Global Recall: {cm_70[1,1] / (cm_70[1,0] + cm_70[1,1]):.3f}")
print(f"  Global Accuracy: {(cm_70[0,0] + cm_70[1,1]) / cm_70.sum():.3f}")
print(f"  Total detections: {cm_70[1,1] + cm_70[0,1]}")
print(f"  True positives: {cm_70[1,1]}")

print(f"\n80% Confidence Threshold ({practical_80_threshold:.4f}):")
print(f"  Global Precision: {cm_80[1,1] / (cm_80[1,1] + cm_80[0,1]) if (cm_80[1,1] + cm_80[0,1]) > 0 else 0:.3f}")
print(f"  Global Recall: {cm_80[1,1] / (cm_80[1,0] + cm_80[1,1]):.3f}")
print(f"  Global Accuracy: {(cm_80[0,0] + cm_80[1,1]) / cm_80.sum():.3f}")
print(f"  Total detections: {cm_80[1,1] + cm_80[0,1]}")
print(f"  True positives: {cm_80[1,1]}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Practical High-Confidence REM Detection: 70% vs 80% Precision Thresholds', fontsize=16, fontweight='bold')

# 1. Global ROC Curve with threshold markers
axes[0, 0].plot(fpr_global, tpr_global, color='blue', lw=3, label=f'ROC Curve (AUC = {roc_auc_global:.3f})')
axes[0, 0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)

# Mark the thresholds on ROC curve
for threshold, color, label in [(practical_70_threshold, 'orange', '70% Conf'), 
                                (practical_80_threshold, 'red', '80% Conf')]:
    threshold_idx = np.argmin(np.abs(thresholds_global - threshold))
    if threshold_idx < len(fpr_global):
        axes[0, 0].scatter(fpr_global[threshold_idx], tpr_global[threshold_idx], 
                          color=color, s=100, label=f'{label} (t={threshold:.3f})', zorder=5)

axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('Global ROC Curve with Practical Thresholds')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Confusion Matrix for 70% confidence
sns.heatmap(cm_70, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], 
            xticklabels=['Non-REM', 'REM'], yticklabels=['Non-REM', 'REM'])
axes[0, 1].set_title(f'70% Confidence Confusion Matrix\n(Threshold: {practical_70_threshold:.3f})')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# 3. Confusion Matrix for 80% confidence
sns.heatmap(cm_80, annot=True, fmt='d', cmap='Reds', ax=axes[0, 2],
            xticklabels=['Non-REM', 'REM'], yticklabels=['Non-REM', 'REM'])
axes[0, 2].set_title(f'80% Confidence Confusion Matrix\n(Threshold: {practical_80_threshold:.3f})')
axes[0, 2].set_xlabel('Predicted')
axes[0, 2].set_ylabel('Actual')

# 4. Per-night precision comparison
axes[1, 0].scatter(results_df['night'], results_df['conf_70_precision'], 
                  color='orange', s=60, alpha=0.8, label='70% Confidence')
axes[1, 0].scatter(results_df['night'], results_df['conf_80_precision'], 
                  color='red', s=60, alpha=0.8, label='80% Confidence')
axes[1, 0].set_xlabel('Night')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Per-Night Precision Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(range(1, 21, 2))

# 5. Per-night recall comparison
axes[1, 1].scatter(results_df['night'], results_df['conf_70_recall'], 
                  color='orange', s=60, alpha=0.8, label='70% Confidence')
axes[1, 1].scatter(results_df['night'], results_df['conf_80_recall'], 
                  color='red', s=60, alpha=0.8, label='80% Confidence')
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

bars1 = axes[1, 2].bar(x - width/2, results_df['conf_70_detections'], width, 
                      label='70% Conf Detections', color='orange', alpha=0.8)
bars2 = axes[1, 2].bar(x + width/2, results_df['conf_80_detections'], width,
                      label='80% Conf Detections', color='red', alpha=0.8)

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
plt.savefig('practical_confidence_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nPractical analysis plot saved as practical_confidence_analysis.png")
plt.close()

# Save results
results_df.to_csv('practical_confidence_results.csv', index=False)

print(f"\n=== FINAL PRACTICAL RECOMMENDATIONS ===")
print(f"For 70% confidence REM detection, use threshold: {practical_70_threshold:.4f}")
print(f"  - Expected precision: ~{cm_70[1,1] / (cm_70[1,1] + cm_70[0,1]) if (cm_70[1,1] + cm_70[0,1]) > 0 else 0:.1%}")
print(f"  - Expected recall: ~{cm_70[1,1] / (cm_70[1,0] + cm_70[1,1]):.1%}")
print(f"  - Will trigger on {(results_df['conf_70_detections'] > 0).sum()}/20 nights")

print(f"\nFor 80% confidence REM detection, use threshold: {practical_80_threshold:.4f}")
print(f"  - Expected precision: ~{cm_80[1,1] / (cm_80[1,1] + cm_80[0,1]) if (cm_80[1,1] + cm_80[0,1]) > 0 else 0:.1%}")
print(f"  - Expected recall: ~{cm_80[1,1] / (cm_80[1,0] + cm_80[1,1]):.1%}")
print(f"  - Will trigger on {(results_df['conf_80_detections'] > 0).sum()}/20 nights")

print(f"\nGlobal ROC AUC: {roc_auc_global:.3f}")
print(f"These thresholds are much more practical than 90%/95% and will actually detect REM!")
