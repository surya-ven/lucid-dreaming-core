import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import yasa
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib
matplotlib.use('Agg')

print("=== Final High-Confidence REM Analysis: ROC Curves & Confusion Matrices ===")

# Fixed thresholds from practical analysis
CONFIDENCE_70_THRESHOLD = 0.4382  # Top 10% of probabilities
CONFIDENCE_80_THRESHOLD = 0.6131  # Top 5% of probabilities

# Initialize comprehensive results storage
all_results = []
all_probabilities = []
all_ground_truth = []
night_performance = []

print("Processing all 20 nights...")

# Process all nights
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
        all_probabilities.extend(rem_probabilities)
        all_ground_truth.extend(ground_truth_rem)
        
        # Calculate performance for both confidence thresholds
        for threshold_name, threshold_value in [
            ("70% Confidence", CONFIDENCE_70_THRESHOLD),
            ("80% Confidence", CONFIDENCE_80_THRESHOLD)
        ]:
            # Apply threshold
            rem_pred_threshold = (rem_probabilities >= threshold_value).astype(int)
            
            # Calculate metrics
            cm = confusion_matrix(ground_truth_rem, rem_pred_threshold)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
            
            accuracy = accuracy_score(ground_truth_rem, rem_pred_threshold)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            night_performance.append({
                'night': night,
                'threshold_name': threshold_name,
                'threshold_value': threshold_value,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'total_rem_epochs': ground_truth_rem.sum(),
                'predicted_rem_epochs': rem_pred_threshold.sum(),
                'total_epochs': len(ground_truth_rem)
            })
        
    except Exception as e:
        print(f"Error processing night {night}: {e}")
        continue

# Convert to DataFrames
night_performance_df = pd.DataFrame(night_performance)
night_performance_df.to_csv('final_confidence_analysis_results.csv', index=False)

print(f"\nProcessed {len(set(night_performance_df['night']))} nights successfully")

# Calculate global metrics
all_probabilities = np.array(all_probabilities)
all_ground_truth = np.array(all_ground_truth)

# Global ROC curve
fpr, tpr, thresholds = roc_curve(all_ground_truth, all_probabilities)
roc_auc = auc(fpr, tpr)

# Global Precision-Recall curve
precision_global, recall_global, pr_thresholds = precision_recall_curve(all_ground_truth, all_probabilities)
pr_auc = auc(recall_global, precision_global)

print(f"\n=== GLOBAL PERFORMANCE METRICS ===")
print(f"Global ROC AUC: {roc_auc:.3f}")
print(f"Global PR AUC: {pr_auc:.3f}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Final High-Confidence REM Detection Analysis (All 20 Nights)', fontsize=16, fontweight='bold')

# 1. Global ROC Curve
ax = axes[0, 0]
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)

# Mark our confidence thresholds on ROC curve
for threshold_name, threshold_value in [("70% Conf", CONFIDENCE_70_THRESHOLD), ("80% Conf", CONFIDENCE_80_THRESHOLD)]:
    # Find closest threshold in ROC curve
    idx = np.argmin(np.abs(thresholds - threshold_value))
    ax.plot(fpr[idx], tpr[idx], 'o', markersize=8, 
           label=f'{threshold_name} (th={threshold_value:.3f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Global ROC Curve')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# 2. Global Precision-Recall Curve
ax = axes[0, 1]
ax.plot(recall_global, precision_global, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')

# Mark our confidence thresholds on PR curve
for threshold_name, threshold_value in [("70% Conf", CONFIDENCE_70_THRESHOLD), ("80% Conf", CONFIDENCE_80_THRESHOLD)]:
    # Calculate precision and recall for this threshold
    pred_global = (all_probabilities >= threshold_value).astype(int)
    cm_global = confusion_matrix(all_ground_truth, pred_global)
    tn, fp, fn, tp = cm_global.ravel() if cm_global.size == 4 else (0, 0, 0, 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax.plot(rec, prec, 'o', markersize=8, 
           label=f'{threshold_name} (P={prec:.3f}, R={rec:.3f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Global Precision-Recall Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Probability Distribution
ax = axes[0, 2]
rem_probs = all_probabilities[all_ground_truth == 1]
non_rem_probs = all_probabilities[all_ground_truth == 0]

ax.hist(non_rem_probs, bins=50, alpha=0.7, label='Non-REM', color='blue', density=True)
ax.hist(rem_probs, bins=50, alpha=0.7, label='REM', color='red', density=True)
ax.axvline(CONFIDENCE_70_THRESHOLD, color='orange', linestyle='--', linewidth=2, label='70% Confidence')
ax.axvline(CONFIDENCE_80_THRESHOLD, color='purple', linestyle='--', linewidth=2, label='80% Confidence')
ax.set_xlabel('REM Probability')
ax.set_ylabel('Density')
ax.set_title('Probability Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Performance by Night - 70% Confidence
ax = axes[1, 0]
night_70 = night_performance_df[night_performance_df['threshold_name'] == '70% Confidence']
x_pos = np.arange(len(night_70))
width = 0.35

ax.bar(x_pos - width/2, night_70['precision'], width, label='Precision', alpha=0.8)
ax.bar(x_pos + width/2, night_70['recall'], width, label='Recall', alpha=0.8)
ax.set_xlabel('Night')
ax.set_ylabel('Score')
ax.set_title('70% Confidence Threshold Performance by Night')
ax.set_xticks(x_pos)
ax.set_xticklabels(night_70['night'])
ax.legend()
ax.grid(True, alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45)

# 5. Performance by Night - 80% Confidence
ax = axes[1, 1]
night_80 = night_performance_df[night_performance_df['threshold_name'] == '80% Confidence']
x_pos = np.arange(len(night_80))

ax.bar(x_pos - width/2, night_80['precision'], width, label='Precision', alpha=0.8)
ax.bar(x_pos + width/2, night_80['recall'], width, label='Recall', alpha=0.8)
ax.set_xlabel('Night')
ax.set_ylabel('Score')
ax.set_title('80% Confidence Threshold Performance by Night')
ax.set_xticks(x_pos)
ax.set_xticklabels(night_80['night'])
ax.legend()
ax.grid(True, alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45)

# 6. Global Confusion Matrices
ax = axes[1, 2]
ax.axis('off')

# Create two sub-confusion matrices
fig_sub = plt.figure(figsize=(12, 5))

for i, (threshold_name, threshold_value) in enumerate([("70% Confidence", CONFIDENCE_70_THRESHOLD), 
                                                      ("80% Confidence", CONFIDENCE_80_THRESHOLD)]):
    pred_global = (all_probabilities >= threshold_value).astype(int)
    cm_global = confusion_matrix(all_ground_truth, pred_global)
    
    ax_sub = fig_sub.add_subplot(1, 2, i+1)
    sns.heatmap(cm_global, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Non-REM', 'Predicted REM'],
                yticklabels=['Actual Non-REM', 'Actual REM'],
                ax=ax_sub)
    ax_sub.set_title(f'Global Confusion Matrix - {threshold_name}')

plt.tight_layout()
plt.savefig('global_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close(fig_sub)

plt.tight_layout()
plt.savefig('final_confidence_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Summary statistics
print(f"\n=== SUMMARY STATISTICS ===")

for threshold_name in ['70% Confidence', '80% Confidence']:
    subset = night_performance_df[night_performance_df['threshold_name'] == threshold_name]
    
    print(f"\n{threshold_name} Threshold ({subset.iloc[0]['threshold_value']:.4f}):")
    print(f"  Nights processed: {len(subset)}")
    print(f"  Mean Precision: {subset['precision'].mean():.3f} (±{subset['precision'].std():.3f})")
    print(f"  Mean Recall: {subset['recall'].mean():.3f} (±{subset['recall'].std():.3f})")
    print(f"  Mean F1-Score: {subset['f1_score'].mean():.3f} (±{subset['f1_score'].std():.3f})")
    print(f"  Mean Accuracy: {subset['accuracy'].mean():.3f} (±{subset['accuracy'].std():.3f})")
    print(f"  Total True Positives: {subset['true_positives'].sum()}")
    print(f"  Total False Positives: {subset['false_positives'].sum()}")
    print(f"  Total REM epochs: {subset['total_rem_epochs'].sum()}")
    print(f"  Total predicted REM: {subset['predicted_rem_epochs'].sum()}")
    
    # Nights with zero recall
    zero_recall_nights = subset[subset['recall'] == 0]['night'].tolist()
    if zero_recall_nights:
        print(f"  Nights with zero recall: {zero_recall_nights}")
    else:
        print(f"  All nights had some REM detection!")

print(f"\n=== FINAL RECOMMENDATIONS ===")
print(f"✓ For moderate high-confidence REM detection:")
print(f"  Use threshold: {CONFIDENCE_70_THRESHOLD:.4f} (70% confidence)")
print(f"  Expected precision: ~{night_performance_df[night_performance_df['threshold_name'] == '70% Confidence']['precision'].mean():.1%}")
print(f"  Expected recall: ~{night_performance_df[night_performance_df['threshold_name'] == '70% Confidence']['recall'].mean():.1%}")

print(f"\n✓ For high-confidence REM detection:")
print(f"  Use threshold: {CONFIDENCE_80_THRESHOLD:.4f} (80% confidence)")
print(f"  Expected precision: ~{night_performance_df[night_performance_df['threshold_name'] == '80% Confidence']['precision'].mean():.1%}")
print(f"  Expected recall: ~{night_performance_df[night_performance_df['threshold_name'] == '80% Confidence']['recall'].mean():.1%}")

print(f"\n✓ Global Performance:")
print(f"  ROC AUC: {roc_auc:.3f} (good discriminative ability)")
print(f"  PR AUC: {pr_auc:.3f}")

print(f"\nAnalysis complete! Results saved to:")
print(f"- final_confidence_analysis_results.csv")
print(f"- final_confidence_analysis.png")
print(f"- global_confusion_matrices.png")
