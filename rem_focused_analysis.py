import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib
import mne
import yasa
matplotlib.use('Agg')  # Use non-interactive backend

print("=== REM-Focused Sleep Stage Analysis ===")

# Load the ground truth labels
labels_df = pd.read_csv('provided_data/night_01_label.csv')
print(f"Label file shape: {labels_df.shape}")

# Get YASA predictions and probabilities
raw = mne.io.read_raw_edf('provided_data/night_01.edf', preload=True, verbose=False)
raw.rename_channels({'LF-FpZ': 'Fp1', 'RF-FpZ': 'Fp2'})
sls = yasa.SleepStaging(raw, eeg_name='Fp2')
yasa_predictions = sls.predict()
yasa_proba = sls.predict_proba()

print(f"YASA predictions shape: {yasa_predictions.shape}")
print(f"YASA probability columns: {yasa_proba.columns.tolist()}")

# Ensure both datasets have the same length
min_length = min(len(labels_df) - 1, len(yasa_predictions))
ground_truth = labels_df['Sleep stage'].iloc[1:min_length+1].values
yasa_pred = yasa_predictions[:min_length]
yasa_prob = yasa_proba.iloc[:min_length]

print(f"\nAnalyzing {min_length} epochs for REM detection")

# Create binary REM classification
ground_truth_rem = (ground_truth == 'REM').astype(int)
yasa_pred_rem = (yasa_pred == 'R').astype(int)

# Get REM probabilities
rem_probabilities = yasa_prob['R'].values

print(f"\nREM Sleep Statistics:")
print(f"Total REM epochs in ground truth: {ground_truth_rem.sum()}")
print(f"Total REM epochs predicted by YASA: {yasa_pred_rem.sum()}")
print(f"REM probability range: {rem_probabilities.min():.3f} - {rem_probabilities.max():.3f}")
print(f"Mean REM probability: {rem_probabilities.mean():.3f}")

# Analyze REM probabilities when truth is REM vs non-REM
rem_true_probs = rem_probabilities[ground_truth_rem == 1]
rem_false_probs = rem_probabilities[ground_truth_rem == 0]

print(f"\nREM Probability Analysis:")
print(f"When truth is REM - Mean prob: {rem_true_probs.mean():.3f}, Std: {rem_true_probs.std():.3f}")
print(f"When truth is non-REM - Mean prob: {rem_true_probs.mean():.3f}, Std: {rem_false_probs.std():.3f}")

# Calculate ROC curve for REM detection
fpr, tpr, thresholds = roc_curve(ground_truth_rem, rem_probabilities)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(ground_truth_rem, rem_probabilities)
pr_auc = auc(recall, precision)

print(f"\nREM Detection Performance:")
print(f"ROC AUC: {roc_auc:.3f}")
print(f"Precision-Recall AUC: {pr_auc:.3f}")

# Find optimal threshold using Youden's index (TPR - FPR)
youden_scores = tpr - fpr
optimal_idx = np.argmax(youden_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal REM threshold (Youden's): {optimal_threshold:.3f}")
print(f"At optimal threshold - TPR: {tpr[optimal_idx]:.3f}, FPR: {fpr[optimal_idx]:.3f}")

# Apply optimal threshold
yasa_pred_rem_optimized = (rem_probabilities >= optimal_threshold).astype(int)

# Original vs optimized performance
original_accuracy = accuracy_score(ground_truth_rem, yasa_pred_rem)
optimized_accuracy = accuracy_score(ground_truth_rem, yasa_pred_rem_optimized)

print(f"\nREM Detection Accuracy:")
print(f"Original YASA threshold: {original_accuracy:.3f} ({original_accuracy*100:.1f}%)")
print(f"Optimized threshold: {optimized_accuracy:.3f} ({optimized_accuracy*100:.1f}%)")

# Detailed classification reports
print(f"\nOriginal YASA REM Classification:")
print(classification_report(ground_truth_rem, yasa_pred_rem, target_names=['Non-REM', 'REM']))

print(f"\nOptimized Threshold REM Classification:")
print(classification_report(ground_truth_rem, yasa_pred_rem_optimized, target_names=['Non-REM', 'REM']))

# Create comprehensive REM-focused visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. REM probability distribution
axes[0, 0].hist(rem_true_probs, bins=20, alpha=0.7, label='True REM', color='red', density=True)
axes[0, 0].hist(rem_false_probs, bins=20, alpha=0.7, label='True Non-REM', color='blue', density=True)
axes[0, 0].axvline(optimal_threshold, color='green', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.3f})')
axes[0, 0].set_xlabel('REM Probability')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('REM Probability Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. ROC Curve
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[0, 1].scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, label=f'Optimal Point')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('REM Detection ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Precision-Recall Curve
axes[0, 2].plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
axes[0, 2].set_xlabel('Recall')
axes[0, 2].set_ylabel('Precision')
axes[0, 2].set_title('REM Detection Precision-Recall Curve')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. REM probabilities over time
epochs = np.arange(min_length)
time_hours = epochs / 120  # 30-second epochs, 120 per hour

axes[1, 0].plot(time_hours, rem_probabilities, 'b-', alpha=0.7, linewidth=0.8, label='REM Probability')
axes[1, 0].axhline(y=optimal_threshold, color='green', linestyle='--', label=f'Optimal Threshold')
axes[1, 0].fill_between(time_hours, 0, ground_truth_rem, alpha=0.3, color='red', label='True REM')
axes[1, 0].set_xlabel('Time (hours)')
axes[1, 0].set_ylabel('REM Probability')
axes[1, 0].set_title('REM Probabilities Over Time')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Threshold vs Performance
thresholds_range = np.linspace(0, 1, 100)
accuracies = []
sensitivities = []
specificities = []

for thresh in thresholds_range:
    pred_thresh = (rem_probabilities >= thresh).astype(int)
    
    # Calculate confusion matrix elements
    tn = np.sum((ground_truth_rem == 0) & (pred_thresh == 0))
    fp = np.sum((ground_truth_rem == 0) & (pred_thresh == 1))
    fn = np.sum((ground_truth_rem == 1) & (pred_thresh == 0))
    tp = np.sum((ground_truth_rem == 1) & (pred_thresh == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)

axes[1, 1].plot(thresholds_range, accuracies, 'g-', label='Accuracy', linewidth=2)
axes[1, 1].plot(thresholds_range, sensitivities, 'r-', label='Sensitivity (Recall)', linewidth=2)
axes[1, 1].plot(thresholds_range, specificities, 'b-', label='Specificity', linewidth=2)
axes[1, 1].axvline(x=optimal_threshold, color='orange', linestyle='--', label=f'Optimal Threshold')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('Performance')
axes[1, 1].set_title('REM Detection Performance vs Threshold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Confusion matrices comparison
cm_original = confusion_matrix(ground_truth_rem, yasa_pred_rem)
cm_optimized = confusion_matrix(ground_truth_rem, yasa_pred_rem_optimized)

# Plot both confusion matrices side by side
ax_cm = axes[1, 2]
ax_cm.text(0.25, 0.8, 'Original YASA', transform=ax_cm.transAxes, ha='center', fontsize=12, fontweight='bold')
ax_cm.text(0.25, 0.7, f'TN: {cm_original[0,0]}  FP: {cm_original[0,1]}', transform=ax_cm.transAxes, ha='center')
ax_cm.text(0.25, 0.6, f'FN: {cm_original[1,0]}  TP: {cm_original[1,1]}', transform=ax_cm.transAxes, ha='center')
ax_cm.text(0.25, 0.5, f'Acc: {original_accuracy:.3f}', transform=ax_cm.transAxes, ha='center')

ax_cm.text(0.75, 0.8, 'Optimized Threshold', transform=ax_cm.transAxes, ha='center', fontsize=12, fontweight='bold')
ax_cm.text(0.75, 0.7, f'TN: {cm_optimized[0,0]}  FP: {cm_optimized[0,1]}', transform=ax_cm.transAxes, ha='center')
ax_cm.text(0.75, 0.6, f'FN: {cm_optimized[1,0]}  TP: {cm_optimized[1,1]}', transform=ax_cm.transAxes, ha='center')
ax_cm.text(0.75, 0.5, f'Acc: {optimized_accuracy:.3f}', transform=ax_cm.transAxes, ha='center')

ax_cm.set_title('REM Detection Confusion Matrix Comparison')
ax_cm.set_xlim(0, 1)
ax_cm.set_ylim(0, 1)
ax_cm.axis('off')

plt.tight_layout()
plt.savefig('rem_focused_analysis.png', dpi=150, bbox_inches='tight')
print("REM-focused analysis saved as rem_focused_analysis.png")
plt.close()

# Create detailed REM analysis report
rem_epochs_idx = np.where(ground_truth_rem == 1)[0]
print(f"\n=== Detailed REM Episode Analysis ===")
print(f"REM epochs found at indices: {rem_epochs_idx}")
print(f"REM epochs timing (hours): {rem_epochs_idx / 120}")

# Analyze REM episodes (consecutive REM epochs)
rem_episodes = []
if len(rem_epochs_idx) > 0:
    episode_start = rem_epochs_idx[0]
    episode_end = rem_epochs_idx[0]
    
    for i in range(1, len(rem_epochs_idx)):
        if rem_epochs_idx[i] == rem_epochs_idx[i-1] + 1:  # Consecutive
            episode_end = rem_epochs_idx[i]
        else:  # Gap found, save current episode and start new one
            rem_episodes.append((episode_start, episode_end))
            episode_start = rem_epochs_idx[i]
            episode_end = rem_epochs_idx[i]
    
    # Don't forget the last episode
    rem_episodes.append((episode_start, episode_end))

print(f"\nREM Episodes Found: {len(rem_episodes)}")
for i, (start, end) in enumerate(rem_episodes):
    duration_min = (end - start + 1) * 0.5  # 30-second epochs
    start_time = start / 120
    end_time = end / 120
    
    # Get YASA performance for this episode
    episode_gt = ground_truth_rem[start:end+1]
    episode_pred_orig = yasa_pred_rem[start:end+1]
    episode_pred_opt = yasa_pred_rem_optimized[start:end+1]
    episode_probs = rem_probabilities[start:end+1]
    
    orig_detected = episode_pred_orig.sum()
    opt_detected = episode_pred_opt.sum()
    
    print(f"  Episode {i+1}: {start_time:.2f}h - {end_time:.2f}h ({duration_min:.1f} min)")
    print(f"    Original YASA detected: {orig_detected}/{len(episode_gt)} epochs ({orig_detected/len(episode_gt)*100:.1f}%)")
    print(f"    Optimized threshold detected: {opt_detected}/{len(episode_gt)} epochs ({opt_detected/len(episode_gt)*100:.1f}%)")
    print(f"    Mean REM probability: {episode_probs.mean():.3f}")

# Save detailed results
results_df = pd.DataFrame({
    'Epoch': range(min_length),
    'Time_Hours': time_hours,
    'Ground_Truth_REM': ground_truth_rem,
    'YASA_Pred_REM_Original': yasa_pred_rem,
    'YASA_Pred_REM_Optimized': yasa_pred_rem_optimized,
    'REM_Probability': rem_probabilities,
    'Above_Optimal_Threshold': (rem_probabilities >= optimal_threshold).astype(int)
})

results_df.to_csv('rem_focused_analysis_results.csv', index=False)
print(f"\nDetailed REM analysis saved as rem_focused_analysis_results.csv")

print(f"\n=== Summary ===")
print(f"REM Detection Performance:")
print(f"  Original accuracy: {original_accuracy:.1%}")
print(f"  Optimized accuracy: {optimized_accuracy:.1%}")
print(f"  ROC AUC: {roc_auc:.3f}")
print(f"  PR AUC: {pr_auc:.3f}")
print(f"  Optimal threshold: {optimal_threshold:.3f}")
print(f"  Total REM epochs: {ground_truth_rem.sum()}")
print(f"  REM episodes: {len(rem_episodes)}")
