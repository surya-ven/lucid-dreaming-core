import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import matplotlib
import mne
import yasa
matplotlib.use('Agg')  # Use non-interactive backend

# Load the ground truth labels
labels_df = pd.read_csv('provided_data/night_01_label.csv')
print(f"Label file shape: {labels_df.shape}")

# Get YASA predictions directly (string format)
raw = mne.io.read_raw_edf('provided_data/night_01.edf', preload=True, verbose=False)
# Only rename the cleaner EEG channels, exclude OTE_L and OTE_R
raw.rename_channels({'LF-FpZ': 'Fp1', 'RF-FpZ': 'Fp2'})
sls = yasa.SleepStaging(raw, eeg_name='Fp2')
yasa_predictions = sls.predict()
yasa_proba = sls.predict_proba()
yasa_confidence = yasa_proba.max(1)

print(f"YASA predictions shape: {yasa_predictions.shape}")
print(f"Unique YASA stages: {np.unique(yasa_predictions)}")
print(f"Unique ground truth labels: {labels_df['Sleep stage'].unique()}")

# Map YASA stages to ground truth format
yasa_to_gt_mapping = {
    'W': 'Wake',
    'N1': 'Light',    # N1 is light sleep
    'N2': 'Light',    # N2 is light sleep  
    'N3': 'Deep',     # N3 is deep sleep
    'R': 'REM'        # REM sleep
}

# Apply mapping
yasa_mapped = [yasa_to_gt_mapping[stage] for stage in yasa_predictions]

# Ensure both datasets have the same length
min_length = min(len(labels_df) - 1, len(yasa_predictions))  # -1 for header in labels
ground_truth = labels_df['Sleep stage'].iloc[1:min_length+1].values  # Skip header
yasa_pred = np.array(yasa_mapped[:min_length])

print(f"\nComparing {min_length} epochs")
print(f"Ground truth shape: {ground_truth.shape}")
print(f"YASA predictions shape: {yasa_pred.shape}")

# Calculate accuracy
accuracy = accuracy_score(ground_truth, yasa_pred)
print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

# Classification report
print("\nDetailed Classification Report:")
print(classification_report(ground_truth, yasa_pred))

# Confusion matrix
cm = confusion_matrix(ground_truth, yasa_pred, labels=['Wake', 'Light', 'Deep', 'REM'])
cm_df = pd.DataFrame(cm, index=['Wake', 'Light', 'Deep', 'REM'], 
                     columns=['Wake', 'Light', 'Deep', 'REM'])

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix: Ground Truth vs YASA Predictions')
plt.ylabel('Ground Truth')
plt.xlabel('YASA Predictions')
plt.tight_layout()
plt.savefig('yasa_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Confusion matrix saved as yasa_confusion_matrix.png")
plt.close()

# Create comparison hypnogram
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

# Map stages to numeric values for plotting
stage_to_num = {'Wake': 4, 'REM': 3, 'Light': 2, 'Deep': 1}
epochs = np.arange(min_length)
ground_truth_num = [stage_to_num[stage] for stage in ground_truth]
yasa_pred_num = [stage_to_num[stage] for stage in yasa_pred]

# Plot ground truth
ax1.plot(epochs/120, ground_truth_num, 'b-', linewidth=0.8, label='Ground Truth')
ax1.set_ylabel('Sleep Stage')
ax1.set_title('Ground Truth Sleep Stages')
ax1.set_yticks([1, 2, 3, 4])
ax1.set_yticklabels(['Deep', 'Light', 'REM', 'Wake'])
ax1.grid(True, alpha=0.3)

# Plot YASA predictions
ax2.plot(epochs/120, yasa_pred_num, 'r-', linewidth=0.8, label='YASA Predictions')
ax2.set_ylabel('Sleep Stage')
ax2.set_title('YASA Predicted Sleep Stages')
ax2.set_yticks([1, 2, 3, 4])
ax2.set_yticklabels(['Deep', 'Light', 'REM', 'Wake'])
ax2.grid(True, alpha=0.3)

# Plot difference (agreement/disagreement)
agreement = [1 if gt == pred else 0 for gt, pred in zip(ground_truth, yasa_pred)]
ax3.fill_between(epochs/120, agreement, color='green', alpha=0.7, label='Agreement')
ax3.fill_between(epochs/120, [1-a for a in agreement], color='red', alpha=0.7, label='Disagreement')
ax3.set_ylabel('Agreement')
ax3.set_xlabel('Time (hours)')
ax3.set_title(f'YASA vs Ground Truth Agreement (Overall Accuracy: {accuracy:.1%})')
ax3.set_ylim(0, 1)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('yasa_vs_ground_truth_comparison.png', dpi=150, bbox_inches='tight')
print("Comparison hypnogram saved as yasa_vs_ground_truth_comparison.png")
plt.close()

# Per-stage accuracy analysis
print("\nPer-stage Analysis:")
for stage in ['Wake', 'Light', 'Deep', 'REM']:
    gt_mask = ground_truth == stage
    if np.sum(gt_mask) > 0:
        stage_accuracy = accuracy_score(ground_truth[gt_mask], yasa_pred[gt_mask])
        print(f"{stage}: {stage_accuracy:.3f} ({stage_accuracy*100:.1f}%) - {np.sum(gt_mask)} epochs")

# Time-based analysis (accuracy over time)
window_size = 60  # 30-minute windows (60 epochs * 30 seconds)
num_windows = len(ground_truth) // window_size
window_accuracies = []

for i in range(num_windows):
    start_idx = i * window_size
    end_idx = (i + 1) * window_size
    window_gt = ground_truth[start_idx:end_idx]
    window_pred = yasa_pred[start_idx:end_idx]
    window_acc = accuracy_score(window_gt, window_pred)
    window_accuracies.append(window_acc)

# Plot accuracy over time
plt.figure(figsize=(12, 6))
window_times = np.arange(num_windows) * window_size / 120  # Convert to hours
plt.plot(window_times, window_accuracies, 'o-', linewidth=2, markersize=6)
plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall Accuracy ({accuracy:.1%})')
plt.xlabel('Time (hours)')
plt.ylabel('Accuracy')
plt.title('YASA Accuracy Over Time (30-minute windows)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('yasa_accuracy_over_time.png', dpi=150, bbox_inches='tight')
print("Accuracy over time plot saved as yasa_accuracy_over_time.png")
plt.close()

# Create detailed epoch-by-epoch comparison CSV
comparison_df = pd.DataFrame({
    'Epoch': range(min_length),
    'Time_Hours': np.arange(min_length) / 120,
    'Ground_Truth': ground_truth,
    'YASA_Raw': yasa_predictions[:min_length],
    'YASA_Mapped': yasa_pred,
    'Correct': [1 if gt == pred else 0 for gt, pred in zip(ground_truth, yasa_pred)],
    'YASA_Confidence': yasa_confidence[:min_length]
})

comparison_df.to_csv('yasa_vs_ground_truth_detailed.csv', index=False)
print("Detailed comparison saved as yasa_vs_ground_truth_detailed.csv")

print(f"\nSummary:")
print(f"Total epochs compared: {min_length}")
print(f"Overall accuracy: {accuracy:.1%}")
print(f"Recording duration: {min_length/120:.1f} hours")
print(f"Average YASA confidence: {yasa_confidence[:min_length].mean():.3f}")

# Calculate Cohen's Kappa for inter-rater agreement
kappa = cohen_kappa_score(ground_truth, yasa_pred)
print(f"Cohen's Kappa (agreement measure): {kappa:.3f}")

if kappa < 0:
    agreement_level = "Poor (worse than random)"
elif kappa < 0.20:
    agreement_level = "Slight"
elif kappa < 0.40:
    agreement_level = "Fair"
elif kappa < 0.60:
    agreement_level = "Moderate"
elif kappa < 0.80:
    agreement_level = "Substantial"
else:
    agreement_level = "Almost perfect"

print(f"Agreement level: {agreement_level}")
