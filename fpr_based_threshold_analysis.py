import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import yasa
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib
matplotlib.use('Agg')

print("=== Empirical FPR-Based REM Threshold Analysis ===")
print("Finding thresholds for FPR < 10% and FPR < 5% across all 20 nights")
print("=" * 70)

# Target false positive rates
TARGET_FPR_10 = 0.10  # 10% FPR
TARGET_FPR_5 = 0.05   # 5% FPR

# Initialize storage
all_results = []
all_probabilities = []
all_ground_truth = []
night_roc_data = []

print("Processing all 20 nights to calculate ROC curves...")

# Process all nights to get comprehensive ROC data
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
        
        # Calculate ROC curve for this night
        fpr, tpr, thresholds = roc_curve(ground_truth_rem, rem_probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Store ROC data for this night
        night_roc_data.append({
            'night': night,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'roc_auc': roc_auc,
            'ground_truth': ground_truth_rem,
            'probabilities': rem_probabilities,
            'total_rem': ground_truth_rem.sum(),
            'total_non_rem': (1 - ground_truth_rem).sum()
        })
        
    except Exception as e:
        print(f"Error processing night {night}: {e}")
        continue

print(f"\nSuccessfully processed {len(night_roc_data)} nights")

# Convert to arrays for global analysis
all_probabilities = np.array(all_probabilities)
all_ground_truth = np.array(all_ground_truth)

# Calculate global ROC curve
print("\nCalculating global ROC curve...")
global_fpr, global_tpr, global_thresholds = roc_curve(all_ground_truth, all_probabilities)
global_roc_auc = auc(global_fpr, global_tpr)

print(f"Global ROC AUC: {global_roc_auc:.3f}")

# Function to find threshold for target FPR
def find_threshold_for_fpr(fpr, tpr, thresholds, target_fpr):
    """Find threshold that achieves target FPR"""
    # Find the index where FPR is closest to but not exceeding target
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return None, None, None
    
    # Get the index with FPR closest to target (but not exceeding)
    best_idx = valid_indices[np.argmax(fpr[valid_indices])]
    
    return fpr[best_idx], tpr[best_idx], thresholds[best_idx]

# Find global thresholds for target FPRs
print(f"\nFinding thresholds for target FPRs...")

# For FPR < 10%
global_fpr_10, global_tpr_10, threshold_fpr_10 = find_threshold_for_fpr(
    global_fpr, global_tpr, global_thresholds, TARGET_FPR_10)

# For FPR < 5%
global_fpr_5, global_tpr_5, threshold_fpr_5 = find_threshold_for_fpr(
    global_fpr, global_tpr, global_thresholds, TARGET_FPR_5)

print(f"\nGLOBAL THRESHOLD RESULTS:")
if threshold_fpr_10 is not None:
    print(f"FPR < 10%: Threshold = {threshold_fpr_10:.4f}, Actual FPR = {global_fpr_10:.3f}, TPR = {global_tpr_10:.3f}")
else:
    print(f"FPR < 10%: No valid threshold found")

if threshold_fpr_5 is not None:
    print(f"FPR < 5%:  Threshold = {threshold_fpr_5:.4f}, Actual FPR = {global_fpr_5:.3f}, TPR = {global_tpr_5:.3f}")
else:
    print(f"FPR < 5%:  No valid threshold found")

# Analyze per-night performance with these thresholds
print(f"\nAnalyzing per-night performance with empirical thresholds...")

night_performance = []

for night_data in night_roc_data:
    night = night_data['night']
    ground_truth_rem = night_data['ground_truth']
    rem_probabilities = night_data['probabilities']
    
    # Test both thresholds if they exist
    for target_name, threshold_value in [
        ("FPR_10", threshold_fpr_10),
        ("FPR_5", threshold_fpr_5)
    ]:
        if threshold_value is None:
            continue
            
        # Apply threshold
        rem_pred_threshold = (rem_probabilities >= threshold_value).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(ground_truth_rem, rem_pred_threshold)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth_rem, rem_pred_threshold)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # This is TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # This is actual FPR
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        night_performance.append({
            'night': night,
            'target_fpr': target_name,
            'threshold_value': threshold_value,
            'actual_fpr': fpr,
            'tpr_recall': recall,
            'precision': precision,
            'specificity': specificity,
            'accuracy': accuracy,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'total_rem_epochs': ground_truth_rem.sum(),
            'total_non_rem_epochs': (1 - ground_truth_rem).sum(),
            'predicted_rem_epochs': rem_pred_threshold.sum()
        })

# Convert to DataFrame
night_performance_df = pd.DataFrame(night_performance)
night_performance_df.to_csv('fpr_based_threshold_analysis.csv', index=False)

# Calculate summary statistics
print(f"\n" + "=" * 70)
print(f"SUMMARY STATISTICS:")
print(f"=" * 70)

for target_fpr in ["FPR_10", "FPR_5"]:
    subset = night_performance_df[night_performance_df['target_fpr'] == target_fpr]
    
    if len(subset) == 0:
        continue
        
    threshold_val = subset.iloc[0]['threshold_value']
    target_fpr_val = 0.10 if target_fpr == "FPR_10" else 0.05
    
    # Global metrics
    total_tp = subset['true_positives'].sum()
    total_fp = subset['false_positives'].sum()
    total_tn = subset['true_negatives'].sum()
    total_fn = subset['false_negatives'].sum()
    
    global_fpr_actual = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    global_tpr_actual = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    
    # Per-night statistics
    mean_fpr = subset['actual_fpr'].mean()
    mean_tpr = subset['tpr_recall'].mean()
    mean_precision = subset['precision'].mean()
    
    nights_meeting_target = len(subset[subset['actual_fpr'] <= target_fpr_val])
    
    print(f"\n🎯 {target_fpr.replace('_', ' < ')}% TARGET:")
    print(f"   Threshold value: {threshold_val:.4f}")
    print(f"   Global actual FPR: {global_fpr_actual:.3f} ({global_fpr_actual*100:.1f}%)")
    print(f"   Global TPR (Recall): {global_tpr_actual:.3f} ({global_tpr_actual*100:.1f}%)")
    print(f"   Global Precision: {global_precision:.3f} ({global_precision*100:.1f}%)")
    print(f"   Mean per-night FPR: {mean_fpr:.3f} (±{subset['actual_fpr'].std():.3f})")
    print(f"   Mean per-night TPR: {mean_tpr:.3f} (±{subset['tpr_recall'].std():.3f})")
    print(f"   Mean per-night Precision: {mean_precision:.3f} (±{subset['precision'].std():.3f})")
    print(f"   Nights meeting FPR target: {nights_meeting_target}/{len(subset)}")
    print(f"   Total true positives: {total_tp}")
    print(f"   Total false positives: {total_fp}")

# Create comprehensive visualization
print(f"\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Empirical FPR-Based REM Threshold Analysis (All 20 Nights)', fontsize=16, fontweight='bold')

# 1. Global ROC Curve with FPR targets
ax = axes[0, 0]
ax.plot(global_fpr, global_tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {global_roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)

# Mark FPR thresholds
if threshold_fpr_10 is not None:
    ax.plot(global_fpr_10, global_tpr_10, 'ro', markersize=10, 
           label=f'FPR<10% (th={threshold_fpr_10:.3f})')
if threshold_fpr_5 is not None:
    ax.plot(global_fpr_5, global_tpr_5, 'bo', markersize=10, 
           label=f'FPR<5% (th={threshold_fpr_5:.3f})')

# Add target FPR lines
ax.axvline(x=0.10, color='red', linestyle=':', alpha=0.7, label='10% FPR target')
ax.axvline(x=0.05, color='blue', linestyle=':', alpha=0.7, label='5% FPR target')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('Global ROC Curve with FPR Targets')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# 2. Per-night FPR distribution
ax = axes[0, 1]
if len(night_performance_df[night_performance_df['target_fpr'] == 'FPR_10']) > 0:
    fpr_10_data = night_performance_df[night_performance_df['target_fpr'] == 'FPR_10']['actual_fpr']
    ax.hist(fpr_10_data, bins=15, alpha=0.7, label='FPR < 10% threshold', color='red')

if len(night_performance_df[night_performance_df['target_fpr'] == 'FPR_5']) > 0:
    fpr_5_data = night_performance_df[night_performance_df['target_fpr'] == 'FPR_5']['actual_fpr']
    ax.hist(fpr_5_data, bins=15, alpha=0.7, label='FPR < 5% threshold', color='blue')

ax.axvline(x=0.10, color='red', linestyle='--', label='10% target')
ax.axvline(x=0.05, color='blue', linestyle='--', label='5% target')
ax.set_xlabel('Actual FPR per night')
ax.set_ylabel('Number of nights')
ax.set_title('Distribution of Actual FPR by Night')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Precision vs Recall scatter
ax = axes[0, 2]
for target_fpr, color, marker in [('FPR_10', 'red', 'o'), ('FPR_5', 'blue', 's')]:
    subset = night_performance_df[night_performance_df['target_fpr'] == target_fpr]
    if len(subset) > 0:
        ax.scatter(subset['tpr_recall'], subset['precision'], 
                  c=color, marker=marker, s=50, alpha=0.7, 
                  label=f'{target_fpr.replace("_", " < ")}%')

ax.set_xlabel('TPR (Recall)')
ax.set_ylabel('Precision')
ax.set_title('Precision vs Recall by Night')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Performance by Night - FPR < 10%
ax = axes[1, 0]
if len(night_performance_df[night_performance_df['target_fpr'] == 'FPR_10']) > 0:
    subset_10 = night_performance_df[night_performance_df['target_fpr'] == 'FPR_10']
    x_pos = np.arange(len(subset_10))
    width = 0.25
    
    ax.bar(x_pos - width, subset_10['actual_fpr'], width, label='Actual FPR', alpha=0.8, color='red')
    ax.bar(x_pos, subset_10['tpr_recall'], width, label='TPR (Recall)', alpha=0.8, color='green')
    ax.bar(x_pos + width, subset_10['precision'], width, label='Precision', alpha=0.8, color='blue')
    
    ax.axhline(y=0.10, color='red', linestyle='--', alpha=0.7, label='10% FPR target')
    ax.set_xlabel('Night')
    ax.set_ylabel('Score')
    ax.set_title('FPR < 10% Performance by Night')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subset_10['night'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45)

# 5. Performance by Night - FPR < 5%
ax = axes[1, 1]
if len(night_performance_df[night_performance_df['target_fpr'] == 'FPR_5']) > 0:
    subset_5 = night_performance_df[night_performance_df['target_fpr'] == 'FPR_5']
    x_pos = np.arange(len(subset_5))
    width = 0.25
    
    ax.bar(x_pos - width, subset_5['actual_fpr'], width, label='Actual FPR', alpha=0.8, color='red')
    ax.bar(x_pos, subset_5['tpr_recall'], width, label='TPR (Recall)', alpha=0.8, color='green')
    ax.bar(x_pos + width, subset_5['precision'], width, label='Precision', alpha=0.8, color='blue')
    
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% FPR target')
    ax.set_xlabel('Night')
    ax.set_ylabel('Score')
    ax.set_title('FPR < 5% Performance by Night')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subset_5['night'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45)

# 6. Threshold comparison
ax = axes[1, 2]
thresholds_data = []
performance_data = []

for target_fpr in ["FPR_10", "FPR_5"]:
    subset = night_performance_df[night_performance_df['target_fpr'] == target_fpr]
    if len(subset) > 0:
        thresholds_data.append(subset.iloc[0]['threshold_value'])
        performance_data.append([
            subset['tpr_recall'].mean(),
            subset['precision'].mean(),
            subset['actual_fpr'].mean()
        ])

if thresholds_data:
    x_labels = ['FPR < 10%', 'FPR < 5%'][:len(thresholds_data)]
    x_pos = np.arange(len(x_labels))
    width = 0.25
    
    tpr_vals = [p[0] for p in performance_data]
    prec_vals = [p[1] for p in performance_data]
    fpr_vals = [p[2] for p in performance_data]
    
    ax.bar(x_pos - width, tpr_vals, width, label='Mean TPR', alpha=0.8)
    ax.bar(x_pos, prec_vals, width, label='Mean Precision', alpha=0.8)
    ax.bar(x_pos + width, fpr_vals, width, label='Mean FPR', alpha=0.8)
    
    ax.set_xlabel('Threshold Type')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fpr_based_threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n" + "=" * 70)
print(f"FINAL EMPIRICAL THRESHOLDS FOR FPR TARGETS:")
print(f"=" * 70)

if threshold_fpr_10 is not None:
    print(f"✅ For FPR < 10%: Use threshold {threshold_fpr_10:.4f}")
    print(f"   Expected TPR (Recall): ~{global_tpr_10*100:.1f}%")
    print(f"   Expected Precision: ~{global_precision*100:.1f}%")
else:
    print(f"❌ FPR < 10% target not achievable with current data")

if threshold_fpr_5 is not None:
    print(f"✅ For FPR < 5%: Use threshold {threshold_fpr_5:.4f}")
    print(f"   Expected TPR (Recall): ~{global_tpr_5*100:.1f}%") 
    print(f"   Expected Precision: ~{global_precision*100:.1f}%")
else:
    print(f"❌ FPR < 5% target not achievable with current data")

print(f"\nFiles generated:")
print(f"- fpr_based_threshold_analysis.csv")
print(f"- fpr_based_threshold_analysis.png")
print(f"=" * 70)
