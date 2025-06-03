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
first_night_data = None  # Store first night for plotting

print("Processing all 20 nights to calculate ROC curves...")


# TODO check this function
def load_process_eeg_datafile(session_folder_path):
    channel_eeg = 6

    # Load the recorded session from your experiment folder
    memmap_eeg_data = np.memmap(session_folder_path, dtype=np.float64, mode='r')
    l_eeg = int(len(memmap_eeg_data)//channel_eeg)

    # EEG DATA
    # Reshape the data into a 2D array
    # l_eeg is the number of samples in the eeg data
    # channel_eeg is the number of channels in the eeg data
    eeg_data = np.array(memmap_eeg_data)[:l_eeg*channel_eeg].reshape((l_eeg, channel_eeg))
    raw_data = eeg_data[:, [0, 1, 3, 4]].T #* 1e-8

    channel_names = ['LF-FpZ', 'OTE_L-FpZ', 'RF-FpZ', 'OTE_R-FpZ']
    sfreq = 125 

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(channel_names)
    )

    raw = mne.io.RawArray(raw_data, info)
    raw.pick_channels(channel_names)

    raw.filter(0.5,40)
    raw.notch_filter(60)
    return raw

# Process all nights to get comprehensive ROC data
for night in range(1, 21):
    print(f"Processing Night {night}...")
    
    try:
        # Load data
        edf_file = f'provided_data/night_{night:02d}.edf'
        dat_file = f'provided_data/night_{night:02d}.dat'

        label_file = f'provided_data/night_{night:02d}_label.csv'
        
        # Load ground truth
        labels_df = pd.read_csv(label_file)
        
        # Load EEG and get YASA predictions
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False) 
        # raw = load_process_eeg_datafile(dat_file) # SWITCHING DATA TYPES TODO check here

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
        
        # Check if there are any REM epochs in this night
        if ground_truth_rem.sum() == 0:
            print(f"  Warning: Night {night} has no REM epochs, skipping ROC calculation")
            continue
        
        # Store first night data for plotting
        if first_night_data is None:
            first_night_data = {
                'night': night,
                'ground_truth': ground_truth,
                'yasa_predictions': yasa_pred,
                'yasa_probabilities': yasa_prob,
                'timestamps': labels_df['Timestamp'].iloc[1:min_length+1].values if 'Timestamp' in labels_df.columns else np.arange(min_length)
            }
        
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
        print(f"  Error processing night {night}: {e}")
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

fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle('Empirical FPR-Based REM Threshold Analysis (All 20 Nights)', fontsize=16, fontweight='bold')

# 1. First Night Sleep Data Comparison
ax = axes[0, 0]
if first_night_data is not None:
    # Create sleep stage mappings for plotting (reordered: Deep, Light, REM, Wake)
    stage_map = {'Deep': 1, 'Light': 2, 'REM': 3, 'Wake': 4}
    
    # Convert ground truth stages to numeric values
    ground_truth_numeric = [stage_map.get(stage, 0) for stage in first_night_data['ground_truth']]
    
    # Convert YASA to binary REM classification using optimal threshold
    rem_probs = first_night_data['yasa_probabilities']['R'].values
    if threshold_fpr_10 is not None:
        yasa_rem_binary = (rem_probs >= threshold_fpr_10).astype(int)
        threshold_used = threshold_fpr_10
        threshold_name = "FPR<10%"
    elif threshold_fpr_5 is not None:
        yasa_rem_binary = (rem_probs >= threshold_fpr_5).astype(int)
        threshold_used = threshold_fpr_5
        threshold_name = "FPR<5%"
    else:
        # Fallback to simple threshold
        yasa_rem_binary = (rem_probs >= 0.5).astype(int)
        threshold_used = 0.5
        threshold_name = "Default"
    
    # Convert binary REM to plotting values (3 for REM, 2.5 for non-REM to distinguish)
    yasa_binary_plot = np.where(yasa_rem_binary == 1, 3.0, 1.5)  # REM=3, Non-REM=1.5
    
    # Create time axis (assuming 30-second epochs)
    time_hours = np.arange(len(ground_truth_numeric)) * 30 / 3600  # Convert to hours
    
    # Plot both traces
    ax.plot(time_hours, ground_truth_numeric, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(time_hours, yasa_binary_plot, 'r-', linewidth=1.5, 
           label=f'YASA REM Binary ({threshold_name} th={threshold_used:.3f})', alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Sleep Stage')
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['Deep', 'Light', 'REM', 'Wake'])
    ax.set_title(f'Night {first_night_data["night"]}: Sleep Stages vs Binary REM Classification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 4.5])
    
    # Add horizontal line to show REM threshold
    ax.axhline(y=3, color='red', linestyle=':', alpha=0.5, label='REM Level')
else:
    ax.text(0.5, 0.5, 'No valid first night data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('First Night Data (Not Available)')

# Generate REM classification comparison plots for all 20 nights
print(f"\nGenerating REM classification plots for all nights...")

# Create directory for plots
import os
plot_dir = 'rem_night_plots'
os.makedirs(plot_dir, exist_ok=True)

# Determine which threshold to use for all plots
if threshold_fpr_10 is not None:
    plot_threshold = threshold_fpr_10
    plot_threshold_name = "FPR<10%"
elif threshold_fpr_5 is not None:
    plot_threshold = threshold_fpr_5
    plot_threshold_name = "FPR<5%"
else:
    plot_threshold = 0.5
    plot_threshold_name = "Default"

# Group nights in sets of 4 for plotting
nights_per_plot = 4
total_plots = (len(night_roc_data) + nights_per_plot - 1) // nights_per_plot

for plot_idx in range(total_plots):
    start_idx = plot_idx * nights_per_plot
    end_idx = min(start_idx + nights_per_plot, len(night_roc_data))
    nights_in_plot = night_roc_data[start_idx:end_idx]
    
    # Create figure with subplots for this group
    fig_group, axes_group = plt.subplots(2, 2, figsize=(16, 12))
    fig_group.suptitle(f'REM Classification Comparison - YASA vs FRENZ', 
                       fontsize=14, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes_group.flatten()
    
    for i, night_data in enumerate(nights_in_plot):
        ax = axes_flat[i]
        
        # Get the ground truth and probabilities for this night
        ground_truth_stages = night_data.get('ground_truth_stages', None)
        rem_probabilities = night_data['probabilities']
        ground_truth_rem = night_data['ground_truth']
        night_num = night_data['night']
        
        # Create sleep stage mappings for plotting
        stage_map = {'Deep': 1, 'Light': 2, 'REM': 3, 'Wake': 4}
        
        # If we have stage labels, use them; otherwise create from binary REM data
        if ground_truth_stages is not None:
            # Convert ground truth stages to binary REM (3 for REM, 1.5 for non-REM)
            ground_truth_numeric = np.where([stage == 'REM' for stage in ground_truth_stages], 3.0, 1.5)
        else:
            # Create simplified ground truth from binary REM (3 for REM, 1.5 for non-REM)
            ground_truth_numeric = np.where(ground_truth_rem == 1, 3.0, 1.5)
        
        # Convert YASA to binary REM classification using the threshold
        yasa_rem_binary = (rem_probabilities >= plot_threshold).astype(int)
        
        # Convert binary REM to plotting values (3 for REM, 1.5 for non-REM to distinguish)
        yasa_binary_plot = np.where(yasa_rem_binary == 1, 3.0, 1.5)
        
        # Create time axis (assuming 30-second epochs)
        time_hours = np.arange(len(ground_truth_numeric)) * 30 / 3600
        
        # Plot both traces
        ax.plot(time_hours, ground_truth_numeric, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(time_hours, yasa_binary_plot, 'r-', linewidth=1.5, 
               label=f'YASA REM Binary ({plot_threshold_name})', alpha=0.7)
        
        # Customize the plot
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Sleep Classification')
        ax.set_yticks([1.5, 3.0])
        ax.set_yticklabels(['Non-REM', 'REM'])
        ax.set_title(f'Night {night_num}: Ground Truth vs Binary REM Classification')
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([1.0, 3.5])
        
        # Add horizontal line to show REM threshold
        ax.axhline(y=3.0, color='red', linestyle=':', alpha=0.5)
    
    # Hide unused subplots if less than 4 nights in this group
    for i in range(len(nights_in_plot), 4):
        axes_flat[i].set_visible(False)
    
    # Save the plot
    plot_filename = f'{plot_dir}/rem_comparison_nights_{nights_in_plot[0]["night"]}-{nights_in_plot[-1]["night"]}.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"  Saved: {plot_filename}")

print(f"All REM classification plots saved to '{plot_dir}/' directory")

# Calculate best AUC threshold using Youden's index (maximizing sensitivity + specificity - 1)
print(f"\nCalculating best AUC threshold using Youden's index...")
youden_scores = global_tpr + (1 - global_fpr) - 1  # Sensitivity + Specificity - 1
best_auc_idx = np.argmax(youden_scores)
best_auc_threshold = global_thresholds[best_auc_idx]
best_auc_fpr = global_fpr[best_auc_idx]
best_auc_tpr = global_tpr[best_auc_idx]
best_auc_specificity = 1 - best_auc_fpr

print(f"Best AUC threshold: {best_auc_threshold:.4f}")
print(f"At this threshold - FPR: {best_auc_fpr:.3f}, TPR: {best_auc_tpr:.3f}, Specificity: {best_auc_specificity:.3f}")
print(f"Youden's J statistic: {youden_scores[best_auc_idx]:.3f}")

# Generate REM classification comparison plots for all 20 nights using BEST AUC threshold
print(f"\nGenerating REM classification plots using best AUC threshold for all nights...")

# Create directory for best AUC plots
plot_dir_auc = 'rem_night_plots_best_auc'
os.makedirs(plot_dir_auc, exist_ok=True)

# Group nights in sets of 4 for plotting with best AUC threshold
for plot_idx in range(total_plots):
    start_idx = plot_idx * nights_per_plot
    end_idx = min(start_idx + nights_per_plot, len(night_roc_data))
    nights_in_plot = night_roc_data[start_idx:end_idx]
    
    # Create figure with subplots for this group
    fig_group, axes_group = plt.subplots(2, 2, figsize=(16, 12))
    fig_group.suptitle(f'REM Classification Comparison - YASA vs FRENZ (Best AUC Threshold)', 
                       fontsize=14, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes_group.flatten()
    
    for i, night_data in enumerate(nights_in_plot):
        ax = axes_flat[i]
        
        # Get the ground truth and probabilities for this night
        ground_truth_stages = night_data.get('ground_truth_stages', None)
        rem_probabilities = night_data['probabilities']
        ground_truth_rem = night_data['ground_truth']
        night_num = night_data['night']
        
        # Create sleep stage mappings for plotting
        stage_map = {'Deep': 1, 'Light': 2, 'REM': 3, 'Wake': 4}
        
        # If we have stage labels, use them; otherwise create from binary REM data
        if ground_truth_stages is not None:
            # Convert ground truth stages to binary REM (3 for REM, 1.5 for non-REM)
            ground_truth_numeric = np.where([stage == 'REM' for stage in ground_truth_stages], 3.0, 1.5)
        else:
            # Create simplified ground truth from binary REM (3 for REM, 1.5 for non-REM)
            ground_truth_numeric = np.where(ground_truth_rem == 1, 3.0, 1.5)
        
        # Convert YASA to binary REM classification using the BEST AUC threshold
        yasa_rem_binary = (rem_probabilities >= best_auc_threshold).astype(int)
        
        # Convert binary REM to plotting values (3 for REM, 1.5 for non-REM to distinguish)
        yasa_binary_plot = np.where(yasa_rem_binary == 1, 3.0, 1.5)
        
        # Create time axis (assuming 30-second epochs)
        time_hours = np.arange(len(ground_truth_numeric)) * 30 / 3600
        
        # Plot both traces
        ax.plot(time_hours, ground_truth_numeric, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(time_hours, yasa_binary_plot, 'g-', linewidth=1.5, 
               label=f'YASA REM Binary (Best AUC)', alpha=0.7)
        
        # Customize the plot
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Sleep Classification')
        ax.set_yticks([1.5, 3.0])
        ax.set_yticklabels(['Non-REM', 'REM'])
        ax.set_title(f'Night {night_num}: Ground Truth vs Binary REM Classification (Best AUC)')
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([1.0, 3.5])
        
        # Add horizontal line to show REM threshold
        ax.axhline(y=3.0, color='green', linestyle=':', alpha=0.5)
    
    # Hide unused subplots if less than 4 nights in this group
    for i in range(len(nights_in_plot), 4):
        axes_flat[i].set_visible(False)
    
    # Save the plot
    plot_filename = f'{plot_dir_auc}/rem_comparison_nights_{nights_in_plot[0]["night"]}-{nights_in_plot[-1]["night"]}_best_auc.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"  Saved: {plot_filename}")

print(f"All REM classification plots (best AUC threshold) saved to '{plot_dir_auc}/' directory")

# Calculate performance metrics for the best AUC threshold across all nights
print(f"\nCalculating performance metrics for best AUC threshold across all nights...")
best_auc_performance = []

for night_data in night_roc_data:
    night = night_data['night']
    ground_truth_rem = night_data['ground_truth']
    rem_probabilities = night_data['probabilities']
    
    # Apply best AUC threshold
    rem_pred_threshold = (rem_probabilities >= best_auc_threshold).astype(int)
    
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
    
    best_auc_performance.append({
        'night': night,
        'target_fpr': 'BEST_AUC',
        'threshold_value': best_auc_threshold,
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

# Add best AUC performance to the main dataframe
best_auc_df = pd.DataFrame(best_auc_performance)
combined_performance_df = pd.concat([night_performance_df, best_auc_df], ignore_index=True)

# Save the combined results
combined_performance_df.to_csv('fpr_based_threshold_analysis_with_best_auc.csv', index=False)

# Calculate summary statistics for best AUC threshold
best_auc_subset = best_auc_df
total_tp_auc = best_auc_subset['true_positives'].sum()
total_fp_auc = best_auc_subset['false_positives'].sum()
total_tn_auc = best_auc_subset['true_negatives'].sum()
total_fn_auc = best_auc_subset['false_negatives'].sum()

global_fpr_auc_actual = total_fp_auc / (total_fp_auc + total_tn_auc) if (total_fp_auc + total_tn_auc) > 0 else 0
global_tpr_auc_actual = total_tp_auc / (total_tp_auc + total_fn_auc) if (total_tp_auc + total_fn_auc) > 0 else 0
global_precision_auc = total_tp_auc / (total_tp_auc + total_fp_auc) if (total_tp_auc + total_fp_auc) > 0 else 0

# Per-night statistics
mean_fpr_auc = best_auc_subset['actual_fpr'].mean()
mean_tpr_auc = best_auc_subset['tpr_recall'].mean()
mean_precision_auc = best_auc_subset['precision'].mean()

print(f"\n🎯 BEST AUC THRESHOLD:")
print(f"   Threshold value: {best_auc_threshold:.4f}")
print(f"   Global actual FPR: {global_fpr_auc_actual:.3f} ({global_fpr_auc_actual*100:.1f}%)")
print(f"   Global TPR (Recall): {global_tpr_auc_actual:.3f} ({global_tpr_auc_actual*100:.1f}%)")
print(f"   Global Precision: {global_precision_auc:.3f} ({global_precision_auc*100:.1f}%)")
print(f"   Mean per-night FPR: {mean_fpr_auc:.3f} (±{best_auc_subset['actual_fpr'].std():.3f})")
print(f"   Mean per-night TPR: {mean_tpr_auc:.3f} (±{best_auc_subset['tpr_recall'].std():.3f})")
print(f"   Mean per-night Precision: {mean_precision_auc:.3f} (±{best_auc_subset['precision'].std():.3f})")
print(f"   Total true positives: {total_tp_auc}")
print(f"   Total false positives: {total_fp_auc}")

print(f"\nFiles generated:")
print(f"- fpr_based_threshold_analysis.csv")
print(f"- fpr_based_threshold_analysis.png")
print(f"- fpr_based_threshold_analysis_with_best_auc.csv")
print(f"=" * 70)

# Generate REM probability score plots for all 20 nights with threshold lines
print(f"\nGenerating REM probability score plots with threshold lines for all nights...")

# Create directory for probability score plots
prob_plot_dir = 'rem_probability_plots'
os.makedirs(prob_plot_dir, exist_ok=True)

# Group nights in sets of 4 for plotting
nights_per_plot = 4
total_plots = (len(night_roc_data) + nights_per_plot - 1) // nights_per_plot

for plot_idx in range(total_plots):
    start_idx = plot_idx * nights_per_plot
    end_idx = min(start_idx + nights_per_plot, len(night_roc_data))
    nights_in_plot = night_roc_data[start_idx:end_idx]
    
    # Create figure with subplots for this group
    fig_group, axes_group = plt.subplots(2, 2, figsize=(16, 12))
    fig_group.suptitle(f'YASA REM Probability Scores vs Ground Truth (FRENZ) with Threshold Lines', 
                       fontsize=14, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes_group.flatten()
    
    for i, night_data in enumerate(nights_in_plot):
        ax = axes_flat[i]
        
        # Get the ground truth and probabilities for this night
        rem_probabilities = night_data['probabilities']
        ground_truth_rem = night_data['ground_truth']
        night_num = night_data['night']
        
        # Create time axis (assuming 30-second epochs)
        time_hours = np.arange(len(ground_truth_rem)) * 30 / 3600
        
        # Convert binary ground truth to plotting values (offset for visibility)
        # REM = 1.0, Non-REM = 0.0, but offset Non-REM slightly for better visibility
        ground_truth_plot = ground_truth_rem.astype(float)
        
        # Plot REM probabilities as a line
        ax.plot(time_hours, rem_probabilities, 'b-', linewidth=1, label='YASA REM Probability', alpha=0.7)
        
        # Plot ground truth as filled areas
        ax.fill_between(time_hours, 0, ground_truth_plot, 
                       where=(ground_truth_rem == 1), alpha=0.3, color='red', 
                       label='Ground Truth REM', interpolate=True)
        
        # Add threshold lines
        if best_auc_threshold is not None:
            ax.axhline(y=best_auc_threshold, color='green', linestyle='-', alpha=0.8, 
                      linewidth=2, label=f'Best AUC ({best_auc_threshold:.3f})')
        
        if threshold_fpr_10 is not None:
            ax.axhline(y=threshold_fpr_10, color='orange', linestyle='--', alpha=0.8, 
                      linewidth=2, label=f'FPR≤10% ({threshold_fpr_10:.3f})')
        
        if threshold_fpr_5 is not None:
            ax.axhline(y=threshold_fpr_5, color='red', linestyle='--', alpha=0.8, 
                      linewidth=2, label=f'FPR≤5% ({threshold_fpr_5:.3f})')
        
        # Customize the plot
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('REM Probability / Ground Truth')
        ax.set_title(f'Night {night_num}: REM Probability vs Ground Truth')
        ax.legend(fontsize='small', loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, 1.0])
        
        # Add text annotation with night statistics
        rem_epochs = ground_truth_rem.sum()
        total_epochs = len(ground_truth_rem)
        rem_percentage = rem_epochs / total_epochs * 100
        ax.text(0.02, 0.95, f'REM: {rem_epochs}/{total_epochs} ({rem_percentage:.1f}%)', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Hide unused subplots if less than 4 nights in this group
    for i in range(len(nights_in_plot), 4):
        axes_flat[i].set_visible(False)
    
    # Save the plot
    plot_filename = f'{prob_plot_dir}/rem_probability_nights_{nights_in_plot[0]["night"]}-{nights_in_plot[-1]["night"]}.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"  Saved: {plot_filename}")

print(f"All REM probability score plots saved to '{prob_plot_dir}/' directory")
