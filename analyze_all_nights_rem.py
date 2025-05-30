import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib
import mne
import yasa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # Use non-interactive backend

print("=== REM Analysis Across All 20 Nights ===")

def analyze_single_night(night_num):
    """Analyze a single night for REM detection performance."""
    try:
        # Load data
        edf_file = f'provided_data/night_{night_num:02d}.edf'
        label_file = f'provided_data/night_{night_num:02d}_label.csv'
        
        # Load ground truth labels
        labels_df = pd.read_csv(label_file)
        
        # Load EDF and process
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        raw.rename_channels({'LF-FpZ': 'Fp1', 'RF-FpZ': 'Fp2'})
        
        # YASA sleep staging
        sls = yasa.SleepStaging(raw, eeg_name='Fp2')
        yasa_predictions = sls.predict()
        yasa_proba = sls.predict_proba()
        
        # Align data lengths
        min_length = min(len(labels_df) - 1, len(yasa_predictions))
        ground_truth = labels_df['Sleep stage'].iloc[1:min_length+1].values
        yasa_pred = yasa_predictions[:min_length]
        yasa_prob = yasa_proba.iloc[:min_length]
        
        # Create binary REM classification
        ground_truth_rem = (ground_truth == 'REM').astype(int)
        yasa_pred_rem = (yasa_pred == 'R').astype(int)
        rem_probabilities = yasa_prob['R'].values
        
        # Calculate ROC curve and optimal threshold
        if ground_truth_rem.sum() > 0:  # Only if there are REM epochs
            fpr, tpr, thresholds = roc_curve(ground_truth_rem, rem_probabilities)
            roc_auc = auc(fpr, tpr)
            
            # Find optimal threshold using Youden's index
            youden_scores = tpr - fpr
            optimal_idx = np.argmax(youden_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Apply optimal threshold
            yasa_pred_rem_optimized = (rem_probabilities >= optimal_threshold).astype(int)
            
            # Calculate performance metrics
            original_accuracy = accuracy_score(ground_truth_rem, yasa_pred_rem)
            optimized_accuracy = accuracy_score(ground_truth_rem, yasa_pred_rem_optimized)
            
            # Confusion matrix for optimized
            cm_opt = confusion_matrix(ground_truth_rem, yasa_pred_rem_optimized)
            tn, fp, fn, tp = cm_opt.ravel() if cm_opt.size == 4 else (0, 0, 0, 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Precision-Recall AUC
            prec, rec, _ = precision_recall_curve(ground_truth_rem, rem_probabilities)
            pr_auc = auc(rec, prec)
            
        else:
            # No REM epochs in this night
            roc_auc = np.nan
            pr_auc = np.nan
            optimal_threshold = np.nan
            original_accuracy = np.nan
            optimized_accuracy = np.nan
            precision = np.nan
            recall = np.nan
            specificity = np.nan
            f1 = np.nan
            tp = fp = tn = fn = 0
        
        # REM episode analysis
        rem_epochs_idx = np.where(ground_truth_rem == 1)[0]
        rem_episodes = []
        
        if len(rem_epochs_idx) > 0:
            episode_start = rem_epochs_idx[0]
            episode_end = rem_epochs_idx[0]
            
            for i in range(1, len(rem_epochs_idx)):
                if rem_epochs_idx[i] == rem_epochs_idx[i-1] + 1:  # Consecutive
                    episode_end = rem_epochs_idx[i]
                else:  # Gap found
                    rem_episodes.append((episode_start, episode_end))
                    episode_start = rem_epochs_idx[i]
                    episode_end = rem_epochs_idx[i]
            
            rem_episodes.append((episode_start, episode_end))
        
        return {
            'night': night_num,
            'total_epochs': min_length,
            'duration_hours': min_length / 120,
            'rem_epochs_total': ground_truth_rem.sum(),
            'rem_episodes_count': len(rem_episodes),
            'rem_percentage': ground_truth_rem.mean() * 100,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'optimal_threshold': optimal_threshold,
            'original_accuracy': original_accuracy,
            'optimized_accuracy': optimized_accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'true_positives': int(tp) if not np.isnan(tp) else 0,
            'false_positives': int(fp) if not np.isnan(fp) else 0,
            'true_negatives': int(tn) if not np.isnan(tn) else 0,
            'false_negatives': int(fn) if not np.isnan(fn) else 0,
            'mean_rem_probability': rem_probabilities.mean(),
            'rem_prob_when_true': rem_probabilities[ground_truth_rem == 1].mean() if ground_truth_rem.sum() > 0 else np.nan,
            'rem_prob_when_false': rem_probabilities[ground_truth_rem == 0].mean(),
            'yasa_rem_predicted': yasa_pred_rem.sum(),
            'yasa_rem_optimized': yasa_pred_rem_optimized.sum() if not np.isnan(optimal_threshold) else 0
        }
        
    except Exception as e:
        print(f"Error processing night {night_num}: {str(e)}")
        return None

# Analyze all nights
results = []
print("Processing all nights...")

for night in tqdm(range(1, 21), desc="Analyzing nights"):
    result = analyze_single_night(night)
    if result is not None:
        results.append(result)

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save detailed results
df_results.to_csv('all_nights_rem_analysis.csv', index=False)
print("Detailed results saved to all_nights_rem_analysis.csv")

# Calculate summary statistics
print("\n=== SUMMARY STATISTICS ACROSS ALL 20 NIGHTS ===")
print(f"Successfully processed: {len(df_results)} nights")

# Filter out nights with no REM for some calculations
df_with_rem = df_results[df_results['rem_epochs_total'] > 0]
print(f"Nights with REM sleep: {len(df_with_rem)}")

# Overall statistics
total_epochs = df_results['total_epochs'].sum()
total_rem_epochs = df_results['rem_epochs_total'].sum()
total_duration = df_results['duration_hours'].sum()

print(f"\nDataset Overview:")
print(f"  Total epochs analyzed: {total_epochs:,}")
print(f"  Total REM epochs: {total_rem_epochs:,}")
print(f"  Total recording time: {total_duration:.1f} hours")
print(f"  Overall REM percentage: {(total_rem_epochs/total_epochs)*100:.1f}%")

# REM detection performance
if len(df_with_rem) > 0:
    print(f"\nREM Detection Performance (nights with REM):")
    print(f"  Mean ROC AUC: {df_with_rem['roc_auc'].mean():.3f} ± {df_with_rem['roc_auc'].std():.3f}")
    print(f"  Mean PR AUC: {df_with_rem['pr_auc'].mean():.3f} ± {df_with_rem['pr_auc'].std():.3f}")
    print(f"  Mean optimal threshold: {df_with_rem['optimal_threshold'].mean():.3f} ± {df_with_rem['optimal_threshold'].std():.3f}")
    
    print(f"\nOriginal YASA Performance:")
    print(f"  Mean accuracy: {df_with_rem['original_accuracy'].mean():.3f} ± {df_with_rem['original_accuracy'].std():.3f}")
    
    print(f"\nOptimized Threshold Performance:")
    print(f"  Mean accuracy: {df_with_rem['optimized_accuracy'].mean():.3f} ± {df_with_rem['optimized_accuracy'].std():.3f}")
    print(f"  Mean precision: {df_with_rem['precision'].mean():.3f} ± {df_with_rem['precision'].std():.3f}")
    print(f"  Mean recall: {df_with_rem['recall'].mean():.3f} ± {df_with_rem['recall'].std():.3f}")
    print(f"  Mean specificity: {df_with_rem['specificity'].mean():.3f} ± {df_with_rem['specificity'].std():.3f}")
    print(f"  Mean F1-score: {df_with_rem['f1_score'].mean():.3f} ± {df_with_rem['f1_score'].std():.3f}")

# REM episode statistics
print(f"\nREM Episode Statistics:")
print(f"  Total REM episodes across all nights: {df_results['rem_episodes_count'].sum()}")
print(f"  Mean REM episodes per night: {df_results['rem_episodes_count'].mean():.1f} ± {df_results['rem_episodes_count'].std():.1f}")
print(f"  Nights with no REM: {len(df_results[df_results['rem_epochs_total'] == 0])}")

# Create comprehensive visualization
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# 1. ROC AUC distribution
axes[0, 0].hist(df_with_rem['roc_auc'], bins=15, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].axvline(df_with_rem['roc_auc'].mean(), color='red', linestyle='--', label=f'Mean: {df_with_rem["roc_auc"].mean():.3f}')
axes[0, 0].set_xlabel('ROC AUC')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('ROC AUC Distribution Across Nights')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. REM percentage per night
night_nums = df_results['night']
rem_percentages = df_results['rem_percentage']
axes[0, 1].bar(night_nums, rem_percentages, alpha=0.7, color='green')
axes[0, 1].set_xlabel('Night')
axes[0, 1].set_ylabel('REM Percentage (%)')
axes[0, 1].set_title('REM Percentage by Night')
axes[0, 1].grid(True, alpha=0.3)

# 3. Optimal threshold distribution
axes[0, 2].hist(df_with_rem['optimal_threshold'], bins=15, alpha=0.7, color='orange', edgecolor='black')
axes[0, 2].axvline(df_with_rem['optimal_threshold'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df_with_rem["optimal_threshold"].mean():.3f}')
axes[0, 2].set_xlabel('Optimal Threshold')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Optimal Threshold Distribution')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Performance comparison (Original vs Optimized)
x_pos = np.arange(len(df_with_rem))
width = 0.35
axes[1, 0].bar(x_pos - width/2, df_with_rem['original_accuracy'], width, 
               label='Original YASA', alpha=0.7, color='red')
axes[1, 0].bar(x_pos + width/2, df_with_rem['optimized_accuracy'], width,
               label='Optimized Threshold', alpha=0.7, color='blue')
axes[1, 0].set_xlabel('Night (with REM)')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('Original vs Optimized Accuracy')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(df_with_rem['night'])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Precision-Recall-F1 scores
metrics = ['precision', 'recall', 'f1_score']
metric_means = [df_with_rem[metric].mean() for metric in metrics]
metric_stds = [df_with_rem[metric].std() for metric in metrics]
axes[1, 1].bar(metrics, metric_means, yerr=metric_stds, capsize=5, alpha=0.7, color=['blue', 'green', 'orange'])
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Mean Performance Metrics (Optimized)')
axes[1, 1].set_ylim(0, 1)
for i, (mean, std) in enumerate(zip(metric_means, metric_stds)):
    axes[1, 1].text(i, mean + std + 0.05, f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
axes[1, 1].grid(True, alpha=0.3)

# 6. REM episodes vs REM percentage correlation
axes[1, 2].scatter(df_results['rem_percentage'], df_results['rem_episodes_count'], alpha=0.7, color='purple')
axes[1, 2].set_xlabel('REM Percentage (%)')
axes[1, 2].set_ylabel('Number of REM Episodes')
axes[1, 2].set_title('REM Episodes vs REM Percentage')
# Add correlation coefficient
corr = np.corrcoef(df_results['rem_percentage'], df_results['rem_episodes_count'])[0, 1]
axes[1, 2].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, 2].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[1, 2].grid(True, alpha=0.3)

# 7. ROC AUC vs REM percentage
axes[2, 0].scatter(df_with_rem['rem_percentage'], df_with_rem['roc_auc'], alpha=0.7, color='red')
axes[2, 0].set_xlabel('REM Percentage (%)')
axes[2, 0].set_ylabel('ROC AUC')
axes[2, 0].set_title('ROC AUC vs REM Percentage')
# Add correlation
corr_roc = np.corrcoef(df_with_rem['rem_percentage'], df_with_rem['roc_auc'])[0, 1]
axes[2, 0].text(0.05, 0.95, f'r = {corr_roc:.3f}', transform=axes[2, 0].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[2, 0].grid(True, alpha=0.3)

# 8. Confusion matrix heatmap (aggregated)
total_tp = df_with_rem['true_positives'].sum()
total_fp = df_with_rem['false_positives'].sum()
total_tn = df_with_rem['true_negatives'].sum()
total_fn = df_with_rem['false_negatives'].sum()

cm_total = np.array([[total_tn, total_fp], [total_fn, total_tp]])
cm_normalized = cm_total.astype('float') / cm_total.sum(axis=1, keepdims=True)

sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
            xticklabels=['Non-REM', 'REM'], yticklabels=['Non-REM', 'REM'], ax=axes[2, 1])
axes[2, 1].set_title('Normalized Confusion Matrix\n(Aggregated Across All Nights)')
axes[2, 1].set_ylabel('True Label')
axes[2, 1].set_xlabel('Predicted Label')

# 9. Performance metrics by night (box plot style)
metric_data = []
metric_names = []
for metric in ['roc_auc', 'precision', 'recall', 'f1_score']:
    metric_data.extend(df_with_rem[metric].tolist())
    metric_names.extend([metric] * len(df_with_rem))

metric_df = pd.DataFrame({'Metric': metric_names, 'Value': metric_data})
sns.boxplot(data=metric_df, x='Metric', y='Value', ax=axes[2, 2])
axes[2, 2].set_title('Performance Metrics Distribution')
axes[2, 2].set_ylabel('Score')
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('all_nights_rem_analysis_comprehensive.png', dpi=150, bbox_inches='tight')
print("Comprehensive analysis plot saved as all_nights_rem_analysis_comprehensive.png")
plt.close()

# Create summary table for best and worst performing nights
print("\n=== BEST AND WORST PERFORMING NIGHTS ===")

if len(df_with_rem) > 0:
    # Sort by ROC AUC
    df_sorted = df_with_rem.sort_values('roc_auc', ascending=False)
    
    print("\nTop 5 Nights by ROC AUC:")
    top_5 = df_sorted.head(5)[['night', 'roc_auc', 'precision', 'recall', 'f1_score', 'rem_epochs_total']]
    print(top_5.to_string(index=False, float_format='%.3f'))
    
    print("\nBottom 5 Nights by ROC AUC:")
    bottom_5 = df_sorted.tail(5)[['night', 'roc_auc', 'precision', 'recall', 'f1_score', 'rem_epochs_total']]
    print(bottom_5.to_string(index=False, float_format='%.3f'))

# Overall aggregated performance
print(f"\n=== OVERALL AGGREGATED PERFORMANCE ===")
if len(df_with_rem) > 0:
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    
    print(f"Aggregated across all nights with REM:")
    print(f"  Total True Positives: {total_tp}")
    print(f"  Total False Positives: {total_fp}")
    print(f"  Total True Negatives: {total_tn}")
    print(f"  Total False Negatives: {total_fn}")
    print(f"  Overall Precision: {overall_precision:.3f}")
    print(f"  Overall Recall: {overall_recall:.3f}")
    print(f"  Overall Specificity: {overall_specificity:.3f}")
    print(f"  Overall F1-Score: {overall_f1:.3f}")
    print(f"  Overall Accuracy: {overall_accuracy:.3f}")

print(f"\n=== FILES GENERATED ===")
print(f"  all_nights_rem_analysis.csv - Detailed per-night results")
print(f"  all_nights_rem_analysis_comprehensive.png - Comprehensive visualization")
print(f"\nAnalysis complete!")
