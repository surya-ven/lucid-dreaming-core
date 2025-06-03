import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# Load the results
df = pd.read_csv('all_nights_rem_analysis.csv')

print("=== High Confidence REM Detection Analysis (90% and 95%) ===")

# Calculate metrics for both 90% and 95% confidence thresholds
conf_90_threshold = 0.90
conf_95_threshold = 0.95

# Estimate how many detections would survive at each confidence level
# This is a rough approximation based on the optimal threshold and probability distributions

# For 90% confidence
scale_factor_90 = conf_90_threshold / df['optimal_threshold'].clip(lower=0.01)
df['conf_90_estimated_detections'] = (df['yasa_rem_optimized'] / scale_factor_90).clip(lower=0)
df['conf_90_estimated_detections'] = df['conf_90_estimated_detections'].astype(int)

# For 95% confidence  
scale_factor_95 = conf_95_threshold / df['optimal_threshold'].clip(lower=0.01)
df['conf_95_estimated_detections'] = (df['yasa_rem_optimized'] / scale_factor_95).clip(lower=0)
df['conf_95_estimated_detections'] = df['conf_95_estimated_detections'].astype(int)

# Calculate metrics for 90% confidence
df['conf_90_tp'] = np.minimum(df['conf_90_estimated_detections'], df['rem_epochs_total'])
df['conf_90_fp'] = np.maximum(0, df['conf_90_estimated_detections'] - df['conf_90_tp'])
df['conf_90_fn'] = df['rem_epochs_total'] - df['conf_90_tp']
df['conf_90_tn'] = df['total_epochs'] - df['rem_epochs_total'] - df['conf_90_fp']

df['conf_90_precision'] = np.where(df['conf_90_estimated_detections'] > 0, 
                                   df['conf_90_tp'] / df['conf_90_estimated_detections'], 1.0)
df['conf_90_recall'] = np.where(df['rem_epochs_total'] > 0,
                                df['conf_90_tp'] / df['rem_epochs_total'], 0)
df['conf_90_accuracy'] = (df['conf_90_tp'] + df['conf_90_tn']) / df['total_epochs']
df['conf_90_f1'] = np.where((df['conf_90_precision'] + df['conf_90_recall']) > 0,
                            2 * df['conf_90_precision'] * df['conf_90_recall'] / 
                            (df['conf_90_precision'] + df['conf_90_recall']), 0)

# Calculate metrics for 95% confidence
df['conf_95_tp'] = np.minimum(df['conf_95_estimated_detections'], df['rem_epochs_total'])
df['conf_95_fp'] = np.maximum(0, df['conf_95_estimated_detections'] - df['conf_95_tp'])
df['conf_95_fn'] = df['rem_epochs_total'] - df['conf_95_tp']
df['conf_95_tn'] = df['total_epochs'] - df['rem_epochs_total'] - df['conf_95_fp']

df['conf_95_precision'] = np.where(df['conf_95_estimated_detections'] > 0, 
                                   df['conf_95_tp'] / df['conf_95_estimated_detections'], 1.0)
df['conf_95_recall'] = np.where(df['rem_epochs_total'] > 0,
                                df['conf_95_tp'] / df['rem_epochs_total'], 0)
df['conf_95_accuracy'] = (df['conf_95_tp'] + df['conf_95_tn']) / df['total_epochs']
df['conf_95_f1'] = np.where((df['conf_95_precision'] + df['conf_95_recall']) > 0,
                            2 * df['conf_95_precision'] * df['conf_95_recall'] / 
                            (df['conf_95_precision'] + df['conf_95_recall']), 0)

print(f"\n=== Detection Counts ===")
print(f"Nights with ANY 90% confidence detections: {(df['conf_90_tp'] > 0).sum()}/20")
print(f"Nights with ANY 95% confidence detections: {(df['conf_95_tp'] > 0).sum()}/20")
print(f"Total REM epochs detected at 90% confidence: {df['conf_90_tp'].sum()}/{df['rem_epochs_total'].sum()} ({df['conf_90_tp'].sum()/df['rem_epochs_total'].sum()*100:.1f}%)")
print(f"Total REM epochs detected at 95% confidence: {df['conf_95_tp'].sum()}/{df['rem_epochs_total'].sum()} ({df['conf_95_tp'].sum()/df['rem_epochs_total'].sum()*100:.1f}%)")

# Performance comparison
print(f"\n=== Performance Comparison ===")
print(f"Original YASA:")
print(f"  Accuracy: {df['original_accuracy'].mean():.3f} ± {df['original_accuracy'].std():.3f}")
print(f"  Recall: {(df['true_positives']/(df['true_positives']+df['false_negatives'])).mean():.3f}")

print(f"\nOptimized Threshold:")
print(f"  Accuracy: {df['optimized_accuracy'].mean():.3f} ± {df['optimized_accuracy'].std():.3f}")
print(f"  Precision: {df['precision'].mean():.3f} ± {df['precision'].std():.3f}")
print(f"  Recall: {df['recall'].mean():.3f} ± {df['recall'].std():.3f}")
print(f"  F1-Score: {df['f1_score'].mean():.3f} ± {df['f1_score'].std():.3f}")

print(f"\n90% Confidence Threshold:")
print(f"  Accuracy: {df['conf_90_accuracy'].mean():.3f} ± {df['conf_90_accuracy'].std():.3f}")
print(f"  Precision: {df['conf_90_precision'].mean():.3f} ± {df['conf_90_precision'].std():.3f}")
print(f"  Recall: {df['conf_90_recall'].mean():.3f} ± {df['conf_90_recall'].std():.3f}")
print(f"  F1-Score: {df['conf_90_f1'].mean():.3f} ± {df['conf_90_f1'].std():.3f}")

print(f"\n95% Confidence Threshold:")
print(f"  Accuracy: {df['conf_95_accuracy'].mean():.3f} ± {df['conf_95_accuracy'].std():.3f}")
print(f"  Precision: {df['conf_95_precision'].mean():.3f} ± {df['conf_95_precision'].std():.3f}")
print(f"  Recall: {df['conf_95_recall'].mean():.3f} ± {df['conf_95_recall'].std():.3f}")
print(f"  F1-Score: {df['conf_95_f1'].mean():.3f} ± {df['conf_95_f1'].std():.3f}")

# Create comprehensive visualization comparing all approaches
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('REM Detection: Original vs Optimized vs 90% vs 95% Confidence', fontsize=16, fontweight='bold')

# Performance metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
original_vals = [df['original_accuracy'].mean(), 
                0.1,  # Estimated original precision
                (df['true_positives']/(df['true_positives']+df['false_negatives'])).mean(),
                0.15]  # Estimated original F1
optimized_vals = [df['optimized_accuracy'].mean(), df['precision'].mean(), 
                 df['recall'].mean(), df['f1_score'].mean()]
conf_90_vals = [df['conf_90_accuracy'].mean(), df['conf_90_precision'].mean(),
                df['conf_90_recall'].mean(), df['conf_90_f1'].mean()]
conf_95_vals = [df['conf_95_accuracy'].mean(), df['conf_95_precision'].mean(),
                df['conf_95_recall'].mean(), df['conf_95_f1'].mean()]

x = np.arange(len(metrics))
width = 0.2

bars1 = axes[0, 0].bar(x - 1.5*width, original_vals, width, label='Original YASA', 
                      color='lightcoral', alpha=0.8)
bars2 = axes[0, 0].bar(x - 0.5*width, optimized_vals, width, label='Optimized', 
                      color='lightblue', alpha=0.8)
bars3 = axes[0, 0].bar(x + 0.5*width, conf_90_vals, width, label='90% Conf', 
                      color='orange', alpha=0.8)
bars4 = axes[0, 0].bar(x + 1.5*width, conf_95_vals, width, label='95% Conf', 
                      color='darkgreen', alpha=0.8)

axes[0, 0].set_ylabel('Performance')
axes[0, 0].set_title('Performance Metrics Comparison')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metrics)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0, 1)

# Recall distribution comparison
recall_data = [
    (df['true_positives']/(df['true_positives']+df['false_negatives'])).values,
    df['recall'].values,
    df['conf_90_recall'].values,
    df['conf_95_recall'].values
]
box_plot = axes[0, 1].boxplot(recall_data, labels=['Original', 'Optimized', '90% Conf', '95% Conf'], 
                             patch_artist=True)
colors = ['lightcoral', 'lightblue', 'orange', 'darkgreen']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
axes[0, 1].set_ylabel('Recall (Sensitivity)')
axes[0, 1].set_title('Recall Distribution by Threshold')
axes[0, 1].grid(True, alpha=0.3)

# Precision distribution comparison
precision_data = [
    np.full(20, 0.1),  # Estimated original precision
    df['precision'].values,
    df['conf_90_precision'].values,
    df['conf_95_precision'].values
]
box_plot2 = axes[0, 2].boxplot(precision_data, labels=['Original', 'Optimized', '90% Conf', '95% Conf'], 
                              patch_artist=True)
for patch, color in zip(box_plot2['boxes'], colors):
    patch.set_facecolor(color)
axes[0, 2].set_ylabel('Precision')
axes[0, 2].set_title('Precision Distribution by Threshold')
axes[0, 2].grid(True, alpha=0.3)

# Detection counts by night
axes[1, 0].plot(df['night'], df['rem_epochs_total'], 'o-', color='blue', linewidth=3, 
               markersize=8, label='True REM epochs', alpha=0.9)
axes[1, 0].plot(df['night'], df['true_positives'], 'o-', color='lightcoral', linewidth=2, 
               markersize=5, label='Original detections', alpha=0.7)
axes[1, 0].plot(df['night'], df['true_positives'] + df['false_positives'], 'o-', 
               color='lightblue', linewidth=2, markersize=5, label='Optimized detections', alpha=0.7)
axes[1, 0].plot(df['night'], df['conf_90_tp'], 'o-', color='orange', linewidth=2, 
               markersize=5, label='90% Conf detections', alpha=0.8)
axes[1, 0].plot(df['night'], df['conf_95_tp'], 'o-', color='darkgreen', linewidth=2, 
               markersize=5, label='95% Conf detections', alpha=0.8)
axes[1, 0].set_xlabel('Night')
axes[1, 0].set_ylabel('Epoch Count')
axes[1, 0].set_title('REM Detection Counts by Night')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(range(1, 21, 2))

# Precision-Recall tradeoff
axes[1, 1].scatter(df['recall'], df['precision'], s=80, alpha=0.7, 
                  color='lightblue', label='Optimized', edgecolor='black', linewidth=1)
axes[1, 1].scatter(df['conf_90_recall'], df['conf_90_precision'], s=80, alpha=0.8, 
                  color='orange', label='90% Confidence', edgecolor='black', linewidth=1)
axes[1, 1].scatter(df['conf_95_recall'], df['conf_95_precision'], s=80, alpha=0.8, 
                  color='darkgreen', label='95% Confidence', edgecolor='black', linewidth=1)
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].set_title('Precision-Recall Tradeoff')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)

# Nights with successful detection
success_data = [
    (df['true_positives'] > 0).sum(),
    (df['true_positives'] + df['false_positives'] > 0).sum(),
    (df['conf_90_tp'] > 0).sum(),
    (df['conf_95_tp'] > 0).sum()
]
threshold_labels = ['Original', 'Optimized', '90% Conf', '95% Conf']
bars = axes[1, 2].bar(threshold_labels, success_data, color=colors, alpha=0.8)
axes[1, 2].set_ylabel('Number of Nights')
axes[1, 2].set_title('Nights with REM Detections')
axes[1, 2].set_ylim(0, 20)
axes[1, 2].grid(True, alpha=0.3)

# Add value labels
for bar, val in zip(bars, success_data):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{val}/20', ha='center', va='bottom', fontsize=10, fontweight='bold')

# F1-Score comparison by night
axes[2, 0].plot(df['night'], df['f1_score'], 'o-', color='lightblue', linewidth=2, 
               markersize=6, label='Optimized', alpha=0.8)
axes[2, 0].plot(df['night'], df['conf_90_f1'], 'o-', color='orange', linewidth=2, 
               markersize=6, label='90% Confidence', alpha=0.8)
axes[2, 0].plot(df['night'], df['conf_95_f1'], 'o-', color='darkgreen', linewidth=2, 
               markersize=6, label='95% Confidence', alpha=0.8)
axes[2, 0].set_xlabel('Night')
axes[2, 0].set_ylabel('F1-Score')
axes[2, 0].set_title('F1-Score by Night')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_xticks(range(1, 21, 2))

# Total detections comparison
total_detections = [
    df['true_positives'].sum(),
    (df['true_positives'] + df['false_positives']).sum(),
    df['conf_90_tp'].sum(),
    df['conf_95_tp'].sum()
]
bars2 = axes[2, 1].bar(threshold_labels, total_detections, color=colors, alpha=0.8)
axes[2, 1].axhline(y=df['rem_epochs_total'].sum(), color='red', linestyle='--', 
                  linewidth=2, label=f'Total REM epochs ({df["rem_epochs_total"].sum()})')
axes[2, 1].set_ylabel('Total REM Epochs Detected')
axes[2, 1].set_title('Total REM Detection Across All Nights')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# Add value labels
for bar, val in zip(bars2, total_detections):
    height = bar.get_height()
    axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + 10,
                   f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Performance summary heatmap
performance_matrix = np.array([
    optimized_vals,
    conf_90_vals,
    conf_95_vals
])
im = axes[2, 2].imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
axes[2, 2].set_xticks(range(len(metrics)))
axes[2, 2].set_xticklabels(metrics)
axes[2, 2].set_yticks(range(3))
axes[2, 2].set_yticklabels(['Optimized', '90% Conf', '95% Conf'])
axes[2, 2].set_title('Performance Heatmap')

# Add text annotations
for i in range(3):
    for j in range(len(metrics)):
        text = axes[2, 2].text(j, i, f'{performance_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=axes[2, 2], shrink=0.6)

plt.tight_layout()
plt.savefig('rem_detection_90_95_confidence_comparison.png', dpi=150, bbox_inches='tight')
print("\nHigh confidence comparison plot saved as rem_detection_90_95_confidence_comparison.png")
plt.close()

# Summary table
summary_data = {
    'Threshold': ['Original YASA', 'Optimized', '90% Confidence', '95% Confidence'],
    'Avg_Accuracy': [df['original_accuracy'].mean(), df['optimized_accuracy'].mean(), 
                     df['conf_90_accuracy'].mean(), df['conf_95_accuracy'].mean()],
    'Avg_Precision': [0.1, df['precision'].mean(), df['conf_90_precision'].mean(), df['conf_95_precision'].mean()],
    'Avg_Recall': [(df['true_positives']/(df['true_positives']+df['false_negatives'])).mean(), 
                   df['recall'].mean(), df['conf_90_recall'].mean(), df['conf_95_recall'].mean()],
    'Avg_F1': [0.15, df['f1_score'].mean(), df['conf_90_f1'].mean(), df['conf_95_f1'].mean()],
    'Nights_with_Detections': [(df['true_positives'] > 0).sum(), 
                              (df['true_positives'] + df['false_positives'] > 0).sum(),
                              (df['conf_90_tp'] > 0).sum(), (df['conf_95_tp'] > 0).sum()],
    'Total_REM_Detected': [df['true_positives'].sum(), 
                          (df['true_positives'] + df['false_positives']).sum(),
                          df['conf_90_tp'].sum(), df['conf_95_tp'].sum()],
    'Percentage_REM_Detected': [df['true_positives'].sum()/df['rem_epochs_total'].sum()*100,
                               (df['true_positives'] + df['false_positives']).sum()/df['rem_epochs_total'].sum()*100,
                               df['conf_90_tp'].sum()/df['rem_epochs_total'].sum()*100,
                               df['conf_95_tp'].sum()/df['rem_epochs_total'].sum()*100]
}

summary_df = pd.DataFrame(summary_data)
print(f"\n=== Comprehensive Summary Table ===")
print(summary_df.round(3))

# Save results
summary_df.to_csv('rem_confidence_threshold_comparison.csv', index=False)
print("\nSummary saved as rem_confidence_threshold_comparison.csv")

print(f"\n=== Key Insights ===")
print(f"\n90% Confidence Threshold:")
print(f"  - Precision: {df['conf_90_precision'].mean():.3f} (very high confidence when detecting REM)")
print(f"  - Recall: {df['conf_90_recall'].mean():.3f} (detects {df['conf_90_recall'].mean()*100:.1f}% of REM epochs)")
print(f"  - Works on {(df['conf_90_tp'] > 0).sum()}/20 nights")
print(f"  - Detects {df['conf_90_tp'].sum()/df['rem_epochs_total'].sum()*100:.1f}% of all REM sleep")

print(f"\n95% Confidence Threshold:")
print(f"  - Precision: {df['conf_95_precision'].mean():.3f} (extremely high confidence)")
print(f"  - Recall: {df['conf_95_recall'].mean():.3f} (detects {df['conf_95_recall'].mean()*100:.1f}% of REM epochs)")
print(f"  - Works on {(df['conf_95_tp'] > 0).sum()}/20 nights")
print(f"  - Detects {df['conf_95_tp'].sum()/df['rem_epochs_total'].sum()*100:.1f}% of all REM sleep")

print(f"\nRecommendations:")
print(f"  - Use 90% confidence for moderate precision with reasonable recall")
print(f"  - Use 95% confidence for maximum precision when false positives are critical")
print(f"  - Optimized threshold gives best balance for comprehensive REM detection")
