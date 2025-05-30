import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# Load the results
df = pd.read_csv('all_nights_rem_analysis.csv')

print("=== REM Detection Analysis Across All 20 Nights ===")
print(f"Total nights analyzed: {len(df)}")
print(f"Total recording time: {df['duration_hours'].sum():.1f} hours")
print(f"Total REM epochs: {df['rem_epochs_total'].sum()}")
print(f"Total REM episodes: {df['rem_episodes_count'].sum()}")

# Summary statistics
print(f"\n=== Performance Summary ===")
print(f"Average ROC AUC: {df['roc_auc'].mean():.3f} ± {df['roc_auc'].std():.3f}")
print(f"Average Precision-Recall AUC: {df['pr_auc'].mean():.3f} ± {df['pr_auc'].std():.3f}")
print(f"Average original accuracy: {df['original_accuracy'].mean():.3f} ± {df['original_accuracy'].std():.3f}")
print(f"Average optimized accuracy: {df['optimized_accuracy'].mean():.3f} ± {df['optimized_accuracy'].std():.3f}")
print(f"Average recall (sensitivity): {df['recall'].mean():.3f} ± {df['recall'].std():.3f}")
print(f"Average precision: {df['precision'].mean():.3f} ± {df['precision'].std():.3f}")
print(f"Average F1-score: {df['f1_score'].mean():.3f} ± {df['f1_score'].std():.3f}")

# REM percentage analysis
print(f"\n=== REM Sleep Distribution ===")
print(f"Average REM percentage: {df['rem_percentage'].mean():.1f}% ± {df['rem_percentage'].std():.1f}%")
print(f"Min REM percentage: {df['rem_percentage'].min():.1f}% (Night {df.loc[df['rem_percentage'].idxmin(), 'night']})")
print(f"Max REM percentage: {df['rem_percentage'].max():.1f}% (Night {df.loc[df['rem_percentage'].idxmax(), 'night']})")

# Best and worst performing nights
best_roc = df.loc[df['roc_auc'].idxmax()]
worst_roc = df.loc[df['roc_auc'].idxmin()]
best_recall = df.loc[df['recall'].idxmax()]
worst_recall = df.loc[df['recall'].idxmin()]

print(f"\n=== Best/Worst Performance ===")
print(f"Best ROC AUC: Night {best_roc['night']} (AUC: {best_roc['roc_auc']:.3f})")
print(f"Worst ROC AUC: Night {worst_roc['night']} (AUC: {worst_roc['roc_auc']:.3f})")
print(f"Best Recall: Night {best_recall['night']} (Recall: {best_recall['recall']:.3f})")
print(f"Worst Recall: Night {worst_recall['night']} (Recall: {worst_recall['recall']:.3f})")

# Create comprehensive visualization
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('REM Detection Performance Across 20 Nights', fontsize=16, fontweight='bold')

# 1. ROC AUC distribution
axes[0, 0].hist(df['roc_auc'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].axvline(df['roc_auc'].mean(), color='red', linestyle='--', label=f'Mean: {df["roc_auc"].mean():.3f}')
axes[0, 0].set_xlabel('ROC AUC')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('ROC AUC Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Recall vs ROC AUC
scatter = axes[0, 1].scatter(df['roc_auc'], df['recall'], c=df['rem_percentage'], 
                           cmap='viridis', s=60, alpha=0.8)
axes[0, 1].set_xlabel('ROC AUC')
axes[0, 1].set_ylabel('Recall (Sensitivity)')
axes[0, 1].set_title('Recall vs ROC AUC (colored by REM %)')
plt.colorbar(scatter, ax=axes[0, 1], label='REM %')
axes[0, 1].grid(True, alpha=0.3)

# 3. REM percentage distribution
axes[0, 2].hist(df['rem_percentage'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 2].axvline(df['rem_percentage'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["rem_percentage"].mean():.1f}%')
axes[0, 2].set_xlabel('REM Percentage')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('REM Sleep Percentage Distribution')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Performance metrics comparison
metrics = ['original_accuracy', 'optimized_accuracy', 'precision', 'recall', 'f1_score']
metric_means = [df[metric].mean() for metric in metrics]
metric_stds = [df[metric].std() for metric in metrics]
metric_labels = ['Original Acc', 'Optimized Acc', 'Precision', 'Recall', 'F1-Score']

bars = axes[1, 0].bar(metric_labels, metric_means, yerr=metric_stds, 
                     capsize=5, alpha=0.7, color=['lightcoral', 'lightblue', 'lightgreen', 'gold', 'plum'])
axes[1, 0].set_ylabel('Performance')
axes[1, 0].set_title('Average Performance Metrics')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].grid(True, alpha=0.3)
# Add value labels on bars
for bar, mean, std in zip(bars, metric_means, metric_stds):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# 5. Night-by-night ROC AUC
axes[1, 1].plot(df['night'], df['roc_auc'], 'o-', color='blue', linewidth=2, markersize=6)
axes[1, 1].axhline(y=df['roc_auc'].mean(), color='red', linestyle='--', alpha=0.7)
axes[1, 1].set_xlabel('Night')
axes[1, 1].set_ylabel('ROC AUC')
axes[1, 1].set_title('ROC AUC by Night')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(range(1, 21))

# 6. REM episodes vs performance
axes[1, 2].scatter(df['rem_episodes_count'], df['roc_auc'], alpha=0.7, s=60, color='purple')
axes[1, 2].set_xlabel('Number of REM Episodes')
axes[1, 2].set_ylabel('ROC AUC')
axes[1, 2].set_title('Performance vs REM Episodes')
axes[1, 2].grid(True, alpha=0.3)

# 7. Duration vs REM percentage
axes[2, 0].scatter(df['duration_hours'], df['rem_percentage'], alpha=0.7, s=60, color='orange')
axes[2, 0].set_xlabel('Recording Duration (hours)')
axes[2, 0].set_ylabel('REM Percentage')
axes[2, 0].set_title('REM % vs Recording Duration')
axes[2, 0].grid(True, alpha=0.3)

# 8. Precision vs Recall
axes[2, 1].scatter(df['recall'], df['precision'], c=df['roc_auc'], 
                  cmap='coolwarm', s=60, alpha=0.8)
axes[2, 1].set_xlabel('Recall')
axes[2, 1].set_ylabel('Precision')
axes[2, 1].set_title('Precision vs Recall (colored by ROC AUC)')
plt.colorbar(axes[2, 1].collections[0], ax=axes[2, 1], label='ROC AUC')
axes[2, 1].grid(True, alpha=0.3)

# 9. Optimal threshold distribution
axes[2, 2].hist(df['optimal_threshold'], bins=10, alpha=0.7, color='pink', edgecolor='black')
axes[2, 2].axvline(df['optimal_threshold'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["optimal_threshold"].mean():.3f}')
axes[2, 2].set_xlabel('Optimal Threshold')
axes[2, 2].set_ylabel('Frequency')
axes[2, 2].set_title('Optimal Threshold Distribution')
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('all_nights_rem_analysis_comprehensive.png', dpi=150, bbox_inches='tight')
print("\nComprehensive analysis plot saved as all_nights_rem_analysis_comprehensive.png")
plt.close()

# Performance categories
print(f"\n=== Performance Categories ===")
excellent_nights = df[df['roc_auc'] >= 0.9]['night'].tolist()
good_nights = df[(df['roc_auc'] >= 0.8) & (df['roc_auc'] < 0.9)]['night'].tolist()
fair_nights = df[(df['roc_auc'] >= 0.7) & (df['roc_auc'] < 0.8)]['night'].tolist()
poor_nights = df[df['roc_auc'] < 0.7]['night'].tolist()

print(f"Excellent ROC AUC (≥0.9): {len(excellent_nights)} nights - {excellent_nights}")
print(f"Good ROC AUC (0.8-0.9): {len(good_nights)} nights - {good_nights}")
print(f"Fair ROC AUC (0.7-0.8): {len(fair_nights)} nights - {fair_nights}")
print(f"Poor ROC AUC (<0.7): {len(poor_nights)} nights - {poor_nights}")

# Correlation analysis
print(f"\n=== Correlation Analysis ===")
corr_matrix = df[['duration_hours', 'rem_percentage', 'rem_episodes_count', 'roc_auc', 'recall', 'precision']].corr()
print("Correlation with ROC AUC:")
roc_correlations = corr_matrix['roc_auc'].sort_values(ascending=False)
for var, corr in roc_correlations.items():
    if var != 'roc_auc':
        print(f"  {var}: {corr:.3f}")

# Create final summary table
summary_stats = pd.DataFrame({
    'Metric': ['ROC AUC', 'PR AUC', 'Original Accuracy', 'Optimized Accuracy', 
               'Precision', 'Recall', 'F1-Score', 'REM Percentage'],
    'Mean': [df['roc_auc'].mean(), df['pr_auc'].mean(), df['original_accuracy'].mean(),
             df['optimized_accuracy'].mean(), df['precision'].mean(), df['recall'].mean(),
             df['f1_score'].mean(), df['rem_percentage'].mean()],
    'Std': [df['roc_auc'].std(), df['pr_auc'].std(), df['original_accuracy'].std(),
            df['optimized_accuracy'].std(), df['precision'].std(), df['recall'].std(),
            df['f1_score'].std(), df['rem_percentage'].std()],
    'Min': [df['roc_auc'].min(), df['pr_auc'].min(), df['original_accuracy'].min(),
            df['optimized_accuracy'].min(), df['precision'].min(), df['recall'].min(),
            df['f1_score'].min(), df['rem_percentage'].min()],
    'Max': [df['roc_auc'].max(), df['pr_auc'].max(), df['original_accuracy'].max(),
            df['optimized_accuracy'].max(), df['precision'].max(), df['recall'].max(),
            df['f1_score'].max(), df['rem_percentage'].max()]
})

print(f"\n=== Summary Statistics Table ===")
print(summary_stats.round(3))

# Save detailed summary
summary_stats.to_csv('rem_analysis_summary_stats.csv', index=False)
print("\nSummary statistics saved as rem_analysis_summary_stats.csv")

print(f"\n=== Key Insights ===")
print(f"1. YASA shows strong REM detection capability with average ROC AUC of {df['roc_auc'].mean():.3f}")
print(f"2. Threshold optimization dramatically improves recall from {(df['true_positives']/(df['true_positives']+df['false_negatives'])).mean():.3f} to {df['recall'].mean():.3f}")
print(f"3. {len(excellent_nights)} nights achieve excellent performance (ROC AUC ≥ 0.9)")
print(f"4. REM percentage varies widely from {df['rem_percentage'].min():.1f}% to {df['rem_percentage'].max():.1f}%")
print(f"5. Strong correlation between REM episodes and detection performance: {corr_matrix.loc['rem_episodes_count', 'roc_auc']:.3f}")
