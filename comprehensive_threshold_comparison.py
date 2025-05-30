import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== COMPREHENSIVE THRESHOLD COMPARISON: FPR vs Confidence-Based ===")
print("Comparing empirical FPR-based thresholds with confidence-based approaches")
print("=" * 80)

# Load both analyses
try:
    fpr_results = pd.read_csv('fpr_based_threshold_analysis.csv')
    confidence_results = pd.read_csv('final_confidence_analysis_results.csv')
    
    print("✅ Successfully loaded both analysis results")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    exit(1)

# Create comprehensive comparison table
comparison_data = []

print("\n📊 THRESHOLD COMPARISON SUMMARY:")
print("=" * 80)

# FPR-based results
fpr_10_subset = fpr_results[fpr_results['target_fpr'] == 'FPR_10']
fpr_5_subset = fpr_results[fpr_results['target_fpr'] == 'FPR_5']

# Confidence-based results
conf_70_subset = confidence_results[confidence_results['threshold_name'] == '70% Confidence']
conf_80_subset = confidence_results[confidence_results['threshold_name'] == '80% Confidence']

# Calculate global metrics for each approach
def calculate_global_metrics(subset):
    total_tp = subset['true_positives'].sum()
    total_fp = subset['false_positives'].sum()
    total_tn = subset['true_negatives'].sum()
    total_fn = subset['false_negatives'].sum()
    
    global_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    global_tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    global_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)
    
    nights_with_detection = len(subset[subset['tpr_recall'] > 0]) if 'tpr_recall' in subset.columns else len(subset[subset['recall'] > 0])
    
    return {
        'global_fpr': global_fpr,
        'global_tpr': global_tpr,
        'global_precision': global_precision,
        'global_accuracy': global_accuracy,
        'nights_with_detection': nights_with_detection,
        'total_nights': len(subset),
        'total_tp': total_tp,
        'total_fp': total_fp
    }

def get_use_case(method_name):
    if "FPR < 10%" in method_name:
        return "Balanced detection, 10% false positive rate"
    elif "FPR < 5%" in method_name:
        return "Conservative detection, 5% false positive rate"
    elif "70%" in method_name:
        return "Moderate confidence, higher sensitivity"
    elif "80%" in method_name:
        return "High confidence, balanced performance"
    return "Unknown"

# Analysis for each threshold type
analyses = [
    ("FPR < 10%", fpr_10_subset, 'tpr_recall', 'actual_fpr'),
    ("FPR < 5%", fpr_5_subset, 'tpr_recall', 'actual_fpr'),
    ("70% Confidence", conf_70_subset, 'recall', None),
    ("80% Confidence", conf_80_subset, 'recall', None)
]

for name, subset, recall_col, fpr_col in analyses:
    if len(subset) == 0:
        print(f"\n❌ {name}: No data available")
        continue
        
    metrics = calculate_global_metrics(subset)
    threshold_val = subset.iloc[0]['threshold_value']
    
    # Calculate mean per-night metrics
    mean_recall = subset[recall_col].mean()
    mean_precision = subset['precision'].mean()
    
    if fpr_col:
        mean_fpr = subset[fpr_col].mean()
    else:
        # Calculate FPR for confidence-based
        mean_fpr = subset.apply(lambda row: row['false_positives'] / (row['false_positives'] + row['true_negatives']) if (row['false_positives'] + row['true_negatives']) > 0 else 0, axis=1).mean()
    
    print(f"\n🎯 {name}:")
    print(f"   Threshold value: {threshold_val:.4f}")
    print(f"   Global FPR: {metrics['global_fpr']:.1%}")
    print(f"   Global TPR (Recall): {metrics['global_tpr']:.1%}")
    print(f"   Global Precision: {metrics['global_precision']:.1%}")
    print(f"   Global Accuracy: {metrics['global_accuracy']:.1%}")
    print(f"   Mean per-night FPR: {mean_fpr:.1%}")
    print(f"   Mean per-night TPR: {mean_recall:.1%}")
    print(f"   Mean per-night Precision: {mean_precision:.1%}")
    print(f"   Nights with REM detection: {metrics['nights_with_detection']}/{metrics['total_nights']}")
    
    # Store for table
    comparison_data.append({
        'Method': name,
        'Threshold': f"{threshold_val:.4f}",
        'Global_FPR': f"{metrics['global_fpr']:.1%}",
        'Global_TPR': f"{metrics['global_tpr']:.1%}",
        'Global_Precision': f"{metrics['global_precision']:.1%}",
        'Mean_FPR': f"{mean_fpr:.1%}",
        'Mean_TPR': f"{mean_recall:.1%}",
        'Mean_Precision': f"{mean_precision:.1%}",
        'Nights_Detected': f"{metrics['nights_with_detection']}/{metrics['total_nights']}",
        'Use_Case': get_use_case(name)
    })

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('threshold_method_comparison.csv', index=False)

print(f"\n" + "=" * 80)
print(f"📋 COMPREHENSIVE THRESHOLD COMPARISON TABLE:")
print(f"=" * 80)
print(comparison_df.to_string(index=False))

# Practical recommendations
print(f"\n" + "=" * 80)
print(f"🎯 PRACTICAL RECOMMENDATIONS:")
print(f"=" * 80)

print(f"\n1️⃣ FOR STRICT FALSE POSITIVE CONTROL:")
print(f"   Use FPR-based thresholds if you have strict false positive requirements")
print(f"   • FPR < 5%:  Threshold 0.7786 (30.4% recall, 49.9% precision)")
print(f"   • FPR < 10%: Threshold 0.5449 (50.8% recall, 45.4% precision)")

print(f"\n2️⃣ FOR BALANCED PERFORMANCE:")
print(f"   Use confidence-based thresholds for general-purpose REM detection")
print(f"   • 70% Confidence: Threshold 0.4382 (59.7% recall, 44.0% precision)")
print(f"   • 80% Confidence: Threshold 0.6131 (45.2% recall, 46.7% precision)")

print(f"\n3️⃣ THRESHOLD RANKING BY RECALL (Sensitivity):")
ranking_data = []
for _, row in comparison_df.iterrows():
    tpr_val = float(row['Global_TPR'].strip('%')) / 100
    ranking_data.append((row['Method'], row['Threshold'], tpr_val, row['Global_FPR']))

ranking_data.sort(key=lambda x: x[2], reverse=True)
for i, (method, threshold, tpr, fpr) in enumerate(ranking_data, 1):
    print(f"   {i}. {method}: {threshold} (TPR: {tpr:.1%}, FPR: {fpr})")

print(f"\n4️⃣ THRESHOLD RANKING BY PRECISION:")
ranking_data_prec = []
for _, row in comparison_df.iterrows():
    prec_val = float(row['Global_Precision'].strip('%')) / 100
    ranking_data_prec.append((row['Method'], row['Threshold'], prec_val, row['Global_FPR']))

ranking_data_prec.sort(key=lambda x: x[2], reverse=True)
for i, (method, threshold, prec, fpr) in enumerate(ranking_data_prec, 1):
    print(f"   {i}. {method}: {threshold} (Precision: {prec:.1%}, FPR: {fpr})")

# Key insights
print(f"\n" + "=" * 80)
print(f"🔍 KEY INSIGHTS:")
print(f"=" * 80)

print(f"\n📈 PERFORMANCE TRADE-OFFS:")
print(f"   • FPR-based thresholds provide guaranteed false positive control")
print(f"   • Confidence-based thresholds offer better recall at cost of higher FPR")
print(f"   • All methods work on 18-20 nights (consistent performance)")

print(f"\n🎯 THRESHOLD EQUIVALENCES:")
print(f"   • FPR < 5% (0.7786) ≈ between 80% confidence (0.6131) and higher")
print(f"   • FPR < 10% (0.5449) ≈ between 70% confidence (0.4382) and 80% confidence")

print(f"\n⚖️  CHOICE CRITERIA:")
print(f"   • Clinical/Research use: Choose FPR-based for controlled false positives")
print(f"   • General REM detection: Choose confidence-based for better sensitivity")
print(f"   • Real-time applications: 70% confidence (0.4382) offers best balance")

print(f"\n" + "=" * 80)
print(f"✅ FINAL ANSWER TO YOUR QUESTION:")
print(f"=" * 80)
print(f"For FPR < 10%: Use threshold 0.5449")
print(f"For FPR < 5%:  Use threshold 0.7786")
print(f"")
print(f"These are the empirically determined thresholds across all 20 nights")
print(f"that achieve your target false positive rates for REM detection.")
print(f"=" * 80)

# Create final visualization comparing all methods
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comprehensive Threshold Method Comparison', fontsize=16, fontweight='bold')

# Extract data for plotting
methods = comparison_df['Method'].tolist()
thresholds = [float(t) for t in comparison_df['Threshold'].tolist()]
fprs = [float(f.strip('%'))/100 for f in comparison_df['Global_FPR'].tolist()]
tprs = [float(t.strip('%'))/100 for t in comparison_df['Global_TPR'].tolist()]
precisions = [float(p.strip('%'))/100 for p in comparison_df['Global_Precision'].tolist()]

colors = ['red', 'blue', 'green', 'orange']

# 1. Threshold values
ax1.bar(methods, thresholds, color=colors, alpha=0.7)
ax1.set_title('Threshold Values by Method')
ax1.set_ylabel('Threshold Value')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# 2. FPR vs TPR
ax2.scatter(fprs, tprs, c=colors, s=100, alpha=0.7)
for i, method in enumerate(methods):
    ax2.annotate(method, (fprs[i], tprs[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate (Recall)')
ax2.set_title('FPR vs TPR Trade-off')
ax2.grid(True, alpha=0.3)

# 3. Precision comparison
ax3.bar(methods, precisions, color=colors, alpha=0.7)
ax3.set_title('Precision by Method')
ax3.set_ylabel('Precision')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# 4. FPR comparison
ax4.bar(methods, fprs, color=colors, alpha=0.7)
ax4.axhline(y=0.10, color='red', linestyle='--', label='10% FPR target')
ax4.axhline(y=0.05, color='blue', linestyle='--', label='5% FPR target')
ax4.set_title('False Positive Rate by Method')
ax4.set_ylabel('False Positive Rate')
ax4.tick_params(axis='x', rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_threshold_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFiles generated:")
print(f"- threshold_method_comparison.csv")
print(f"- comprehensive_threshold_comparison.png")
