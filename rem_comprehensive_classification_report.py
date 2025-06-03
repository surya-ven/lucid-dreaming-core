"""
Comprehensive Classification Report for REM Detection using YASA Model.
Based on the structure from example_from_LRLR_classification.py, this script provides
a detailed analysis of REM detection performance across different thresholds.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from datetime import datetime
import os

def plot_confusion_matrix(y_true, y_scores, threshold, title, save_path=None):
    """Plot confusion matrix for a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Non-REM', 'Predicted REM'],
                yticklabels=['Actual Non-REM', 'Actual REM'])
    plt.title(f'Confusion Matrix - {title}\nThreshold: {threshold:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_with_thresholds(y_true, y_scores, thresholds_dict, model_name, save_path=None):
    """Plot ROC curve with optimal threshold points marked."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6)
    
    # Mark optimal threshold points
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'v']
    threshold_names = ['Conservative (FPR≤5%)', 'Balanced (FPR≤10%)', 'Sensitive (70% conf)', 
                      'High Conf (80% conf)', 'Very High Conf (90% conf)']
    threshold_keys = ['conservative', 'balanced', 'sensitive', 'high_conf', 'very_high_conf']
    
    for i, (name, key) in enumerate(zip(threshold_names, threshold_keys)):
        threshold = thresholds_dict.get(key)
        if threshold is not None and i < len(colors):
            # Find closest point on ROC curve
            threshold_idx = np.argmin(np.abs(roc_thresholds - threshold))
            if threshold_idx < len(fpr):
                plt.plot(fpr[threshold_idx], tpr[threshold_idx], 
                        color=colors[i], marker=markers[i], markersize=10, 
                        label=f'{name} (t={threshold:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def find_threshold_for_fpr(y_true, y_scores, target_fpr):
    """Find threshold that achieves target false positive rate."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find the threshold that gives us the target FPR or lower
    valid_indices = fpr <= target_fpr
    if np.any(valid_indices):
        # Among valid thresholds, pick the one with highest TPR (sensitivity)
        best_idx = np.argmax(tpr[valid_indices])
        # Get the actual index in the original arrays
        valid_idx_positions = np.where(valid_indices)[0]
        actual_idx = valid_idx_positions[best_idx]
        
        return thresholds[actual_idx], fpr[actual_idx], tpr[actual_idx]
    else:
        return None, None, None

def find_best_auc_threshold(y_true, y_scores):
    """Find threshold that maximizes Youden's J statistic (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr  # Youden's J statistic
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_sensitivity = tpr[best_idx]
    best_specificity = 1 - fpr[best_idx]
    return best_threshold, best_sensitivity, best_specificity

def find_threshold_for_confidence(y_true, y_scores, target_confidence):
    """Find threshold that achieves target confidence level (precision)."""
    # Try different thresholds from high to low
    thresholds = np.arange(0.99, 0.01, -0.01)
    
    best_threshold = None
    best_recall = 0
    best_metrics = None
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate precision (confidence)
        if tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # If we achieve target confidence and this has better recall
            if precision >= target_confidence and recall > best_recall:
                best_threshold = threshold
                best_recall = recall
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                }
    
    if best_threshold is not None:
        return best_threshold, best_metrics['precision'], best_metrics['recall']
    else:
        return None, None, None

def evaluate_threshold(y_true, y_scores, threshold):
    """Evaluate performance at a specific threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def plot_prediction_distribution(y_true, y_scores, model_name):
    """Plot distribution of prediction scores for each class."""
    plt.figure(figsize=(10, 6))
    
    # Separate scores by true class
    scores_class_0 = y_scores[y_true == 0]
    scores_class_1 = y_scores[y_true == 1]
    
    plt.hist(scores_class_0, bins=50, alpha=0.6, label='Non-REM (Class 0)', color='blue', density=True)
    plt.hist(scores_class_1, bins=50, alpha=0.6, label='REM (Class 1)', color='red', density=True)
    
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title(f'Prediction Score Distribution - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_threshold_performance(y_true, y_scores):
    """Analyze model performance across different thresholds."""
    thresholds = np.arange(0.05, 1.0, 0.05)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        n_pred_pos = np.sum(y_pred)
        
        if n_pred_pos > 0:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            
            # Calculate FPR
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'accuracy': acc,
                'fpr': fpr,
                'n_pred_pos': n_pred_pos
            })
    
    # Plot threshold analysis
    if results:
        thresholds_plot = [r['threshold'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1s = [r['f1'] for r in results]
        fprs = [r['fpr'] for r in results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1.plot(thresholds_plot, precisions, 'b-', label='Precision', linewidth=2)
        ax1.plot(thresholds_plot, recalls, 'r-', label='Recall', linewidth=2)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Precision and Recall vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(thresholds_plot, f1s, 'g-', label='F1 Score', linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(thresholds_plot, fprs, 'orange', label='False Positive Rate', linewidth=2)
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='FPR = 0.05')
        ax3.axhline(y=0.1, color='blue', linestyle='--', alpha=0.7, label='FPR = 0.1')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('False Positive Rate')
        ax3.set_title('False Positive Rate vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        n_pred_pos_list = [r['n_pred_pos'] for r in results]
        ax4.plot(thresholds_plot, n_pred_pos_list, 'purple', label='Positive Predictions', linewidth=2)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Number of Positive Predictions')
        ax4.set_title('Positive Predictions vs Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def print_comprehensive_classification_table(y_true, y_scores, thresholds_dict):
    """
    Print a comprehensive classification table comparing all thresholds.
    
    Args:
        y_true: True labels (0=Non-REM, 1=REM)
        y_scores: Prediction scores from REM detection model
        thresholds_dict: Dictionary containing thresholds for different criteria
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE REM CLASSIFICATION REPORT - ALL THRESHOLDS")
    print(f"{'='*80}")
    
    # Define threshold names and values
    threshold_configs = [
        ("Conservative (FPR ≤ 5%)", thresholds_dict.get('conservative')),
        ("Balanced (FPR ≤ 10%)", thresholds_dict.get('balanced')),
        ("Sensitive (70% confidence)", thresholds_dict.get('sensitive')),
        ("High Confidence (80%)", thresholds_dict.get('high_conf')),
        ("Very High Confidence (90%)", thresholds_dict.get('very_high_conf'))
    ]
    
    # Collect all metrics for table
    table_data = []
    
    for name, threshold in threshold_configs:
        if threshold is not None:
            # Calculate predictions and metrics
            y_pred = (y_scores >= threshold).astype(int)
            metrics = evaluate_threshold(y_true, y_scores, threshold)
            
            # Calculate additional metrics
            fpr = 1 - metrics['specificity']
            
            table_data.append({
                'name': name,
                'threshold': threshold,
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'precision': metrics['precision'],
                'accuracy': metrics['accuracy'],
                'fpr': fpr,
                'tp': metrics['tp'],
                'tn': metrics['tn'],
                'fp': metrics['fp'],
                'fn': metrics['fn']
            })
        else:
            # Handle case where threshold is not achievable
            table_data.append({
                'name': name,
                'threshold': None,
                'sensitivity': None,
                'specificity': None,
                'precision': None,
                'accuracy': None,
                'fpr': None,
                'tp': None,
                'tn': None,
                'fp': None,
                'fn': None
            })
    
    # Print summary metrics table
    print(f"\nSUMMARY METRICS TABLE")
    print(f"{'-'*80}")
    print(f"{'Threshold Type':<25} {'Thresh':<8} {'Sens':<6} {'Spec':<6} {'Prec':<6} {'Acc':<6} {'FPR':<6}")
    print(f"{'-'*80}")
    
    for data in table_data:
        if data['threshold'] is not None:
            print(f"{data['name']:<25} {data['threshold']:<8.4f} {data['sensitivity']:<6.3f} "
                  f"{data['specificity']:<6.3f} {data['precision']:<6.3f} {data['accuracy']:<6.3f} "
                  f"{data['fpr']:<6.3f}")
        else:
            print(f"{data['name']:<25} {'N/A':<8} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<6}")
    
    # Print confusion matrix table
    print(f"\nCONFUSION MATRIX TABLE")
    print(f"{'-'*80}")
    print(f"{'Threshold Type':<25} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} {'Total Pos':<10} {'Total Neg':<10}")
    print(f"{'-'*80}")
    
    for data in table_data:
        if data['threshold'] is not None:
            total_pos = data['tp'] + data['fp']
            total_neg = data['tn'] + data['fn']
            print(f"{data['name']:<25} {data['tp']:<6} {data['tn']:<6} {data['fp']:<6} "
                  f"{data['fn']:<6} {total_pos:<10} {total_neg:<10}")
        else:
            print(f"{data['name']:<25} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<10} {'N/A':<10}")
    
    # Print detailed classification reports for each threshold
    print(f"\nDETAILED CLASSIFICATION REPORTS")
    print(f"{'='*80}")
    
    for data in table_data:
        if data['threshold'] is not None:
            print(f"\n{data['name']} (Threshold = {data['threshold']:.4f}):")
            print(f"{'-'*50}")
            y_pred = (y_scores >= data['threshold']).astype(int)
            report = classification_report(y_true, y_pred, target_names=['Non-REM', 'REM'], 
                                         digits=4, zero_division=0)
            print(report)
        else:
            print(f"\n{data['name']}: Not achievable with this model")
            print(f"{'-'*50}")
    
    print(f"\n{'='*80}")
    print(f"Legend:")
    print(f"  Sens = Sensitivity (Recall/TPR) - Proportion of actual REM epochs correctly identified")
    print(f"  Spec = Specificity (TNR) - Proportion of actual Non-REM epochs correctly identified")
    print(f"  Prec = Precision (PPV) - Proportion of REM predictions that are correct")
    print(f"  Acc  = Accuracy - Overall proportion of correct predictions")
    print(f"  FPR  = False Positive Rate (1 - Specificity) - Proportion of Non-REM epochs incorrectly classified as REM")
    print(f"  TP   = True Positives - Correctly identified REM epochs")
    print(f"  TN   = True Negatives - Correctly identified Non-REM epochs")
    print(f"  FP   = False Positives - Non-REM epochs incorrectly classified as REM")
    print(f"  FN   = False Negatives - REM epochs incorrectly classified as Non-REM")
    print(f"{'='*80}")

def export_comprehensive_report_to_md(y_true, y_scores, thresholds_dict, model_name="YASA REM Detection Model", output_path=None):
    """
    Export comprehensive classification report to a Markdown file.
    
    Args:
        y_true: True labels (0=Non-REM, 1=REM)
        y_scores: Prediction scores from REM detection model
        thresholds_dict: Dictionary containing thresholds for different criteria
        model_name: Name of the model for the report title
        output_path: Path to save the markdown file (optional)
    """
    # Generate filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"rem_classification_report_{timestamp}.md"
    
    # Define threshold names and values
    threshold_configs = [
        ("Conservative (FPR ≤ 5%)", thresholds_dict.get('conservative')),
        ("Balanced (FPR ≤ 10%)", thresholds_dict.get('balanced')),
        ("Sensitive (70% confidence)", thresholds_dict.get('sensitive')),
        ("High Confidence (80%)", thresholds_dict.get('high_conf')),
        ("Very High Confidence (90%)", thresholds_dict.get('very_high_conf'))
    ]
    
    # Collect all metrics for table
    table_data = []
    
    for name, threshold in threshold_configs:
        if threshold is not None:
            # Calculate predictions and metrics
            y_pred = (y_scores >= threshold).astype(int)
            metrics = evaluate_threshold(y_true, y_scores, threshold)
            
            # Calculate additional metrics
            fpr = 1 - metrics['specificity']
            
            table_data.append({
                'name': name,
                'threshold': threshold,
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'precision': metrics['precision'],
                'accuracy': metrics['accuracy'],
                'fpr': fpr,
                'tp': metrics['tp'],
                'tn': metrics['tn'],
                'fp': metrics['fp'],
                'fn': metrics['fn']
            })
        else:
            # Handle case where threshold is not achievable
            table_data.append({
                'name': name,
                'threshold': None,
                'sensitivity': None,
                'specificity': None,
                'precision': None,
                'accuracy': None,
                'fpr': None,
                'tp': None,
                'tn': None,
                'fp': None,
                'fn': None
            })
    
    # Calculate overall model metrics
    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_scores)
    model_auc = auc(fpr_roc, tpr_roc)
    
    # Generate Markdown content
    md_content = f"""# {model_name} - Comprehensive Classification Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Overview

- **Model AUC:** {model_auc:.4f}
- **Total Epochs:** {len(y_true)}
- **REM Epochs (Class 1):** {np.sum(y_true == 1)} ({np.sum(y_true == 1)/len(y_true)*100:.1f}%)
- **Non-REM Epochs (Class 0):** {np.sum(y_true == 0)} ({np.sum(y_true == 0)/len(y_true)*100:.1f}%)
- **Score Range:** [{np.min(y_scores):.4f}, {np.max(y_scores):.4f}]

## Summary Metrics Table

| Threshold Type | Threshold | Sensitivity | Specificity | Precision | Accuracy | FPR |
|---|---|---|---|---|---|---|
"""
    
    # Add summary metrics rows
    for data in table_data:
        if data['threshold'] is not None:
            md_content += f"| {data['name']} | {data['threshold']:.4f} | {data['sensitivity']:.3f} | {data['specificity']:.3f} | {data['precision']:.3f} | {data['accuracy']:.3f} | {data['fpr']:.3f} |\n"
        else:
            md_content += f"| {data['name']} | N/A | N/A | N/A | N/A | N/A | N/A |\n"
    
    # Add confusion matrix table
    md_content += f"""
## Confusion Matrix Table

| Threshold Type | TP | TN | FP | FN | Total Pos | Total Neg |
|---|---|---|---|---|---|---|
"""
    
    for data in table_data:
        if data['threshold'] is not None:
            total_pos = data['tp'] + data['fp']
            total_neg = data['tn'] + data['fn']
            md_content += f"| {data['name']} | {data['tp']} | {data['tn']} | {data['fp']} | {data['fn']} | {total_pos} | {total_neg} |\n"
        else:
            md_content += f"| {data['name']} | N/A | N/A | N/A | N/A | N/A | N/A |\n"
    
    # Add detailed classification reports
    md_content += f"""
## Detailed Classification Reports

"""
    
    for data in table_data:
        if data['threshold'] is not None:
            md_content += f"""### {data['name']} (Threshold = {data['threshold']:.4f})

```
"""
            y_pred = (y_scores >= data['threshold']).astype(int)
            report = classification_report(y_true, y_pred, target_names=['Non-REM', 'REM'], 
                                         digits=4, zero_division=0)
            md_content += report
            md_content += """```

"""
        else:
            md_content += f"""### {data['name']}
**Status:** Not achievable with this model

"""
    
    # Add threshold comparison analysis
    # Helper function to safely format percentages
    def safe_format_percent(value):
        return f"{value*100:.1f}" if value is not None else "N/A"
    
    def safe_format_threshold(value):
        return f"{value:.4f}" if value is not None else "N/A"
    
    # Safely access table data with bounds checking
    conservative_fpr = safe_format_percent(table_data[0]['fpr'] if len(table_data) > 0 else None)
    conservative_sens = safe_format_percent(table_data[0]['sensitivity'] if len(table_data) > 0 else None)
    conservative_prec = safe_format_percent(table_data[0]['precision'] if len(table_data) > 0 else None)
    conservative_thresh = safe_format_threshold(thresholds_dict.get('conservative'))
    
    high_conf_prec = safe_format_percent(table_data[4]['precision'] if len(table_data) > 4 and table_data[4]['precision'] is not None else None)
    
    sensitive_thresh = safe_format_threshold(thresholds_dict.get('sensitive'))
    sensitive_sens = safe_format_percent(table_data[2]['sensitivity'] if len(table_data) > 2 else None)
    sensitive_prec = safe_format_percent(table_data[2]['precision'] if len(table_data) > 2 else None)
    
    balanced_thresh = safe_format_threshold(thresholds_dict.get('balanced'))
    balanced_sens = safe_format_percent(table_data[1]['sensitivity'] if len(table_data) > 1 else None)
    balanced_prec = safe_format_percent(table_data[1]['precision'] if len(table_data) > 1 else None)
    
    md_content += f"""## Threshold Strategy Comparison

### FPR-Based vs Confidence-Based Thresholds

**Conservative Strategy (FPR ≤ 5%):**
- **Goal:** Minimize false alarms from Non-REM epochs
- **Result:** Only {conservative_fpr}% of Non-REM epochs incorrectly classified as REM
- **Trade-off:** Lower sensitivity ({conservative_sens}%) - only catches {conservative_sens}% of actual REM epochs

**High Confidence Strategy (90% confidence):**
- **Goal:** High confidence when predicting REM
- **Result:** When predicting REM, correct {high_conf_prec}% of the time
- **Trade-off:** Lower sensitivity - only catches a subset of actual REM epochs

### Recommendations

**For Clinical Research (High Precision Required):**
- Use **Conservative (FPR ≤ 5%)** threshold ({conservative_thresh})
- Achieves {conservative_prec}% precision with {conservative_sens}% sensitivity

**For Real-time Lucid Dream Detection (High Sensitivity Required):**
- Use **Sensitive (70% confidence)** threshold ({sensitive_thresh})
- Achieves {sensitive_sens}% sensitivity with {sensitive_prec}% precision

**For Balanced Performance:**
- Use **Balanced (FPR ≤ 10%)** threshold ({balanced_thresh})
- Achieves {balanced_sens}% sensitivity with {balanced_prec}% precision

## REM Detection Context

REM sleep detection has specific requirements compared to general classification tasks:

- **REM constitutes ~20-25% of total sleep** - Class imbalance is expected
- **False positives can disrupt sleep** - Conservative thresholds preferred for sleep applications
- **Missing REM periods reduces study validity** - Balance needed between precision and recall
- **Real-time detection requires fast response** - Consider latency vs accuracy trade-offs

## Metrics Legend

- **Sens (Sensitivity):** True Positive Rate (Recall) - Proportion of actual REM epochs correctly identified
- **Spec (Specificity):** True Negative Rate - Proportion of actual Non-REM epochs correctly identified  
- **Prec (Precision):** Positive Predictive Value - Proportion of REM predictions that are correct
- **Acc (Accuracy):** Overall proportion of correct predictions
- **FPR:** False Positive Rate (1 - Specificity) - Proportion of Non-REM epochs incorrectly classified as REM
- **TP:** True Positives - Correctly identified REM epochs
- **TN:** True Negatives - Correctly identified Non-REM epochs
- **FP:** False Positives - Non-REM epochs incorrectly classified as REM
- **FN:** False Negatives - REM epochs incorrectly classified as Non-REM

---
*Report generated by REM Detection Classification Analysis Script*
"""
    
    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"\n✅ Comprehensive REM classification report exported to: {output_path}")
        return output_path
    except Exception as e:
        print(f"\n❌ Error writing report to {output_path}: {e}")
        return None

def analyze_rem_classification_performance(y_true, y_scores, model_name="YASA REM Detection Model"):
    """
    Main function to analyze REM classification performance using empirically determined thresholds.
    
    Args:
        y_true: True labels (0=Non-REM, 1=REM)
        y_scores: REM probability scores from YASA model
        model_name: Name of the model for reporting
    
    Returns:
        dict: Dictionary containing all threshold values and metrics
    """
    print("="*80)
    print("REM CLASSIFICATION PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    model_auc = auc(fpr, tpr)
    print(f"\nModel AUC: {model_auc:.4f}")
    print(f"Total epochs: {len(y_true)}")
    print(f"REM epochs: {np.sum(y_true == 1)} ({np.sum(y_true == 1)/len(y_true)*100:.1f}%)")
    print(f"Non-REM epochs: {np.sum(y_true == 0)} ({np.sum(y_true == 0)/len(y_true)*100:.1f}%)")
    
    # Find optimal thresholds based on empirical analysis
    print("\nFinding optimal thresholds...")
    
    # 1. Conservative threshold (FPR ≤ 5%) - Empirically determined from 20-night analysis
    conservative_threshold = 0.7786  # From fpr_based_threshold_analysis.py
    conservative_metrics = evaluate_threshold(y_true, y_scores, conservative_threshold)
    
    print(f"\n1. CONSERVATIVE THRESHOLD (FPR ≤ 5%):")
    print(f"   Threshold: {conservative_threshold:.4f} (empirically determined)")
    print(f"   Sensitivity: {conservative_metrics['sensitivity']:.4f}")
    print(f"   Specificity: {conservative_metrics['specificity']:.4f}")
    print(f"   Precision: {conservative_metrics['precision']:.4f}")
    print(f"   Accuracy: {conservative_metrics['accuracy']:.4f}")
    print(f"   FPR: {1 - conservative_metrics['specificity']:.4f}")
    
    # 2. Balanced threshold (FPR ≤ 10%)
    balanced_threshold = 0.5449  # From fpr_based_threshold_analysis.py
    balanced_metrics = evaluate_threshold(y_true, y_scores, balanced_threshold)
    
    print(f"\n2. BALANCED THRESHOLD (FPR ≤ 10%):")
    print(f"   Threshold: {balanced_threshold:.4f} (empirically determined)")
    print(f"   Sensitivity: {balanced_metrics['sensitivity']:.4f}")
    print(f"   Specificity: {balanced_metrics['specificity']:.4f}")
    print(f"   Precision: {balanced_metrics['precision']:.4f}")
    print(f"   Accuracy: {balanced_metrics['accuracy']:.4f}")
    print(f"   FPR: {1 - balanced_metrics['specificity']:.4f}")
    
    # 3. Sensitive threshold (70% confidence)
    sensitive_threshold = 0.4382  # From fpr_based_threshold_analysis.py
    sensitive_metrics = evaluate_threshold(y_true, y_scores, sensitive_threshold)
    
    print(f"\n3. SENSITIVE THRESHOLD (70% confidence):")
    print(f"   Threshold: {sensitive_threshold:.4f} (empirically determined)")
    print(f"   Sensitivity: {sensitive_metrics['sensitivity']:.4f}")
    print(f"   Specificity: {sensitive_metrics['specificity']:.4f}")
    print(f"   Precision: {sensitive_metrics['precision']:.4f}")
    print(f"   Accuracy: {sensitive_metrics['accuracy']:.4f}")
    print(f"   FPR: {1 - sensitive_metrics['specificity']:.4f}")
    
    # 4. High confidence threshold (80%)
    high_conf_threshold, high_conf_prec, high_conf_recall = find_threshold_for_confidence(y_true, y_scores, 0.8)
    if high_conf_threshold is not None:
        high_conf_metrics = evaluate_threshold(y_true, y_scores, high_conf_threshold)
        print(f"\n4. HIGH CONFIDENCE THRESHOLD (80%):")
        print(f"   Threshold: {high_conf_threshold:.4f}")
        print(f"   Sensitivity: {high_conf_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {high_conf_metrics['specificity']:.4f}")
        print(f"   Precision: {high_conf_metrics['precision']:.4f}")
        print(f"   Accuracy: {high_conf_metrics['accuracy']:.4f}")
        print(f"   FPR: {1 - high_conf_metrics['specificity']:.4f}")
    else:
        print(f"\n4. HIGH CONFIDENCE THRESHOLD (80%): Not achievable with this model")
    
    # 5. Very high confidence threshold (90%)
    very_high_conf_threshold, very_high_conf_prec, very_high_conf_recall = find_threshold_for_confidence(y_true, y_scores, 0.9)
    if very_high_conf_threshold is not None:
        very_high_conf_metrics = evaluate_threshold(y_true, y_scores, very_high_conf_threshold)
        print(f"\n5. VERY HIGH CONFIDENCE THRESHOLD (90%):")
        print(f"   Threshold: {very_high_conf_threshold:.4f}")
        print(f"   Sensitivity: {very_high_conf_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {very_high_conf_metrics['specificity']:.4f}")
        print(f"   Precision: {very_high_conf_metrics['precision']:.4f}")
        print(f"   Accuracy: {very_high_conf_metrics['accuracy']:.4f}")
        print(f"   FPR: {1 - very_high_conf_metrics['specificity']:.4f}")
    else:
        print(f"\n5. VERY HIGH CONFIDENCE THRESHOLD (90%): Not achievable with this model")
    
    # Create thresholds dictionary
    thresholds_dict = {
        'conservative': conservative_threshold,
        'balanced': balanced_threshold,
        'sensitive': sensitive_threshold,
        'high_conf': high_conf_threshold,
        'very_high_conf': very_high_conf_threshold
    }
    
    # Generate plots
    print(f"\nGenerating visualizations...")
    
    # Plot prediction distribution
    plot_prediction_distribution(y_true, y_scores, model_name)
    
    # Plot ROC curve with threshold points
    plot_roc_with_thresholds(y_true, y_scores, thresholds_dict, model_name)
    
    # Plot threshold analysis
    analyze_threshold_performance(y_true, y_scores)
    
    # Print comprehensive classification table for all thresholds
    print_comprehensive_classification_table(y_true, y_scores, thresholds_dict)
    
    # Export comprehensive report to Markdown
    export_comprehensive_report_to_md(y_true, y_scores, thresholds_dict, model_name)
    
    print(f"\nREM classification analysis completed!")
    
    return thresholds_dict

if __name__ == "__main__":
    # Example usage with dummy data
    print("REM Classification Report Generator")
    print("This script provides comprehensive analysis for REM detection models.")
    print("To use with real data, call analyze_rem_classification_performance(y_true, y_scores)")
    
    # Generate example data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate REM detection scores (higher scores = more likely REM)
    # 25% REM epochs (realistic for sleep data)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    
    # Generate realistic scores: REM epochs tend to have higher scores
    y_scores = np.random.beta(2, 5, n_samples)  # Base scores
    y_scores[y_true == 1] += np.random.beta(3, 2, np.sum(y_true == 1)) * 0.4  # Boost REM scores
    y_scores = np.clip(y_scores, 0, 1)  # Ensure [0,1] range
    
    print(f"\nExample with {n_samples} simulated sleep epochs:")
    print(f"REM epochs: {np.sum(y_true == 1)} ({np.sum(y_true == 1)/len(y_true)*100:.1f}%)")
    
    # Run analysis
    analyze_rem_classification_performance(y_true, y_scores, "Example REM Detection Model")
