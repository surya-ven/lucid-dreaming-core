"""
Test script for the trained Conv1D LRLR model.
Evaluates the model on lstm_training_data.npz and generates comprehensive metrics and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def plot_confusion_matrix(y_true, y_scores, threshold, title, save_path=None):
    """Plot confusion matrix for a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
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
    colors = ['red', 'green', 'blue']
    markers = ['o', 's', '^']
    threshold_names = ['Best AUC', 'FPR ≤ 0.05', 'FPR ≤ 0.1']
    threshold_values = [thresholds_dict['best_auc'], thresholds_dict['fpr005'], thresholds_dict['fpr01']]
    
    for i, (name, threshold) in enumerate(zip(threshold_names, threshold_values)):
        if threshold is not None:
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
    
    plt.hist(scores_class_0, bins=50, alpha=0.6, label='Non-LRLR (Class 0)', color='blue', density=True)
    plt.hist(scores_class_1, bins=50, alpha=0.6, label='LRLR (Class 1)', color='red', density=True)
    
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
        y_true: True labels
        y_scores: Prediction scores
        thresholds_dict: Dictionary containing thresholds for different criteria
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE CLASSIFICATION REPORT - ALL THRESHOLDS")
    print(f"{'='*80}")
    
    # Define threshold names and values
    threshold_configs = [
        ("Best AUC (Youden's J)", thresholds_dict['best_auc']),
        ("FPR ≤ 0.05", thresholds_dict['fpr005']),
        ("FPR ≤ 0.1", thresholds_dict['fpr01']),
        ("Precision ≥ 0.9", thresholds_dict['prec90']),
        ("Precision ≥ 0.95", thresholds_dict['prec95'])
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
    print(f"{'Threshold Type':<20} {'Thresh':<8} {'Sens':<6} {'Spec':<6} {'Prec':<6} {'Acc':<6} {'FPR':<6}")
    print(f"{'-'*80}")
    
    for data in table_data:
        if data['threshold'] is not None:
            print(f"{data['name']:<20} {data['threshold']:<8.4f} {data['sensitivity']:<6.3f} "
                  f"{data['specificity']:<6.3f} {data['precision']:<6.3f} {data['accuracy']:<6.3f} "
                  f"{data['fpr']:<6.3f}")
        else:
            print(f"{data['name']:<20} {'N/A':<8} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<6}")
    
    # Print confusion matrix table
    print(f"\nCONFUSION MATRIX TABLE")
    print(f"{'-'*80}")
    print(f"{'Threshold Type':<20} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} {'Total Pos':<10} {'Total Neg':<10}")
    print(f"{'-'*80}")
    
    for data in table_data:
        if data['threshold'] is not None:
            total_pos = data['tp'] + data['fp']
            total_neg = data['tn'] + data['fn']
            print(f"{data['name']:<20} {data['tp']:<6} {data['tn']:<6} {data['fp']:<6} "
                  f"{data['fn']:<6} {total_pos:<10} {total_neg:<10}")
        else:
            print(f"{data['name']:<20} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<10} {'N/A':<10}")
    
    # Print detailed classification reports for each threshold
    print(f"\nDETAILED CLASSIFICATION REPORTS")
    print(f"{'='*80}")
    
    for data in table_data:
        if data['threshold'] is not None:
            print(f"\n{data['name']} (Threshold = {data['threshold']:.4f}):")
            print(f"{'-'*50}")
            y_pred = (y_scores >= data['threshold']).astype(int)
            report = classification_report(y_true, y_pred, target_names=['Non-LRLR', 'LRLR'], 
                                         digits=4, zero_division=0)
            print(report)
        else:
            print(f"\n{data['name']}: Not achievable with this model")
            print(f"{'-'*50}")
    
    print(f"\n{'='*80}")
    print(f"Legend:")
    print(f"  Sens = Sensitivity (Recall/TPR)")
    print(f"  Spec = Specificity (TNR)")
    print(f"  Prec = Precision (PPV)")
    print(f"  Acc  = Accuracy")
    print(f"  FPR  = False Positive Rate (1 - Specificity)")
    print(f"  TP   = True Positives")
    print(f"  TN   = True Negatives")
    print(f"  FP   = False Positives")
    print(f"  FN   = False Negatives")
    print(f"{'='*80}")

def find_threshold_for_precision(y_true, y_scores, target_precision):
    """Find threshold that achieves target precision."""
    # Try different thresholds from high to low
    thresholds = np.arange(0.99, 0.01, -0.01)
    
    best_threshold = None
    best_recall = 0
    best_metrics = None
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate precision
        if tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # If we achieve target precision and this has better recall
            if precision >= target_precision and recall > best_recall:
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

def export_comprehensive_report_to_md(y_true, y_scores, thresholds_dict, model_name="Conv1D Model", output_path=None):
    """
    Export comprehensive classification report to a Markdown file.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        thresholds_dict: Dictionary containing thresholds for different criteria
        model_name: Name of the model for the report title
        output_path: Path to save the markdown file (optional)
    """
    from datetime import datetime
    
    # Generate filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"classification_report_{model_name.lower().replace(' ', '_')}_{timestamp}.md"
    
    # Define threshold names and values
    threshold_configs = [
        ("Best AUC (Youden's J)", thresholds_dict['best_auc']),
        ("FPR ≤ 0.05", thresholds_dict['fpr005']),
        ("FPR ≤ 0.1", thresholds_dict['fpr01']),
        ("Precision ≥ 0.9", thresholds_dict['prec90']),
        ("Precision ≥ 0.95", thresholds_dict['prec95'])
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
- **Total Samples:** {len(y_true)}
- **LRLR Samples (Class 1):** {np.sum(y_true == 1)} ({np.sum(y_true == 1)/len(y_true)*100:.1f}%)
- **Non-LRLR Samples (Class 0):** {np.sum(y_true == 0)} ({np.sum(y_true == 0)/len(y_true)*100:.1f}%)
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
            report = classification_report(y_true, y_pred, target_names=['Non-LRLR', 'LRLR'], 
                                         digits=4, zero_division=0)
            md_content += report
            md_content += """```

"""
        else:
            md_content += f"""### {data['name']}

**Status:** Not achievable with this model

"""
    
    # Add threshold comparison analysis
    md_content += f"""## Threshold Strategy Comparison

### FPR-Based vs Precision-Based Thresholds

**FPR ≤ 0.1 Strategy:**
- **Goal:** Minimize false alarms from Non-LRLR samples
- **Result:** Only {table_data[2]['fpr']*100:.1f}% of Non-LRLR samples incorrectly classified as LRLR
- **Trade-off:** Lower precision ({table_data[2]['precision']*100:.1f}%) - when predicting LRLR, only correct {table_data[2]['precision']*100:.1f}% of the time

**Precision ≥ 0.9 Strategy:**
- **Goal:** High confidence when predicting LRLR
- **Result:** When predicting LRLR, correct {table_data[3]['precision']*100:.1f}% of the time
- **Trade-off:** Lower sensitivity ({table_data[3]['sensitivity']*100:.1f}%) - only catches {table_data[3]['sensitivity']*100:.1f}% of actual LRLR events

### Recommendations

**For Medical Diagnosis (High Precision Required):**
- Use **Precision ≥ 0.9** threshold ({thresholds_dict['prec90']:.4f})
- Achieves {table_data[3]['precision']*100:.1f}% precision with {table_data[3]['sensitivity']*100:.1f}% sensitivity

**For Screening Tool (High Sensitivity Required):**
- Use **Best AUC** threshold ({thresholds_dict['best_auc']:.4f})
- Achieves {table_data[0]['sensitivity']*100:.1f}% sensitivity with {table_data[0]['precision']*100:.1f}% precision

**For Balanced Performance:**
- Use **FPR ≤ 0.1** threshold ({thresholds_dict['fpr01']:.4f})
- Achieves {table_data[2]['sensitivity']*100:.1f}% sensitivity with {table_data[2]['precision']*100:.1f}% precision

## Metrics Legend

- **Sens (Sensitivity):** True Positive Rate (Recall) - Proportion of actual LRLR events correctly identified
- **Spec (Specificity):** True Negative Rate - Proportion of actual Non-LRLR events correctly identified
- **Prec (Precision):** Positive Predictive Value - Proportion of LRLR predictions that are correct
- **Acc (Accuracy):** Overall proportion of correct predictions
- **FPR:** False Positive Rate (1 - Specificity) - Proportion of Non-LRLR events incorrectly classified as LRLR
- **TP:** True Positives - Correctly identified LRLR events
- **TN:** True Negatives - Correctly identified Non-LRLR events
- **FP:** False Positives - Non-LRLR events incorrectly classified as LRLR
- **FN:** False Negatives - LRLR events incorrectly classified as Non-LRLR

---
*Report generated by {model_name} evaluation script*
"""
    
    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"\n✅ Comprehensive report exported to: {output_path}")
        return output_path
    except Exception as e:
        print(f"\n❌ Error writing report to {output_path}: {e}")
        return None

def main():
    """Main function to test Conv1D model."""
    print("Testing Conv1D LRLR Model")
    print("=" * 50)
    
    # Load model
    model_path = 'models/lrlr_conv1d_model_fold__final_all_data_v2.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test data
    data_path = 'lstm_training_data.npz'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    try:
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        print(f"Data loaded successfully from {data_path}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Clean data if needed
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: Found NaN/Inf values in data. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Class distribution:")
    print(f"  LRLR samples (1): {np.sum(y == 1)}")
    print(f"  Non-LRLR samples (0): {np.sum(y == 0)}")
    
    # Make predictions
    print("\nMaking predictions...")
    try:
        y_scores = model.predict(X, verbose=1)
        y_scores = y_scores.flatten()  # Ensure 1D array
        print(f"Predictions completed. Score range: [{np.min(y_scores):.4f}, {np.max(y_scores):.4f}]")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y, y_scores)
    model_auc = auc(fpr, tpr)
    print(f"\nModel AUC: {model_auc:.4f}")
    
    # Find optimal thresholds
    print("\nFinding optimal thresholds...")
    
    # 1. Best AUC threshold (Youden's J)
    best_auc_threshold, best_sens, best_spec = find_best_auc_threshold(y, y_scores)
    auc_metrics = evaluate_threshold(y, y_scores, best_auc_threshold)
    
    print(f"\n1. BEST AUC THRESHOLD (Youden's J):")
    print(f"   Threshold: {best_auc_threshold:.4f}")
    print(f"   Sensitivity: {auc_metrics['sensitivity']:.4f}")
    print(f"   Specificity: {auc_metrics['specificity']:.4f}")
    print(f"   Precision: {auc_metrics['precision']:.4f}")
    print(f"   Accuracy: {auc_metrics['accuracy']:.4f}")
    
    # 2. FPR ≤ 0.05 threshold
    fpr005_threshold, fpr005_actual, fpr005_sens = find_threshold_for_fpr(y, y_scores, 0.05)
    if fpr005_threshold is not None:
        fpr005_metrics = evaluate_threshold(y, y_scores, fpr005_threshold)
        print(f"\n2. FPR ≤ 0.05 THRESHOLD:")
        print(f"   Threshold: {fpr005_threshold:.4f}")
        print(f"   Sensitivity: {fpr005_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {fpr005_metrics['specificity']:.4f}")
        print(f"   Precision: {fpr005_metrics['precision']:.4f}")
        print(f"   Accuracy: {fpr005_metrics['accuracy']:.4f}")
        print(f"   FPR: {1 - fpr005_metrics['specificity']:.4f}")
    else:
        print(f"\n2. FPR ≤ 0.05: Not achievable with this model")
    
    # 3. FPR ≤ 0.1 threshold
    fpr01_threshold, fpr01_actual, fpr01_sens = find_threshold_for_fpr(y, y_scores, 0.1)
    if fpr01_threshold is not None:
        fpr01_metrics = evaluate_threshold(y, y_scores, fpr01_threshold)
        print(f"\n3. FPR ≤ 0.1 THRESHOLD:")
        print(f"   Threshold: {fpr01_threshold:.4f}")
        print(f"   Sensitivity: {fpr01_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {fpr01_metrics['specificity']:.4f}")
        print(f"   Precision: {fpr01_metrics['precision']:.4f}")
        print(f"   Accuracy: {fpr01_metrics['accuracy']:.4f}")
        print(f"   FPR: {1 - fpr01_metrics['specificity']:.4f}")
    else:
        print(f"\n3. FPR ≤ 0.1: Not achievable with this model")
    
    # 4. Precision ≥ 0.9 threshold
    prec90_threshold, prec90_actual, prec90_recall = find_threshold_for_precision(y, y_scores, 0.9)
    if prec90_threshold is not None:
        prec90_metrics = evaluate_threshold(y, y_scores, prec90_threshold)
        print(f"\n4. PRECISION ≥ 0.9 THRESHOLD:")
        print(f"   Threshold: {prec90_threshold:.4f}")
        print(f"   Sensitivity: {prec90_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {prec90_metrics['specificity']:.4f}")
        print(f"   Precision: {prec90_metrics['precision']:.4f}")
        print(f"   Accuracy: {prec90_metrics['accuracy']:.4f}")
        print(f"   FPR: {1 - prec90_metrics['specificity']:.4f}")
    else:
        print(f"\n4. PRECISION ≥ 0.9: Not achievable with this model")
    
    # 5. Precision ≥ 0.95 threshold
    prec95_threshold, prec95_actual, prec95_recall = find_threshold_for_precision(y, y_scores, 0.95)
    if prec95_threshold is not None:
        prec95_metrics = evaluate_threshold(y, y_scores, prec95_threshold)
        print(f"\n5. PRECISION ≥ 0.95 THRESHOLD:")
        print(f"   Threshold: {prec95_threshold:.4f}")
        print(f"   Sensitivity: {prec95_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {prec95_metrics['specificity']:.4f}")
        print(f"   Precision: {prec95_metrics['precision']:.4f}")
        print(f"   Accuracy: {prec95_metrics['accuracy']:.4f}")
        print(f"   FPR: {1 - prec95_metrics['specificity']:.4f}")
    else:
        print(f"\n5. PRECISION ≥ 0.95: Not achievable with this model")
    
    # Generate plots
    print(f"\nGenerating plots...")
    
    # Plot prediction distribution
    plot_prediction_distribution(y, y_scores, "Conv1D Model")
    
    # Plot ROC curve with threshold points
    thresholds_dict = {
        'best_auc': best_auc_threshold,
        'fpr005': fpr005_threshold,
        'fpr01': fpr01_threshold,
        'prec90': prec90_threshold,
        'prec95': prec95_threshold
    }
    plot_roc_with_thresholds(y, y_scores, thresholds_dict, "Conv1D Model")
    
    # Plot confusion matrices for each threshold # TODO restore
    # plot_confusion_matrix(y, y_scores, best_auc_threshold, 
    #                      "Conv1D Model - Best AUC Threshold")
    
    # if fpr005_threshold is not None:
    #     plot_confusion_matrix(y, y_scores, fpr005_threshold, 
    #                          "Conv1D Model - FPR ≤ 0.05 Threshold")
    
    # if fpr01_threshold is not None:
    #     plot_confusion_matrix(y, y_scores, fpr01_threshold, 
    #                          "Conv1D Model - FPR ≤ 0.1 Threshold")
    
    # Plot threshold analysis
    # analyze_threshold_performance(y, y_scores) # TODO restore
    
    # Print comprehensive classification table for all thresholds
    print_comprehensive_classification_table(y, y_scores, thresholds_dict)
    
    # Export comprehensive report to Markdown
    export_comprehensive_report_to_md(y, y_scores, thresholds_dict, "Conv1D Model")
    
    print(f"\nTesting completed!")

if __name__ == "__main__":
    main()
