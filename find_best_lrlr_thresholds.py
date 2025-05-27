

"""
Script to find optimal LRLR classification thresholds for Conv1D and LSTM models.
Identifies best thresholds for:
1. Maximum AUC
2. False Positive Rate ≤ 0.05
3. False Positive Rate ≤ 0.1
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

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

def create_threshold_summary_plot(conv1d_results, lstm_results):
    """Create a summary plot comparing thresholds between models."""
    models = ['Conv1D', 'LSTM']
    threshold_types = ['Best AUC', 'FPR ≤ 0.05', 'FPR ≤ 0.1']
    
    conv1d_thresholds = [
        conv1d_results['best_auc_threshold'],
        conv1d_results['fpr005_threshold'],
        conv1d_results['fpr01_threshold']
    ]
    
    lstm_thresholds = [
        lstm_results['best_auc_threshold'],
        lstm_results['fpr005_threshold'],
        lstm_results['fpr01_threshold']
    ]
    
    # Replace None values with NaN for plotting
    conv1d_thresholds = [t if t is not None else np.nan for t in conv1d_thresholds]
    lstm_thresholds = [t if t is not None else np.nan for t in lstm_thresholds]
    
    x = np.arange(len(threshold_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, conv1d_thresholds, width, label='Conv1D', alpha=0.8)
    rects2 = ax.bar(x + width/2, lstm_thresholds, width, label='LSTM', alpha=0.8)
    
    ax.set_ylabel('Threshold Value')
    ax.set_title('Optimal Thresholds Comparison Between Models')
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            else:
                ax.annotate('N/A',
                           xy=(rect.get_x() + rect.get_width() / 2, 0),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_and_clean_data(csv_path):
    """Load CSV data and remove NA values."""
    df = pd.read_csv(csv_path)
    # Remove rows with NA values in key columns
    df = df.dropna(subset=['true_label', 'prediction_score'])
    print(f"Loaded {len(df)} valid samples from {csv_path}")
    return df

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

def analyze_model(csv_path, model_name):
    """Analyze a single model's performance and find optimal thresholds."""
    print(f"\n{'='*50}")
    print(f"ANALYZING {model_name}")
    print(f"{'='*50}")
    
    # Load data
    df = load_and_clean_data(csv_path)
    y_true = df['true_label'].values
    y_scores = df['prediction_score'].values
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    model_auc = auc(fpr, tpr)
    print(f"Model AUC: {model_auc:.4f}")
    
    # Find best threshold for maximum AUC (Youden's J)
    best_auc_threshold, best_sens, best_spec = find_best_auc_threshold(y_true, y_scores)
    auc_metrics = evaluate_threshold(y_true, y_scores, best_auc_threshold)
    
    print(f"\n1. BEST AUC THRESHOLD (Youden's J):")
    print(f"   Threshold: {best_auc_threshold:.4f}")
    print(f"   Sensitivity: {auc_metrics['sensitivity']:.4f}")
    print(f"   Specificity: {auc_metrics['specificity']:.4f}")
    print(f"   Precision: {auc_metrics['precision']:.4f}")
    print(f"   Accuracy: {auc_metrics['accuracy']:.4f}")
    
    # Find threshold for FPR ≤ 0.05
    fpr005_threshold, fpr005_actual, fpr005_sens = find_threshold_for_fpr(y_true, y_scores, 0.05)
    if fpr005_threshold is not None:
        fpr005_metrics = evaluate_threshold(y_true, y_scores, fpr005_threshold)
        print(f"\n2. FPR ≤ 0.05 THRESHOLD:")
        print(f"   Threshold: {fpr005_threshold:.4f}")
        print(f"   Sensitivity: {fpr005_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {fpr005_metrics['specificity']:.4f}")
        print(f"   Precision: {fpr005_metrics['precision']:.4f}")
        print(f"   Accuracy: {fpr005_metrics['accuracy']:.4f}")
        print(f"   FPR: {1 - fpr005_metrics['specificity']:.4f}")
    else:
        print(f"\n2. FPR ≤ 0.05: Not achievable with this model")
    
    # Find threshold for FPR ≤ 0.1
    fpr01_threshold, fpr01_actual, fpr01_sens = find_threshold_for_fpr(y_true, y_scores, 0.1)
    if fpr01_threshold is not None:
        fpr01_metrics = evaluate_threshold(y_true, y_scores, fpr01_threshold)
        print(f"\n3. FPR ≤ 0.1 THRESHOLD:")
        print(f"   Threshold: {fpr01_threshold:.4f}")
        print(f"   Sensitivity: {fpr01_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {fpr01_metrics['specificity']:.4f}")
        print(f"   Precision: {fpr01_metrics['precision']:.4f}")
        print(f"   Accuracy: {fpr01_metrics['accuracy']:.4f}")
        print(f"   FPR: {1 - fpr01_metrics['specificity']:.4f}")
    else:
        print(f"\n3. FPR ≤ 0.1: Not achievable with this model")
    
    # Create plots for this model
    print(f"\nGenerating plots for {model_name}...")
    
    # Plot ROC curve with threshold points
    thresholds_dict = {
        'best_auc': best_auc_threshold,
        'fpr005': fpr005_threshold,
        'fpr01': fpr01_threshold
    }
    plot_roc_with_thresholds(y_true, y_scores, thresholds_dict, model_name, 
                            f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    
    # Plot confusion matrices for each threshold
    plot_confusion_matrix(y_true, y_scores, best_auc_threshold, 
                         f'{model_name} - Best AUC Threshold',
                         f'cm_{model_name.lower().replace(" ", "_")}_best_auc.png')
    
    if fpr005_threshold is not None:
        plot_confusion_matrix(y_true, y_scores, fpr005_threshold, 
                             f'{model_name} - FPR ≤ 0.05 Threshold',
                             f'cm_{model_name.lower().replace(" ", "_")}_fpr_005.png')
    
    if fpr01_threshold is not None:
        plot_confusion_matrix(y_true, y_scores, fpr01_threshold, 
                             f'{model_name} - FPR ≤ 0.1 Threshold',
                             f'cm_{model_name.lower().replace(" ", "_")}_fpr_01.png')
    
    return {
        'model_name': model_name,
        'auc': model_auc,
        'best_auc_threshold': best_auc_threshold,
        'fpr005_threshold': fpr005_threshold,
        'fpr01_threshold': fpr01_threshold
    }

def main():
    """Main function to analyze both models."""
    print("LRLR Classification Threshold Optimization")
    print("Finding optimal thresholds for Conv1D and LSTM models")
    
    # Data file paths
    conv1d_path = "lrlr_test_results/results_Conv1D_filter_remove_artifacts_ensemble.csv"
    lstm_path = "lrlr_test_results/results_LSTM_final_model_ensemble.csv"
    
    # Analyze both models
    conv1d_results = analyze_model(conv1d_path, "Conv1D Model")
    lstm_results = analyze_model(lstm_path, "LSTM Model")
    
    # Summary comparison
    print(f"\n{'='*50}")
    print("SUMMARY COMPARISON")
    print(f"{'='*50}")
    print(f"{'Model':<15} {'AUC':<8} {'Best AUC':<12} {'FPR≤0.05':<12} {'FPR≤0.1':<12}")
    print(f"{'-'*60}")
    
    for results in [conv1d_results, lstm_results]:
        fpr005_str = f"{results['fpr005_threshold']:.4f}" if results['fpr005_threshold'] is not None else "N/A"
        fpr01_str = f"{results['fpr01_threshold']:.4f}" if results['fpr01_threshold'] is not None else "N/A"
        
        print(f"{results['model_name']:<15} {results['auc']:<8.4f} {results['best_auc_threshold']:<12.4f} {fpr005_str:<12} {fpr01_str:<12}")
    
    # Create threshold comparison plot
    print(f"\nGenerating threshold comparison plot...")
    create_threshold_summary_plot(conv1d_results, lstm_results)
    
    print(f"\nAll plots saved to current directory!")
    print(f"Generated files:")
    print(f"- roc_curve_conv1d_model.png")
    print(f"- roc_curve_lstm_model.png")
    print(f"- cm_conv1d_model_best_auc.png")
    print(f"- cm_lstm_model_best_auc.png")
    print(f"- cm_conv1d_model_fpr_005.png (if achievable)")
    print(f"- cm_lstm_model_fpr_005.png (if achievable)")
    print(f"- cm_conv1d_model_fpr_01.png (if achievable)")
    print(f"- cm_lstm_model_fpr_01.png (if achievable)")
    print(f"- threshold_comparison.png")

if __name__ == "__main__":
    main()

