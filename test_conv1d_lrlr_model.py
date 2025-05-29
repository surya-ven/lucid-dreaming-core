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

def main():
    """Main function to test Conv1D model."""
    print("Testing Conv1D LRLR Model")
    print("=" * 50)
    
    # Load model
    model_path = 'lrlr_conv1d_model_fold__final_all_data_v2.keras'
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
    
    # Generate plots
    print(f"\nGenerating plots...")
    
    # Plot prediction distribution
    plot_prediction_distribution(y, y_scores, "Conv1D Model")
    
    # Plot ROC curve with threshold points
    thresholds_dict = {
        'best_auc': best_auc_threshold,
        'fpr005': fpr005_threshold,
        'fpr01': fpr01_threshold
    }
    plot_roc_with_thresholds(y, y_scores, thresholds_dict, "Conv1D Model")
    
    # Plot confusion matrices for each threshold
    plot_confusion_matrix(y, y_scores, best_auc_threshold, 
                         "Conv1D Model - Best AUC Threshold")
    
    if fpr005_threshold is not None:
        plot_confusion_matrix(y, y_scores, fpr005_threshold, 
                             "Conv1D Model - FPR ≤ 0.05 Threshold")
    
    if fpr01_threshold is not None:
        plot_confusion_matrix(y, y_scores, fpr01_threshold, 
                             "Conv1D Model - FPR ≤ 0.1 Threshold")
    
    # Plot threshold analysis
    analyze_threshold_performance(y, y_scores)
    
    # Print detailed classification report for best threshold
    print(f"\nDetailed Classification Report (Best AUC Threshold = {best_auc_threshold:.4f}):")
    y_pred_best = (y_scores >= best_auc_threshold).astype(int)
    print(classification_report(y, y_pred_best, target_names=['Non-LRLR', 'LRLR']))
    
    print(f"\nTesting completed!")

if __name__ == "__main__":
    main()
