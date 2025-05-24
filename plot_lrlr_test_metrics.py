import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

def plot_roc_curve(y_true, y_scores, file_path_roc):
    """
    Plots and saves an ROC curve.

    Args:
        y_true (pd.Series): True binary labels.
        y_scores (pd.Series): Target scores, can either be probability estimates of the positive class,
                              confidence values, or non-thresholded measure of decisions.
        file_path_roc (str): Path to save the ROC curve plot.
    """
    # Remove NaN and infinite values from both arrays
    valid_mask = ~(pd.isna(y_true) | pd.isna(y_scores) | np.isinf(y_true) | np.isinf(y_scores))
    y_true_clean = y_true[valid_mask]
    y_scores_clean = y_scores[valid_mask]
    
    if len(y_true_clean) == 0 or len(np.unique(y_true_clean)) < 2 or len(np.unique(y_scores_clean)) < 2:
        print(f"Skipping ROC curve for {os.path.basename(file_path_roc)} due to invalid/empty scores or insufficient class diversity.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'ROC curve cannot be plotted\n(Invalid scores, insufficient data, or single class)',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=10, color='red')
        ax.set_title(f"ROC Curve - {os.path.basename(file_path_roc).replace('_roc.png', '')}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(file_path_roc)
        plt.close(fig)
        return

    fpr, tpr, _ = roc_curve(y_true_clean, y_scores_clean)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {os.path.basename(file_path_roc).replace("_roc.png", "")}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path_roc)
    plt.close()
    print(f"Saved ROC curve to {file_path_roc}")

def plot_confusion_matrix_with_metrics(y_true, y_pred, file_path_cm, class_names=None):
    """
    Plots and saves a confusion matrix with key classification metrics.

    Args:
        y_true (pd.Series): True binary labels.
        y_pred (pd.Series): Predicted binary labels.
        file_path_cm (str): Path to save the confusion matrix plot.
        class_names (list, optional): Names of the classes. Defaults to ['Non-LRLR', 'LRLR'].
    """
    if class_names is None:
        class_names = ['Non-LRLR', 'LRLR']
    
    # Remove NaN and infinite values from both arrays
    valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0 or len(y_pred_clean) == 0:
        print(f"Skipping Confusion Matrix for {os.path.basename(file_path_cm)} due to empty or invalid true/predicted labels after cleaning.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Confusion Matrix cannot be plotted\n(No valid true or predicted labels)',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=10, color='red')
        ax.set_title(f"Confusion Matrix - {os.path.basename(file_path_cm).replace('_cm.png', '')}")
        plt.tight_layout()
        plt.savefig(file_path_cm)
        plt.close(fig)
        return

    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=[0, 1]) # Ensure labels are 0 and 1 for consistency

    accuracy = accuracy_score(y_true_clean, y_pred_clean)
    precision = precision_score(y_true_clean, y_pred_clean, zero_division=0, labels=[0,1], pos_label=1)
    recall = recall_score(y_true_clean, y_pred_clean, zero_division=0, labels=[0,1], pos_label=1)
    f1 = f1_score(y_true_clean, y_pred_clean, zero_division=0, labels=[0,1], pos_label=1)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 7)) # Adjusted figure size
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    
    metrics_text = (f"Accuracy: {accuracy:.3f}\n"
                    f"Precision (LRLR): {precision:.3f}\n"
                    f"Recall (LRLR): {recall:.3f}\n"
                    f"F1-score (LRLR): {f1:.3f}")
    
    # Place text below the plot
    plt.text(0.5, -0.2, metrics_text, ha='center', va='bottom', 
             transform=ax.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8))
    
    plt.title(f'Confusion Matrix - {os.path.basename(file_path_cm).replace("_cm.png", "")}')
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for text
    plt.savefig(file_path_cm)
    plt.close(fig)
    print(f"Saved Confusion Matrix to {file_path_cm}")

if __name__ == "__main__":
    # Example usage:
    # This part is for standalone testing of the plotting functions.
    # It assumes you have a CSV file in 'lrlr_test_results' to test with.
    print("Running plot_lrlr_test_metrics.py as standalone script.")
    test_results_dir = "lrlr_test_results_old"
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
        print(f"Created directory: {test_results_dir}")
        # Create a dummy CSV for testing if it doesn't exist
        dummy_data = {
            'dataset_name': ['test_data'] * 10,
            'window_start_sample': range(0, 100, 10),
            'window_end_sample': range(10, 110, 10),
            'true_label': [0,1,0,1,0,1,0,0,1,1],
            'predicted_label': [0,0,1,1,0,1,1,0,1,0],
            'prediction_score': [0.1, 0.8, 0.4, 0.9, 0.2, 0.7, 0.6, 0.3, 0.95, 0.4]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_csv_path = os.path.join(test_results_dir, "dummy_results.csv")
        if not os.path.exists(dummy_csv_path):
            dummy_df.to_csv(dummy_csv_path, index=False)
            print(f"Created dummy CSV: {dummy_csv_path}")


    for filename in os.listdir(test_results_dir):
        if filename.endswith(".csv"):
            csv_path = os.path.join(test_results_dir, filename)
            print(f"\nProcessing {filename} for standalone test...")
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    print(f"Skipping {filename}: CSV is empty.")
                    continue
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue

            if not all(col in df.columns for col in ['true_label', 'predicted_label', 'prediction_score']):
                print(f"Skipping {filename}: missing required columns.")
                continue
            
            y_true = df['true_label'].astype(int)
            y_pred = df['predicted_label'].astype(int)
            y_scores = df['prediction_score']

            valid_indices = ~y_scores.isnull()
            y_true_roc = y_true[valid_indices]
            y_scores_roc = y_scores[valid_indices]
            
            base_name = os.path.splitext(filename)[0]
            
            roc_file_path = os.path.join(test_results_dir, f"{base_name}_roc_standalone.png")
            plot_roc_curve(y_true_roc, y_scores_roc, roc_file_path)
            
            cm_file_path = os.path.join(test_results_dir, f"{base_name}_cm_standalone.png")
            plot_confusion_matrix_with_metrics(y_true, y_pred, cm_file_path)
    print("Standalone plotting test finished.")
