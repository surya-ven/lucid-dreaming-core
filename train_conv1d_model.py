import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import os

def train_lrlr_conv1d(data_path='lstm_training_data.npz', model_save_path_base='lrlr_conv1d_model_fold_'):
    """
    Trains a 1D CNN model to detect LRLR events using Stratified K-Fold cross-validation.

    Args:
        data_path (str): Path to the .npz file containing training data (X, y).
        model_save_path_base (str): Base path to save the trained Keras models.
    """
    print(f"Loading data from {data_path}...")
    try:
        data = np.load(data_path)
        X = data['X']
        y_orig = data['y']
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please run lstm_data_extraction.py first.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if X.size == 0 or y_orig.size == 0:
        print("Error: Loaded data is empty. Cannot train model.")
        return

    # Clean NaNs and Infs from X data
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        num_corrupted_elements = np.sum(np.isnan(X)) + np.sum(np.isinf(X))
        total_elements = X.size
        percentage_corrupted = (num_corrupted_elements / total_elements) * 100
        print(f"Warning: Found {num_corrupted_elements} NaN/Inf values ({percentage_corrupted:.2f}%) in loaded X data. Cleaning them by replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Data loaded. X shape: {X.shape}, y shape: {y_orig.shape}")
    print(f"LRLR samples (1): {np.sum(y_orig == 1)}, Non-LRLR samples (0): {np.sum(y_orig == 0)}")

    # K-Fold Cross-Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_no = 1
    all_fold_val_loss = []
    all_fold_val_accuracy = []
    all_fold_val_precision = []
    all_fold_val_recall = []

    for train_index, val_index in skf.split(X, y_orig):
        print(f"\n--- Fold {fold_no}/{n_splits} ---")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y_orig[train_index], y_orig[val_index]

        # Reshape y for Keras
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        print(f"Training set: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"Validation set: X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"LRLR in Val (1): {np.sum(y_val == 1)}, Non-LRLR in Val (0): {np.sum(y_val == 0)}")

        # Calculate class weights to handle imbalance for the current fold
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train.flatten()), y=y_train.flatten())
        class_weights_dict = {i: weights[i] for i in range(len(weights))}
        print(f"Calculated class weights for fold {fold_no}: {class_weights_dict}")
        
        # More conservative manual class weights
        pos_samples = np.sum(y_train == 1)
        neg_samples = np.sum(y_train == 0)
        manual_class_weights = {
            0: 1.0,
            1: (neg_samples / pos_samples) * 0.5  # Reduced from 2.0 to 0.5
        }
        print(f"Manual class weights for fold {fold_no}: {manual_class_weights}")

        # Define the 1D CNN model with bias initialization
        model = Sequential([
            # First convolutional block
            Conv1D(filters=16, kernel_size=7, activation='relu',  # Increased filters and kernel size
                   input_shape=(X_train.shape[1], X_train.shape[2])),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),  # Reduced dropout
            
            # Second convolutional block
            Conv1D(filters=32, kernel_size=5, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # Third convolutional block
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')  # Remove bias initialization
        ])

        if fold_no == 1:  # Print summary only for the first fold
            model.summary()

        # Use focal loss to encourage more confident predictions
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.BinaryFocalCrossentropy(
                          apply_class_balancing=True,
                          alpha=0.25,  # Weight for positive class
                          gamma=2.0),  # Focus on hard examples
                      metrics=['accuracy', 
                               tf.keras.metrics.Precision(name='precision'), 
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc_pr', curve='PR')])

        # Callbacks - monitor AUC-PR which is better for imbalanced data
        callbacks = [
            EarlyStopping(
                monitor='val_auc_pr',
                patience=30,
                min_delta=1e-4,
                mode='max',
                restore_best_weights=True,
                verbose=1),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1)
        ]

        # Train the model with moderate class weights
        moderate_class_weights = {
            0: 1.0,
            1: (neg_samples / pos_samples) * 0.8  # Moderate boost
        }
        print(f"Using moderate class weights for fold {fold_no}: {moderate_class_weights}")
        
        print(f"\nStarting model training for fold {fold_no}...")
        history = model.fit(X_train, y_train,
                            epochs=100,
                            batch_size=16,  # Moderate batch size
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            class_weight=moderate_class_weights,
                            verbose=1)

        print(f"\nTraining finished for fold {fold_no}.")

        # Evaluate the best model for the current fold
        print(f"\nEvaluating model on validation set for fold {fold_no}:")
        eval_results = model.evaluate(X_val, y_val, verbose=0)
        loss = eval_results[0]
        accuracy = eval_results[1]
        precision = eval_results[2]
        recall = eval_results[3]

        print(f"  Fold {fold_no} Val Loss: {loss:.4f}")
        print(f"  Fold {fold_no} Val Accuracy: {accuracy:.4f}")
        print(f"  Fold {fold_no} Val Precision: {precision:.4f}")
        print(f"  Fold {fold_no} Val Recall: {recall:.4f}")

        # Add prediction analysis with different thresholds
        y_pred = model.predict(X_val, verbose=0)
        print(f"  Prediction stats - Min: {np.min(y_pred):.4f}, Max: {np.max(y_pred):.4f}, Mean: {np.mean(y_pred):.4f}")
        print(f"  Std: {np.std(y_pred):.4f}, Median: {np.median(y_pred):.4f}")
        
        # Check predictions at different thresholds
        from sklearn.metrics import precision_score, recall_score, f1_score
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        print("  Threshold analysis:")
        for threshold in thresholds:
            y_pred_thresh = (y_pred > threshold).astype(int)
            n_pred_pos = np.sum(y_pred_thresh)
            if n_pred_pos > 0:
                prec = precision_score(y_val, y_pred_thresh, zero_division=0)
                rec = recall_score(y_val, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
                print(f"    {threshold:.1f}: Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, Pred={n_pred_pos}")
            else:
                print(f"    {threshold:.1f}: No positive predictions")

        # Use best threshold based on F1 score for final metrics
        best_f1 = 0
        best_threshold = 0.5
        best_prec = 0
        best_rec = 0
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred > threshold).astype(int)
            if np.sum(y_pred_thresh) > 0:
                prec = precision_score(y_val, y_pred_thresh, zero_division=0)
                rec = recall_score(y_val, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_prec = prec
                    best_rec = rec
        
        print(f"  Best threshold: {best_threshold:.1f} (F1={best_f1:.3f}, Prec={best_prec:.3f}, Rec={best_rec:.3f})")
        
        # Store the best metrics instead of default threshold
        all_fold_val_loss.append(loss)
        all_fold_val_accuracy.append(accuracy)
        all_fold_val_precision.append(best_prec)  # Use best threshold precision
        all_fold_val_recall.append(best_rec)      # Use best threshold recall

        # Plot training & validation curves for each fold
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model Accuracy - Fold {fold_no}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss - Fold {fold_no}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plot_save_path = os.path.splitext(model_save_path_base)[0] + f'_fold_{fold_no}_training_curves_v2.png'
        plt.savefig(plot_save_path)
        print(f"Training curves plot for fold {fold_no} saved to {plot_save_path}")
        plt.close()

        fold_no += 1

    # # Calculate average optimal threshold across folds
    optimal_global_threshold = 0.5
    
    print(f"\n--- K-Fold Cross-Validation Results ---")
    print(f"Average Validation Loss: {np.mean(all_fold_val_loss):.4f} (+/- {np.std(all_fold_val_loss):.4f})")
    print(f"Average Validation Accuracy: {np.mean(all_fold_val_accuracy):.4f} (+/- {np.std(all_fold_val_accuracy):.4f})")
    print(f"Average Validation Precision: {np.mean(all_fold_val_precision):.4f} (+/- {np.std(all_fold_val_precision):.4f})")
    print(f"Average Validation Recall: {np.mean(all_fold_val_recall):.4f} (+/- {np.std(all_fold_val_recall):.4f})")
    print(f"Recommended threshold: {optimal_global_threshold}")

    # Train final model with same architecture
    print("\n--- Training final model on all data ---")
    
    y_full_train = y_orig.reshape(-1, 1)
    
    pos_samples_full = np.sum(y_orig == 1)
    neg_samples_full = np.sum(y_orig == 0)
    
    final_model = Sequential([
        Conv1D(filters=16, kernel_size=7, activation='relu', 
               input_shape=(X.shape[1], X.shape[2])),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Same compilation
    final_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    final_model.compile(optimizer=final_optimizer,
                        loss=tf.keras.losses.BinaryFocalCrossentropy(
                            apply_class_balancing=True,
                            alpha=0.25,
                            gamma=2.0),
                        metrics=['accuracy', 
                                 tf.keras.metrics.Precision(name='precision'), 
                                 tf.keras.metrics.Recall(name='recall'),
                                 tf.keras.metrics.AUC(name='auc_pr', curve='PR')])

    final_class_weights = {
        0: 1.0,
        1: (neg_samples_full / pos_samples_full) * 0.8
    }
    
    print("\nStarting training of the final model on all data...")
    final_model.fit(X, y_full_train,
                    epochs=33,
                    batch_size=16,
                    class_weight=final_class_weights,
                    callbacks=[],  # No callbacks - train for full epochs
                    verbose=1)

    # Save model and optimal threshold
    final_model_save_path = os.path.splitext(model_save_path_base)[0] + '_final_all_data_v2.keras'
    final_model.save(final_model_save_path)
    print(f"Final model saved to: {final_model_save_path}")
    
    # Save threshold info
    threshold_save_path = os.path.splitext(model_save_path_base)[0] + '_optimal_threshold_v2.npz'
    np.savez(threshold_save_path, optimal_threshold=optimal_global_threshold)
    print(f"Optimal threshold ({optimal_global_threshold}) saved to: {threshold_save_path}")

    # Evaluate final model with optimal threshold
    print("\nEvaluating final model on full training dataset:")
    y_pred_final = final_model.predict(X, verbose=0)
    y_pred_final_thresh = (y_pred_final > optimal_global_threshold).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    final_accuracy = accuracy_score(y_full_train, y_pred_final_thresh)
    final_precision = precision_score(y_full_train, y_pred_final_thresh, zero_division=0)
    final_recall = recall_score(y_full_train, y_pred_final_thresh, zero_division=0)
    final_f1 = f1_score(y_full_train, y_pred_final_thresh, zero_division=0)
    
    print(f"  With threshold {optimal_global_threshold}:")
    print(f"  Final Accuracy: {final_accuracy:.4f}")
    print(f"  Final Precision: {final_precision:.4f}")
    print(f"  Final Recall: {final_recall:.4f}")
    print(f"  Final F1 Score: {final_f1:.4f}")

if __name__ == '__main__':
    train_lrlr_conv1d()



