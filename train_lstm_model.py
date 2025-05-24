import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # ModelCheckpoint removed for k-fold, can be added back strategically
from sklearn.model_selection import StratifiedKFold # Changed from train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import os

def train_lrlr_lstm(data_path='lstm_training_data.npz', model_save_path_base='lrlr_lstm_model_fold_'):
    """
    Trains an LSTM model to detect LRLR events using Stratified K-Fold cross-validation.

    Args:
        data_path (str): Path to the .npz file containing training data (X, y).
        model_save_path_base (str): Base path to save the trained Keras models (e.g., 'lrlr_lstm_model_fold_').
                                     Each fold's model will be saved as model_save_path_base + k + .keras.
    """
    print(f"Loading data from {data_path}...")
    try:
        data = np.load(data_path)
        X = data['X']
        y_orig = data['y'] # Keep original y for k-fold splitting
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
    n_splits = 5 # You can adjust the number of folds
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
        class_weights_dict = {i : weights[i] for i in range(len(weights))}
        print(f"Calculated class weights for fold {fold_no}: {class_weights_dict}")

        # Define the LSTM model (re-initialize for each fold)
        model = Sequential([
            Bidirectional(LSTM(
                64,
                input_shape=(X_train.shape[1], X_train.shape[2]),
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.2,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )),
            LayerNormalization(),
            Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        if fold_no == 1: # Print summary only for the first fold
            model.summary()

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4) # Consider making this tunable
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.25, gamma=2.0, from_logits=False), # UPDATED LOSS
                      metrics=['accuracy', 
                               tf.keras.metrics.Precision(name='precision'), 
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc_pr', curve='PR')])

        # Callbacks
        # ModelCheckpoint can be added here if you want to save each fold's best model
        # checkpoint = ModelCheckpoint(model_save_path_base + str(fold_no) + '.keras', monitor='val_recall', save_best_only=True, mode='max', verbose=1)
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                min_delta=1e-3,
                mode='min',
                restore_best_weights=True,
                verbose=1),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-5,
                verbose=1)
        ]

        # Train the model
        print(f"\nStarting model training for fold {fold_no}...")
        history = model.fit(X_train, y_train,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks, # Add checkpoint here if using
                            class_weight=class_weights_dict,
                            verbose=1)

        print(f"\nTraining finished for fold {fold_no}.")

        # Evaluate the best model (restored by EarlyStopping) for the current fold
        print(f"\nEvaluating model on validation set for fold {fold_no}:")
        # loss, accuracy, precision, recall = model.evaluate(X_val, y_val, verbose=0) # Old evaluation
        eval_results = model.evaluate(X_val, y_val, verbose=0)
        loss = eval_results[0]
        accuracy = eval_results[1]
        precision = eval_results[2]
        recall = eval_results[3]
        # auc_pr = eval_results[4] # If you want to store/print AUC_PR per fold

        print(f"  Fold {fold_no} Val Loss: {loss:.4f}")
        print(f"  Fold {fold_no} Val Accuracy: {accuracy:.4f}")
        print(f"  Fold {fold_no} Val Precision: {precision:.4f}")
        print(f"  Fold {fold_no} Val Recall: {recall:.4f}")
        # print(f"  Fold {fold_no} Val AUC_PR: {auc_pr:.4f}")


        all_fold_val_loss.append(loss)
        all_fold_val_accuracy.append(accuracy)
        all_fold_val_precision.append(precision)
        all_fold_val_recall.append(recall)

        # Plot training & validation curves for each fold (optional)
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
        plot_save_path = os.path.splitext(model_save_path_base)[0] + f'_fold_{fold_no}_training_curves.png'
        plt.savefig(plot_save_path)
        print(f"Training curves plot for fold {fold_no} saved to {plot_save_path}")
        plt.close() # Close the plot to avoid displaying multiple plots if running interactively

        fold_no += 1

    print("\n--- K-Fold Cross-Validation Results ---")
    print(f"Average Validation Loss: {np.mean(all_fold_val_loss):.4f} (+/- {np.std(all_fold_val_loss):.4f})")
    print(f"Average Validation Accuracy: {np.mean(all_fold_val_accuracy):.4f} (+/- {np.std(all_fold_val_accuracy):.4f})")
    print(f"Average Validation Precision: {np.mean(all_fold_val_precision):.4f} (+/- {np.std(all_fold_val_precision):.4f})")
    print(f"Average Validation Recall: {np.mean(all_fold_val_recall):.4f} (+/- {np.std(all_fold_val_recall):.4f})")


    # --- Train a final model on all data ---
    print("\n--- Training final model on all data ---")
    
    # Prepare full dataset
    y_full_train = y_orig.reshape(-1, 1)
    
    # Calculate class weights for the full dataset
    full_train_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_orig.flatten()),
        y=y_orig.flatten()
    )
    full_class_weights_dict = {i: full_train_weights[i] for i in range(len(full_train_weights))}
    print(f"Calculated class weights for full dataset: {full_class_weights_dict}")

    # Define the LSTM model (using the same architecture as in K-Fold)
    final_model = Sequential([
        Bidirectional(LSTM(
            64,
            input_shape=(X.shape[1], X.shape[2]),
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )),
        LayerNormalization(),
        Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compile the final model
    final_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4) # Use the same LR or one found optimal
    final_model.compile(optimizer=final_optimizer,
                        loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.25, gamma=2.0, from_logits=False), # UPDATED LOSS
                        metrics=['accuracy', 
                                 tf.keras.metrics.Precision(name='precision'), 
                                 tf.keras.metrics.Recall(name='recall'),
                                 tf.keras.metrics.AUC(name='auc_pr', curve='PR')])
    
    print("\nFinal model summary:")
    final_model.summary()

    # Callbacks for the final model (monitoring 'loss' as no validation data)
    final_model_callbacks = [
        EarlyStopping(
            monitor='loss', # Monitor training loss
            patience=5,     # Increased patience slightly as it's training loss
            min_delta=1e-3, # Or adjust as needed
            mode='min',
            restore_best_weights=True, # Restores based on min training loss
            verbose=1),
        ReduceLROnPlateau(
            monitor='loss', # Monitor training loss
            factor=0.5,
            patience=3,    # Patience for reducing LR based on training loss
            min_lr=1e-5,
            verbose=1)
    ]

    # Train the final model on all data
    print("\nStarting training of the final model on all data for 30 epochs...")
    final_model.fit(X, y_full_train,
                    epochs=30, # Or adjust as needed, EarlyStopping might stop it sooner
                    batch_size=32,
                    class_weight=full_class_weights_dict,
                    callbacks=final_model_callbacks, # Added callbacks
                    verbose=1) # No validation data here as we are using all data for training

    print("\nTraining of final model finished.")

    # Save the final model
    final_model_save_path = os.path.splitext(model_save_path_base)[0] + '_final_all_data.keras'
    final_model.save(final_model_save_path)
    print(f"Final model trained on all data saved to: {final_model_save_path}")

    # Optionally, evaluate on the (entire) training data - this is not a true validation
    print("\nEvaluating final model on the full training dataset (not a true validation):")
    # loss, accuracy, precision, recall = final_model.evaluate(X, y_full_train, verbose=0) # Old evaluation
    final_eval_results = final_model.evaluate(X, y_full_train, verbose=0)
    loss = final_eval_results[0]
    accuracy = final_eval_results[1]
    precision = final_eval_results[2]
    recall = final_eval_results[3]
    auc_pr = final_eval_results[4]

    print(f"  Full Training Set Loss: {loss:.4f}")
    print(f"  Full Training Set Accuracy: {accuracy:.4f}")
    print(f"  Full Training Set Precision: {precision:.4f}")
    print(f"  Full Training Set Recall: {recall:.4f}")
    print(f"  Full Training Set AUC_PR: {auc_pr:.4f}")

    # Note: Saving a single "best" model from k-fold requires a strategy.
    # For example, you could train a final model on all data using the best epoch count found,
    # or select the model from the fold with the best validation recall/precision.
    # For now, this script evaluates and reports average performance.

if __name__ == '__main__':
    train_lrlr_lstm()