import time
import os

import numpy as np
import tensorflow as tf
from scipy.signal import butter, filtfilt, medfilt


rec_data_names = [
    "v2_LRLR_once_3_mix", ## NOTE I included mix here even though it is eyes open because it is not trained on
    "v2_LRLR_once_4_mix",
    "v2_LRLR_once_5_mix",
    "v2_LRLR_once_6_closed",
    "v2_LRLR_once_7_closed",
    "v2_LRLR_once_8_closed",
    "20250523_183954_966852", #LRLR ? 
    "20250523_192852_995272", #LRLR x1
    "20250523_193034_228556", #LRLR x4
    "20250523_193526_401634", #LRLR x4
    "20250523_194915_602917", #LRLR x4
    "20250523_200210_295876", #LRLR x4
    "20250524_015512_029025", #LRLR x9
    "20250524_020422_208890", #LRLR x12
    "20250524_022100_630075", #LRLR x12
    "20250524_033027_138563", #LRLR x9
    "20250524_033637_296315", #LRLR x7
]


BEST_MODEL_PATH = 'models/lrlr_conv1d_model_fold__final_all_data.keras'


MODEL_THRESHOLD_BEST = 0.4039 # Threshold for best model to classify LRLR as True
MODEL_THRESHOLD_90 = 0.5575 # Threshold for best model to classify LRLR as True
MODEL_THRESHOLD_95 = 0.5894 # Threshold for best model to classify LRLR as True


MODEL_SAMPLE_LENGTH = 750

LRLR_MODEL = None # Initialize the model variable globally





def load_custom_data(session_folder_path):
    """
    Loads the combined data and metadata from a custom recording session.

    Args:
        session_folder_path (str): Path to the session folder.

    Returns:
        tuple: (data_array, metadata_dict)
               - data_array (np.ndarray): The loaded data. Timestamps and TargetEvent columns might be prepended.
               - metadata_dict (dict): The loaded metadata, including session_info and potentially processed_column_names.
               Returns (None, None) if loading fails at an early stage.
    """
    data_filename = "custom_combined_data.dat"
    metadata_filename = "custom_metadata.npz"

    data_filepath = os.path.join(session_folder_path, data_filename)
    metadata_filepath = os.path.join(session_folder_path, metadata_filename)

    if not os.path.exists(metadata_filepath):
        print(f"Error: Metadata file not found: {metadata_filepath}")
        return None, None
    
    session_info_loaded = None
    metadata_loaded = None
    try:
        metadata_loaded = np.load(metadata_filepath, allow_pickle=True)
        session_info_loaded = metadata_loaded['session_info'].item()
    except Exception as e_meta:
        print(f"Error loading metadata from {metadata_filepath}: {e_meta}")
        return None, None # Cannot proceed without metadata

    if not os.path.exists(data_filepath):
        print(f"Error: Data file not found: {data_filepath}")
        return None, session_info_loaded 

    try:
        num_channels = session_info_loaded['expected_columns'] 
        data_type = np.dtype(session_info_loaded['custom_data_type'])
        data_shape_on_save = session_info_loaded.get('data_shape_on_save', 'samples_first') 
        original_column_names = list(session_info_loaded.get('column_names', [f'Ch{j+1}' for j in range(num_channels)]))

        loaded_flat_data = np.fromfile(data_filepath, dtype=data_type)

        if loaded_flat_data.size == 0:
            print("Warning: Data file is empty.")
            # Return empty array matching expected original channels, plus session_info
            empty_data = np.array([]).reshape(0, num_channels) 
            session_info_loaded['processed_column_names'] = original_column_names
            return empty_data, session_info_loaded

        processed_data = None
        final_column_names = []

        if data_shape_on_save == 'channels_first':
            if loaded_flat_data.size % num_channels == 0:
                num_samples_loaded_total = loaded_flat_data.size // num_channels
                reshaped_data_channels_first = loaded_flat_data.reshape(num_channels, num_samples_loaded_total)
                current_data_array = reshaped_data_channels_first.T  
                final_column_names = list(original_column_names) # Start with original channel names

                # Attempt to prepend timestamps and event data
                block_ts = metadata_loaded.get('data_block_timestamps', None)
                block_counts = metadata_loaded.get('data_block_sample_counts', None)
                target_event_transitions = metadata_loaded.get('target_event_transitions', None)

                if block_ts is not None and block_counts is not None and len(block_ts) > 0:
                    if sum(block_counts) == current_data_array.shape[0]: # Validate counts match data length
                        sample_timestamps = np.concatenate([np.full(int(cnt), float(ts)) for ts, cnt in zip(block_ts, block_counts)])
                        current_data_array = np.column_stack((sample_timestamps, current_data_array))
                        final_column_names.insert(0, "Timestamp")

                        # If timestamps were added, try to add event data
                        if target_event_transitions is not None and len(target_event_transitions) > 0:
                            target_event_values = np.full(len(sample_timestamps), False, dtype=bool)
                            current_event_state = False 
                            transition_idx = 0
                            for i in range(len(sample_timestamps)):
                                sample_ts_val = sample_timestamps[i]
                                while transition_idx < len(target_event_transitions) and \
                                      target_event_transitions[transition_idx][0] <= sample_ts_val:
                                    current_event_state = target_event_transitions[transition_idx][1]
                                    transition_idx += 1
                                target_event_values[i] = current_event_state
                            
                            # Insert event data after timestamp column
                            current_data_array = np.column_stack((current_data_array[:,0], target_event_values, current_data_array[:,1:]))
                            final_column_names.insert(1, "TargetEvent")
                        else:
                            print("Note: No target event transitions found in metadata or transitions array is empty.")
                    else:
                        print("Warning: Sum of block_counts does not match data length. Timestamps/Events not prepended.")
                else:
                    print("Note: data_block_timestamps or data_block_sample_counts not found or empty in metadata. Timestamps/Events not prepended.")
                
                processed_data = current_data_array
            else:
                print(f"Error: Cannot reshape data saved as 'channels_first'. Total elements ({loaded_flat_data.size}) not divisible by num_channels ({num_channels}).")
                return None, session_info_loaded
        
        else: # Assuming 'samples_first' or old format
            if loaded_flat_data.size % num_channels == 0: 
                num_samples_loaded = loaded_flat_data.size // num_channels
                processed_data = loaded_flat_data.reshape(num_samples_loaded, num_channels)
                final_column_names = list(original_column_names)
            else:
                print(f"Error: Cannot reshape data saved as 'samples_first'. Total elements ({loaded_flat_data.size}) not divisible by num_columns ({num_channels}).")
                return None, session_info_loaded

        session_info_loaded['processed_column_names'] = final_column_names
        return processed_data, session_info_loaded

    except Exception as e:
        print(f"Error processing data from {data_filepath} or applying metadata: {e}")
        import traceback
        traceback.print_exc()
        return None, session_info_loaded


# # --- Display loaded data and metadata ---
def display_loaded_data_and_metadata(loaded_data, session_metadata):
    if session_metadata is not None: 
        print("\nSession Information (from metadata):")
        for key, value in session_metadata.items():
            if key != 'processed_column_names': # Don't print this internal-use key here
                print(f"  {key}: {value}")

        if loaded_data is not None:
            print("\nSuccessfully loaded data.")
            display_column_names = session_metadata.get('processed_column_names', 
                                                    [f'Col{i+1}' for i in range(loaded_data.shape[1])])
            print(f"Data shape (samples, columns): {loaded_data.shape}")
            print(f"Columns: {display_column_names}")
            
            if loaded_data.shape[0] > 0: 
                print("\nFirst 5 rows of loaded data:")
                header = " | ".join(display_column_names)
                print(header)
                print("-" * len(header))
                for row in loaded_data[:5, :]:
                    # Format each element in the row for display
                    formatted_row = []
                    for i, item in enumerate(row):
                        col_name = display_column_names[i] if i < len(display_column_names) else ""
                        if col_name == "Timestamp":
                            formatted_row.append(f"{item:.2f}") # Timestamp with 2 decimal places
                        elif isinstance(item, bool) or col_name == "TargetEvent":
                            formatted_row.append(str(item))    # Boolean as True/False
                        elif isinstance(item, float) or isinstance(item, np.floating):
                            formatted_row.append(f"{item:.3f}" if not np.isnan(item) else "NaN") # Floats with 3 decimal places
                        else:
                            formatted_row.append(str(item))
                    print(" | ".join(formatted_row))
            else:
                print("\nData loaded, but no samples to display (data shape is 0 rows).")
        else: 
            print(f"\nFailed to load data array from SESSION_FOLDER_PATH, but metadata was available.")
            print("Please check data file integrity and error messages above.")
    else: 
        print(f"\nFailed to load any data or metadata from SESSION_FOLDER_PATH.")




def filter_signal_data(data, srate, mft=5, lowcut=0.5, highcut=15, artifact_threshold=125, apply_ica=True):
    # Make a copy to avoid modifying the original data
    data = np.copy(data)
    n_channels = data.shape[1]
    print(f"Filtering data with {n_channels} channels")
    
    # 1. Artifact removal/rejection with thresholding
    for channel in range(data.shape[1]):
        # Find artifact indices where signal exceeds threshold
        artifacts = np.abs(data[:, channel]) > artifact_threshold
        
        if np.any(artifacts):
            # Get indices of artifacts
            artifact_indices = np.where(artifacts)[0]
            
            # Process each continuous segment of artifacts
            segments = np.split(artifact_indices, np.where(np.diff(artifact_indices) > 1)[0] + 1)
            
            for segment in segments:
                if len(segment) > 0:
                    # Get start and end indices of segment
                    start_idx = segment[0]
                    end_idx = segment[-1]
                    
                    # Get values before and after the artifact segment for interpolation
                    # Handle edge cases where artifact is at beginning or end
                    if start_idx == 0:
                        # Artifact at beginning, use first non-artifact value
                        non_artifact_idx = np.where(~artifacts)[0]
                        if len(non_artifact_idx) > 0:
                            pre_value = data[non_artifact_idx[0], channel]
                        else:
                            pre_value = 0  # All values are artifacts; use 0
                    else:
                        pre_value = data[start_idx-1, channel]
                        
                    if end_idx == len(data) - 1:
                        # Artifact at end, use last non-artifact value
                        non_artifact_idx = np.where(~artifacts)[0]
                        if len(non_artifact_idx) > 0:
                            post_value = data[non_artifact_idx[-1], channel]
                        else:
                            post_value = 0  # All values are artifacts; use 0
                    else:
                        post_value = data[end_idx+1, channel]
                    
                    # Linear interpolation across the artifact segment
                    segment_length = len(segment)
                    for i, idx in enumerate(segment):
                        weight = i / segment_length
                        data[idx, channel] = pre_value * (1 - weight) + post_value * weight

    # 3. Median filter to kill isolated spikes
    data = medfilt(data, kernel_size=(mft,1))

    # 4. Zero‑phase band‑pass (0.5–15 Hz)
    b, a = butter(4, [lowcut, highcut], btype='band', fs=srate)
    data = filtfilt(b, a, data, axis=0)
    
    return data



# Helper function for preprocessing, adapted from lstm_data_extraction.py
def _preprocess_input_for_model(data_segment_raw, model_srate_for_filter=118, mft=5, lowcut=0.5, highcut=15, artifact_threshold=125):
    """
    Preprocesses a 2-channel data segment (LSTM_SAMPLE_LENGTH, 2) exactly as done for LSTM training.
    model_srate_for_filter is the sampling rate used for designing the butterworth filter during training.
    """
    data = np.copy(data_segment_raw).astype(np.float32) 
    
    try:
        data = filter_signal_data(data, srate=model_srate_for_filter, mft=mft, lowcut=lowcut, highcut=highcut, artifact_threshold=artifact_threshold)
    except ValueError as e:
        print(f"LSTM Preprocessing Error: Bandpass filtering failed: {e}. Check highcut vs model_srate_for_filter/2.")
        return np.zeros_like(data_segment_raw) # Return zeros or raise error

    # Normalization (per-channel z-score)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Add epsilon to std to prevent division by zero if a channel is flat after filtering
    normalized_data = (data - mean) / (std + 1e-8)

    return normalized_data



def detect_lrlr_window_FINAL_MODEL(eog_data, srate, model_path=BEST_MODEL_PATH, detection_threshold=MODEL_THRESHOLD_BEST):
    """
    Detects LRLR patterns using the trained model.

    Args:
        eog_data (np.ndarray): EOG data array, expected to have at least 2 channels.
                               The first two channels will be used.
        srate (float): Sampling rate of the input eog_data. (Currently used for context,
                       filter design uses a fixed srate from training).
        model_path (str): Path to the Keras model file.
        detection_threshold (float): Threshold for classifying a prediction as LRLR.

    Returns:
        bool: True if LRLR is detected, False otherwise.
    """
    global LRLR_MODEL
    if LRLR_MODEL is None:
        try:
            LRLR_MODEL = tf.keras.models.load_model(model_path)
            print(f"model '{model_path}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model from '{model_path}': {e}")
            return False, 0.0 # Cannot proceed without the model

    if eog_data.shape[0] < MODEL_SAMPLE_LENGTH:
        print(f"Warning: EOG data too short for ({eog_data.shape[0]} samples, need {MODEL_SAMPLE_LENGTH}).")
        return False, 0.0
    
    if eog_data.shape[1] < 2:
        print(f"Warning: EOG data has fewer than 2 channels ({eog_data.shape[1]}). requires 2.")
        return False, 0.0

    # Extract the last LSTM_SAMPLE_LENGTH samples from the first two channels
    # Assumes eog_data's first two columns are the ones the Model was trained on.
    eog_segment_raw = eog_data[-MODEL_SAMPLE_LENGTH:, 0:2]

    # Preprocess the segment
    # model_srate_for_filter=118 is used internally as that's what the model was trained with.
    preprocessed_segment = _preprocess_input_for_model(eog_segment_raw, model_srate_for_filter=118)

    # Reshape for model prediction: (1, timesteps, features)
    model_input = np.expand_dims(preprocessed_segment, axis=0)

    # Predict
    try:
        # print(f"Model input shape: {model_input.shape}")
        prediction_value = LRLR_MODEL.predict(model_input, verbose=0)[0][0]
        is_lrlr = prediction_value > detection_threshold
        # print(f"Model Prediction: {prediction_value:.4f} -> Detected: {is_lrlr}") # For debugging
        return is_lrlr, prediction_value
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return False, 0.0 # Return a default value for the prediction if an error occurs




    
def test1():
    # <<< PLEASE UPDATE THIS PATH >>> ## TO COPY LRLR_1_time_1 ALERTNESS_3minmark_1
    for filepath in rec_data_names:

        # Load Data
        SESSION_FOLDER_PATH = f"recorded_data/{filepath}"
        loaded_data, session_metadata = load_custom_data(SESSION_FOLDER_PATH)
        if loaded_data is None or session_metadata is None:
            print("Failed to load data or metadata. Exiting.")
            return

        ## Display loaded data
        eog_data = loaded_data[:,-4:]
        # display_loaded_data_and_metadata(loaded_data, session_metadata)


        # Detect LRLR
        start_time = time.time()
        pred_lrlr = detect_lrlr_window_FINAL_MODEL(eog_data, srate=118, model_path=BEST_MODEL_PATH, detection_threshold=MODEL_THRESHOLD_90)
        end_time = time.time()

        # Print results
        execution_time = end_time - start_time
        print(f"LRLR detected: {pred_lrlr[0]}, Score: {pred_lrlr[1]:.4f}")
        print(f"Detection time: {execution_time:.4f} seconds")
        target_event = loaded_data[:, 1]

        # Print true LRLR event metrics
        last_750 = target_event[-750:]
        print(f"Average value of last 750 target events: {np.mean(last_750):.4f}")
        print(f"Max value of last 750 target events: {np.max(last_750):.4f}")
        print('\n')

if __name__ == "__main__":
    test1()