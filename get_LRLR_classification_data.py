# Author: Benjamin Grayzel

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import butter, filtfilt, medfilt, find_peaks, welch
from sklearn.decomposition import FastICA
import tensorflow as tf # Add this import

import time
import csv
import traceback


LRLR_LSTM_MODEL = None
DEFAULT_LSTM_MODEL_PATH = 'lrlr_lstm_model.keras'
LSTM_SAMPLE_LENGTH = 750
DEFAULT_SRATE = 125.0     # recordings were all made at 125 Hz


rec_data_names = [
    "v2_LRLR_once_1_mix",
    "v2_LRLR_once_2_mix",
    "v2_LRLR_once_3_mix",
    "v2_LRLR_once_4_mix",
    "v2_LRLR_once_5_mix",
    "v2_LRLR_once_6_closed",
    "v2_LRLR_once_7_closed",
    "v2_LRLR_once_8_closed",
    "v2_LRLR_once_9_closed",
    "v2_LRLR_once_10_closed",
    "v2_LRLR_once_11_mix_rapid",
    "v2_LRLR_once_12_closed_rapid",
    "v2_LRLR_once_13_mix_rapid",
    "v2_LRLR_once_14_mix_rapid",
    "v2_LRLR_once_15_mix_rapid",
]

ylim = 500


THRESH_WINDOW_SEC = 3.0          # 3‑second windows work well for LRLR
THRESH_STEP_FRACTION = 0.50      # 50 % overlap

def _get_srate(meta: dict):
    """Return sampling‑rate, trying several possible field‑names."""
    for k in ('srate', 'sample_rate', 'sampling_rate', 's_rate', 'fs'):
        if k in meta:
            return float(meta[k])
    return DEFAULT_SRATE


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
        print(f"[load_custom_data] {os.path.basename(session_folder_path)} » metadata missing")
        return None, None 
    
    session_info_loaded = None
    metadata_loaded = None
    try:
        metadata_loaded = np.load(metadata_filepath, allow_pickle=True)
        session_info_loaded = metadata_loaded['session_info'].item()
    except Exception as e_meta:
        print(f"Error loading metadata from {metadata_filepath}: {e_meta}")
        print(f"[load_custom_data] {os.path.basename(session_folder_path)} » metadata missing")
        return None, None 

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
            empty_data = np.array([]).reshape(0, num_channels) 
            session_info_loaded['processed_column_names'] = original_column_names
            return empty_data, session_info_loaded

        processed_data = None
        final_column_names = list(original_column_names) 

        if data_shape_on_save == 'channels_first':
            if loaded_flat_data.size % num_channels == 0:
                num_samples_loaded_total = loaded_flat_data.size // num_channels
                reshaped_data_channels_first = loaded_flat_data.reshape(num_channels, num_samples_loaded_total)
                current_data_array = reshaped_data_channels_first.T  
                
                block_ts = metadata_loaded.get('data_block_timestamps', None)
                block_counts = metadata_loaded.get('data_block_sample_counts', None)
                target_event_transitions = metadata_loaded.get('target_event_transitions', None)

                if block_ts is not None and block_counts is not None and len(block_ts) > 0 and len(block_ts) == len(block_counts):
                    if sum(block_counts) == current_data_array.shape[0]:
                        sample_timestamps = np.concatenate([np.full(int(cnt), float(ts)) for ts, cnt in zip(block_ts, block_counts)])
                        current_data_array = np.column_stack((sample_timestamps, current_data_array))
                        final_column_names.insert(0, "Timestamp")

                        if target_event_transitions is not None and len(target_event_transitions) > 0:
                            target_event_values = np.full(len(sample_timestamps), False, dtype=bool)
                            current_event_state = False 
                            transition_idx = 0
                            for i_ts in range(len(sample_timestamps)):
                                sample_ts_val = sample_timestamps[i_ts]
                                while transition_idx < len(target_event_transitions) and \
                                      target_event_transitions[transition_idx][0] <= sample_ts_val:
                                    current_event_state = target_event_transitions[transition_idx][1]
                                    transition_idx += 1
                                target_event_values[i_ts] = current_event_state
                            
                            current_data_array = np.column_stack((current_data_array[:,0], target_event_values, current_data_array[:,1:]))
                            final_column_names.insert(1, "TargetEvent")
                processed_data = current_data_array
            else:
                return None, session_info_loaded
        
        else: 
            if loaded_flat_data.size % num_channels == 0: 
                num_samples_loaded = loaded_flat_data.size // num_channels
                processed_data = loaded_flat_data.reshape(num_samples_loaded, num_channels)
            else:
                return None, session_info_loaded

        session_info_loaded['processed_column_names'] = final_column_names
        return processed_data, session_info_loaded

    except Exception as e:
        print(f"Error processing data from {data_filepath} or applying metadata: {e}")
        # traceback.print_exc()
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

    # # 2. Apply ICA for artifact removal if requested and we have enough channels
    # if apply_ica and n_channels >= 2:
    #     try:            
    #         # Initial filtering to improve ICA performance
    #         # Apply a lenient bandpass filter before ICA
    #         b, a = butter(4, [0.1, 20], btype='band', fs=srate)
    #         data_pre_ica = filtfilt(b, a, data, axis=0)
            
    #         # Apply ICA
    #         ica = FastICA(n_components=n_channels, random_state=42)
    #         components = ica.fit_transform(data_pre_ica)
            
    #         # Identify artifact components automatically
    #         # Method 1: Components with unusually high kurtosis (peaky/spikey components)
    #         from scipy.stats import kurtosis
    #         kurt_values = [kurtosis(components[:, i]) for i in range(n_channels)]
            
    #         # Components with kurtosis higher than 2 standard deviations from mean
    #         # are likely artifact components
    #         kurt_mean = np.mean(kurt_values)
    #         kurt_std = np.std(kurt_values)
    #         artifact_components = [i for i, k in enumerate(kurt_values) 
    #                              if k > kurt_mean + 2*kurt_std]
            
    #         print(f"ICA identified {len(artifact_components)} likely artifact components")
            
    #         # Reconstruct signal without artifact components
    #         mixing_matrix = ica.mixing_
    #         unmixing_matrix = ica.components_
            
    #         # Zero out the artifact components
    #         clean_components = np.copy(components)
    #         for artifact_idx in artifact_components:
    #             clean_components[:, artifact_idx] = 0
                
    #         # Reconstruct signal
    #         reconstructed = np.matmul(clean_components, mixing_matrix.T)
    #         data = reconstructed
            
    #     except (ImportError, ValueError, np.linalg.LinAlgError) as e:
    #         print(f"ICA processing failed: {e}. Continuing with thresholded data.")

    # 3. Median filter to kill isolated spikes
    data = medfilt(data, kernel_size=(mft,1))

    # 4. Zero‑phase band‑pass (0.5–15 Hz)
    b, a = butter(4, [lowcut, highcut], btype='band', fs=srate)
    data = filtfilt(b, a, data, axis=0)
    
    return data


def filter_signal_data_and_remove_artifacts(data, srate, mft=5, lowcut=0.5, highcut=15, artifact_threshold=125, apply_ica=True):
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



## function to detect LRLR patterns. ---
# Input is EOG data only, sampling rate, and seconds to check.
# Function checks the last N seconds of EOG data for LRLR patterns threshold is min count of LRLR patterns to detect for positive result (across all channels).
def detect_lrlr_in_window(eog_data, srate, seconds, threshold=2, test=False):
    """
    Detects LRLR patterns in the last N seconds of EOG data.
    
    Args:
        eog_data: EOG data with channels in columns
        srate: Sampling rate in Hz
        seconds: Number of seconds to check from the end of the data
        threshold: Minimum number of LRLR patterns to detect for a positive result
        
    Returns:
        bool: True if LRLR patterns detected above threshold, False otherwise
    """
    # Get the window from the last N seconds
    window_size = int(seconds * srate)
    if eog_data.shape[0] < window_size:
        eog_data_window = eog_data  # Use all data if less than window_size
    else:
        eog_data_window = eog_data[-window_size:, :]  # Last N seconds
    
    # Initialize count of LRLR patterns
    lrlr_count = 0
    
    # Process each channel
    for channel in range(eog_data_window.shape[1]):
        signal = eog_data_window[:, channel]
        
        # Check if the channel has enough data
        if len(signal) < srate:  # Need at least 1 second of data
            continue
            
        
        # Check for data continuity before applying filter
        if np.all(np.isfinite(signal)) and np.sum(np.abs(np.diff(signal)) > 1000) < len(signal) * 0.05:
            # Apply zero-phase bandpass filter (0.5-15 Hz)
            b, a = butter(4, [0.5, 15], btype='band', fs=srate)
            filtered_signal = filtfilt(b, a, signal)
        else:
            # Skip filtering for discontinuous data
            filtered_signal = signal
            print(f"Warning: Skipping filter for channel {channel} due to discontinuities")
        
        # Apply median filter to remove spikes
        filtered_signal = medfilt(filtered_signal, kernel_size=5)
        
        # Find peaks and troughs
        pos_peaks, _ = find_peaks(filtered_signal, height=100, distance=int(srate * 0.2))  # Min 0.2s between peaks
        neg_peaks, _ = find_peaks(-filtered_signal, height=100, distance=int(srate * 0.2))
        
        # Combine and sort peaks
        events = sorted([(i, 'pos') for i in pos_peaks] + [(i, 'neg') for i in neg_peaks])
        
        # Search for LRLR patterns (neg-pos-neg-pos or pos-neg-pos-neg)
        found_pattern = False
        for i in range(len(events) - 3):
            types = [events[i+j][1] for j in range(4)]
            times = [events[i+j][0] / srate for j in range(4)]
            duration = times[-1] - times[0]
            
            # Check if the pattern occurs within 2.5 seconds
            if duration <= 2.5:
                if (types == ['neg', 'pos', 'neg', 'pos'] or 
                    types == ['pos', 'neg', 'pos', 'neg']):
                    found_pattern = True
                    lrlr_count += 1
                    break

        # REMOVE THIS FOR ONLINE TESTING
        if test:
            print(f"Channel {channel}: Found LRLR pattern: {found_pattern}")
            print(f"  Events: {events}")
            # plot_single_channel_data(filtered_signal, srate=125, LRLR=found_pattern)

    
    # Return True if count exceeds threshold
    return lrlr_count >= threshold, lrlr_count

# Helper function for preprocessing, adapted from lstm_data_extraction.py
def _preprocess_input_for_lstm(data_segment_raw, model_srate_for_filter=118, mft=5, lowcut=0.5, highcut=15, artifact_threshold=125):
    """
    Preprocesses a 2-channel data segment (LSTM_SAMPLE_LENGTH, 2) exactly as done for LSTM training.
    model_srate_for_filter is the sampling rate used for designing the butterworth filter during training.
    The lowcut and highcut parameters are now passed to filter_signal_data.
    """
    data = np.copy(data_segment_raw).astype(np.float32) 
    
    try:
        # Use the passed lowcut and highcut for filtering
        data = filter_signal_data(data, srate=model_srate_for_filter, mft=mft, lowcut=lowcut, highcut=highcut, artifact_threshold=artifact_threshold, apply_ica=False) # ICA not typically used in this specific LSTM preprocess
    except ValueError as e:
        print(f"LSTM Preprocessing Error: Bandpass filtering failed: {e}. Check highcut vs model_srate_for_filter/2.")
        return np.zeros_like(data_segment_raw) 

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / (std + 1e-8)

    return normalized_data


def detect_lrlr_window_from_lstm(eog_data_segment, srate, model_path=DEFAULT_LSTM_MODEL_PATH, detection_threshold=0.5, lstm_lowcut=0.5, lstm_highcut=15):
    """
    Detects LRLR patterns using the trained LSTM model on a given EOG data segment.

    Args:
        eog_data_segment (np.ndarray): EOG data array segment, must be (LSTM_SAMPLE_LENGTH, 2).
        srate (float): Sampling rate of the input eog_data.
        model_path (str): Path to the Keras LSTM model file.
        detection_threshold (float): Threshold for classifying a prediction as LRLR.
        lstm_lowcut (float): Lowcut for LSTM's internal bandpass filter.
        lstm_highcut (float): Highcut for LSTM's internal bandpass filter.

    Returns:
        tuple: (bool, float) -> (is_lrlr_detected, prediction_value) or None if error.
    """
    global LRLR_LSTM_MODEL
    if LRLR_LSTM_MODEL is None:
        try:
            from tensorflow.keras.models import load_model # type: ignore
            LRLR_LSTM_MODEL = load_model(model_path)
            print(f"LSTM Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading LSTM model from {model_path}: {e}")
            # traceback.print_exc()
            return None

    if eog_data_segment.shape[0] != LSTM_SAMPLE_LENGTH:
        print(f"Error: EOG data segment length is incorrect for LSTM ({eog_data_segment.shape[0]} samples, need {LSTM_SAMPLE_LENGTH}).")
        return None 
    
    if eog_data_segment.shape[1] != 2:
        print(f"Error: EOG data segment has incorrect channels ({eog_data_segment.shape[1]}). LSTM requires 2.")
        return None

    # Preprocess the segment using the provided lowcut and highcut
    preprocessed_segment = _preprocess_input_for_lstm(
        eog_data_segment, 
        model_srate_for_filter=118, # This srate is for filter design, matching training
        lowcut=lstm_lowcut, 
        highcut=lstm_highcut
    )

    model_input = np.expand_dims(preprocessed_segment, axis=0)

    try:
        prediction_value = LRLR_LSTM_MODEL.predict(model_input, verbose=0)[0][0]
        is_lrlr = prediction_value > detection_threshold
        return is_lrlr, float(prediction_value) 
    except Exception as e:
        print(f"Error during LSTM prediction: {e}")
        # traceback.print_exc()
        return None


def get_values_for_development():
    pass

def write_csv_results(filepath, data_rows, fieldnames):
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data_rows:
                writer.writerow(row)
        print(f"Successfully wrote results to {filepath}")
    except IOError:
        print(f"Error writing CSV to {filepath}")
        traceback.print_exc()

def run_single_test_configuration(config, rec_data_names_list, base_output_dir):
    config_name = config['name']
    algo_type = config['algo_type']
    filter_func_name = config['filter_function_name'] # Name to select function
    use_ensemble = config['is_ensemble'] # For threshold, affects interpretation. For LSTM, implies 2 EOG.
    use_hdiff = config['is_hdiff']
    dataset_keyword_filter = config['dataset_keyword_filter']
    bandpass_params = config['bandpass'] # (lowcut, highcut) for primary filter
    lstm_bandpass_params = config.get('lstm_bandpass', bandpass_params) # Specific for LSTM internal, defaults to primary
    csv_filename = config['csv_name']

    results_for_csv = []
    csv_filepath = os.path.join(base_output_dir, csv_filename)
    fieldnames = ['dataset_name', 'window_start_sample', 'window_end_sample', 'true_label', 'predicted_label', 'prediction_score']

    print(f"--- Running test: {config_name} ---")

    # Select the filter function
    if filter_func_name == 'filter_signal_data':
        filter_function_to_use = filter_signal_data
    elif filter_func_name == 'filter_signal_data_and_remove_artifacts':
        filter_function_to_use = filter_signal_data_and_remove_artifacts
    else:
        print(f"Error: Unknown filter function name '{filter_func_name}' in config '{config_name}'. Skipping.")
        return

    for rec_name in rec_data_names_list:
        if dataset_keyword_filter and dataset_keyword_filter not in rec_name:
            continue
        
        session_folder_path = os.path.join("recorded_data", rec_name)
        loaded_data, session_metadata = load_custom_data(session_folder_path)

        if loaded_data is None or session_metadata is None or loaded_data.size == 0:
            print(f"Skipping {rec_name} for {config_name}: missing data, metadata, srate, or empty data.")
            continue
        
        # use the tolerant getter
        srate = _get_srate(session_metadata)
        if srate is None:
            print(f"  ✗ {rec_name}: sampling‑rate not found in metadata → skipping")
            continue

        all_col_names = session_metadata.get('processed_column_names', [])
        
        if 'TargetEvent' not in all_col_names:
            print(f"Skipping {rec_name} for {config_name}: 'TargetEvent' column not found.")
            continue
        target_event_col_idx = all_col_names.index('TargetEvent')
        target_events_full = loaded_data[:, target_event_col_idx].astype(bool)

        data_col_indices = [
            i for i, name in enumerate(all_col_names)
            if name.upper().startswith('CH') or \
               name.upper().startswith('EOG') or \
               name.upper() in ['LEFT', 'RIGHT', 'HORIZ', 'L', 'R', 'HORIZONTAL', 'VERTICAL']
        ]
        if not data_col_indices:
            searched_patterns = "names starting with 'CH' or 'EOG', or matching 'LEFT', 'RIGHT', 'HORIZ', 'L', 'R', 'HORIZONTAL', 'VERTICAL' (case-insensitive)"
            print(f"Skipping {rec_name} for {config_name}: No EOG data channels found. Searched for {searched_patterns} in columns: {all_col_names}.")
            continue
        
        eog_data_full_original_channels = loaded_data[:, data_col_indices]

        # Prepare EOG input for filtering based on hdiff
        eog_input_for_filter = None
        if use_hdiff:
            if eog_data_full_original_channels.shape[1] >= 2:
                # Assuming first two EOG channels are Left and Right or equivalent for hdiff
                hdiff_signal = eog_data_full_original_channels[:, 0] - eog_data_full_original_channels[:, 1]
                eog_input_for_filter = hdiff_signal.reshape(-1, 1)
            else:
                print(f"Skipping {rec_name} for {config_name} (hdiff): needs at least 2 EOG channels for difference.")
                continue
        else:
            eog_input_for_filter = eog_data_full_original_channels
        
        if eog_input_for_filter is None or eog_input_for_filter.size == 0 :
            print(f"Skipping {rec_name} for {config_name}: EOG data for filter is empty or None.")
            continue

        # Apply primary filter
        lowcut, highcut = bandpass_params
        filtered_eog_for_windowing = filter_function_to_use(np.copy(eog_input_for_filter), srate, lowcut=lowcut, highcut=highcut)
        
        if filtered_eog_for_windowing.size == 0:
            print(f"Skipping {rec_name} for {config_name}: EOG data became empty after filtering.")
            continue

        # Prepare data for the specific detection algorithm (channel selection)
        eog_for_detection_func = None
        if algo_type == 'lstm':
            if use_hdiff: 
                if filtered_eog_for_windowing.shape[1] == 1: # hdiff result is 1 channel
                    eog_for_detection_func = np.column_stack((filtered_eog_for_windowing.flatten(), filtered_eog_for_windowing.flatten())) # Duplicate for LSTM's 2ch input
                else:
                    print(f"Skipping {rec_name} for {config_name} (LSTM hdiff): filtered hdiff data not 1 channel.")
                    continue
            else: # Standard LSTM: use first 2 channels from (potentially multi-channel) filtered EOG
                if filtered_eog_for_windowing.shape[1] >= 2:
                    eog_for_detection_func = filtered_eog_for_windowing[:, :2]
                else:
                    print(f"Skipping {rec_name} for {config_name} (LSTM): needs at least 2 channels post-filter for non-hdiff.")
                    continue
        elif algo_type == 'threshold':
            # Threshold uses all channels passed to it (either original filtered, or single hdiff filtered)
            eog_for_detection_func = filtered_eog_for_windowing
        
        if eog_for_detection_func is None or eog_for_detection_func.size == 0:
            print(f"Skipping {rec_name} for {config_name}: data for detection function is empty or None.")
            continue
            
        # pick a window length appropriate for the algorithm ---------------------
        if algo_type == 'lstm':
            window_len = LSTM_SAMPLE_LENGTH            # must stay 750 for the model
            step_size  = window_len // 2               # 50 % overlap
        else:   # threshold family – size in seconds → samples
            window_len = int(THRESH_WINDOW_SEC * srate)
            window_len = max(window_len,     int(1.0 * srate))   # never < 1 s
            window_len = min(window_len, eog_for_detection_func.shape[0])
            if window_len < 2:          # still too short – skip this dataset
                continue
            step_size  = int(window_len * THRESH_STEP_FRACTION)
            
        num_windows = (eog_for_detection_func.shape[0] - window_len) // step_size + 1

        for i_win in range(num_windows):
            start = i_win * step_size
            end = start + window_len
            
            window_eog_segment = eog_for_detection_func[start:end, :]
            true_label_val = np.any(target_events_full[start:end]) # True if any part of the window has a target event
            
            pred_label_bool = False
            pred_score_float = 0.0

            if algo_type == 'threshold':
                window_duration_sec = window_len / srate
                threshold_val_for_detection = 2 
                if use_ensemble and window_eog_segment.shape[1] > 1:
                     # This is where ensemble logic for threshold could be adjusted if needed.
                     # For now, detect_lrlr_in_window's default behavior of summing counts is used.
                     pass 

                detection_result = detect_lrlr_in_window(window_eog_segment, srate, seconds=window_duration_sec, threshold=threshold_val_for_detection)
                pred_label_bool, count = detection_result
                pred_score_float = float(count)

            elif algo_type == 'lstm':
                lstm_bp_low, lstm_bp_high = lstm_bandpass_params
                result_tuple = detect_lrlr_window_from_lstm(window_eog_segment, srate, lstm_lowcut=lstm_bp_low, lstm_highcut=lstm_bp_high)
                if result_tuple:
                    pred_label_bool, pred_score_float = result_tuple
                else: 
                    pred_label_bool, pred_score_float = False, 0.0 
            
            results_for_csv.append({
                'dataset_name': rec_name,
                'window_start_sample': start,
                'window_end_sample': end,
                'true_label': int(true_label_val),
                'predicted_label': int(pred_label_bool),
                'prediction_score': pred_score_float
            })

    if results_for_csv:
        write_csv_results(csv_filepath, results_for_csv, fieldnames)
    else:
        print(f"No results generated for {config_name}. CSV not written.")
    print(f"--- Finished test: {config_name} ---")


def run_all_lrlr_tests():
    output_dir = "lrlr_test_results"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            traceback.print_exc()
            return

    default_lowcut = 0.5
    default_highcut = 15
    hdiff_bp_lowcut = 0.2
    hdiff_bp_highcut = 4

    configurations = [
        {
            'name': 'threshold_filter', 'algo_type': 'threshold', 
            'filter_function_name': 'filter_signal_data', 'is_ensemble': False, 'is_hdiff': False,
            'dataset_keyword_filter': None, 'bandpass': (default_lowcut, default_highcut),
            'csv_name': 'results_threshold_filter.csv'
        },
        {
            'name': 'threshold_filter_ensemble', 'algo_type': 'threshold',
            'filter_function_name': 'filter_signal_data', 'is_ensemble': True, 'is_hdiff': False,
            'dataset_keyword_filter': None, 'bandpass': (default_lowcut, default_highcut),
            'csv_name': 'results_threshold_filter_ensemble.csv'
        },
        {
            'name': 'threshold_filter_remove_artifacts_ensemble', 'algo_type': 'threshold',
            'filter_function_name': 'filter_signal_data_and_remove_artifacts', 'is_ensemble': True, 'is_hdiff': False,
            'dataset_keyword_filter': None, 'bandpass': (default_lowcut, default_highcut),
            'csv_name': 'results_threshold_filter_remove_artifacts_ensemble.csv'
        },
        {
            'name': 'LSTM_filter_remove_artifacts_ensemble', 'algo_type': 'lstm',
            'filter_function_name': 'filter_signal_data_and_remove_artifacts', 'is_ensemble': True, 'is_hdiff': False, # is_ensemble for LSTM implies using 2 EOG channels, not hdiff.
            'dataset_keyword_filter': None, 'bandpass': (default_lowcut, default_highcut), # Primary filter for the data before LSTM
            'lstm_bandpass': (default_lowcut, default_highcut), # Bandpass for LSTM's internal preprocessing
            'csv_name': 'results_LSTM_filter_remove_artifacts_ensemble.csv'
        },
        {
            'name': 'threshold_filter_remove_artifacts_hdiff', 'algo_type': 'threshold',
            'filter_function_name': 'filter_signal_data_and_remove_artifacts', 'is_ensemble': False, 'is_hdiff': True,
            'dataset_keyword_filter': None, 'bandpass': (hdiff_bp_lowcut, hdiff_bp_highcut),
            'csv_name': 'results_threshold_filter_remove_artifacts_hdiff.csv'
        },
        {
            'name': 'LSTM_filter_remove_artifacts_hdiff', 'algo_type': 'lstm',
            'filter_function_name': 'filter_signal_data_and_remove_artifacts', 'is_ensemble': False, 'is_hdiff': True,
            'dataset_keyword_filter': None, 'bandpass': (hdiff_bp_lowcut, hdiff_bp_highcut), # Primary filter for the hdiff data
            'lstm_bandpass': (hdiff_bp_lowcut, hdiff_bp_highcut), # Bandpass for LSTM's internal preprocessing on hdiff
            'csv_name': 'results_LSTM_filter_remove_artifacts_hdiff.csv'
        },
        {
            'name': 'threshold_filter_remove_artifacts_hdiff_closed', 'algo_type': 'threshold',
            'filter_function_name': 'filter_signal_data_and_remove_artifacts', 'is_ensemble': False, 'is_hdiff': True,
            'dataset_keyword_filter': 'closed', 'bandpass': (hdiff_bp_lowcut, hdiff_bp_highcut),
            'csv_name': 'results_threshold_filter_remove_artifacts_hdiff_closed.csv'
        }
    ]

    global rec_data_names
    if not rec_data_names:
        print("rec_data_names list is empty. No data to process.")
        return

    for config in configurations:
        run_single_test_configuration(config, rec_data_names, output_dir)

    print("All LRLR classification tests completed.")

def main():
    print("Starting LRLR classification tests...")
    run_all_lrlr_tests()
    print("LRLR classification tests finished.")

if __name__ == "__main__":
    main()
    #get_values_for_development()