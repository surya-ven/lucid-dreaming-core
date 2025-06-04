# Author: Benjamin Grayzel

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import butter, filtfilt, medfilt, find_peaks, welch

import time


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


def filter_signal_data(data, srate = 118, mft=5, lowcut=0.5, highcut=15, artifact_threshold=125):
    data = np.copy(data)
    n_channels = data.shape[1]
    # print(f"Filtering data with {n_channels} channels") # Optional: reduce verbosity
    
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


def create_lstm_data():
    all_samples_list = []
    all_labels_list = []
    SAMPLE_LENGTH = 750
    
    for filepath_short in rec_data_names:
        SESSION_FOLDER_PATH = f"recorded_data/{filepath_short}"
        print(f"Processing: {SESSION_FOLDER_PATH}")
        loaded_data, session_info = load_custom_data(SESSION_FOLDER_PATH)

        if loaded_data is None or loaded_data.shape[0] < SAMPLE_LENGTH:
            print(f"  Failed to load data or data too short for {filepath_short} (shape: {loaded_data.shape if loaded_data is not None else 'None'}). Skipping.")
            continue

        srate = 118  # Default srate

        if loaded_data.shape[1] < 4: # Need at least Timestamp, TargetEvent, and 2 EOG channels from the end
            print(f"  Not enough columns in loaded_data for {filepath_short} (shape: {loaded_data.shape}) to extract EOG channels. Skipping.")
            continue
        
        try:
            # Select 3rd and 2nd to last columns as EOG data
            eog_raw_data = loaded_data[:, -4:-2].astype(np.float32)
        except IndexError:
            print(f"  Could not select EOG channels -4:-2 from data with shape {loaded_data.shape} for {filepath_short}. Skipping.")
            continue
        
        if eog_raw_data.shape[1] != 2:
            print(f"  Selected EOG data does not have 2 channels for {filepath_short}. Shape: {eog_raw_data.shape}. Skipping.")
            continue

        target_event = loaded_data[:, 1].astype(int)  # TargetEvent is the second column

        # Find LRLR event segments
        diff_target = np.diff(np.concatenate(([target_event[0]], target_event, [target_event[-1]])))
        event_starts = np.where(diff_target == 1)[0]
        event_ends = np.where(diff_target == -1)[0] -1


        # Extract LRLR samples
        for start, end in zip(event_starts, event_ends):
            if start > end : continue # Should not happen with correct diff logic but good check
            event_len = end - start + 1
            
            if 0 < event_len <= SAMPLE_LENGTH:
                event_center = start + event_len // 2
                
                sample_start_idx = event_center - SAMPLE_LENGTH // 2
                sample_end_idx = sample_start_idx + SAMPLE_LENGTH

                if sample_start_idx < 0:
                    sample_end_idx += (0 - sample_start_idx)
                    sample_start_idx = 0
                if sample_end_idx > len(eog_raw_data):
                    sample_start_idx -= (sample_end_idx - len(eog_raw_data))
                    sample_end_idx = len(eog_raw_data)

                if sample_start_idx >= 0 and (sample_end_idx - sample_start_idx) == SAMPLE_LENGTH:
                    sample_data = eog_raw_data[sample_start_idx:sample_end_idx, :]
                    filtered_sample = filter_signal_data(sample_data, srate=srate)
                    
                    mean = np.mean(filtered_sample, axis=0)
                    std = np.std(filtered_sample, axis=0)
                    normalized_sample = (filtered_sample - mean) / (std + 1e-8)
                    
                    all_samples_list.append(normalized_sample)
                    all_labels_list.append(1)
                else:
                    print(f"  Skipping LRLR event at {start}-{end} in {filepath_short} due to windowing issues after centering (final window {sample_start_idx}-{sample_end_idx}).")
            elif event_len > SAMPLE_LENGTH:
                print(f"  Skipping LRLR event at {start}-{end} in {filepath_short} because it's longer ({event_len}) than {SAMPLE_LENGTH} samples.")

        # Extract non-LRLR samples
        non_lrlr_stride = SAMPLE_LENGTH // 2 
        for i in range(0, len(eog_raw_data) - SAMPLE_LENGTH + 1, non_lrlr_stride):
            window_start = i
            window_end = i + SAMPLE_LENGTH
            
            if not np.any(target_event[window_start:window_end]): # Ensure entire window is non-LRLR
                sample_data = eog_raw_data[window_start:window_end, :]
                filtered_sample = filter_signal_data(sample_data, srate=srate)
                
                mean = np.mean(filtered_sample, axis=0)
                std = np.std(filtered_sample, axis=0)
                normalized_sample = (filtered_sample - mean) / (std + 1e-8)
                
                all_samples_list.append(normalized_sample)
                all_labels_list.append(0)

    if not all_samples_list:
        print("No samples were extracted. NPZ file will not be created.")
        return

    X = np.array(all_samples_list, dtype=np.float32)
    y = np.array(all_labels_list, dtype=np.int32)

    print(f"\nTotal samples extracted: {X.shape[0]}")
    print(f"LRLR samples (label 1): {np.sum(y == 1)}")
    print(f"Non-LRLR samples (label 0): {np.sum(y == 0)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    output_filename = 'lstm_training_data.npz'
    np.savez(output_filename, X=X, y=y)
    print(f"Saved data to {output_filename}")


if __name__ == '__main__':
    create_lstm_data()
