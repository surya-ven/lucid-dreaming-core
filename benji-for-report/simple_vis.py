# Author: Benjamin Grayzel

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import butter, filtfilt, medfilt, find_peaks, welch, savgol_filter, firwin
from sklearn.decomposition import FastICA
import tensorflow as tf
import time
import csv
import traceback
from datetime import datetime

# Disable matplotlib toolbar
plt.rcParams['toolbar'] = 'None'

LRLR_LSTM_MODEL = None
DEFAULT_LSTM_MODEL_PATH = 'lrlr_lstm_model.keras'
BEST_MODEL_PATH = 'models/lrlr_conv1d_model_fold__final_all_data.keras'
BEST_MODEL_THRESHOLD = 0.5575 # Threshold for best model to classify LRLR as True

LSTM_SAMPLE_LENGTH = 750
MODEL_SAMPLE_LENGTH = 750

rec_data_name_surya = "v2_LRLR_once_7_closed"
sy_start = 300
sy_end = 2800  # Changed to use 2500 samples (300 + 2500)
sy_viz_start = 300
sy_viz_end = 1800  # Original visualization range

rec_data_name_benji = "20250524_033027_138563"
bj_start = 0
bj_end = 2500  # Changed to use 2500 samples
bj_viz_start = 0
bj_viz_end = 1500  # Original visualization range


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



def plot_power_spectrum(data, srate, name="???"):
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):
        f, Pxx = welch(data[:, i], fs=srate)
        plt.semilogy(f, Pxx, label=f'Channel {i}')
    
    plt.axvline(60, color='r', linestyle='--', alpha=0.7, label='60 Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density')
    plt.title(f'Power Spectrum of {name} Channels')
    plt.legend()
    plt.grid(True)
    plt.show()



# --- Plotting the last 4 channels (EOG OR EEG) ---
def plot_eeg_eog_data(loaded_data, session_metadata, colstart=1, colend=4, target_event=None, title="All Channels"):
    if loaded_data is not None and loaded_data.shape[0] > 0 and loaded_data.shape[1] >= colend:
        num_samples = loaded_data.shape[0]
        time_vector = np.arange(num_samples)  # Timestep of 1/100th of a second

        data_to_plot = loaded_data[:, colstart:colend]
        column_names = session_metadata.get('column_names', [f'Ch{i+1}' for i in range(loaded_data.shape[1])])
        column_names = column_names
        print(column_names)

        fig, axes = plt.subplots(colend-colstart, 1, figsize=(15, 10), sharex=True)
        if data_to_plot.shape[1] == 1: 
            axes = [axes] 

        for i in range(0,colend-colstart):
            ax = axes[i]
            ax.plot(time_vector, data_to_plot[:, i])
            if target_event is not None:
                ax.fill_between(range(len(target_event)), target_event, color='darkred', edgecolor='darkred', 
                                linewidth=4,alpha=1, where=(target_event == 1), label="Target Event (filled)")

            ax.set_title(f"Plot of: {column_names[i]}") #if i != 2 else ax.set_title(f"Plot of: Horz. Difference")
            ax.set_ylabel("Value")
            
            # Set y-limits with proper margins to prevent line cutoff
            data_min = np.min(data_to_plot[:, i])
            data_max = np.max(data_to_plot[:, i])
            data_range = data_max - data_min
            margin = data_range * 0.1 if data_range > 0 else 1.0  # 10% margin or 1.0 if flat
            ax.set_ylim([data_min - margin, data_max + margin])
            
            ax.grid(True)
            
            # Remove x-axis labels and ticks from all but the last subplot
            if i < colend-colstart-1:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    axes[-1].set_xlabel("Time (samples)")
    plt.tight_layout()



def CURRENT_process_eog_for_plotting(data, srate, lowcut=0.5, highcut=4.0,
                             artifact_threshold=1000, mft_kernel_size=5):
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.shape[0] < 15:
        return data

    for ch in range(data.shape[1]):
        cd = data[:, ch]
        artifacts = np.abs(cd) > artifact_threshold
        if artifacts.any():
            idx = np.where(artifacts)[0]
            splits = np.where(np.diff(idx) > 1)[0] + 1
            segs = np.split(idx, splits)
            for seg in segs:
                if len(seg) == 0:
                    continue
                start, end = seg[0], seg[-1]
                pre = cd[start-1] if start > 0 else cd[end +
                                                       1] if end+1 < len(cd) else 0
                post = cd[end+1] if end + \
                    1 < len(cd) else cd[start-1] if start > 0 else 0
                data[seg, ch] = np.linspace(pre, post, len(seg))


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

    k = mft_kernel_size + (1 - mft_kernel_size % 2)
    try:
        data = medfilt(data, kernel_size=(k, 1))
    except ValueError:
        pass

    b, a = butter(4, [lowcut, highcut], btype='band', fs=srate)
    data = filtfilt(b, a, data, axis=0)
    return data


def alternative_process_eog_for_plotting(data, srate, low=0.5, high=4.0,
                                     clip=500, sg_win=11, sg_ord=3):

    # ---- 0) shape + float ----
    data = np.atleast_2d(data).T if data.ndim == 1 else data.copy()
    data = data.astype(float)

    # ---- 1) mark clips as NaN ----
    clip_mask = np.abs(data) > clip
    data[clip_mask] = np.nan

    # ---- 2) fill NaNs ONCE (linear) ----
    for ch in range(data.shape[1]):
        bad = np.isnan(data[:, ch])
        if bad.any():
            good = ~bad
            if good.sum() == 0:                 # full‑channel clip
                data[:, ch] = 0.0               # flat‑fill (or leave NaN)
            else:
                data[bad, ch] = np.interp(np.flatnonzero(bad),
                                          np.flatnonzero(good),data[good, ch])

    # ---- 3) Savitzky–Golay (safe: no NaNs left) ----
    if sg_win >= data.shape[0]:
        sg_win = data.shape[0] - 1 if data.shape[0] % 2 == 0 else data.shape[0] - 2
        sg_win = max(sg_win, 3)  # Minimum window size
    data = savgol_filter(data, sg_win, sg_ord, axis=0, mode='interp')

    # ---- 4) FIR band‑pass 0.5‑4 Hz with reasonable filter length ----
    # Use shorter filter lengths that work with the data size
    data_len = data.shape[0]
    max_filter_len = min(201, data_len // 3)  # Reasonable filter length
    
    if max_filter_len < 11:
        # If data is too short for FIR, fall back to Butterworth
        try:
            b, a = butter(4, [low, high], btype='band', fs=srate)
            data = filtfilt(b, a, data, axis=0)
        except ValueError:
            print(f"Warning: Could not apply bandpass filter to short data ({data_len} samples)")
    else:
        # Make filter length odd
        if max_filter_len % 2 == 0:
            max_filter_len -= 1
            
        try:
            hp = firwin(max_filter_len, low, fs=srate, pass_zero=False)
            lp = firwin(max_filter_len, high, fs=srate)
            # Use shorter convolution to avoid excessive filter length
            bp = np.convolve(hp, lp, mode='same')  # Use 'same' instead of 'full'
            data = filtfilt(bp, [1.0], data, axis=0)
        except ValueError:
            # Fall back to Butterworth if FIR still fails
            try:
                b, a = butter(4, [low, high], btype='band', fs=srate)
                data = filtfilt(b, a, data, axis=0)
            except ValueError:
                print(f"Warning: Could not apply any bandpass filter to data ({data_len} samples)")

    return data



# Helper function for preprocessing, adapted from lstm_data_extraction.py
def _preprocess_input_for_lstm(data_segment_raw, model_srate_for_filter=118, mft=5, lowcut=0.5, highcut=15, artifact_threshold=125):
    """
    Preprocesses a 2-channel data segment (LSTM_SAMPLE_LENGTH, 2) exactly as done for LSTM training.
    model_srate_for_filter is the sampling rate used for designing the butterworth filter during training.
    """
    data = np.copy(data_segment_raw).astype(np.float32) 
    
    try:
        data = CURRENT_process_eog_for_plotting(data, srate=model_srate_for_filter, mft=mft, lowcut=lowcut, highcut=highcut, artifact_threshold=artifact_threshold)
    except ValueError as e:
        print(f"LSTM Preprocessing Error: Bandpass filtering failed: {e}. Check highcut vs model_srate_for_filter/2.")
        return np.zeros_like(data_segment_raw) # Return zeros or raise error

    # Normalization (per-channel z-score)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Add epsilon to std to prevent division by zero if a channel is flat after filtering
    normalized_data = (data - mean) / (std + 1e-8)

    return normalized_data

def detect_lrlr_window_from_lstm(eog_data, srate, model_path, detection_threshold=0.5):
    """
    Detects LRLR patterns using the trained LSTM model.

    Args:
        eog_data (np.ndarray): EOG data array, expected to have at least 2 channels.
                               The first two channels will be used.
        srate (float): Sampling rate of the input eog_data. (Currently used for context,
                       filter design uses a fixed srate from training).
        model_path (str): Path to the Keras LSTM model file.
        detection_threshold (float): Threshold for classifying a prediction as LRLR.

    Returns:
        bool: True if LRLR is detected, False otherwise.
    """
    global LRLR_LSTM_MODEL
    if LRLR_LSTM_MODEL is None:
        try:
            LRLR_LSTM_MODEL = tf.keras.models.load_model(model_path)
            print(f"LSTM model '{model_path}' loaded successfully.")
        except Exception as e:
            print(f"Error loading LSTM model from '{model_path}': {e}")
            return False, 0.0 # Cannot proceed without the model

    if eog_data.shape[0] < LSTM_SAMPLE_LENGTH:
        print(f"Warning: EOG data too short for LSTM ({eog_data.shape[0]} samples, need {LSTM_SAMPLE_LENGTH}).")
        return False, 0.0
    
    if eog_data.shape[1] < 2:
        print(f"Warning: EOG data has fewer than 2 channels ({eog_data.shape[1]}). LSTM requires 2.")
        return False, 0.0

    # Extract the last LSTM_SAMPLE_LENGTH samples from the first two channels
    # Assumes eog_data's first two columns are the ones the LSTM was trained on.
    eog_segment_raw = eog_data[-LSTM_SAMPLE_LENGTH:, 0:2]

    # Preprocess the segment
    # model_srate_for_filter=118 is used internally as that's what the model was trained with.
    preprocessed_segment = _preprocess_input_for_lstm(eog_segment_raw, model_srate_for_filter=118)

    # Reshape for model prediction: (1, timesteps, features)
    model_input = np.expand_dims(preprocessed_segment, axis=0)

    # Predict
    try:
        # print(f"Model input shape: {model_input.shape}")
        prediction_value = LRLR_LSTM_MODEL.predict(model_input, verbose=0)[0][0]
        is_lrlr = prediction_value > detection_threshold
        # print(f"LSTM Prediction: {prediction_value:.4f} -> Detected: {is_lrlr}") # For debugging
        return is_lrlr, prediction_value
    except Exception as e:
        print(f"Error during LSTM prediction: {e}")
        return False, 0.0 # Return a default value for the prediction if an error occurs


# --- Enhanced plotting function for EOG data with LRLR detection ---
def plot_eog_with_lrlr_detection(loaded_data, session_metadata, colstart=0, colend=3, 
                                target_event=None, title="EOG Channels with LRLR Detection",
                                data_start_idx=0, srate=125):
    """
    Enhanced plotting function that shows EOG channels with proper timestamps,
    LRLR segment marking, and improved visualization.
    """
    if loaded_data is not None and loaded_data.shape[0] > 0 and loaded_data.shape[1] >= colend:
        num_samples = loaded_data.shape[0]
        
        # Create proper time vector - use sample-based time for now
        time_vector = np.arange(num_samples) / srate
        
        # Debug: Print some info
        print(f"Debug: Data shape: {loaded_data.shape}")
        print(f"Debug: Time vector range: {time_vector[0]:.2f} to {time_vector[-1]:.2f} seconds")
        print(f"Debug: Data range - min: {np.min(loaded_data[:, colstart:colend]):.3f}, max: {np.max(loaded_data[:, colstart:colend]):.3f}")
        
        data_to_plot = loaded_data[:, colstart:colend]
        column_names = session_metadata.get('processed_column_names', [f'Ch{i+1}' for i in range(loaded_data.shape[1])])
        plot_column_names = column_names[colstart:colend] if colstart < len(column_names) else [f'Ch{i+colstart}' for i in range(colend-colstart)]
        
        # Create figure with smaller size and styling
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot channels with different styling
        colors = ['lightblue', 'lightcoral', 'black']
        alphas = [0.4, 0.4, 1.0]
        linewidths = [1, 1, 2]
        labels = ['Left EOG (LF_FpZ)', 'Right EOG (RF_FpZ)', 'Horizontal Difference (L-R)']
        
        for i in range(colend - colstart):
            ax.plot(time_vector, data_to_plot[:, i], 
                   color=colors[i % len(colors)], 
                   alpha=alphas[i % len(alphas)],
                   linewidth=linewidths[i % len(linewidths)],
                   label=labels[i] if i < len(labels) else plot_column_names[i])
        
        # Set y-limits with proper margins to prevent line cutoff
        data_min = np.min(data_to_plot)
        data_max = np.max(data_to_plot)
        data_range = data_max - data_min
        margin = data_range * 0.15 if data_range > 0 else 1.0  # 15% margin or 1.0 if flat
        ax.set_ylim([data_min - margin, data_max + margin])
        
        lrlr_time = None

        # Add LRLR event highlighting
        if target_event is not None:
            # Find LRLR segments
            target_event_int = target_event.astype(int)
            lrlr_transitions = np.diff(target_event_int)
            lrlr_starts = np.where(lrlr_transitions == 1)[0] + 1
            lrlr_ends = np.where(lrlr_transitions == -1)[0] + 1


            # Handle edge cases
            if target_event_int[0] == 1:
                lrlr_starts = np.concatenate([[0], lrlr_starts])
            if target_event_int[-1] == 1:
                lrlr_ends = np.concatenate([lrlr_ends, [len(target_event) - 1]])
            
            # Ensure equal number of starts and ends
            min_len = min(len(lrlr_starts), len(lrlr_ends))
            lrlr_starts = lrlr_starts[:min_len]
            lrlr_ends = lrlr_ends[:min_len]
            
            print(f"Debug: Found {len(lrlr_starts)} LRLR segments")
            
            # Highlight LRLR segments with less aggressive styling
            for idx, (start_idx, end_idx) in enumerate(zip(lrlr_starts, lrlr_ends)):
                start_time = time_vector[start_idx]
                end_time = time_vector[end_idx]
                duration = end_time - start_time
                lrlr_time = duration
                
                print(f"Debug: LRLR segment {idx+1}: {start_time:.1f}s to {end_time:.1f}s (duration: {duration:.1f}s)")
                
                # Add vertical lines at boundaries
                ax.axvline(start_time, color='orange', linestyle='--', alpha=0.6, linewidth=1.5)
                ax.axvline(end_time, color='orange', linestyle='--', alpha=0.6, linewidth=1.5)
                
                # Add shaded region with less aggressive styling
                ax.axvspan(start_time, end_time, alpha=0.15, color='orange', 
                          label='LRLR Event' if idx == 0 else "")
        
        # Determine subject name from title
        subject_name = "Benji" if "benji" in title.lower() else "Surya" if "surya" in title.lower() else "Subject"
        
        # Formatting with clearer titles
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude (µV)', fontsize=12)
        ax.set_title(f'{subject_name} - LRLR Eye Movement Detection', fontsize=14, fontweight='bold')
        
        # Add sampling rate as subtitle
        ax.text(0.05, 0.95, f'Sampling Rate: {srate} Hz', transform=ax.transAxes, 
               ha='left', va='top', fontsize=10, style='italic')
        
        ax.text(0.05, 0.9, f'LRLR Window Duration: {lrlr_time} Seconds', transform=ax.transAxes, 
               ha='left', va='top', fontsize=10, style='italic')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()  # Actually display the plot
        return fig, ax

def testmydat():
    # Test both datasets with full processing range and truncated visualization
    datasets = [
        (f"../recorded_data/{rec_data_name_surya}", sy_start, sy_end, sy_viz_start, sy_viz_end, "Surya"),
        (f"../recorded_data/{rec_data_name_benji}", bj_start, bj_end, bj_viz_start, bj_viz_end, "Benji")
    ]
    
    for SESSION_FOLDER_PATH, proc_start, proc_end, viz_start, viz_end, subject_name in datasets:
        print("\n","=="*50)
        print(f"Processing {subject_name} data from: {SESSION_FOLDER_PATH}")
        print(f"Processing range: {proc_start} to {proc_end} ({proc_end - proc_start} samples)")
        print(f"Visualization range: {viz_start} to {viz_end} ({viz_end - viz_start} samples)")

        loaded_data, session_metadata = load_custom_data(SESSION_FOLDER_PATH)
        if loaded_data is None:
            print(f"Failed to load data for {subject_name}")
            continue
            
        display_loaded_data_and_metadata(loaded_data, session_metadata)

        # Extract EOG data for full processing range (2500 samples)
        eog_data_full = loaded_data[proc_start:proc_end, [6, 8]]  # Left and right EOG channels
        target_events_full = loaded_data[proc_start:proc_end, 1]  # Target events
        
        # Extract data for visualization including horizontal difference calculation
        eog_data_viz = loaded_data[viz_start:viz_end, [6, 8]]  # Left and right EOG channels for viz
        # Calculate horizontal difference (L-R) and add as third column
        # processed_eog_data = CURRENT_process_eog_for_plotting(eog_data_viz, srate=125,
        #                                                       lowcut=0.5, highcut=4.0, 
        #                                                       artifact_threshold=500, mft_kernel_size=5)
        # horizontal_diff = processed_eog_data[:, 0] - processed_eog_data[:, 1]
        # eog_data_viz_with_diff = np.column_stack([processed_eog_data, horizontal_diff])
        target_events_viz = loaded_data[viz_start:viz_end, 1]  # Target events for viz
        
        print(f"Full processing EOG data shape: {eog_data_full.shape}")
        # print(f"Visualization EOG data shape (with diff): {eog_data_viz_with_diff.shape}")
        print(f"Full processing target events shape: {target_events_full.shape}")
        print(f"Visualization target events shape: {target_events_viz.shape}")
        print(f"Column names: {session_metadata['processed_column_names'][6:10]}")

        # Create preprocessing comparison using visualization data for cleaner plots
        print(f"\nCreating preprocessing comparison for {subject_name} (using visualization range)...")
        save_filename = f"processing_compare_{subject_name.lower()}.png"

        # Plot EOG with LRLR detection using the data with horizontal difference
        # plot_eog_with_lrlr_detection(eog_data_viz_with_diff, session_metadata, colstart=0, colend=3,
        #                                target_event=target_events_viz, 
        #                                title=f"{subject_name} EOG Data with LRLR Detection",
        #                                srate=125)

        compare_preprocessing_techniques(eog_data_viz, srate=125, 
                                       target_event=target_events_viz,
                                       save_path=save_filename)
        plt.show()
        
        print(f"Saved comparison plot to: {save_filename}")
        
        # Optional: You can add additional processing here using the full 2500 samples
        # For example, LRLR detection or other analysis that benefits from more data
        print(f"\nFull dataset available for advanced processing: {eog_data_full.shape[0]} samples")

def run_analysis():
    """Run the complete analysis with both datasets"""
    testmydat()


def compare_preprocessing_techniques(eog_data, srate=125, target_event=None, save_path="processing_compare.png"):
    """
    Compare 10 different preprocessing techniques to visualize LRLR patterns.
    Only plots the horizontal difference channel to avoid clutter.
    
    Args:
        eog_data: Raw EOG data with shape (samples, channels) - expects at least 2 channels
        srate: Sampling rate
        target_event: Binary array marking LRLR events
        save_path: Path to save the comparison plot
    """
    from scipy.stats import kurtosis
    
    # Ensure we have at least 2 channels for left/right EOG
    if eog_data.shape[1] < 2:
        print("Error: Need at least 2 EOG channels for comparison")
        return
    
    # Extract left and right channels
    left_channel = eog_data[:, 0].copy()
    right_channel = eog_data[:, 1].copy()
    
    # Create time vector
    time_vector = np.arange(len(left_channel)) / srate
    
    # Define 10 different preprocessing techniques
    techniques = []
    
    # 1. No processing (raw difference)
    raw_diff = left_channel - right_channel
    techniques.append(("1. Raw Frenz EOG (No Processing)", raw_diff))

    # 4. CURRENT method (0.5-4 Hz + median filter + artifact removal)
    current_data = np.column_stack([left_channel, right_channel]).copy()
    current_processed = CURRENT_process_eog_for_plotting(current_data, srate=srate, 
                                                        lowcut=0.5, highcut=4.0, 
                                                        artifact_threshold=500, mft_kernel_size=5)
    current_diff = current_processed[:, 0] - current_processed[:, 1]
    techniques.append(("4. Final Method: BP (0.5-4Hz) + AR (th=500) + IP + Medfilt (k=5)", current_diff))

    # # 3. Conservative bandpass 0.1-40 Hz
    # data_thresh_1000_cbp = np.column_stack([left_channel, right_channel]).copy()
    # for ch in range(2):
    #     cd = data_thresh_1000_cbp[:, ch]
    #     artifacts = np.abs(cd) > 1000
    #     if artifacts.any():
    #         idx = np.where(artifacts)[0]
    #         splits = np.where(np.diff(idx) > 1)[0] + 1
    #         segs = np.split(idx, splits)
    #         for seg in segs:
    #             if len(seg) > 0:
    #                 start, end = seg[0], seg[-1]
    #                 pre = cd[start-1] if start > 0 else 0
    #                 post = cd[end+1] if end+1 < len(cd) else 0
    #                 data_thresh_1000_cbp[seg, ch] = np.linspace(pre, post, len(seg))
    # try:
    #     data_wide_bp = data_thresh_1000_cbp
    #     b, a = butter(4, [1, 2], btype='band', fs=srate)
    #     data_wide_bp = filtfilt(b, a, data_wide_bp, axis=0)
    #     wide_bp_diff = data_wide_bp[:, 0] - data_wide_bp[:, 1]
    #     techniques.append(("3. Wide Artifact Removal + BP (0.5-4 Hz)", wide_bp_diff))
    # except:
    #     techniques.append(("3. Wide Artifact Removal + BP (0.5-4 Hz)", raw_diff))
        
    
    data_thresh_1000_nbp = np.column_stack([left_channel, right_channel]).copy()
    for ch in range(2):
        cd = data_thresh_1000_nbp[:, ch]
        artifacts = np.abs(cd) > 1000
        if artifacts.any():
            idx = np.where(artifacts)[0]
            splits = np.where(np.diff(idx) > 1)[0] + 1
            segs = np.split(idx, splits)
            for seg in segs:
                if len(seg) > 0:
                    start, end = seg[0], seg[-1]
                    pre = cd[start-1] if start > 0 else 0
                    post = cd[end+1] if end+1 < len(cd) else 0
                    data_thresh_1000_nbp[seg, ch] = np.linspace(pre, post, len(seg))
    try:
        data_narrow_bp = data_thresh_1000_nbp
        b, a = butter(4, [1, 2], btype='band', fs=srate)
        data_narrow_bp = filtfilt(b, a, data_narrow_bp, axis=0)
        narrow_bp_diff = data_narrow_bp[:, 0] - data_narrow_bp[:, 1]
        techniques.append(("10. Wide AR (th=1000) + Narrow BP (1-2 Hz)", narrow_bp_diff))
    except:
        techniques.append(("10. Wide AR (th=1000) + Narrow BP (1-2 Hz)", raw_diff))
    
# 9. Median filter only (no bandpass)
    data_thresh_100 = np.column_stack([left_channel, right_channel]).copy()
    for ch in range(2):
        cd = data_thresh_100[:, ch]
        artifacts = np.abs(cd) > 100
        if artifacts.any():
            idx = np.where(artifacts)[0]
            splits = np.where(np.diff(idx) > 1)[0] + 1
            segs = np.split(idx, splits)
            for seg in segs:
                if len(seg) > 0:
                    start, end = seg[0], seg[-1]
                    pre = cd[start-1] if start > 0 else 0
                    post = cd[end+1] if end+1 < len(cd) else 0
                    data_thresh_100[seg, ch] = np.linspace(pre, post, len(seg))

    data_med = np.column_stack([left_channel, right_channel]).copy()
    try:
        data_med = medfilt(data_thresh_100, kernel_size=(5, 1))
    except ValueError:
        pass
    med_diff = data_med[:, 0] - data_med[:, 1]
    techniques.append(("9. Tight AR (th=100) + Medfilt", med_diff))


    # 5. Alternative method with Savitzky-Golay
    alt_data = np.column_stack([left_channel, right_channel]).copy()
    alt_processed = alternative_process_eog_for_plotting(alt_data, srate=srate, 
                                                        low=0.5, high=4.0, 
                                                        clip=500, sg_win=11, sg_ord=3)
    alt_diff = alt_processed[:, 0] - alt_processed[:, 1]
    techniques.append(("5. Alternative (Savitzky-Golay filter + FIR BP)", alt_diff))

    # Double ICA + BP
    try:
        if len(left_channel) > 1000:  # Only try ICA with sufficient data
            data_ica = np.column_stack([left_channel, right_channel]).copy().astype(float)
            # Pre-filter for ICA
            b, a = butter(4, [0.1, 20], btype='band', fs=srate)
            data_pre_ica = filtfilt(b, a, data_ica, axis=0)
            
            # Apply ICA
            ica = FastICA(n_components=2, random_state=42, max_iter=200)
            components = ica.fit_transform(data_pre_ica)
            
            # Identify artifact components using kurtosis
            kurt_values = [kurtosis(components[:, i]) for i in range(2)]
            kurt_mean = np.mean(kurt_values)
            kurt_std = np.std(kurt_values)
            artifact_components = [i for i, k in enumerate(kurt_values) 
                                 if k > kurt_mean + 1*kurt_std]  # Less aggressive threshold
            
            # Reconstruct without artifacts
            clean_components = np.copy(components)
            for artifact_idx in artifact_components:
                clean_components[:, artifact_idx] = 0
            
            reconstructed = np.matmul(clean_components, ica.mixing_.T)
            ica_diff = reconstructed[:, 0] - reconstructed[:, 1]
            techniques.append(("8. Double ICA + BP", ica_diff))
        else:
            techniques.append(("8. Double ICA + BP", raw_diff))
    except Exception as e:
        print(f"ICA failed: {e}")
        techniques.append(("8. Double ICA + BP (Failed)", raw_diff))

    
    # 6. Tight artifact removal (threshold 100)
    # data_thresh_100 = np.column_stack([left_channel, right_channel]).copy()
    # for ch in range(2):
    #     cd = data_thresh_100[:, ch]
    #     artifacts = np.abs(cd) > 100
    #     if artifacts.any():
    #         idx = np.where(artifacts)[0]
    #         splits = np.where(np.diff(idx) > 1)[0] + 1
    #         segs = np.split(idx, splits)
    #         for seg in segs:
    #             if len(seg) > 0:
    #                 start, end = seg[0], seg[-1]
    #                 pre = cd[start-1] if start > 0 else 0
    #                 post = cd[end+1] if end+1 < len(cd) else 0
    #                 data_thresh_100[seg, ch] = np.linspace(pre, post, len(seg))
    # b, a = butter(4, [0.5, 4.0], btype='band', fs=srate)
    # data_thresh_100 = filtfilt(b, a, data_thresh_100, axis=0)
    # thresh_100_diff = data_thresh_100[:, 0] - data_thresh_100[:, 1]
    # techniques.append(("6. Tight Artifact Removal (thresh=100) + 0.5-4 BP", thresh_100_diff))
    
    
    # Create the comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.flatten()
    
    for i, (title, data) in enumerate(techniques):
        ax = axes[i]
        
        # Plot the processed difference signal
        ax.plot(time_vector, data, 'b-', linewidth=1, alpha=0.8, label='EOG Signal (L-R)')
        
        # Add LRLR event highlighting if provided
        lrlr_highlighted = False
        if target_event is not None:
            target_event_int = target_event.astype(int)
            lrlr_transitions = np.diff(target_event_int)
            lrlr_starts = np.where(lrlr_transitions == 1)[0] + 1
            lrlr_ends = np.where(lrlr_transitions == -1)[0] + 1
            
            # Handle edge cases
            if target_event_int[0] == 1:
                lrlr_starts = np.concatenate([[0], lrlr_starts])
            if target_event_int[-1] == 1:
                lrlr_ends = np.concatenate([lrlr_ends, [len(target_event) - 1]])
            
            # Ensure equal number of starts and ends
            min_len = min(len(lrlr_starts), len(lrlr_ends))
            lrlr_starts = lrlr_starts[:min_len]
            lrlr_ends = lrlr_ends[:min_len]
            
            # Highlight LRLR segments
            for j, (start_idx, end_idx) in enumerate(zip(lrlr_starts, lrlr_ends)):
                start_time = time_vector[start_idx]
                end_time = time_vector[end_idx]
                ax.axvspan(start_time, end_time, alpha=0.2, color='red', 
                          label='LRLR Window' if j == 0 else "")
                lrlr_highlighted = True
        
        # Formatting
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Amplitude (µV)', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add legend to each subplot
        if lrlr_highlighted:
            ax.legend(loc='upper right', fontsize=8)
        
        # Only add x-axis labels and ticks to bottom row (indices 4 and 5 in 3x2 grid)
        if i >= 4:  # Bottom row
            ax.set_xlabel('Time (s)', fontsize=9)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        
        # Set y-limits with proper margins to prevent line cutoff
        if np.any(np.isfinite(data)):  # Check if data contains finite values
            data_min = np.min(data[np.isfinite(data)])
            data_max = np.max(data[np.isfinite(data)])
            data_range = data_max - data_min
            margin = data_range * 0.15 if data_range > 0 else 1.0  # 15% margin or 1.0 if flat
            ax.set_ylim([data_min - margin, data_max + margin])
        else:
            ax.set_ylim([-1, 1])  # Default range if no finite data
        
        # Add basic statistics
        rms = np.sqrt(np.mean(data[np.isfinite(data)]**2)) if np.any(np.isfinite(data)) else 0
        ax.text(0.02, 0.96, f'RMS: {rms:.1f}', transform=ax.transAxes, 
               verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # plt.suptitle('EOG Preprocessing Techniques Comparison\nHorizontal Difference Channel (Left - Right)', 
    #              fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Preprocessing comparison saved to: {save_path}")
    
    return fig

if __name__ == "__main__":
    run_analysis()