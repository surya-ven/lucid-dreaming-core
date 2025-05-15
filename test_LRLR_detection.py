# Author: Benjamin Grayzel

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import butter, filtfilt, medfilt, find_peaks, welch
from sklearn.decomposition import FastICA
import tensorflow as tf # Added import

import time


LRLR_LSTM_MODEL = None
DEFAULT_LSTM_MODEL_PATH = 'lrlr_lstm_model.keras'
LSTM_SAMPLE_LENGTH = 750

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
    "v2_LR_once_16_mix",
    "v2_LR_once_17_mix",
    "v2_L_once_17_mix",
    "v2_R_once_18_mix",
    "v2_REM_once_19_closed",
    "v2_REM_once_20_closed"
]

ylim = 500



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


# --- Plotting all 4 EOG channels at once with events ---
def plot_eog_events(loaded_data, session_metadata, title = "All Channels"):
    display_column_names = session_metadata.get('processed_column_names', 
                                                [f'Col{i+1}' for i in range(loaded_data.shape[1])])
    if loaded_data is not None:
        plt.figure(figsize=(12, 6))
        for i in range(6, loaded_data.shape[1]):
            col_name = display_column_names[i] if i < len(display_column_names) else f"Col{i+1}"
            plt.plot(loaded_data[:, i], label=col_name, alpha=0.15)

        # plot the target event with a different color for if the value is 0 or 1
        target_event = loaded_data[:, 1]
        # plt.plot(target_event, label="Target Event", color='red', alpha=0.5)
        plt.fill_between(range(len(target_event)), target_event, color='red', where=(target_event == 1), label="Target Event (filled)")
        plt.fill_between(range(len(target_event)), target_event, color='blue', where=(target_event == 0), label="Non-Target Event (filled)")
        # plt.axhline(0, color='black', linewidth=0.5, linestyle='--')

        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        
        plt.show()

def plot_exg_channels(data, colnames=None, target_event=None, ylim=None):
    fig, axes = plt.subplots(data.shape[1], 1, figsize=(15, 10), sharex=True)
    if data.shape[1] == 1: # Handle case if there's only 1 column to plot (though we expect 4)
        axes = [axes] 

    time_vector = np.arange(data.shape[0])  # Timestep of 1/100th of a second
    for i in range(0,data.shape[1]):
        ax = axes[i]
        ax.plot(time_vector, data[:, i])
        if target_event is not None:
            ax.fill_between(range(len(target_event)), target_event, color='darkred', edgecolor='darkred', 
                            linewidth=4,alpha=1, where=(target_event == 1), label="Target Event (filled)")

        title = colnames[i] if colnames is not None else f"Channel {i+1}"

        ax.set_title(title)
        ax.set_ylabel("Value")
        if ylim is not None:
            ax.set_ylim([-ylim, ylim])

        ax.grid(True)
    plt.tight_layout()
    plt.show()



# --- Plotting the last 4 channels (EOG OR EEG) ---
def plot_eeg_eog_data(loaded_data, session_metadata, colstart=0, colend=4, target_event=None, title="All Channels"):
    if loaded_data is not None and loaded_data.shape[0] > 0 and loaded_data.shape[1] >= colend:
        num_samples = loaded_data.shape[0]
        time_vector = np.arange(num_samples)  # Timestep of 1/100th of a second

        # Select the last 4 columns
        data_to_plot = loaded_data[:, colstart:colend]

        # Get column names for the last 4 columns, if available
        column_names = session_metadata.get('column_names', [f'Ch{i+1}' for i in range(loaded_data.shape[1])])
        plot_column_names = column_names[-4:]

        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        if data_to_plot.shape[1] == 1: # Handle case if there's only 1 column to plot (though we expect 4)
            axes = [axes] 

        for i in range(0,4):
            ax = axes[i]
            ax.plot(time_vector, data_to_plot[:, i])
            if target_event is not None:
                ax.fill_between(range(len(target_event)), target_event, color='darkred', edgecolor='darkred', 
                                linewidth=4,alpha=1, where=(target_event == 1), label="Target Event (filled)")


            ax.set_title(f"Plot of: {plot_column_names[i]}")
            ax.set_ylabel("Value")
            # ax.set_ylim([-ylim, ylim])
            ax.grid(True)

        # # Plot 3: LF - RF ## NOTE I DON'T THINK THIS WORKS/HELPS
        # # ------------------------------------------------------
        # ax = axes[2]
        # LH_RH_vector = data_to_plot[:, 0]-data_to_plot[:, 1]
        # ax.plot(time_vector, LH_RH_vector, label="LH-RH")
        # if target_event is not None:
        #     ax.fill_between(range(len(target_event)), target_event, color='darkred', edgecolor='darkred', 
        #                     linewidth=2,alpha=1, where=(target_event == 1), label="Target Event (filled)")

        # ax.set_title(f"Plot of: LH-RH ({title})")
        # ax.set_ylabel("Value")
        # # ax.set_ylim([-ylim, ylim])
        # ax.grid(True)

        # # Plot 4: OTEL + LF - OTER - RF
        # ax = axes[3]
        # diff_vector = data_to_plot[:, 0] + data_to_plot[:, 2] - data_to_plot[:, 1] - data_to_plot[:, 3]
        # ax.plot(time_vector, diff_vector, label="L side - R side")
        # if target_event is not None:
        #     ax.fill_between(range(len(target_event)), target_event, color='darkred', edgecolor='darkred', 
        #                     linewidth=2,alpha=1, where=(target_event == 1), label="Target Event (filled)")

        # ax.set_title(f"Plot of: LH-RH")
        # ax.set_ylabel("Value")
        # ax.grid(True)
        ## ------------------------------------------------------

        axes[-1].set_xlabel("Time (118*seconds)")
        plt.tight_layout()

        # # Save the plot to a PDF file
        # pdf_filename = f"{title}_event_plot.pdf"
        # plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        # print(f"Plot saved to {pdf_filename}")
        # plt.show()

    elif loaded_data is None:
        print("\nSkipping plot: `loaded_data` is None.")
    elif loaded_data.shape[0] == 0:
        print("\nSkipping plot: `loaded_data` has no samples.")
    else: # loaded_data.shape[1] < 4
        print(f"\nSkipping plot: `loaded_data` has fewer than 4 columns (shape: {loaded_data.shape}).")


# --- Plotting a single first channel of EOG data ---
def plot_single_channel_data(data, srate, LRLR=None):
    if data is not None and data.shape[0] > 0:
        num_samples = data.shape[0]
        time_vector = np.arange(num_samples) / srate  # Timestep of 1/100th of a second

        plt.figure(figsize=(15, 5))
        plt.plot(time_vector, data)
        plt.title(f"Plot of channel & window")
        if LRLR is not None:
            plt.suptitle(f"LRLR Pattern: {LRLR}", fontsize=10)
            
        plt.xlabel("Time (seconds)")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()
    else:
        print("\nSkipping plot: `data` is None or has no samples.")


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


# OLD Basic peak detection function for LRLR-like detection. OLD
def detect_lrlr_OLD(signal, srate):
    # Find peaks and troughs
    pos_peaks, _ = find_peaks(signal, height=150, distance=10)
    neg_peaks, _ = find_peaks(-signal, height=150, distance=10)
    print(f"Detected {len(pos_peaks)} positive peaks and {len(neg_peaks)} negative peaks.")

    # Combine and sort peaks
    events = sorted([(i, 'pos') for i in pos_peaks] + [(i, 'neg') for i in neg_peaks])
    
    # Search for neg-pos-neg-pos (or reverse) within ~2 seconds
    lrlr_detected = False
    for i in range(len(events) - 3):
        types = [events[i+j][1] for j in range(4)]
        times = [events[i+j][0] / srate for j in range(4)]
        duration = times[-1] - times[0]
        if duration <= 2.5:
            if types == ['neg', 'pos', 'neg', 'pos'] or types == ['pos', 'neg', 'pos', 'neg']:
                lrlr_detected = True
                lrlr_window = signal[events[i][0]-25:events[i+3][0]+25]
                break

    return lrlr_detected, lrlr_window if lrlr_detected else None


## --- IMPORTANT main function to detect LRLR patterns. ---
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


def detect_REM_in_window(eog_data, srate, seconds, 
                         activity_threshold=3, 
                         peak_height_uv=50, 
                         min_saccade_duration_s=0.05, # Min time from peak1 to peak2 for a saccade
                         max_saccade_duration_s=1.0,  # Max time from peak1 to peak2 for a saccade
                         test=False):
    """
    Detects REM-like activity in the last N seconds of EOG data.
    A REM-like event is defined as a pair of opposite polarity peaks
    occurring within a specified time window.

    Args:
        eog_data: EOG data with channels in columns (expected in microvolts).
        srate: Sampling rate in Hz.
        seconds: Number of seconds of data to check from the end.
        activity_threshold: Minimum number of REM-like movements summed across all channels
                             to classify the window as REM.
        peak_height_uv: Minimum height of individual peaks (µV) to be considered 
                        part of a REM event. Standard AASM criteria for REMs is >= 50-75uV.
        min_saccade_duration_s: Minimum duration (in seconds) between the two opposing peaks 
                                of a saccade-like event.
        max_saccade_duration_s: Maximum duration (in seconds) between the two opposing peaks 
                                of a saccade-like event.
        test: Boolean, if True, prints debug info.

    Returns:
        bool: True if REM-like activity detected above activity_threshold, False otherwise.
        int: Total count of detected REM-like movements across all channels.
    """
    # Get the window from the last N seconds
    window_size = int(seconds * srate)
    if eog_data.ndim == 1: # Handle single channel case
        eog_data = eog_data[:, np.newaxis]
        
    if eog_data.shape[0] < window_size:
        eog_data_window = eog_data
    else:
        eog_data_window = eog_data[-window_size:, :]

    total_rem_like_movements = 0

    # Process each channel
    for channel_idx in range(eog_data_window.shape[1]):
        signal = eog_data_window[:, channel_idx]

        if len(signal) < srate:  # Need at least 1 second of data
            if test: print(f"REM Channel {channel_idx}: Signal too short ({len(signal)} samples).")
            continue
        
        # Basic check for flat signal or excessive NaNs before filtering
        if np.all(signal == signal[0]) or np.isnan(signal).sum() > 0.1 * len(signal):
            if test: print(f"Warning: Skipping REM channel {channel_idx} due to flat signal or excessive NaNs.")
            continue

        # Apply zero-phase bandpass filter (e.g., 0.5-12 Hz for EOG REMs)
        # AASM guidelines often suggest 0.3Hz high-pass, 35Hz low-pass.
        # For detecting saccadic components, 0.5-12 Hz is often effective.
        filtered_signal = signal # Default to original if filtering fails
        if np.all(np.isfinite(signal)) and np.sum(np.abs(np.diff(signal)) > 2000) < len(signal) * 0.10: # Looser discontinuity check
            try:
                b, a = butter(4, [0.5, 12], btype='band', fs=srate)
                filtered_signal = filtfilt(b, a, signal)
            except ValueError as e:
                if test: print(f"Warning: Filtering failed for REM channel {channel_idx}: {e}. Using raw signal.")
                # filtered_signal remains 'signal'
        else:
            if test: print(f"Warning: Skipping filter for REM channel {channel_idx} due to discontinuities or non-finite values.")
            # filtered_signal remains 'signal'

        # Apply median filter to remove sharp spikes (kernel_size typically odd)
        # Make kernel size adaptive to sampling rate, e.g., 30-50ms
        kernel_s = int(0.04 * srate) # e.g. 40ms, results in kernel_size=5 for srate=125Hz
        if kernel_s % 2 == 0: kernel_s += 1 # Ensure odd
        kernel_s = max(3, kernel_s) # Minimum kernel size of 3

        if len(filtered_signal) > kernel_s :
             filtered_signal = medfilt(filtered_signal, kernel_size=kernel_s)
        else:
            if test: print(f"Warning: Signal too short for median filter on REM channel {channel_idx}")
            # Continue with unfiltered or partially filtered signal

        # Find positive and negative peaks
        # Distance: min time between peaks of the same type (e.g., 0.1s to avoid multiple detections on one wave)
        min_peak_dist_samples = int(srate * 0.1) 
        pos_peaks, _ = find_peaks(filtered_signal, height=peak_height_uv, distance=min_peak_dist_samples)
        neg_peaks, _ = find_peaks(-filtered_signal, height=peak_height_uv, distance=min_peak_dist_samples) # height is positive

        # Combine and sort all peaks by time index
        events = []
        for p_idx in pos_peaks: events.append({'index': p_idx, 'type': 'pos', 'value': filtered_signal[p_idx]})
        for n_idx in neg_peaks: events.append({'index': n_idx, 'type': 'neg', 'value': filtered_signal[n_idx]})
        
        events.sort(key=lambda x: x['index'])

        channel_rem_movements = 0
        # Search for neg-pos or pos-neg patterns (simple saccade-like events)
        used_peak_indices = set() # To avoid using the same peak in multiple events

        for i in range(len(events) - 1):
            if events[i]['index'] in used_peak_indices:
                continue

            event1 = events[i]
            
            # Find the next UNUSED event of OPPOSITE polarity
            for j in range(i + 1, len(events)):
                if events[j]['index'] in used_peak_indices:
                    continue
                
                event2 = events[j]
                
                if event1['type'] != event2['type']: # Opposite polarity
                    duration_s = (event2['index'] - event1['index']) / srate
                    
                    if min_saccade_duration_s <= duration_s <= max_saccade_duration_s:
                        channel_rem_movements += 1
                        used_peak_indices.add(event1['index'])
                        used_peak_indices.add(event2['index']) # Mark both peaks as used
                        if test:
                             print(f"  REM Channel {channel_idx}: Found REM-like event: {event1['type']} at {event1['index']/srate:.2f}s ({event1['value']:.1f}uV) -> {event2['type']} at {event2['index']/srate:.2f}s ({event2['value']:.1f}uV), duration {duration_s:.3f}s")
                        break # Move to the event after event1 (or rather, the outer loop will increment i)
                # If not opposite or not in duration, keep searching for a partner for event1
                # If event2 is too far, break inner loop to save computation (implicit in max_saccade_duration_s check)
                if (event2['index'] - event1['index']) / srate > max_saccade_duration_s:
                    break


        total_rem_like_movements += channel_rem_movements
        if test:
            print(f"Channel {channel_idx}: Found {channel_rem_movements} REM-like movements. Peaks: {len(pos_peaks)} pos, {len(neg_peaks)} neg.")
            # Add plotting logic if needed, similar to plot_single_channel_data
            # Example: plot_single_channel_data(filtered_signal, srate=srate, REM_events=channel_rem_movements > 0)

    if test:
        print(f"Total REM-like movements: {total_rem_like_movements}, Threshold: {activity_threshold}")

    return total_rem_like_movements >= activity_threshold, total_rem_like_movements



# Find the pcts of signal missing
def detect_signal_integrity(eog_data, nchannels, print_missing_pcts=False):

    # Check for large jumps in signal
    pcts = []
    for channel in range(nchannels):
        signal = eog_data[:, channel]

        # Check for missing values (NaN or Inf)
        missing_values = np.isnan(signal) | np.isinf(signal)
        missing_percentage = np.mean(missing_values) * 100
        if print_missing_pcts:
            print(f"Channel {channel}: {missing_percentage}% missing values")
        pcts.append(missing_percentage)

        # Check for large jumps that might indicate signal discontinuity
        if len(signal) > 1:
            jumps = np.abs(np.diff(signal))
            large_jumps = jumps > 1000  # Threshold for large jumps
            large_jump_percentage = np.mean(large_jumps) * 100
            print(f"Channel {channel}: {large_jump_percentage}% large jumps") if print_missing_pcts else None
    
    
    return sum(pcts)/len(pcts), pcts





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



# Helper function for preprocessing, adapted from lstm_data_extraction.py
def _preprocess_input_for_lstm(data_segment_raw, model_srate_for_filter=118, mft=5, lowcut=0.5, highcut=15, artifact_threshold=125):
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

def detect_lrlr_window_from_lstm(eog_data, srate, model_path=DEFAULT_LSTM_MODEL_PATH, detection_threshold=0.5):
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
            return False # Cannot proceed without the model

    if eog_data.shape[0] < LSTM_SAMPLE_LENGTH:
        # print(f"Warning: EOG data too short for LSTM ({eog_data.shape[0]} samples, need {LSTM_SAMPLE_LENGTH}).")
        return False
    
    if eog_data.shape[1] < 2:
        print(f"Warning: EOG data has fewer than 2 channels ({eog_data.shape[1]}). LSTM requires 2.")
        return False

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
        prediction_value = LRLR_LSTM_MODEL.predict(model_input, verbose=0)[0][0]
        is_lrlr = prediction_value > detection_threshold
        # print(f"LSTM Prediction: {prediction_value:.4f} -> Detected: {is_lrlr}") # For debugging
        return is_lrlr, prediction_value
    except Exception as e:
        print(f"Error during LSTM prediction: {e}")
        return False, 0.0 # Return a default value for the prediction if an error occurs


def main2():

    l = []

    # Loop through recorded data files
    for filepath in rec_data_names[0:16]:
        SESSION_FOLDER_PATH = f"recorded_data/{filepath}"
        print(f"Attempting to load data from: {SESSION_FOLDER_PATH}")
        loaded_data, session_metadata = load_custom_data(SESSION_FOLDER_PATH)

        if loaded_data is None or session_metadata is None:
            print("Failed to load data or metadata. Exiting.")
            return
        
        display_column_names = session_metadata.get('processed_column_names', 
                                                    [f'Col{i+1}' for i in range(loaded_data.shape[1])])
        print(f"Data shape (samples, columns): {loaded_data.shape}")
        print(f"Columns: {display_column_names}")


        ## Filtering and processing
        eog_data = loaded_data[:2000,-4:]

        eog_data = filter_signal_data(eog_data, srate=118, mft=5, lowcut=0.5, highcut=15, artifact_threshold=75)
        # srate = 118
        # b, a  = butter(4, [0.5, 15], btype='band', fs=srate)
        # eog_data    = filtfilt(b, a, eog_data, axis=0)

        # eog_data = medfilt(eog_data, kernel_size=(71,1))




        
        print(f"  --using cols: {display_column_names[-4:]}")
        l.append(detect_signal_integrity(eog_data, 4, True)[0])
        plot_eeg_eog_data(eog_data, session_metadata, colstart=0, colend=4, target_event=loaded_data[:2000, 1])

    print(f"Average Average % missing values across all channels: {np.mean(l):.2f}%")
    print(f"Average % missing values across all channels: {l}")
    


# # --- Example of filtering and processing and LRLR checking the loaded data ---
def main():

    for filepath in rec_data_names[4:11]:
        SESSION_FOLDER_PATH = f"recorded_data/{filepath}"


        print(f"Attempting to load data from: {SESSION_FOLDER_PATH}")
        loaded_data, session_metadata = load_custom_data(SESSION_FOLDER_PATH)

        if loaded_data is None or session_metadata is None:
            print("Failed to load data or metadata. Exiting.")
            return

        srate = 118  # Sampling frequency in Hz
        eog_data = loaded_data[:,-4:]


        ## -- TESTING ALGS -- ##

        # LRLR detection & time
        print(f"\nDetecting LRLR patterns in the last 750 instances of EOG data...")
        start_time = time.time()
        test, count = detect_lrlr_window_from_lstm(eog_data, srate)
        end_time = time.time()
        execution_time = end_time - start_time # Execution time: 0.0018 seconds; 1.8 ms/milliseconds

        print(f"  --LRLR detected: {test}, Count: {count}")
        print(f"  --Execution time: {execution_time:.4f} seconds, {1000*execution_time:.4f} ms")

        # # REM detection & time
        # print(f"\nDetecting REM patterns in the last {s} seconds of EOG data...")
        # start_time = time.time()
        # REM_test, REM_count = detect_REM_in_window(eog_data, srate, seconds=s, test=False)
        # end_time = time.time()
        # REM_execution_time = end_time - start_time

        # print(f"  --REM detected: {REM_test}, Count: {REM_count}")
        # print(f"  --Execution time: {REM_execution_time:.4f} seconds, {1000*REM_execution_time:.4f} ms")


        # # Plot the channels
        # plot_eeg_eog_data(eog_data, session_metadata, colstart=0, colend=4)

def test1():
    # <<< PLEASE UPDATE THIS PATH >>> ## TO COPY LRLR_1_time_1 ALERTNESS_3minmark_1
    for filepath in rec_data_names[4:11]:
        SESSION_FOLDER_PATH = f"recorded_data/{filepath}"


        loaded_data, session_metadata = load_custom_data(SESSION_FOLDER_PATH)

        if loaded_data is None or session_metadata is None:
            print("Failed to load data or metadata. Exiting.")
            return

        ## Filtering and processing
        eog_data = loaded_data[:2000,-4:]
        display_loaded_data_and_metadata(loaded_data, session_metadata)
        # data = np.concatenate((eeg_data[:,0:2], eog_data[:,0:2]), axis=1)
        # print(f"shape of data: {data.shape}")
        # plot_exg_channels(data, target_event=loaded_data[:2000, 1])


        plot_eeg_eog_data(eog_data, session_metadata, title="pre filter", target_event=loaded_data[:2000, 1])
        plt.show()

        
        eog_filt_data = filter_signal_data(eog_data, srate=118, mft=7, lowcut=0.2, highcut=3, artifact_threshold=75)

        plot_eeg_eog_data(eog_filt_data, session_metadata, title="post filter", target_event=loaded_data[:2000, 1])
        # pdf_filename = f"{filepath}_postprocessing_plot.pdf"
        # plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    main()

