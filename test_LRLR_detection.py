# Author: Benjamin Grayzel

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import butter, filtfilt, medfilt, find_peaks, welch
import time



def load_custom_data(session_folder_path):
    """
    Loads the combined data and metadata from a custom recording session.

    Args:
        session_folder_path (str): Path to the session folder.

    Returns:
        tuple: (data_array, metadata)
               - data_array (np.ndarray): The loaded data, transposed to (samples, channels).
               - metadata (dict): The loaded session_info from metadata.
               Returns (None, None) if loading fails.
    """
    data_filename = "custom_combined_data.dat"
    metadata_filename = "custom_metadata.npz"

    data_filepath = os.path.join(session_folder_path, data_filename)
    metadata_filepath = os.path.join(session_folder_path, metadata_filename)

    if not os.path.exists(metadata_filepath):
        print(f"Error: Metadata file not found: {metadata_filepath}")
        return None, None
    
    session_info_loaded = None
    try:
        metadata_loaded = np.load(metadata_filepath, allow_pickle=True)
        session_info_loaded = metadata_loaded['session_info'].item()
    except Exception as e_meta:
        print(f"Error loading metadata from {metadata_filepath}: {e_meta}")
        return None, None # Cannot proceed without metadata

    if not os.path.exists(data_filepath):
        print(f"Error: Data file not found: {data_filepath}")
        # Return metadata if it was loaded, so user can inspect session_info
        return None, session_info_loaded 

    try:
        num_channels = session_info_loaded['expected_columns'] # Should be 8
        data_type = np.dtype(session_info_loaded['custom_data_type'])
        data_shape_on_save = session_info_loaded.get('data_shape_on_save', 'samples_first') 

        loaded_flat_data = np.fromfile(data_filepath, dtype=data_type)

        if loaded_flat_data.size == 0:
            print("Warning: Data file is empty.")
            return np.array([]).reshape(0, num_channels), session_info_loaded 

        if data_shape_on_save == 'channels_first':
            if loaded_flat_data.size % num_channels == 0:
                num_samples_loaded_total = loaded_flat_data.size // num_channels
                # Reshape as (channels, total_samples_in_file)
                reshaped_data_channels_first = loaded_flat_data.reshape(num_channels, num_samples_loaded_total)
                # Transpose to (total_samples_in_file, channels)
                final_data_samples_first = reshaped_data_channels_first.T 
                return final_data_samples_first, session_info_loaded
            else:
                print(f"Error: Cannot reshape data saved as 'channels_first'. Total elements ({loaded_flat_data.size}) "
                      f"not divisible by num_channels ({num_channels}).")
                return None, session_info_loaded
        else: # Assuming 'samples_first' or old format where num_channels was num_cols
            if loaded_flat_data.size % num_channels == 0: 
                num_samples_loaded = loaded_flat_data.size // num_channels
                reshaped_data = loaded_flat_data.reshape(num_samples_loaded, num_channels)
                return reshaped_data, session_info_loaded
            else:
                print(f"Error: Cannot reshape data saved as 'samples_first'. Total elements ({loaded_flat_data.size}) "
                      f"not divisible by num_columns ({num_channels}).")
                return None, session_info_loaded

    except Exception as e:
        print(f"Error processing data from {data_filepath} or applying metadata: {e}")
        import traceback
        traceback.print_exc()
        # Return metadata if it was loaded, so user can inspect session_info
        return None, session_info_loaded

# --- Configuration for loading ---
# !!! IMPORTANT !!!
# Replace 'YOUR_SESSION_FOLDER_HERE' with the actual name of the 
# subfolder in 'recorded_data/' that was created by the modified 'test_custom_save.py'.
# For example: SESSION_FOLDER_PATH = "recorded_data/20250507_103045_123456"


# <<< PLEASE UPDATE THIS PATH >>> ## TO COPY LRLR_1_time_1 ALERTNESS_3minmark_1
# SESSION_FOLDER_PATH = "recorded_data/LRLR_1_time_1"

# print(f"Attempting to load data from: {SESSION_FOLDER_PATH}")
# loaded_data, session_metadata = load_custom_data(SESSION_FOLDER_PATH)


# # --- Display loaded data and metadata ---
# if session_metadata is not None: 
#     print("\nSession Information (from metadata):")
#     for key, value in session_metadata.items():
#         print(f"  {key}: {value}")

#     if loaded_data is not None:
#         print("\nSuccessfully loaded data.")
#         print(f"Data shape (samples, channels): {loaded_data.shape}")
        
#         if loaded_data.shape[0] > 0: 
#             print("\nFirst 5 rows of loaded data (transposed to samples, channels):")
#             column_names = session_metadata.get('column_names', [f'Ch{i+1}' for i in range(loaded_data.shape[1])])
#             header = " | ".join(column_names)
#             print(header)
#             print("-" * len(header))
#             for row in loaded_data[:5, :]:
#                 print(" | ".join(map(lambda x: f"{x:.3f}" if not np.isnan(x) else "NaN", row)))
#         else:
#             print("\nData loaded, but no samples to display (data shape is 0 rows).")

#         # --- Optional: Example of plotting the first EEG channel ---
#         # if 'column_names' in session_metadata and loaded_data.shape[0] > 0 and loaded_data.shape[1] > 0:
#         #     first_eeg_channel_name = 'EEG_Filt_1' 
#         #     if first_eeg_channel_name in session_metadata['column_names']:
#         #         try:
#         #             eeg_channel_index = session_metadata['column_names'].index(first_eeg_channel_name)
#         #             plt.figure(figsize=(15, 5))
#         #             plt.plot(loaded_data[:, eeg_channel_index]) 
#         #             plt.title(f"Plot of: {session_metadata['column_names'][eeg_channel_index]}")
#         #             plt.xlabel("Sample Index")
#         #             plt.ylabel("Value")
#         #             plt.grid(True)
#         #             plt.show()
#         #         except IndexError:
#         #              print(f"\nSkipping plot: Channel index for '{first_eeg_channel_name}' out of bounds for loaded data shape {loaded_data.shape}.")
#         #     else:
#         #         print(f"\nSkipping plot: Channel '{first_eeg_channel_name}' not found in column names: {session_metadata['column_names']}.")
#         # else:
#         #     print("\nSkipping plot: Conditions not met (column names missing, no samples, or no channels).")

#     else: 
#         print(f"\nFailed to load data array from {SESSION_FOLDER_PATH}, but metadata was available.")
#         print("Please check data file integrity and error messages above.")
# else: 
#     print(f"\nFailed to load any data or metadata from {SESSION_FOLDER_PATH}.")
#     print("Please check the following:")
#     print("1. The 'test_custom_save.py' script has been run successfully to generate data.")
#     print("2. The 'SESSION_FOLDER_PATH' variable in this cell is correctly set to the generated session folder.")
#     print("   (e.g., 'recorded_data/YYYYMMDD_HHMMSS_micros')")
#     print("3. The session folder contains 'custom_combined_data.dat' and 'custom_metadata.npz'.")


# --- Plotting the last 4 channels (EOG) ---
def plot_eeg_eog_data(loaded_data, session_metadata, colstart=0, colend=3):
    if loaded_data is not None and loaded_data.shape[0] > 0 and loaded_data.shape[1] >= colend:
        num_samples = loaded_data.shape[0]
        time_vector = np.arange(num_samples) / 125.0  # Timestep of 1/100th of a second

        # Select the last 4 columns
        data_to_plot = loaded_data[:, colstart:colend]

        # Get column names for the last 4 columns, if available
        column_names = session_metadata.get('column_names', [f'Ch{i+1}' for i in range(loaded_data.shape[1])])
        plot_column_names = column_names[colstart:colend]

        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        if data_to_plot.shape[1] == 1: # Handle case if there's only 1 column to plot (though we expect 4)
            axes = [axes] 

        for i in range(data_to_plot.shape[1]):
            ax = axes[i]
            ax.plot(time_vector, data_to_plot[:, i])
            ax.set_title(f"Plot of: {plot_column_names[i]}")
            ax.set_ylabel("Value")
            ax.set_ylim(-400, 400)  # Set y-axis limits
            ax.grid(True)

        axes[-1].set_xlabel("Time (seconds)")
        plt.tight_layout()
        plt.show()
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



# Pcts Missing
def detect_signal_integrity(eog_data, nchannels, print=False):

    # Check for large jumps in signal
    pcts = []
    for channel in range(nchannels):
        signal = eog_data[:, channel]

        # Check for missing values (NaN or Inf)
        missing_values = np.isnan(signal) | np.isinf(signal)
        missing_percentage = np.mean(missing_values) * 100
        print(f"Channel {channel}: {missing_percentage:.2f}% missing values") if print else None
        pcts.append(missing_percentage)

        # Check for large jumps that might indicate signal discontinuity
        if len(signal) > 1:
            jumps = np.abs(np.diff(signal))
            large_jumps = jumps > 1000  # Threshold for large jumps
            large_jump_percentage = np.mean(large_jumps) * 100
            print(f"Channel {channel}: {large_jump_percentage:.2f}% large jumps") if print else None
    
    
    return sum(pcts)/len(pcts), pcts




# # --- Example of filtering and processing and LRLR checking the loaded data ---
def main():
    srate = 125  # Sampling frequency in Hz
    eog_data = loaded_data[:,4:8]
    print("Length of Data in seconds: ",len(loaded_data)/srate)


    # # 1. Remove 60 Hz noise using a notch filter/Look to see if this is needed
    # f, Pxx = welch(eog_data[:, 0], fs=srate) 

    # 2. Zero‑phase band‑pass (0.5–15 Hz)
    # b, a  = butter(4, [0.5, 15], btype='band', fs=srate)
    # eog_data    = filtfilt(b, a, eog_data, axis=0)

    # 3. Median filter to kill isolated spikes
    # eog_data = medfilt(eog_data, kernel_size=(3,1))


    ## -- TESTING ALGS -- ##

    s = 10
    # LRLR detection & time
    print(f"\nDetecting LRLR patterns in the last {s} seconds of EOG data...")
    start_time = time.time()
    test, count = detect_lrlr_in_window(eog_data, srate, seconds=s, threshold=1, test=False)
    end_time = time.time()
    execution_time = end_time - start_time # Execution time: 0.0018 seconds; 1.8 ms/milliseconds

    print(f"  --LRLR detected: {test}, Count: {count}")
    print(f"  --Execution time: {execution_time:.4f} seconds, {1000*execution_time:.4f} ms")

    # REM detection & time
    print(f"\nDetecting REM patterns in the last {s} seconds of EOG data...")
    start_time = time.time()
    REM_test, REM_count = detect_REM_in_window(eog_data, srate, seconds=s, test=False)
    end_time = time.time()
    REM_execution_time = end_time - start_time

    print(f"  --REM detected: {REM_test}, Count: {REM_count}")
    print(f"  --Execution time: {REM_execution_time:.4f} seconds, {1000*REM_execution_time:.4f} ms")


    # # Plot the channels
    # plot_eeg_eog_data(eog_data, session_metadata, colstart=0, colend=4)


    print(f"LRLR detected: {test}, Count: {count}")

if __name__ == "__main__":
    main()

