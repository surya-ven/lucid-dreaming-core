#!/usr/bin/env python3
"""
Data extraction pipeline for creating training data for 1D convolutional neural network REM classifier.

Extracts 15-second time intervals from 20 nights of sleep data (night_01.edf to night_20.edf with corresponding
label files). Applies preprocessing (notch filter and band-pass filter). Labels windows as 1 if the last 2 seconds
are all REM, 0 otherwise. Maximizes positive cases through overlapping windows.

Output file: "extracted_REM_windows.npz"

Author: Benjamin Grayzel
"""

import os
import numpy as np
import pandas as pd
import mne
from glob import glob
from scipy.signal import butter, filtfilt, medfilt
from scipy import signal
from tqdm import tqdm
import warnings

# Suppress MNE log messages for cleaner output
mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')

# Configuration parameters
WINDOW_LENGTH_SEC = 15      # 15-second windows
LABEL_CHECK_SEC = 2         # Check last 2 seconds for REM labeling
OVERLAP_SEC = 1             # 1-second overlap to maximize positive cases
SAMPLING_RATE = 125         # Hz (from EDF files)
NOTCH_FREQ = 60            # 60 Hz powerline noise
NOTCH_Q = 30               # Quality factor for notch filter
BANDPASS_LOW = 0.5         # Low cutoff for bandpass filter
BANDPASS_HIGH = 15         # High cutoff for bandpass filter
FILTER_ORDER = 4           # Filter order
ARTIFACT_THRESHOLD = 150   # Threshold for artifact detection
MED_FILTER_SIZE = 5        # Median filter kernel size

def apply_preprocessing(data, srate=SAMPLING_RATE, 
                       notch_freq=NOTCH_FREQ, notch_q=NOTCH_Q,
                       lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH, 
                       order=FILTER_ORDER, artifact_threshold=ARTIFACT_THRESHOLD,
                       mft_size=MED_FILTER_SIZE):
    """
    Apply preprocessing filters to EEG data.
    
    Args:
        data: EEG data array of shape (n_samples, n_channels)
        srate: Sampling rate
        notch_freq: Notch filter frequency  
        notch_q: Notch filter quality factor
        lowcut: Bandpass low cutoff
        highcut: Bandpass high cutoff
        order: Filter order
        artifact_threshold: Threshold for artifact removal
        mft_size: Median filter size
        
    Returns:
        Preprocessed data array
    """
    data = np.copy(data).astype(np.float32)
    n_samples, n_channels = data.shape
    
    # 1. Artifact removal with linear interpolation
    for ch in range(n_channels):
        artifacts = np.abs(data[:, ch]) > artifact_threshold
        if np.any(artifacts):
            artifact_indices = np.where(artifacts)[0]
            # Split into continuous segments
            segments = np.split(artifact_indices, np.where(np.diff(artifact_indices) > 1)[0] + 1)
            
            for segment in segments:
                if len(segment) > 0:
                    start_idx, end_idx = segment[0], segment[-1]
                    
                    # Get interpolation values
                    if start_idx == 0:
                        non_artifacts = np.where(~artifacts)[0]
                        pre_value = data[non_artifacts[0], ch] if len(non_artifacts) > 0 else 0
                    else:
                        pre_value = data[start_idx - 1, ch]
                        
                    if end_idx == n_samples - 1:
                        non_artifacts = np.where(~artifacts)[0]
                        post_value = data[non_artifacts[-1], ch] if len(non_artifacts) > 0 else 0
                    else:
                        post_value = data[end_idx + 1, ch]
                    
                    # Linear interpolation
                    for i, idx in enumerate(segment):
                        weight = i / len(segment) if len(segment) > 1 else 0
                        data[idx, ch] = pre_value * (1 - weight) + post_value * weight
    
    # 2. Notch filter (60 Hz powerline noise)
    if n_samples > 3 * order:
        b_notch, a_notch = signal.iirnotch(notch_freq, notch_q, srate)
        for ch in range(n_channels):
            data[:, ch] = signal.filtfilt(b_notch, a_notch, data[:, ch])
    
    # 3. Median filter to remove spikes
    if n_samples >= mft_size:
        for ch in range(n_channels):
            data[:, ch] = medfilt(data[:, ch], kernel_size=mft_size)
    
    # 4. Bandpass filter (0.5-15 Hz)
    if n_samples > 3 * order:
        nyquist = srate / 2
        low = lowcut / nyquist
        high = min(highcut / nyquist, 0.99)  # Ensure below Nyquist
        
        if low < high:
            b_band, a_band = butter(order, [low, high], btype='band')
            for ch in range(n_channels):
                data[:, ch] = filtfilt(b_band, a_band, data[:, ch])
    
    return data

def load_edf_and_labels(night_id):
    """
    Load EDF file and corresponding label CSV for a given night.
    
    Args:
        night_id: Night identifier (e.g., "night_01")
        
    Returns:
        Tuple of (eeg_data, label_data, start_timestamp)
    """
    edf_path = f"provided_data/{night_id}.edf"
    label_path = f"provided_data/{night_id}_label.csv"
    
    # Load EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    eeg_data = raw.get_data().T  # Shape: (n_samples, n_channels)
    
    # Load label file  
    labels_df = pd.read_csv(label_path)
    
    # Get start timestamp for synchronization
    start_timestamp = labels_df['Timestamp'].iloc[0]
    
    return eeg_data, labels_df, start_timestamp

def get_label_for_window(labels_df, start_timestamp, window_start_sec, window_end_sec):
    """
    Determine if the last 2 seconds of a window are all REM.
    
    Args:
        labels_df: DataFrame with timestamp and sleep stage labels
        start_timestamp: Recording start timestamp
        window_start_sec: Window start time in seconds from recording start
        window_end_sec: Window end time in seconds from recording start
        
    Returns:
        1 if last 2 seconds are all REM, 0 otherwise
    """
    # Calculate timestamps for the last 2 seconds of the window
    label_start_time = start_timestamp + (window_end_sec - LABEL_CHECK_SEC)
    label_end_time = start_timestamp + window_end_sec
    
    # Find labels that overlap with the last 2 seconds
    relevant_labels = labels_df[
        (labels_df['Timestamp'] >= label_start_time) & 
        (labels_df['Timestamp'] < label_end_time)
    ]
    
    # Check if all relevant labels are REM
    if len(relevant_labels) == 0:
        return 0
    
    return 1 if all(relevant_labels['Sleep stage'] == 'REM') else 0

def extract_windows_from_night(night_id):
    """
    Extract all 15-second windows from a single night of data.
    
    Args:
        night_id: Night identifier
        
    Returns:
        Tuple of (windows_array, labels_array)
    """
    print(f"Processing {night_id}...")
    
    try:
        # Load data
        eeg_data, labels_df, start_timestamp = load_edf_and_labels(night_id)
        
        # Apply preprocessing
        processed_data = apply_preprocessing(eeg_data)
        
        # Calculate window parameters
        window_samples = int(WINDOW_LENGTH_SEC * SAMPLING_RATE)
        overlap_samples = int(OVERLAP_SEC * SAMPLING_RATE)
        step_samples = window_samples - overlap_samples
        
        total_samples = processed_data.shape[0]
        num_windows = (total_samples - window_samples) // step_samples + 1
        
        # Extract windows
        windows = []
        labels = []
        
        for i in range(num_windows):
            start_sample = i * step_samples
            end_sample = start_sample + window_samples
            
            if end_sample <= total_samples:
                # Extract window
                window = processed_data[start_sample:end_sample, :]
                
                # Calculate time bounds
                window_start_sec = start_sample / SAMPLING_RATE
                window_end_sec = end_sample / SAMPLING_RATE
                
                # Get label
                label = get_label_for_window(labels_df, start_timestamp, 
                                           window_start_sec, window_end_sec)
                
                windows.append(window)
                labels.append(label)
        
        windows_array = np.array(windows)  # Shape: (n_windows, window_samples, n_channels)
        labels_array = np.array(labels)
        
        print(f"  {night_id}: {len(windows)} windows extracted, "
              f"{np.sum(labels_array)} REM windows ({np.mean(labels_array)*100:.1f}%)")
        
        return windows_array, labels_array
        
    except Exception as e:
        print(f"  Error processing {night_id}: {e}")
        return np.array([]), np.array([])

def main():
    """
    Main function to extract REM training data from all nights.
    """
    print("Starting REM data extraction pipeline...")
    print(f"Configuration:")
    print(f"  Window length: {WINDOW_LENGTH_SEC}s")
    print(f"  Label check period: {LABEL_CHECK_SEC}s (last seconds of window)")
    print(f"  Overlap: {OVERLAP_SEC}s")
    print(f"  Sampling rate: {SAMPLING_RATE} Hz")
    print(f"  Filters: Notch {NOTCH_FREQ}Hz, Bandpass {BANDPASS_LOW}-{BANDPASS_HIGH}Hz")
    print()
    
    # Initialize storage
    all_windows = []
    all_labels = []
    
    # Process each night
    night_ids = [f"night_{i:02d}" for i in range(1, 21)]  # night_01 to night_20
    
    for night_id in tqdm(night_ids, desc="Processing nights"):
        # Check if files exist
        edf_path = f"provided_data/{night_id}.edf"
        label_path = f"provided_data/{night_id}_label.csv"
        
        if os.path.exists(edf_path) and os.path.exists(label_path):
            windows, labels = extract_windows_from_night(night_id)
            
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.append(labels)
        else:
            print(f"  Warning: Files for {night_id} not found, skipping...")
    
    # Combine all data
    if all_windows:
        combined_windows = np.concatenate(all_windows, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        # Print summary statistics
        total_windows = len(combined_labels)
        rem_windows = np.sum(combined_labels)
        rem_percentage = (rem_windows / total_windows) * 100
        
        print(f"\nExtraction Summary:")
        print(f"  Total windows extracted: {total_windows}")
        print(f"  REM windows: {rem_windows} ({rem_percentage:.1f}%)")
        print(f"  Non-REM windows: {total_windows - rem_windows} ({100-rem_percentage:.1f}%)")
        print(f"  Window shape: {combined_windows.shape}")
        print(f"  Data type: {combined_windows.dtype}")
        
        # Save data
        output_file = "extracted_REM_windows.npz"
        np.savez_compressed(output_file,
                           windows=combined_windows,
                           labels=combined_labels,
                           sampling_rate=SAMPLING_RATE,
                           window_length_sec=WINDOW_LENGTH_SEC,
                           label_check_sec=LABEL_CHECK_SEC,
                           overlap_sec=OVERLAP_SEC)
        
        print(f"\nData saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024**2):.1f} MB")
        
        # Verify saved data
        loaded = np.load(output_file)
        print(f"\nVerification:")
        print(f"  Loaded windows shape: {loaded['windows'].shape}")
        print(f"  Loaded labels shape: {loaded['labels'].shape}")
        print(f"  Loaded sampling rate: {loaded['sampling_rate']}")
        
    else:
        print("\nNo data was successfully extracted!")

if __name__ == "__main__":
    main()