import time
import os

import numpy as np
from scipy.signal import butter, filtfilt, medfilt
import mne
import yasa
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta


# Import real-time optimized REM detector
try:
    from detect_rem_realtime import RealTimeREMDetector, detect_rem_realtime
    REALTIME_AVAILABLE = True
except ImportError:
    print("Real-time optimized detector not available. Install detect_rem_realtime.py")
    REALTIME_AVAILABLE = False


rec_data_names = [
    "v2_LRLR_once_3_mix", ## NOTE I included mix here even though it is eyes open because it is not trained on
    "v2_LRLR_once_4_mix",
    "v2_LRLR_once_5_mix",
    "v2_LRLR_once_6_closed",
    "v2_LRLR_once_7_closed",
    "v2_LRLR_once_8_closed",
    "20250523_183954_966852", 
    "20250523_192852_995272", 
    "20250523_193034_228556", 
    "20250523_193526_401634", 
    "20250523_194915_602917", 
    "20250523_200210_295876", 
    "20250524_015512_029025", 
    "20250524_020422_208890", 
    "20250524_022100_630075", 
    "20250524_033027_138563", 
    "20250524_033637_296315",
]


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


def load_process_eeg_datafile(session_folder_path):
    channel_eeg = 6

    # Load the recorded session from your experiment folder
    memmap_eeg_data = np.memmap(session_folder_path, dtype=np.float64, mode='r')
    l_eeg = int(len(memmap_eeg_data)//channel_eeg)

    # EEG DATA
    # Reshape the data into a 2D array
    # l_eeg is the number of samples in the eeg data
    # channel_eeg is the number of channels in the eeg data
    eeg_data = np.array(memmap_eeg_data)[:l_eeg*channel_eeg].reshape((l_eeg, channel_eeg))
    raw_data = eeg_data[:, [0, 1, 3, 4]].T * 1e-8

    channel_names = ['LF-FpZ', 'OTE_L-FpZ', 'RF-FpZ', 'OTE_R-FpZ']
    sfreq = 125 

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(channel_names)
    )

    raw = mne.io.RawArray(raw_data, info)
    raw.pick_channels(['LF-FpZ','RF-FpZ'])
    
    raw.filter(0.5,40)
    raw.notch_filter(60)
    return raw
    


def detect_rem_window_REALTIME_OPTIMIZED(data_array, srate=250, threshold_mode='balanced', window_duration=15):
    """
    Real-time optimized REM detection for lucid dreaming applications.
    
    This function uses fast signal processing techniques optimized for low-latency
    real-time REM detection, designed specifically for lucid dreaming triggers.
    
    Args:
        data_array (np.ndarray): Raw EEG data array (samples x 2 channels)
                                Expected format: nx2 array with [LF, RF] channels
                                Channel 0: Left frontal (LF-FpZ equivalent)
                                Channel 1: Right frontal (RF-FpZ equivalent)
        srate (int): Sampling rate in Hz (default: 250)
        threshold_mode (str): Detection threshold mode:
            - 'conservative': Higher threshold, fewer false positives
            - 'balanced': Good balance of sensitivity and specificity
            - 'sensitive': Lower threshold, higher sensitivity
        window_duration (int): Analysis window duration in seconds (default: 15)
    
    Returns:
        tuple: (is_rem_detected, rem_score, confidence_info)
            - is_rem_detected (bool): True if REM detected above threshold
            - rem_score (float): Composite REM score (0-1)
            - confidence_info (dict): Additional metrics and processing details
    """
    
    if not REALTIME_AVAILABLE:
        return False, 0.0, {
            'error': 'Real-time optimized detector not available. Install detect_rem_realtime.py',
            'method': 'realtime_optimized'
        }
    
    try:
        # Validate input data format
        if data_array.ndim != 2 or data_array.shape[1] != 2:
            return False, 0.0, {
                'error': f'Expected nx2 data array, got shape {data_array.shape}',
                'method': 'realtime_optimized'
            }
        
        # Use left frontal channel (index 0) as primary, as it typically shows stronger REM signals
        # This corresponds to the original EEG_Filt_1 (LF-FpZ -> Fp1)
        eeg_channel_idx = 0
        eeg_channel_name = "LF_Channel_0"
        channel_priority = "frontal (optimal)"
            
        eeg_data = data_array[:, eeg_channel_idx]
        
        # Check minimum data requirements (more permissive for real-time)
        min_samples = int(5 * srate)  # 5 seconds minimum
        if len(eeg_data) < min_samples:
            return False, 0.0, {
                'error': f'Insufficient data: {len(eeg_data)} samples < {min_samples} required',
                'method': 'realtime_optimized'
            }
        
        # Use real-time optimized detection
        is_rem, rem_score, detailed_info = detect_rem_realtime(
            eeg_data, 
            srate=srate,
            threshold_mode=threshold_mode,
            window_size_sec=min(window_duration, len(eeg_data) / srate)
        )
        
        # Enhance confidence info with channel information
        if 'error' not in detailed_info:
            detailed_info.update({
                'eeg_channel_used': eeg_channel_name,
                'channel_priority': channel_priority,
                'method': 'realtime_optimized',
                'data_duration_seconds': len(eeg_data) / srate,
                'optimization_notes': 'Low-latency processing optimized for lucid dreaming applications'
            })
        
        return is_rem, rem_score, detailed_info
        
    except Exception as e:
        return False, 0.0, {
            'error': f'Real-time REM detection failed: {str(e)}',
            'method': 'realtime_optimized'
        }


def detect_rem_window_FINAL_MODEL(data_array, srate=250, threshold_mode='balanced', window_duration=30):
    """
    Detect REM sleep in a window of EEG data using YASA sleep staging model.
    
    This function uses empirically determined thresholds from analysis of 20 nights
    of sleep data to provide reliable REM detection with controlled false positive rates.
    
    Args:
        data_array (np.ndarray): Raw EEG data array (samples x 2 channels)
                                Expected format: nx2 array with [LF, RF] channels
                                Channel 0: Left frontal (LF-FpZ equivalent)
                                Channel 1: Right frontal (RF-FpZ equivalent)
        srate (int): Sampling rate in Hz (default: 250)
        threshold_mode (str): Detection threshold mode:
            - 'conservative': FPR < 5%, threshold = 0.7786 (30.4% recall, 49.9% precision)
            - 'balanced': FPR < 10%, threshold = 0.5449 (50.8% recall, 45.4% precision)  
            - 'sensitive': 70% confidence, threshold = 0.4382 (59.7% recall, 44.0% precision)
        window_duration (int): Expected window duration in seconds (default: 30)
    
    Returns:
        tuple: (is_rem_detected, rem_probability, confidence_info)
            - is_rem_detected (bool): True if REM detected above threshold
            - rem_probability (float): Raw REM probability from YASA model (0-1)
            - confidence_info (dict): Additional metrics including threshold used and detection strength
    """
    
    # Empirically determined thresholds from 20-night analysis
    THRESHOLDS = {
        'conservative': 0.7786,  # FPR < 5%
        'balanced': 0.5449,      # FPR < 10%  
        'sensitive': 0.4382      # 70% confidence
    }
    
    if threshold_mode not in THRESHOLDS:
        raise ValueError(f"threshold_mode must be one of {list(THRESHOLDS.keys())}")
    
    selected_threshold = THRESHOLDS[threshold_mode]
    
    try:
        # Validate input data format
        if data_array.ndim != 2 or data_array.shape[1] != 2:
            return False, 0.0, {
                'error': f'Expected nx2 data array, got shape {data_array.shape}',
                'threshold_used': selected_threshold,
                'threshold_mode': threshold_mode,
                'method': 'yasa_model'
            }
        
        # Use left frontal channel (index 0) as primary for YASA analysis
        # This corresponds to the original EEG_Filt_1 (LF-FpZ -> Fp1)
        # Channel mapping: 0=LF-FpZ->Fp1 (left frontal), 1=RF-FpZ->Fp2 (right frontal)
        eeg_channel_idx = 0
        eeg_channel_name = "LF_Channel_0"
        yasa_channel_name = 'Fp1'  # Map to standard YASA electrode name
        channel_priority = "frontal (optimal)"
        
        eeg_data = data_array[:, eeg_channel_idx]
        
        # Check if we have enough data for at least one YASA epoch (30 seconds)
        # Note: YASA works with 30-second epochs, but can handle shorter recordings
        # A full REM cycle can be as short as 90 seconds, so we'll be more permissive
        min_samples = int(15 * srate)  # Reduced to 15 seconds minimum for real-time detection
        if len(eeg_data) < min_samples:
            return False, 0.0, {
                'error': f'Insufficient data: {len(eeg_data)} samples < {min_samples} required (need at least 15 seconds)',
                'threshold_used': selected_threshold,
                'threshold_mode': threshold_mode,
                'method': 'yasa_model'
            }
        
        # Note about data duration for interpretation
        data_duration_sec = len(eeg_data) / srate
        duration_note = ""
        if data_duration_sec < 60:
            duration_note = "(short window - consider longer data for better reliability)"
        elif data_duration_sec < 300:  # less than 5 minutes
            duration_note = "(moderate window - YASA recommends 5+ min for best results)"
        
        # Create MNE Raw object
        # YASA expects channels x samples format
        eeg_data_reshaped = eeg_data.reshape(1, -1)
        
        # Create info object with YASA-compatible channel name directly
        ch_names = [yasa_channel_name]  # Use 'Fp1' directly as channel name
        ch_types = ['eeg']
        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
        
        # Create Raw object
        raw = mne.io.RawArray(eeg_data_reshaped, info, verbose=False)
        
        # Apply YASA sleep staging
        sls = yasa.SleepStaging(raw, eeg_name=yasa_channel_name)
        
        # Get predictions and probabilities
        yasa_predictions = sls.predict()
        yasa_proba = sls.predict_proba()
        
        # Extract REM probability
        # YASA outputs probabilities for each epoch (usually 30-second epochs)
        rem_probabilities = yasa_proba['R'].values
        
        # For real-time detection, use the mean REM probability across all epochs
        # or the maximum if we want to be more sensitive to REM periods
        mean_rem_prob = np.mean(rem_probabilities)
        max_rem_prob = np.max(rem_probabilities)
        
        # Use maximum probability for detection (more sensitive to REM periods)
        final_rem_prob = max_rem_prob
        
        # Apply threshold for binary decision
        is_rem_detected = final_rem_prob >= selected_threshold
        
        # Calculate confidence metrics
        detection_strength = 'weak'
        if final_rem_prob >= 0.8:
            detection_strength = 'very_strong'
        elif final_rem_prob >= 0.6:
            detection_strength = 'strong'
        elif final_rem_prob >= 0.4:
            detection_strength = 'moderate'
        elif final_rem_prob >= 0.2:
            detection_strength = 'weak'
        else:
            detection_strength = 'very_weak'
        
        confidence_info = {
            'threshold_used': selected_threshold,
            'threshold_mode': threshold_mode,
            'mean_rem_probability': mean_rem_prob,
            'max_rem_probability': max_rem_prob,
            'detection_strength': detection_strength,
            'num_epochs': len(rem_probabilities),
            'eeg_channel_used': eeg_channel_name,
            'yasa_channel_mapped': yasa_channel_name,
            'channel_priority': channel_priority,
            'data_duration_seconds': len(eeg_data) / srate,
            'duration_note': duration_note,
            'epochs_above_threshold': np.sum(rem_probabilities >= selected_threshold),
            'method': 'yasa_model'
        }
        
        return is_rem_detected, final_rem_prob, confidence_info
        
    except Exception as e:
        return False, 0.0, {
            'error': f'YASA processing failed: {str(e)}',
            'threshold_used': selected_threshold,
            'threshold_mode': threshold_mode,
            'method': 'yasa_model'
        }

def test():
    # Test REM detection on recorded data sessions using both methods
    print("=== Testing REM Detection: YASA Model vs Real-Time Optimized ===")
    
    for filepath in rec_data_names[:2]:  # Test on first 2 sessions for demonstration

        print(f"\n" + "="*80)
        print(f"Processing session: {filepath}")
        print("="*80)

        # Load Data
        SESSION_FOLDER_PATH = f"recorded_data/{filepath}"
        loaded_data, session_metadata = load_custom_data(SESSION_FOLDER_PATH)
        if loaded_data is None or session_metadata is None:
            print("Failed to load data or metadata. Skipping...")
            continue

        ## Display loaded data
        display_loaded_data_and_metadata(loaded_data, session_metadata)

        # Extract column names for display only
        display_column_names = session_metadata.get('processed_column_names', 
                                                [f'Col{i+1}' for i in range(loaded_data.shape[1])])
        
        # Extract sampling rate from session info
        srate = session_metadata.get('sample_rate', 250)  # Default to 250 Hz if not specified
        
        print(f"\n🧠 Running REM Detection Analysis...")
        print(f"Data shape: {loaded_data.shape}")
        print(f"Sampling rate: {srate} Hz")
        print(f"Available channels: {display_column_names}")
        
        # Extract only the EEG channels (assume first 2 channels are LF and RF)
        # For raw EEG data, we expect the format to be [LF, RF] at indices [0, 1]
        if loaded_data.shape[1] < 2:
            print("❌ Error: Need at least 2 EEG channels (LF, RF) for REM detection")
            continue
            
        # Extract the first 2 columns as raw EEG data [LF, RF]
        eeg_data = loaded_data[:, :2]
        
        print(f"📡 Using EEG data shape: {eeg_data.shape} (samples x 2 channels: LF, RF)")
        
        # Test different threshold modes with both methods
        threshold_modes = ['conservative', 'balanced', 'sensitive']
        
        print(f"\n" + "="*50)
        print("🎯 YASA MODEL RESULTS")
        print("="*50)
        
        for mode in threshold_modes:
            print(f"\n🎚️ {mode.upper()} threshold mode (YASA):")
            
            start_time = time.time()
            
            # Run YASA REM detection with raw EEG data
            is_rem, rem_prob, confidence_info = detect_rem_window_FINAL_MODEL(
                eeg_data,  # Now passing nx2 EEG data directly 
                srate=srate,
                threshold_mode=mode
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Print results
            if 'error' in confidence_info:
                print(f"   ❌ Error: {confidence_info['error']}")
            else:
                print(f"   🔍 REM Detected: {'YES' if is_rem else 'NO'}")
                print(f"   📊 REM Probability: {rem_prob:.4f}")
                print(f"   🎚️  Threshold Used: {confidence_info['threshold_used']:.4f}")
                print(f"   💪 Detection Strength: {confidence_info['detection_strength']}")
                print(f"   📏 Data Duration: {confidence_info['data_duration_seconds']:.1f} seconds")
                print(f"   🧮 Epochs Analyzed: {confidence_info['num_epochs']}")
                print(f"   ✅ Epochs Above Threshold: {confidence_info['epochs_above_threshold']}")
                print(f"   📡 EEG Channel: {confidence_info['eeg_channel_used']} -> {confidence_info['yasa_channel_mapped']} ({confidence_info['channel_priority']})")
                print(f"   ⏱️  Processing Time: {execution_time:.4f} seconds")
                
                # Additional analysis
                if confidence_info['num_epochs'] > 0:
                    pct_above_threshold = (confidence_info['epochs_above_threshold'] / confidence_info['num_epochs']) * 100
                    print(f"   📈 Percentage of epochs above threshold: {pct_above_threshold:.1f}%")
        
        print(f"\n" + "="*50)
        print("🚀 REAL-TIME OPTIMIZED RESULTS")
        print("="*50)
        
        for mode in threshold_modes:
            print(f"\n⚡ {mode.upper()} threshold mode (Real-time):")
            
            start_time = time.time()
            
            # Run real-time optimized REM detection with raw EEG data
            is_rem_rt, rem_score_rt, confidence_info_rt = detect_rem_window_REALTIME_OPTIMIZED(
                eeg_data,  # Now passing nx2 EEG data directly
                srate=srate,
                threshold_mode=mode
            )
            
            end_time = time.time()
            execution_time_rt = end_time - start_time
            
            # Print results
            if 'error' in confidence_info_rt:
                print(f"   ❌ Error: {confidence_info_rt['error']}")
            else:
                print(f"   🔍 REM Detected: {'YES' if is_rem_rt else 'NO'}")
                print(f"   📊 REM Score: {rem_score_rt:.4f}")
                print(f"   🎚️  Threshold Used: {confidence_info_rt.get('threshold_used', 'N/A')}")
                print(f"   📏 Data Duration: {confidence_info_rt['data_duration_seconds']:.1f} seconds")
                print(f"   📡 EEG Channel: {confidence_info_rt['eeg_channel_used']} ({confidence_info_rt['channel_priority']})")
                print(f"   ⏱️  Processing Time: {execution_time_rt:.4f} seconds")
                print(f"   🚀 Speedup: {execution_time / execution_time_rt:.1f}x faster than YASA")
                
                # Feature breakdown
                if 'feature_scores' in confidence_info_rt:
                    print(f"   📈 Top Features:")
                    feature_scores = confidence_info_rt['feature_scores']
                    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    for feature, score in sorted_features:
                        print(f"      • {feature}: {score:.3f}")
        
        # Check if target event data is available for comparison
        if 'TargetEvent' in display_column_names:
            target_event_idx = display_column_names.index('TargetEvent')
            target_events = loaded_data[:, target_event_idx]
            has_target_event = np.any(target_events)
            print(f"\n🎯 Target Event Present: {'YES' if has_target_event else 'NO'}")
            if has_target_event:
                print(f"   Target event epochs: {np.sum(target_events)}")

        print(f"\n" + "="*80)

    print(f"\n🏁 REM Detection Testing Complete!")
    print(f"\n📋 METHOD COMPARISON SUMMARY:")
    print(f"   • YASA MODEL:")
    print(f"     - Pros: Clinically validated, empirical thresholds from 20-night analysis")
    print(f"     - Cons: Higher latency (~0.1-1s), requires 30s+ epochs for best results")
    print(f"     - Use case: Offline analysis, clinical validation, research")
    print(f"   • REAL-TIME OPTIMIZED:")
    print(f"     - Pros: Ultra-low latency (~0.001s), sliding window, continuous processing")
    print(f"     - Cons: Less validated, custom thresholds, potentially more false positives")
    print(f"     - Use case: Real-time lucid dreaming triggers, immediate response required")
    print(f"\n💡 RECOMMENDATION FOR LUCID DREAMING:")
    print(f"   • Use REAL-TIME OPTIMIZED for live detection and immediate triggers")
    print(f"   • Use YASA MODEL for validation and post-session analysis")
    print(f"   • Consider hybrid approach: real-time for triggers + YASA for confirmation")

def test2():
    raw = load_process_eeg_datafile("recorded_data/liu_sleep/1748599265.051497/eeg.dat")
    raw.compute_psd().plot()

def test_raw_eeg_detection():
    """
    Test REM detection with raw EEG data format (nx2 array).
    This function demonstrates the new interface for direct EEG data input.
    """
    print("=== Testing REM Detection with Raw EEG Data (nx2 format) ===")
    
    # Example: Load EEG data and convert to nx2 format
    try:
        raw = load_process_eeg_datafile("recorded_data/liu_sleep/1748599265.051497/eeg.dat")
        
        # Convert MNE Raw to numpy array in nx2 format [LF, RF]
        raw_data = raw.get_data().T  # Transpose to get samples x channels
        
        # Ensure we have exactly 2 channels
        if raw_data.shape[1] != 2:
            print(f"❌ Expected 2 channels, got {raw_data.shape[1]}. Selecting first 2 channels.")
            raw_data = raw_data[:, :2]
        
        srate = int(raw.info['sfreq'])
        
        print(f"📊 Raw EEG data shape: {raw_data.shape}")
        print(f"📡 Sampling rate: {srate} Hz")
        print(f"⏱️ Duration: {raw_data.shape[0] / srate:.1f} seconds")
        
        # Test both detection methods
        threshold_modes = ['conservative', 'balanced', 'sensitive']
        
        print(f"\n🎯 YASA MODEL with Raw EEG")
        print("="*40)
        
        for mode in threshold_modes:
            print(f"\n🎚️ {mode.upper()} threshold:")
            
            start_time = time.time()
            is_rem, rem_prob, confidence_info = detect_rem_window_FINAL_MODEL(
                raw_data, 
                srate=srate,
                threshold_mode=mode
            )
            end_time = time.time()
            
            if 'error' in confidence_info:
                print(f"   ❌ Error: {confidence_info['error']}")
            else:
                print(f"   🔍 REM Detected: {'YES' if is_rem else 'NO'}")
                print(f"   📊 REM Probability: {rem_prob:.4f}")
                print(f"   💪 Detection Strength: {confidence_info['detection_strength']}")
                print(f"   ⏱️ Processing Time: {end_time - start_time:.4f} seconds")
        
        print(f"\n🚀 REAL-TIME OPTIMIZED with Raw EEG")
        print("="*40)
        
        for mode in threshold_modes:
            print(f"\n⚡ {mode.upper()} threshold:")
            
            start_time = time.time()
            is_rem_rt, rem_score_rt, confidence_info_rt = detect_rem_window_REALTIME_OPTIMIZED(
                raw_data,
                srate=srate, 
                threshold_mode=mode
            )
            end_time = time.time()
            
            if 'error' in confidence_info_rt:
                print(f"   ❌ Error: {confidence_info_rt['error']}")
            else:
                print(f"   🔍 REM Detected: {'YES' if is_rem_rt else 'NO'}")
                print(f"   📊 REM Score: {rem_score_rt:.4f}")
                print(f"   ⏱️ Processing Time: {end_time - start_time:.4f} seconds")
                
                if 'feature_scores' in confidence_info_rt:
                    print(f"   📈 Top Features:")
                    sorted_features = sorted(confidence_info_rt['feature_scores'].items(), 
                                           key=lambda x: x[1], reverse=True)[:3]
                    for feature, score in sorted_features:
                        print(f"      • {feature}: {score:.3f}")
        
        print(f"\n✅ Raw EEG REM detection test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during raw EEG test: {e}")
        import traceback
        traceback.print_exc()

def plot_rem_detection_timeline(data_array, srate=250, window_size_minutes=5, 
                              overlap_percent=90, threshold_mode='balanced',
                              session_start_time=None, show_plot=True, save_path=None):
    """
    Create a timeline plot of REM detection across the duration of a sleep session.
    
    This function analyzes the entire sleep session using sliding windows and plots
    REM detection probabilities over time using the YASA classifier.
    
    Args:
        data_array (np.ndarray): Raw EEG data array (samples x 2 channels)
                                Expected format: nx2 array with [LF, RF] channels
        srate (int): Sampling rate in Hz (default: 250)
        window_size_minutes (float): Size of analysis windows in minutes (default: 5)
        overlap_percent (float): Overlap between consecutive windows (0-99, default: 90)
        threshold_mode (str): Detection threshold mode ('conservative', 'balanced', 'sensitive')
        session_start_time (datetime or None): Start time of the session for timeline labeling
        show_plot (bool): Whether to display the plot (default: True)
        save_path (str or None): Path to save the plot figure
    
    Returns:
        dict: Analysis results containing:
            - timeline_minutes: Array of time points in minutes
            - rem_probabilities: Array of REM probabilities for each window
            - rem_detections: Boolean array indicating REM detection for each window
            - window_info: Detailed information about each analysis window
            - session_summary: Overall session statistics
    """
    print(f"🕐 Creating REM Detection Timeline...")
    
    # Validate input data format
    if data_array.ndim != 2 or data_array.shape[1] != 2:
        raise ValueError(f'Expected nx2 data array, got shape {data_array.shape}')
    
    # Calculate window parameters
    window_size_samples = int(window_size_minutes * 60 * srate)
    step_size_samples = int(window_size_samples * (1 - overlap_percent/100))
    total_samples = data_array.shape[0]
    session_duration_minutes = total_samples / (srate * 60)
    
    print(f"📊 Session duration: {session_duration_minutes:.1f} minutes")
    print(f"🪟 Window size: {window_size_minutes} minutes ({window_size_samples} samples)")
    print(f"👣 Step size: {step_size_samples} samples ({overlap_percent}% overlap)")
    
    # Initialize results arrays
    timeline_minutes = []
    rem_probabilities = []
    rem_detections = []
    window_info = []
    
    # Process each window
    window_count = 0
    for start_sample in range(0, total_samples - window_size_samples + 1, step_size_samples):
        end_sample = start_sample + window_size_samples
        window_data = data_array[start_sample:end_sample, :]
        
        # Calculate time point for this window (center of window)
        center_sample = start_sample + window_size_samples // 2
        time_minutes = center_sample / (srate * 60)
        
        # Run REM detection on this window
        try:
            is_rem, rem_prob, confidence_info = detect_rem_window_FINAL_MODEL(
                window_data,
                srate=srate,
                threshold_mode=threshold_mode
            )
            
            # Store results
            timeline_minutes.append(time_minutes)
            rem_probabilities.append(rem_prob)
            rem_detections.append(is_rem)
            
            # Store detailed window information
            window_info.append({
                'window_id': window_count,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'time_minutes': time_minutes,
                'rem_probability': rem_prob,
                'rem_detected': is_rem,
                'detection_strength': confidence_info.get('detection_strength', 'unknown'),
                'num_epochs': confidence_info.get('num_epochs', 0),
                'processing_status': 'success'
            })
            
        except Exception as e:
            print(f"⚠️ Error processing window {window_count} at {time_minutes:.1f}min: {e}")
            timeline_minutes.append(time_minutes)
            rem_probabilities.append(0.0)
            rem_detections.append(False)
            
            window_info.append({
                'window_id': window_count,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'time_minutes': time_minutes,
                'rem_probability': 0.0,
                'rem_detected': False,
                'detection_strength': 'error',
                'num_epochs': 0,
                'processing_status': f'error: {e}'
            })
        
        window_count += 1
        
        # Progress indicator - more frequent updates for smaller windows
        if window_count % 5 == 0:
            progress = (start_sample / total_samples) * 100
            print(f"⏳ Processing... {progress:.1f}% complete ({window_count} windows)")
    
    # Convert to numpy arrays
    timeline_minutes = np.array(timeline_minutes)
    rem_probabilities = np.array(rem_probabilities)
    rem_detections = np.array(rem_detections)
    
    # Calculate session summary statistics
    rem_periods = []
    in_rem = False
    rem_start = None
    
    for i, is_rem in enumerate(rem_detections):
        if is_rem and not in_rem:
            # Start of REM period
            in_rem = True
            rem_start = timeline_minutes[i]
        elif not is_rem and in_rem:
            # End of REM period
            in_rem = False
            rem_end = timeline_minutes[i-1]
            rem_periods.append((rem_start, rem_end, rem_end - rem_start))
    
    # Handle case where session ends during REM
    if in_rem:
        rem_end = timeline_minutes[-1]
        rem_periods.append((rem_start, rem_end, rem_end - rem_start))
    
    session_summary = {
        'total_duration_minutes': session_duration_minutes,
        'total_windows_analyzed': len(timeline_minutes),
        'rem_windows_detected': np.sum(rem_detections),
        'rem_percentage': (np.sum(rem_detections) / len(rem_detections)) * 100 if len(rem_detections) > 0 else 0,
        'mean_rem_probability': np.mean(rem_probabilities),
        'max_rem_probability': np.max(rem_probabilities),
        'rem_periods': rem_periods,
        'num_rem_periods': len(rem_periods),
        'total_rem_duration_minutes': sum([duration for _, _, duration in rem_periods]),
        'threshold_mode': threshold_mode
    }
    
    print(f"✅ Timeline analysis complete!")
    print(f"📈 REM Summary:")
    print(f"   • Total REM windows: {session_summary['rem_windows_detected']}/{session_summary['total_windows_analyzed']}")
    print(f"   • REM percentage: {session_summary['rem_percentage']:.1f}%")
    print(f"   • REM periods detected: {session_summary['num_rem_periods']}")
    print(f"   • Total REM duration: {session_summary['total_rem_duration_minutes']:.1f} minutes")
    print(f"   • Mean REM probability: {session_summary['mean_rem_probability']:.3f}")
    
    # Create timeline plot
    if show_plot or save_path:
        create_rem_timeline_plot(
            timeline_minutes, rem_probabilities, rem_detections, 
            session_summary, session_start_time, threshold_mode,
            show_plot, save_path
        )
    
    return {
        'timeline_minutes': timeline_minutes,
        'rem_probabilities': rem_probabilities,
        'rem_detections': rem_detections,
        'window_info': window_info,
        'session_summary': session_summary
    }


def create_rem_timeline_plot(timeline_minutes, rem_probabilities, rem_detections, 
                           session_summary, session_start_time, threshold_mode,
                           show_plot=True, save_path=None):
    """
    Create the actual timeline plot visualization.
    """
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'REM Detection Timeline - {threshold_mode.title()} Threshold', 
                 fontsize=16, fontweight='bold')
    
    # Convert timeline to hours for better readability
    timeline_hours = timeline_minutes / 60
    
    # Top subplot: REM probability over time
    ax1.plot(timeline_hours, rem_probabilities, 'b-', linewidth=1.5, alpha=0.7, label='REM Probability')
    
    # Highlight REM detection periods
    rem_mask = rem_detections
    ax1.fill_between(timeline_hours, 0, rem_probabilities, where=rem_mask, 
                     alpha=0.3, color='red', label='REM Detected')
    
    # Add threshold line based on mode (using actual threshold values)
    threshold_values = {'conservative': 0.7786, 'balanced': 0.5449}  # Updated with actual values
    threshold = threshold_values.get(threshold_mode, 0.5449)
    ax1.axhline(y=threshold, color='orange', linestyle='--', alpha=0.8, 
                label=f'Threshold ({threshold:.3f})')
    
    # Add time markers if session start time is available
    if session_start_time is not None:
        start_dt = datetime.fromtimestamp(session_start_time) if isinstance(session_start_time, (int, float)) else session_start_time
        
        # Define marker times
        marker_times = [
            ("9:10-9:30 AM", 9, 10, 9, 30),  # 9:10-9:30 AM
            ("7:36-7:37 AM", 7, 36, 7, 37)   # 7:36-7:37 AM
        ]
        
        for label, start_hour, start_min, end_hour, end_min in marker_times:
            # Create datetime objects for marker times
            marker_start = start_dt.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
            marker_end = start_dt.replace(hour=end_hour, minute=end_min, second=0, microsecond=0)
            
            # Convert to hours from session start
            start_hours_from_start = (marker_start - start_dt).total_seconds() / 3600
            end_hours_from_start = (marker_end - start_dt).total_seconds() / 3600
            
            # Only plot if within the timeline range
            max_hours = timeline_hours[-1] if len(timeline_hours) > 0 else 0
            if start_hours_from_start >= 0 and start_hours_from_start <= max_hours:
                # Add vertical span for the time period
                for ax in [ax1, ax2]:
                    ax.axvspan(start_hours_from_start, end_hours_from_start, 
                              alpha=0.2, color='green', label=label if ax == ax1 else "")
                    ax.axvline(start_hours_from_start, color='green', linestyle=':', alpha=0.8)
                    if end_hours_from_start != start_hours_from_start:
                        ax.axvline(end_hours_from_start, color='green', linestyle=':', alpha=0.8)
    
    ax1.set_ylabel('REM Probability', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('REM Probability Timeline', fontsize=14)
    
    # Bottom subplot: Binary REM detection
    ax2.fill_between(timeline_hours, 0, rem_detections.astype(int), 
                     alpha=0.6, color='red', step='mid', label='REM Periods')
    ax2.set_ylabel('REM Detected', fontsize=12)
    ax2.set_xlabel('Time (hours from start)', fontsize=12)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('REM Detection Events', fontsize=14)
    
#     # Add session statistics as text
#     stats_text = f"""Session Statistics:
# Duration: {session_summary['total_duration_minutes']:.1f} min
# REM Windows: {session_summary['rem_windows_detected']}/{session_summary['total_windows_analyzed']} ({session_summary['rem_percentage']:.1f}%)
# REM Periods: {session_summary['num_rem_periods']}
# Total REM: {session_summary['total_rem_duration_minutes']:.1f} min
# Mean Prob: {session_summary['mean_rem_probability']:.3f}"""
    
#     ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, 
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Format x-axis with time labels
    if session_start_time is not None:
        # Convert relative hours to actual time
        start_dt = datetime.fromtimestamp(session_start_time) if isinstance(session_start_time, (int, float)) else session_start_time
        # Create hourly ticks for precise labeling
        max_hours = timeline_hours[-1]
        hour_ticks = np.arange(0, max_hours + 0.5, 0.5)  # Every 30 minutes
        time_labels = [start_dt + timedelta(hours=h) for h in hour_ticks]
        
        for ax in [ax1, ax2]:
            ax.set_xticks(hour_ticks)
            ax.set_xticklabels([t.strftime('%H:%M') for t in time_labels], rotation=45)
    else:
        # Add precise hour labels even without start time
        max_hours = timeline_hours[-1]
        hour_ticks = np.arange(0, max_hours + 0.5, 0.5)  # Every 30 minutes
        
        for ax in [ax1, ax2]:
            ax.set_xticks(hour_ticks)
            ax.set_xticklabels([f'{h:.1f}h' for h in hour_ticks], rotation=45)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def extract_session_start_time(filepath):
    """
    Extract session start time from filepath containing Unix timestamp.
    
    Args:
        filepath (str): Path containing Unix timestamp (e.g., "recorded_data/liu_sleep/1748599265.051497/eeg.dat")
    
    Returns:
        datetime or None: Session start time as datetime object, or None if not found
    """
    import re
    
    # Look for Unix timestamp pattern in filepath (10 digits for seconds since epoch)
    timestamp_pattern = r'(\d{10})\.?\d*'
    match = re.search(timestamp_pattern, filepath)
    
    if match:
        unix_timestamp = int(match.group(1))
        try:
            session_start = datetime.fromtimestamp(unix_timestamp)
            print(f"📅 Session start time detected: {session_start.strftime('%Y-%m-%d %H:%M:%S')} (timestamp: {unix_timestamp})")
            
            # Verify this is the expected date/time
            if session_start.year == 2025 and session_start.month == 5 and session_start.day == 30:
                print(f"✅ Confirmed: May 30th, 2025 session at {session_start.strftime('%H:%M:%S')}")
            else:
                print(f"⚠️ Unexpected date: Expected May 30, 2025 but got {session_start.strftime('%Y-%m-%d')}")
            
            return session_start
        except (ValueError, OSError) as e:
            print(f"⚠️ Invalid timestamp {unix_timestamp}: {e}")
            return None
    else:
        print(f"⚠️ No timestamp found in filepath: {filepath}")
        return None

def test_timeline_detection():
    """
    Test the timeline REM detection functionality with sleep session data.
    """
    print("=== Testing REM Detection Timeline ===")
    
    try:
        # Load sleep session data
        sleep_data_path = "recorded_data/liu_sleep/1748599265.051497/eeg.dat"
        print("📂 Loading sleep session data...")
        
        # Extract session start time from filepath
        session_start_time = extract_session_start_time(sleep_data_path)
        
        raw = load_process_eeg_datafile(sleep_data_path)
        
        # Convert MNE Raw to numpy array in nx2 format [LF, RF]
        raw_data = raw.get_data().T  # Transpose to get samples x channels
        
        # Ensure we have exactly 2 channels
        if raw_data.shape[1] != 2:
            print(f"❌ Expected 2 channels, got {raw_data.shape[1]}. Selecting first 2 channels.")
            raw_data = raw_data[:, :2]
        
        # Use the actual sampling rate from the loaded data (should be ~125 Hz)
        srate = int(raw.info['sfreq'])
        session_duration_hours = raw_data.shape[0] / (srate * 3600)
        
        print(f"📊 Loaded sleep session:")
        print(f"   • Data shape: {raw_data.shape}")
        print(f"   • Sampling rate: {srate} Hz (using actual rate from data)")
        print(f"   • Duration: {session_duration_hours:.2f} hours")
        
        if session_start_time:
            session_end_time = session_start_time + timedelta(hours=session_duration_hours)
            print(f"   • Start time: {session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   • End time: {session_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Use the full session data instead of limiting to 2 hours
        print(f"🕐 Analyzing full session duration of {session_duration_hours:.2f} hours...")
        
        # Run timeline analysis with low FPR threshold modes only
        threshold_modes = ['conservative', 'balanced']  # 5% FPR and 10% FPR only
        
        for mode in threshold_modes:
            fpr_label = "5% FPR" if mode == 'conservative' else "10% FPR"
            print(f"\n🎯 Running timeline analysis with {mode} threshold ({fpr_label})...")
            
            # Create timeline analysis with actual session start time and sampling rate
            timeline_results = plot_rem_detection_timeline(
                raw_data,  # Use full dataset instead of subset
                srate=srate,  # Use actual sampling rate from data (~125 Hz)
                window_size_minutes=5,  # 5-minute windows for higher granularity
                overlap_percent=98.33,  # 98.33% overlap for 5-second steps (5min window, step every 5sec)
                threshold_mode=mode,
                session_start_time=session_start_time,  # Pass actual start time
                show_plot=True,
                save_path=f"rem_timeline_{mode}_threshold_{fpr_label.replace('%', 'pct').replace(' ', '_')}.png"
            )
            
            # Print detailed results
            print(f"\n📋 {mode.upper()} Results ({fpr_label}):")
            summary = timeline_results['session_summary']
            print(f"   • Windows analyzed: {summary['total_windows_analyzed']}")
            print(f"   • REM detected: {summary['rem_windows_detected']} windows ({summary['rem_percentage']:.1f}%)")
            print(f"   • REM periods: {summary['num_rem_periods']}")
            print(f"   • Total REM time: {summary['total_rem_duration_minutes']:.1f} minutes")
            
            if summary['rem_periods'] and session_start_time:
                print(f"   • REM period details (actual times):")
                for i, (start_min, end_min, duration) in enumerate(summary['rem_periods']):
                    start_time = session_start_time + timedelta(minutes=start_min)
                    end_time = session_start_time + timedelta(minutes=end_min)
                    print(f"     {i+1}. {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')} ({duration:.1f} min duration)")
            elif summary['rem_periods']:
                print(f"   • REM period details:")
                for i, (start, end, duration) in enumerate(summary['rem_periods']):
                    print(f"     {i+1}. {start:.1f}-{end:.1f} min ({duration:.1f} min duration)")
        
        print(f"\n✅ Timeline detection test completed successfully!")
        print(f"📊 Check the generated PNG files for visual timeline plots with time markers")
        print(f"🕐 Time markers added:")
        print(f"   • 9:10-9:30 AM (green shaded region)")
        print(f"   • 7:36-7:37 AM (green shaded region)")
        
    except Exception as e:
        print(f"❌ Error during timeline test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Uncomment the test you want to run:
    # test()  # Original test with loaded session data
    # test_raw_eeg_detection()  # New test with raw EEG format
    test_timeline_detection()  # Test timeline detection functionality
