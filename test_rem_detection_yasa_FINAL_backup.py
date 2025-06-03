import time
import os

import numpy as np
from scipy.signal import butter, filtfilt, medfilt
import mne
import yasa
import warnings
warnings.filterwarnings('ignore')

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


def detect_rem_window_REALTIME_OPTIMIZED(data_array, column_names, srate=250, threshold_mode='balanced', window_duration=15):
    """
    Real-time optimized REM detection for lucid dreaming applications.
    
    This function uses fast signal processing techniques optimized for low-latency
    real-time REM detection, designed specifically for lucid dreaming triggers.
    
    Args:
        data_array (np.ndarray): EEG data array (samples x channels)
        column_names (list): List of column names corresponding to data_array columns
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
    
    try:
        # Find EEG channels, prioritizing frontal channels (1 and 3)
        eeg_channels = []
        preferred_channels = []
        
        for i, col in enumerate(column_names):
            if 'EEG' in col.upper() and 'FILT' in col.upper():
                eeg_channels.append((i, col))
                # Prioritize frontal channels (EEG_Filt_1 = Fp1, EEG_Filt_3 = Fp2)
                if 'EEG_FILT_1' in col.upper() or 'EEG_FILT_3' in col.upper():
                    preferred_channels.append((i, col))
        
        if len(eeg_channels) == 0:
            # Fallback: look for any EEG-like channels
            for i, col in enumerate(column_names):
                if 'EEG' in col.upper():
                    eeg_channels.append((i, col))
        
        if len(eeg_channels) == 0:
            return False, 0.0, {
                'error': 'No EEG channels found',
                'method': 'realtime_optimized'
            }
        
        # Use preferred frontal channels if available
        if len(preferred_channels) > 0:
            eeg_channel_idx, eeg_channel_name = preferred_channels[0]
            channel_priority = "frontal (optimal)"
        else:
            eeg_channel_idx, eeg_channel_name = eeg_channels[0]
            channel_priority = "fallback"
            
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


def detect_rem_window_REALTIME_OPTIMIZED(data_array, column_names, srate=250, threshold_mode='balanced', window_duration=15):
    """
    Real-time optimized REM detection for lucid dreaming applications.
    
    This function uses fast signal processing techniques optimized for low-latency
    real-time REM detection, designed specifically for lucid dreaming triggers.
    
    Args:
        data_array (np.ndarray): EEG data array (samples x channels)
        column_names (list): List of column names corresponding to data_array columns
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
        # Find EEG channels, prioritizing frontal channels (1 and 3)
        eeg_channels = []
        preferred_channels = []
        
        for i, col in enumerate(column_names):
            if 'EEG' in col.upper() and 'FILT' in col.upper():
                eeg_channels.append((i, col))
                # Prioritize frontal channels (EEG_Filt_1 = Fp1, EEG_Filt_3 = Fp2)
                if 'EEG_FILT_1' in col.upper() or 'EEG_FILT_3' in col.upper():
                    preferred_channels.append((i, col))
        
        if len(eeg_channels) == 0:
            # Fallback: look for any EEG-like channels
            for i, col in enumerate(column_names):
                if 'EEG' in col.upper():
                    eeg_channels.append((i, col))
        
        if len(eeg_channels) == 0:
            return False, 0.0, {
                'error': 'No EEG channels found',
                'method': 'realtime_optimized'
            }
        
        # Use preferred frontal channels if available
        if len(preferred_channels) > 0:
            eeg_channel_idx, eeg_channel_name = preferred_channels[0]
            channel_priority = "frontal (optimal)"
        else:
            eeg_channel_idx, eeg_channel_name = eeg_channels[0]
            channel_priority = "fallback"
            
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


def detect_rem_window_FINAL_MODEL(data_array, column_names, srate=250, threshold_mode='balanced', window_duration=30):
    """
    Detect REM sleep in a window of EEG data using YASA sleep staging model.
    
    This function uses empirically determined thresholds from analysis of 20 nights
    of sleep data to provide reliable REM detection with controlled false positive rates.
    
    Args:
        data_array (np.ndarray): EEG data array (samples x channels)
        column_names (list): List of column names corresponding to data_array columns
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
        # Find EEG channels in the data, prioritizing frontal channels (1 and 3)
        # Channel mapping: 1='LF–FpZ'->Fp1, 2='OTE_L–FpZ'->T3, 3='RF–FpZ'->Fp2, 4='OTE_R–FpZ'->T4
        # Best results from channels 1 and 3 (frontal: Fp1, Fp2)
        
        eeg_channels = []
        preferred_channels = []  # For frontal channels (1, 3)
        
        for i, col in enumerate(column_names):
            if 'EEG' in col.upper() and 'FILT' in col.upper():
                eeg_channels.append((i, col))
                # Prioritize frontal channels (EEG_Filt_1' in col.upper() or 'EEG_FILT_3' in col.upper():
                    preferred_channels.append((i, col))
        
        if len(eeg_channels) == 0:
            # Fallback: look for any EEG-like channels
            for i, col in enumerate(column_names):
                if 'EEG' in col.upper():
                    eeg_channels.append((i, col))
        
        if len(eeg_channels) == 0:
            return False, 0.0, {
                'error': 'No EEG channels found',
                'threshold_used': selected_threshold,
                'threshold_mode': threshold_mode
            }
        
        # Use preferred frontal channels if available, otherwise use first available
        if len(preferred_channels) > 0:
            # Use EEG_Filt_1 (Fp1) as first priority, then EEG_Filt_3 (Fp2)
            eeg_channel_idx, eeg_channel_name = preferred_channels[0]
            channel_priority = "frontal (optimal)"
        else:
            eeg_channel_idx, eeg_channel_name = eeg_channels[0]
            channel_priority = "fallback"
        eeg_data = data_array[:, eeg_channel_idx]
        
        # Check if we have enough data for at least one YASA epoch (30 seconds)
        # Note: YASA works with 30-second epochs, but can handle shorter recordings
        # A full REM cycle can be as short as 90 seconds, so we'll be more permissive
        min_samples = int(15 * srate)  # Reduced to 15 seconds minimum for real-time detection
        if len(eeg_data) < min_samples:
            return False, 0.0, {
                'error': f'Insufficient data: {len(eeg_data)} samples < {min_samples} required (need at least 15 seconds)',
                'threshold_used': selected_threshold,
                'threshold_mode': threshold_mode
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
        
        # Create info object
        ch_names = [eeg_channel_name]
        ch_types = ['eeg']
        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
        
        # Create Raw object
        raw = mne.io.RawArray(eeg_data_reshaped, info, verbose=False)
        
        # Rename channel to match YASA expectations
        # Map to appropriate frontal electrode name based on channel used
        if 'EEG_FILT_1' in eeg_channel_name.upper():
            yasa_channel_name = 'Fp1'  # Left frontal
        elif 'EEG_FILT_3' in eeg_channel_name.upper():
            yasa_channel_name = 'Fp2'  # Right frontal  
        else:
            yasa_channel_name = 'Fp1'  # Default to Fp1 for other channels
            
        raw.rename_channels({eeg_channel_name: yasa_channel_name})
        
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
            'epochs_above_threshold': np.sum(rem_probabilities >= selected_threshold)
        }
        
        return is_rem_detected, final_rem_prob, confidence_info
        
    except Exception as e:
        return False, 0.0, {
            'error': f'YASA processing failed: {str(e)}',
            'threshold_used': selected_threshold,
            'threshold_mode': threshold_mode
        }

    
def test():
    # Test REM detection on recorded data sessions
    print("=== Testing REM Detection with YASA Final Model ===")
    
    for filepath in rec_data_names[:3]:  # Test on first 3 sessions for demonstration

        print(f"\n" + "="*60)
        print(f"Processing session: {filepath}")
        print("="*60)

        # Load Data
        SESSION_FOLDER_PATH = f"recorded_data/{filepath}"
        loaded_data, session_metadata = load_custom_data(SESSION_FOLDER_PATH)
        if loaded_data is None or session_metadata is None:
            print("Failed to load data or metadata. Skipping...")
            continue

        ## Display loaded data
        display_loaded_data_and_metadata(loaded_data, session_metadata)

        # Extract column names
        display_column_names = session_metadata.get('processed_column_names', 
                                                [f'Col{i+1}' for i in range(loaded_data.shape[1])])
        
        # Extract sampling rate from session info
        srate = session_metadata.get('sample_rate', 250)  # Default to 250 Hz if not specified
        
        print(f"\n🧠 Running REM Detection Analysis...")
        print(f"Data shape: {loaded_data.shape}")
        print(f"Sampling rate: {srate} Hz")
        print(f"Available channels: {display_column_names}")
        
        # Test different threshold modes
        threshold_modes = ['conservative', 'balanced', 'sensitive']
        
        for mode in threshold_modes:
            print(f"\n🎯 Testing {mode.upper()} threshold mode:")
            
            start_time = time.time()
            
            # Run REM detection
            is_rem, rem_prob, confidence_info = detect_rem_window_FINAL_MODEL(
                loaded_data, 
                display_column_names, 
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
                print(f"   📡 EEG Channel Used: {confidence_info['eeg_channel_used']} -> {confidence_info['yasa_channel_mapped']} ({confidence_info['channel_priority']})")
                print(f"   ⏱️  Processing Time: {execution_time:.4f} seconds")
                
                # Additional analysis
                if confidence_info['num_epochs'] > 0:
                    pct_above_threshold = (confidence_info['epochs_above_threshold'] / confidence_info['num_epochs']) * 100
                    print(f"   📈 Percentage of epochs above threshold: {pct_above_threshold:.1f}%")
                
        # Check if target event data is available for comparison
        if 'TargetEvent' in display_column_names:
            target_event_idx = display_column_names.index('TargetEvent')
            target_events = loaded_data[:, target_event_idx]
            has_target_event = np.any(target_events)
            print(f"\n🎯 Target Event Present: {'YES' if has_target_event else 'NO'}")
            if has_target_event:
                print(f"   Target event epochs: {np.sum(target_events)}")

        print(f"\n" + "="*60)

    print(f"\n🏁 REM Detection Testing Complete!")
    print(f"\n📋 THRESHOLD SUMMARY:")
    print(f"   • Conservative (FPR < 5%):  Threshold 0.7786 - High precision, lower sensitivity")
    print(f"   • Balanced (FPR < 10%):     Threshold 0.5449 - Good balance of precision/sensitivity")  
    print(f"   • Sensitive (70% conf):     Threshold 0.4382 - Higher sensitivity, more false positives")
    print(f"\n💡 Choose threshold mode based on your application requirements:")
    print(f"   • Clinical/Research: Use 'conservative' for minimal false positives")
    print(f"   • General purpose: Use 'balanced' for good overall performance")
    print(f"   • Real-time detection: Use 'sensitive' for maximum REM capture")


if __name__ == "__main__":
    test()