from frenztoolkit import Streamer
import time
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
from alertness_detection import compute_alertness_score, calculate_DL_based_alertness_score
from test_custom_save import last_alertness_compute_time

# import matplotlib.pyplot as plt # Uncomment for debugging plots

PRODUCT_KEY = "RUtYA4W3kpXi0i9C7VZCQJY5_GRhm4XL2rKp6cviwQI="
DEVICE_ID = "FRENZI40"

# --- EEG Preprocessing Parameters ---
# !!! IMPORTANT: Based on your terminal output, FS appears to be around 126 Hz.
# (e.g., (225 samples - 99 samples) / (9.25s - 8.24s) approx = 126 samples/sec)
# PLEASE VERIFY THIS with official FRENZ Brainband specs. This is CRITICAL.
FS = 126.0  # Sampling Frequency in Hz (Observed, NEEDS VERIFICATION)

NOTCH_FREQ = 50.0
NOTCH_QUALITY_FACTOR = 30.0
BANDPASS_LOWCUT = 0.5
# Max useful EEG is often around 40-50Hz. FS/2 is Nyquist.
BANDPASS_HIGHCUT = 40.0
# If FS = 126, Nyquist = 63Hz. So 40Hz is fine.
BANDPASS_ORDER = 4  # Lowered order slightly, can be tuned.
ICA_N_COMPONENTS = 4  # We expect 4 EEG channels after selection
ICA_RANDOM_STATE = 42

# Indices of the 4 relevant EEG channels from the 6 columns provided by the SDK
# (assuming columns 2 and 5 are the zero-padded ones as per your description)
EEG_COLUMN_INDICES = [0, 1, 3, 4]

last_alertness_compute_time = 0


# --- Preprocessing Functions (expect data as [num_channels, num_samples]) ---

def apply_notch_filter(data, fs, notch_freq, quality_factor):
    # Data is expected as [num_channels, num_samples]
    if data is None or data.ndim < 2 or data.shape[0] == 0 or data.shape[1] == 0:
        print("Notch Filter: Invalid or empty channel/sample data provided.")
        return data
    filtered_data = np.copy(data)
    # Design filter once
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
    padlen_notch = 3 * max(len(a_notch), len(b_notch))

    for i in range(filtered_data.shape[0]):  # Iterate over channels
        if filtered_data.shape[1] > padlen_notch:
            filtered_data[i, :] = signal.filtfilt(
                b_notch, a_notch, filtered_data[i, :])
        else:
            print(
                f"Notch Filter: Not enough samples ({filtered_data.shape[1]}) in channel {i} for filtfilt (padlen {padlen_notch}). Skipping channel.")
    return filtered_data


def apply_bandpass_filter(data, fs, lowcut, highcut, order):
    # Data is expected as [num_channels, num_samples]
    if data is None or data.ndim < 2 or data.shape[0] == 0 or data.shape[1] == 0:
        print("Band-pass Filter: Invalid or empty channel/sample data provided.")
        return data
    filtered_data = np.copy(data)
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    if high >= 1.0:  # Ensure highcut is less than Nyquist
        print(
            f"Warning: highcut frequency {highcut}Hz is >= Nyquist {nyquist_freq}Hz. Clamping highcut.")
        # Clamp slightly below Nyquist
        high = 0.999 * (nyquist_freq - 0.1) / nyquist_freq
        if high <= low:
            high = low + 0.01  # ensure high > low
    if low <= 0:
        print(
            f"Warning: lowcut frequency {lowcut}Hz is <= 0. Clamping lowcut.")
        low = 0.001  # A very small positive number relative to Nyquist
    if low >= high:
        print(
            f"Warning: Band-pass lowcut ({lowcut}Hz) is >= highcut ({highcut}Hz after adjustments). Skipping bandpass filter.")
        return data

    b_bandpass, a_bandpass = signal.butter(order, [low, high], btype='band')
    padlen_bandpass = 3 * max(len(a_bandpass), len(b_bandpass))

    for i in range(filtered_data.shape[0]):  # Iterate over channels
        if filtered_data.shape[1] > padlen_bandpass:
            filtered_data[i, :] = signal.filtfilt(
                b_bandpass, a_bandpass, filtered_data[i, :])
        else:
            print(
                f"Band-pass Filter: Not enough samples ({filtered_data.shape[1]}) in channel {i} for filtfilt (padlen {padlen_bandpass}). Skipping channel.")
    return filtered_data


def apply_adaptive_filter_stub(data):
    # Data is expected as [num_channels, num_samples]
    print("Skipping adaptive EMG filter (stub function).")
    if data is None or data.ndim < 2 or data.shape[0] == 0 or data.shape[1] == 0:
        return data
    return data


def apply_ica_eog_removal_stub(data, n_components, random_state):
    # Data is expected as [num_channels, num_samples]
    print("Attempting ICA for EOG removal (stub function).")
    if data is None or data.ndim < 2 or data.shape[0] == 0 or data.shape[1] < 2:
        print("ICA: Invalid, empty, or insufficient sample data for ICA.")
        return data

    num_actual_channels = data.shape[0]
    num_samples_available = data.shape[1]

    current_n_components = n_components
    if current_n_components is None or current_n_components > num_actual_channels:
        # Default to number of available channels
        current_n_components = num_actual_channels

    if current_n_components == 0:
        print("ICA: No channels available for ICA (current_n_components is 0). Skipping.")
        return data
    # Heuristic: need at least e.g. 2x samples as components
    if num_samples_available < current_n_components * 2:
        print(
            f"ICA: Not enough samples ({num_samples_available}) for {current_n_components} ICA components. Skipping ICA.")
        return data

    # FastICA expects data as (n_samples, n_features/channels)
    eeg_data_for_ica = data.T

    try:
        ica = FastICA(n_components=current_n_components,
                      random_state=random_state, whiten='unit-variance', max_iter=500)
        print(
            f"ICA: Attempting to fit with {current_n_components} components on data of shape {eeg_data_for_ica.shape}")
        sources = ica.fit_transform(eeg_data_for_ica)
        print(
            f"ICA: Fitted successfully. Identified {sources.shape[1]} sources.")
        # Actual EOG component identification and removal is complex.
        return data  # Placeholder: return original data as removal is complex
    except Exception as e:
        print(f"ICA: Error during processing: {e}")
        return data


# --- Main Streaming and Processing Loop ---
if __name__ == "__main__":
    print(
        f"Attempting to connect to Frenz Brainband: {DEVICE_ID} with FS={FS}Hz")
    streamer = Streamer(
        device_id=DEVICE_ID,
        product_key=PRODUCT_KEY,
        data_folder="./recorded_data"
    )

    try:
        print("Starting EEG streaming session...")
        streamer.start()
        print("Streamer started successfully.")

        while True:
            current_time = time.time()
            if streamer.session_dur > 10*60:  # Run for 10 minutes
                print("Desired session duration reached.")
                break

            # raw_sdk_eeg_data shape is (num_samples, num_raw_columns=6)
            raw_sdk_eeg_data = streamer.DATA["RAW"]["EEG"]
            filtered_sdk_eeg_data = streamer.DATA["FILTERED"]["EEG"]

            # --- Robust check for valid SDK output ---
            # Expects (num_samples, 6_columns_from_sdk)
            min_samples_to_process = 10  # Arbitrary small number of samples to start processing
            if (raw_sdk_eeg_data is None or
                    raw_sdk_eeg_data.ndim < 2 or
                    # Check for expected 6 columns
                    raw_sdk_eeg_data.shape[1] != 6 or
                    raw_sdk_eeg_data.shape[0] < min_samples_to_process):
                print(
                    f"Waiting for valid SDK EEG data (expecting X samples, 6 columns). Current shape: {raw_sdk_eeg_data.shape if raw_sdk_eeg_data is not None else 'None'}. Session time: {streamer.session_dur:.2f}s")
                time.sleep(1)
                continue

            # --- Select relevant channels and transpose ---
            # raw_sdk_eeg_data is (num_samples, 6)
            # We want columns 0, 1, 3, 4 based on user info (indices 2 and 5 are zeros)
            # Shape: (num_samples, 4)
            selected_eeg_data_cols = raw_sdk_eeg_data[:, EEG_COLUMN_INDICES]

            # Transpose to (num_channels, num_samples) for filter functions
            # Shape: (4, num_samples)
            eeg_for_processing = selected_eeg_data_cols.T

            print(
                f"\n--- Processing EEG data at session time: {streamer.session_dur:.2f}s ---")
            print(
                f"SDK Raw EEG shape: {raw_sdk_eeg_data.shape}, Selected & Transposed EEG shape for processing: {eeg_for_processing.shape}")

            # Pass eeg_for_processing (shape: 4, num_samples) to filters
            processed_eeg = eeg_for_processing

            # 1. Notch Filter
            processed_eeg = apply_notch_filter(
                processed_eeg, FS, NOTCH_FREQ, NOTCH_QUALITY_FACTOR)
            if processed_eeg.shape[0] != len(EEG_COLUMN_INDICES) or processed_eeg.shape[1] == 0:
                print(
                    "EEG data became invalid after Notch Filter. Skipping further processing.")
                time.sleep(1)
                continue
            print(f"EEG shape after Notch Filter: {processed_eeg.shape}")

            # 2. Band-pass Filter
            processed_eeg = apply_bandpass_filter(
                processed_eeg, FS, BANDPASS_LOWCUT, BANDPASS_HIGHCUT, BANDPASS_ORDER)
            if processed_eeg.shape[0] != len(EEG_COLUMN_INDICES) or processed_eeg.shape[1] == 0:
                print(
                    "EEG data became invalid after Band-pass Filter. Skipping further processing.")
                time.sleep(1)
                continue
            print(f"EEG shape after Band-pass Filter: {processed_eeg.shape}")

            # 3. Adaptive Filter (Stub)
            processed_eeg = apply_adaptive_filter_stub(processed_eeg)
            # Shape won't change

            # 4. ICA for EOG Removal (Stub)
            # ICA_N_COMPONENTS is set to 4 (number of EEG channels)
            # Heuristic
            if processed_eeg.shape[0] == ICA_N_COMPONENTS and processed_eeg.shape[1] > ICA_N_COMPONENTS * 2:
                processed_eeg = apply_ica_eog_removal_stub(
                    processed_eeg, ICA_N_COMPONENTS, ICA_RANDOM_STATE)
            else:
                print(
                    f"Not enough data or channels for ICA. Shape: {processed_eeg.shape}, ICA components: {ICA_N_COMPONENTS}. Skipping ICA.")

            print(f"Final processed EEG data shape: {processed_eeg.shape}")

            if processed_eeg.shape[0] > 0 and processed_eeg.shape[1] >= 5:
                print(
                    f"Example processed EEG (1st chan, 1st 5 samples): {processed_eeg[0, :5]}")
            elif processed_eeg.shape[0] > 0:
                print(
                    f"Example processed EEG (1st chan, all samples): {processed_eeg[0, :]}")
            else:
                print("No channel data in processed_eeg to display example.")

            # detecting alertness using filtered eeg data
            # data source could be replaced further

            if (current_time - last_alertness_compute_time) >= 5.0:
                if selected_eeg_data_cols.shape[0] > 125 * 30:
                    last_alertness_compute_time = current_time
                    raw_alertness_score, ema_alertness_score, is_not_alert \
                        = calculate_DL_based_alertness_score(selected_eeg_data_cols.transpose())

                    print(
                        f"Raw Latest alertness score: {raw_alertness_score:.2f}")
                    print(
                        f"Smoothed (EMA) alertness score: {ema_alertness_score:.2f}")


            posture = streamer.SCORES.get("posture")
            poas = streamer.SCORES.get("poas")
            sleep_stage = streamer.SCORES.get("sleep_stage")

            print(
                f"Latest POSTURE: {posture}, POAS: {poas}, Sleep Stage: {sleep_stage}")
            time.sleep(5)

    except KeyboardInterrupt:
        print("Session stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping streamer...")
        if 'streamer' in locals() and streamer is not None and streamer.is_streaming:
            streamer.stop()
            print("Streamer stopped. Data should be saved if session ran.")
        else:
            print("Streamer was not active or already stopped.")
