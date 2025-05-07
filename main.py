from frenztoolkit import Streamer
import time
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
# import matplotlib.pyplot as plt # Uncomment for debugging plots

# Please ensure this is your actual product key
PRODUCT_KEY = "RUtYA4W3kpXi0i9C7VZCQJY5_GRhm4XL2rKp6cviwQI="
DEVICE_ID = "FRENZI40"  # Please ensure this is your actual device ID

# --- EEG Preprocessing Parameters ---
FS = 250
NOTCH_FREQ = 50.0
NOTCH_QUALITY_FACTOR = 30.0
BANDPASS_LOWCUT = 0.5
BANDPASS_HIGHCUT = 40.0
BANDPASS_ORDER = 5
ICA_N_COMPONENTS = None
ICA_RANDOM_STATE = 42

# --- Preprocessing Functions (with added robustness for empty channel data) ---


def apply_notch_filter(data, fs, notch_freq, quality_factor):
    if data is None or data.ndim < 2 or data.shape[0] == 0 or data.shape[1] == 0:
        print("Notch Filter: Invalid or empty data provided.")
        return data  # Return original or empty data
    filtered_data = np.copy(data)
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
    for i in range(filtered_data.shape[0]):
        if filtered_data.shape[1] > 0:  # Ensure samples exist for the channel
            # filtfilt requires len(x) > padlen which is 3 * max(len(a), len(b)).
            # For iirnotch, len(a) and len(b) are 3. So padlen is 9.
            if filtered_data.shape[1] > 3 * max(len(a_notch), len(b_notch)):
                filtered_data[i, :] = signal.filtfilt(
                    b_notch, a_notch, filtered_data[i, :])
            else:
                print(
                    f"Notch Filter: Not enough samples ({filtered_data.shape[1]}) in channel {i} for filtfilt. Skipping.")
    return filtered_data


def apply_bandpass_filter(data, fs, lowcut, highcut, order):
    if data is None or data.ndim < 2 or data.shape[0] == 0 or data.shape[1] == 0:
        print("Band-pass Filter: Invalid or empty data provided.")
        return data
    filtered_data = np.copy(data)
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    if high >= 1.0:
        high = 0.99
        print(
            f"Warning: highcut frequency {highcut}Hz is too high for Nyquist {nyquist_freq}Hz. Clamped to {high*nyquist_freq}Hz.")
    if low <= 0:  # Low cut must be > 0 for bandpass
        low = 0.001  # A very small positive number
        print(
            f"Warning: lowcut frequency {lowcut}Hz is too low. Clamped to {low*nyquist_freq}Hz.")
    if low >= high:
        print(
            f"Warning: Band-pass lowcut {lowcut}Hz is >= highcut {highcut}Hz. Skipping bandpass filter.")
        return data

    b_bandpass, a_bandpass = signal.butter(order, [low, high], btype='band')
    for i in range(filtered_data.shape[0]):
        if filtered_data.shape[1] > 0:  # Ensure samples exist for the channel
            # Check padlen for filtfilt for butterworth filter
            padlen = 3 * max(len(a_bandpass), len(b_bandpass))
            if filtered_data.shape[1] > padlen:
                filtered_data[i, :] = signal.filtfilt(
                    b_bandpass, a_bandpass, filtered_data[i, :])
            else:
                print(
                    f"Band-pass Filter: Not enough samples ({filtered_data.shape[1]}) in channel {i} for filtfilt (padlen {padlen}). Skipping.")
    return filtered_data


def apply_adaptive_filter_stub(data, eeg_channels_for_emg_removal=None, emg_reference_channels=None):
    print("Skipping adaptive EMG filter (stub function).")
    if data is None or data.ndim < 2 or data.shape[0] == 0 or data.shape[1] == 0:
        return data
    return data


def apply_ica_eog_removal_stub(data, n_components, random_state, eeg_channel_indices=None):
    print("Attempting ICA for EOG removal (stub function). This requires careful implementation.")
    if data is None or data.ndim < 2 or data.shape[0] == 0 or data.shape[1] < 2:
        print("ICA: Invalid, empty, or insufficient data for ICA.")
        return data

    num_channels_available = data.shape[0]
    num_samples_available = data.shape[1]

    # Determine the actual number of components for ICA
    current_n_components = n_components
    if current_n_components is None or current_n_components > num_channels_available:
        current_n_components = num_channels_available

    if current_n_components == 0:  # No channels to perform ICA on
        print("ICA: No channels available for ICA (current_n_components is 0). Skipping.")
        return data
    if num_samples_available < current_n_components:
        print(
            f"ICA: Not enough samples ({num_samples_available}) for {current_n_components} ICA components. Skipping ICA.")
        return data

    # Transpose data for ICA: (n_samples, n_features/channels)
    eeg_data_for_ica = data.T  # Assuming all channels in 'data' are to be used by default
    if eeg_channel_indices:  # If specific indices are provided
        if not all(idx < num_channels_available for idx in eeg_channel_indices):
            print("ICA: Invalid eeg_channel_indices. Skipping.")
            return data
        eeg_data_for_ica = data[eeg_channel_indices, :].T
        # Update current_n_components if it was based on all channels but now we have a subset
        if n_components is None or n_components > eeg_data_for_ica.shape[1]:
            current_n_components = eeg_data_for_ica.shape[1]
        if current_n_components == 0:
            print("ICA: No channels selected via eeg_channel_indices. Skipping.")
            return data
        if num_samples_available < current_n_components:
            print(
                f"ICA: Not enough samples ({num_samples_available}) for {current_n_components} ICA components with selected channels. Skipping ICA.")
            return data

    try:
        ica = FastICA(n_components=current_n_components,
                      random_state=random_state, whiten='unit-variance', max_iter=500)
        print(
            f"ICA: Attempting to fit with {current_n_components} components on data of shape {eeg_data_for_ica.shape}")
        sources = ica.fit_transform(eeg_data_for_ica)
        print(
            f"ICA: Fitted successfully. Identified {sources.shape[1]} sources.")
        # Actual EOG component identification and removal is complex and not implemented in this stub.
        # Conceptually, you would modify 'sources' and then:
        # cleaned_eeg_data_ica_transformed = ica.inverse_transform(sources_with_eog_removed)
        # if eeg_channel_indices:
        #     # Reconstruct original data shape if a subset of channels was used
        #     reconstructed_data = np.copy(data)
        #     reconstructed_data[eeg_channel_indices, :] = cleaned_eeg_data_ica_transformed.T
        #     return reconstructed_data
        # else:
        #     return cleaned_eeg_data_ica_transformed.T
        return data  # Placeholder: return original data as removal is complex
    except Exception as e:
        print(f"ICA: Error during processing: {e}")
        return data


# --- Main Streaming and Processing Loop ---
if __name__ == "__main__":
    print(f"Attempting to connect to Frenz Brainband: {DEVICE_ID}")
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
            if streamer.session_dur > 10*60:
                print("Desired session duration reached.")
                break

            raw_eeg_data = streamer.DATA["RAW"]["EEG"]

            # --- Robust check for valid EEG data ---
            # e.g. require at least 10 samples
            if raw_eeg_data is None or raw_eeg_data.ndim < 2 or raw_eeg_data.shape[0] == 0 or raw_eeg_data.shape[1] < 10:
                print(
                    f"Waiting for valid EEG data. Current shape: {raw_eeg_data.shape if raw_eeg_data is not None else 'None'}. Session time: {streamer.session_dur:.2f}s")
                time.sleep(1)
                continue

            print(
                f"\n--- Processing EEG data at session time: {streamer.session_dur:.2f}s ---")
            print(f"Raw EEG data shape: {raw_eeg_data.shape}")

            processed_eeg = raw_eeg_data

            # 1. Notch Filter
            processed_eeg = apply_notch_filter(
                processed_eeg, FS, NOTCH_FREQ, NOTCH_QUALITY_FACTOR)
            # Check if still valid
            if processed_eeg.shape[0] > 0 and processed_eeg.shape[1] > 0:
                print(f"EEG shape after Notch Filter: {processed_eeg.shape}")
            else:
                print(
                    "EEG data became invalid after Notch Filter. Skipping further processing in this iteration.")
                time.sleep(1)
                continue

            # 2. Band-pass Filter
            processed_eeg = apply_bandpass_filter(
                processed_eeg, FS, BANDPASS_LOWCUT, BANDPASS_HIGHCUT, BANDPASS_ORDER)
            if processed_eeg.shape[0] > 0 and processed_eeg.shape[1] > 0:
                print(
                    f"EEG shape after Band-pass Filter: {processed_eeg.shape}")
            else:
                print(
                    "EEG data became invalid after Band-pass Filter. Skipping further processing in this iteration.")
                time.sleep(1)
                continue

            # 3. Adaptive Filter (Stub)
            processed_eeg = apply_adaptive_filter_stub(processed_eeg)
            # Shape won't change with stub

            # 4. ICA for EOG Removal (Stub)
            num_eeg_channels = processed_eeg.shape[0]
            ica_n_components_to_use = ICA_N_COMPONENTS if ICA_N_COMPONENTS is not None else num_eeg_channels

            # Ensure enough data and channels for ICA before calling
            # (additional checks are also inside the function)
            # Heuristic: e.g., 2 seconds or twice components
            min_samples_for_ica = max(2 * FS, ica_n_components_to_use * 2)
            if processed_eeg.shape[1] > min_samples_for_ica and num_eeg_channels >= 2 and \
               (ica_n_components_to_use > 0 and num_eeg_channels >= ica_n_components_to_use):
                processed_eeg = apply_ica_eog_removal_stub(
                    processed_eeg, ica_n_components_to_use, ICA_RANDOM_STATE)
            else:
                print(
                    f"Not enough data or channels for ICA. Samples: {processed_eeg.shape[1]}/{min_samples_for_ica}, Channels: {num_eeg_channels}/{ica_n_components_to_use if ica_n_components_to_use > 0 else 2 } needed. Skipping ICA.")
            # Shape might change if ICA was effective and reconstructed. With stub, it won't.

            print(
                f"Final processed EEG data shape to be used for features: {processed_eeg.shape}")

            # --- Guarded print for example data ---
            if processed_eeg.shape[0] > 0 and processed_eeg.shape[1] >= 5:
                print(
                    f"Example of first 5 samples from first channel of processed EEG: {processed_eeg[0, :5]}")
            elif processed_eeg.shape[0] > 0:
                print(
                    f"Example of all available samples from first channel: {processed_eeg[0, :]}")
            else:
                print("No channel data in processed_eeg to display example.")

            posture = streamer.SCORES.get("posture")
            poas = streamer.SCORES.get("poas")
            sleep_stage = streamer.SCORES.get("sleep_stage")

            print(f"Latest POSTURE: {posture}")
            print(f"Latest POAS: {poas}")
            print(f"Latest Sleep Stage: {sleep_stage}")

            time.sleep(5)

    except KeyboardInterrupt:
        print("Session stopped by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
    finally:
        print("Stopping streamer and saving data...")
        if 'streamer' in locals() and streamer is not None and streamer.is_streaming:
            streamer.stop()
            print("Streamer stopped. Data should be saved in './recorded_data' if the session ran and data_folder was specified.")
        else:
            print("Streamer was not active or already stopped.")
