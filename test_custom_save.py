import sys
import select
import argparse
from frenztoolkit import Streamer
import time
import numpy as np
import os
import pandas as pd
import joblib
from mne.preprocessing import read_ica
from datetime import datetime
from alertness_detection import compute_alertness_score, calculate_ML_based_alertness_score, save_alertness_score
from test_LRLR_detection import detect_lrlr_window_from_lstm, LSTM_SAMPLE_LENGTH

# --- Configuration ---
PRODUCT_KEY = "RUtYA4W3kpXi0i9C7VZCQJY5_GRhm4XL2rKp6cviwQI="  # Your actual product key
DEVICE_ID = "FRENZI40"  # Your actual device ID
BASE_RECORDING_FOLDER = "./recorded_data"
METADATA_SAVE_INTERVAL_SECONDS = 60  # Save metadata every 60 seconds
EEG_DATA_TYPE = np.float32  # Data type for saving EEG samples
# EEG_filt(4) + EOG_filt(4) + RAW_EEG_SELECTED(4)
# Now 12 columns: 4 filtered EEG, 4 filtered EOG, 4 raw EEG (cols 0,1,3,4)
NUM_COMBINED_COLUMNS = 12
FS = 125.0  # Sampling Frequency in Hz (Observed, NEEDS VERIFICATION)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="FrenzToolkit Custom Data Streamer with Event Logging")
parser.add_argument(
    "--log-events",
    action="store_true",
    help="Enable keyboard input (S/E) for logging target event periods."
)
cli_args = parser.parse_args()

# --- Session Setup ---
session_start_time_obj = datetime.now()
session_timestamp_str = session_start_time_obj.strftime(
    "%Y%m%d_%H%M%S_%f")
session_data_path = os.path.join(BASE_RECORDING_FOLDER, session_timestamp_str)
os.makedirs(session_data_path, exist_ok=True)

# Changed file name to reflect combined data
custom_data_filepath = os.path.join(
    session_data_path, "custom_combined_data.dat")
custom_metadata_filepath = os.path.join(
    session_data_path, "custom_metadata.npz")

print(
    f"Custom streaming session starting. Data will be saved in: {session_data_path}")

alertness_log_df = pd.DataFrame(columns=["Timestamp", "AlertnessScore"])
last_alertness_compute_time = 0
last_LRLR_compute_time = 0
lrlr_result = None

# --- Initialize Streamer ---
streamer = Streamer(
    device_id=DEVICE_ID,
    product_key=PRODUCT_KEY,
    data_folder=BASE_RECORDING_FOLDER
)

# --- In-memory data accumulators ---
samples_written_count = 0
# Timestamps for each main loop iteration (for session_duration_log)
metadata_timestamps = []
metadata_session_dur = []
data_block_timestamps = []  # Timestamps for each block of data written to .dat
data_block_sample_counts = []  # Number of samples in each block written

target_event_active = False
target_event_transitions = []  # List of (timestamp, boolean_state)

# Initial session info for metadata
session_info = {
    "product_key": PRODUCT_KEY,
    "device_id": DEVICE_ID,
    "session_start_iso": session_start_time_obj.isoformat(),
    "custom_data_type": EEG_DATA_TYPE.__name__,
    "expected_columns": NUM_COMBINED_COLUMNS,  # Number of channels (8)
    # Indicates (channels, samples_per_write)
    "data_shape_on_save": "channels_first",
    "column_names": [
        "EEG_Filt_1", "EEG_Filt_2", "EEG_Filt_3", "EEG_Filt_4",
        "EOG_Filt_1", "EOG_Filt_2", "EOG_Filt_3", "EOG_Filt_4",
        "RAW_EEG_1", "RAW_EEG_2", "RAW_EEG_3", "RAW_EEG_4"
    ]
}

ica_model = read_ica("models/ica_artifact_cleaning.fif")
lgb_model = joblib.load("models/alertness_lgbm_light.pkl")

data_file_handle = None
last_metadata_save_time = time.time()

try:
    streamer.start()
    # Open file immediately after streamer starts
    data_file_handle = open(custom_data_filepath, 'ab')
    initial_metadata_saved = False

    if cli_args.log_events:
        print("\nEvent logging enabled. Press 'S' then Enter to START an event, 'E' then Enter to END an event.")
        print("You can do this multiple times during the session.\n")

    while True:
        current_time = time.time()
        session_duration_seconds = streamer.session_dur

        if session_duration_seconds > 10 * 60:  # Example: run for 10 minutes
            break

        if cli_args.log_events:
            # Non-blocking check for keyboard input (0.01s timeout)
            # User needs to press Enter after S or E
            if select.select([sys.stdin], [], [], 0.01)[0]:
                user_input = sys.stdin.readline().strip().upper()
                if user_input == 'S':
                    if not target_event_active:
                        target_event_active = True
                        ts = time.time()
                        target_event_transitions.append((ts, True))
                        print(f"--- EVENT STARTED at {ts} ---")
                elif user_input == 'E':
                    if target_event_active:
                        target_event_active = False
                        ts = time.time()
                        target_event_transitions.append((ts, False))
                        print(f"--- EVENT ENDED at {ts} ---")

        # --- Data Handling ---
        # streamer.DATA["FILTERED"]["EEG"] is expected to be (4, N_total_accumulated_samples)
        # streamer.DATA["FILTERED"]["EOG"] is expected to be (4, N_total_accumulated_samples)
        filtered_eeg_buffer = streamer.DATA["FILTERED"]["EEG"]
        filtered_eog_buffer = streamer.DATA["FILTERED"]["EOG"]
        # streamer.DATA["RAW"]["EEG"] is expected to be (N_total_accumulated_samples, 6)
        raw_eeg_buffer = streamer.DATA["RAW"]["EEG"]

        # detecting alertness using filtered eeg data
        # data source could be replaced further

        # Scores are no longer collected or saved

        # Determine the total number of samples currently available in the buffer (per channel)
        current_total_samples_in_buffer = 0
        if filtered_eeg_buffer is not None and filtered_eeg_buffer.ndim == 2 and filtered_eeg_buffer.shape[0] == 4:
            current_total_samples_in_buffer = filtered_eeg_buffer.shape[1]
        else:
            # If EEG buffer is not as expected, assume no new valid samples can be determined from it.
            # samples_written_count will remain unchanged, so num_new_samples will be <= 0.
            current_total_samples_in_buffer = samples_written_count

        num_new_samples = current_total_samples_in_buffer - samples_written_count

        if num_new_samples > 0:

            # Slice the new samples from the EEG buffer: (4, num_new_samples)
            new_eeg_data = filtered_eeg_buffer[:,
                                               samples_written_count:current_total_samples_in_buffer]

            # Slice the new samples from the EOG buffer: (4, num_new_samples)
            if filtered_eog_buffer is not None and filtered_eog_buffer.ndim == 2 and \
               filtered_eog_buffer.shape[0] == 4 and filtered_eog_buffer.shape[1] >= current_total_samples_in_buffer:
                new_eog_data = filtered_eog_buffer[:,
                                                   samples_written_count:current_total_samples_in_buffer]
                
                if (current_time - last_LRLR_compute_time) >= 1.0:
                    if (filtered_eog_buffer.shape[1] > LSTM_SAMPLE_LENGTH):
                        # Detect LRLR in the new EOG data
                        lrlr_result = detect_lrlr_window_from_lstm(
                            filtered_eog_buffer.T, srate=118)
                        if lrlr_result is not None:
                            test, count = lrlr_result

                        last_LRLR_compute_time = current_time
                
            else:
                new_eog_data = np.full(
                    (4, num_new_samples), np.nan, dtype=EEG_DATA_TYPE)

            # Slice the new samples from the RAW EEG buffer: (num_new_samples, 6)
            # Save columns [0, 1, 3, 4] as (4, num_new_samples)
            if raw_eeg_buffer is not None and raw_eeg_buffer.ndim == 2 and raw_eeg_buffer.shape[1] >= 5:
                # Only take the last num_new_samples rows
                new_raw_eeg_data = raw_eeg_buffer[-num_new_samples:,
                                                  [0, 1, 3, 4]].T
                
                # Detect alertness using the latest raw EEG data
                raw_eeg_data = raw_eeg_buffer[:, [0, 1, 3, 4]].T
                if (current_time - last_alertness_compute_time) >= 1.0:
                    if (raw_eeg_buffer.shape[0] > 500):
                        # make sure data is in shape of (4, N)
                        latest_alertness_score = calculate_ML_based_alertness_score(
                            raw_eeg_data, ica_model, lgb_model)

                        save_alertness_score(
                            alertness_log_df, latest_alertness_score)

                        smoothed_score = alertness_log_df["AlertnessScore_EMA"].iloc[-1]
                        print(
                            f"Raw Latest alertness score: {latest_alertness_score:.2f}"
                        )
                        print(
                            f"Smoothed (EMA) alertness score: {smoothed_score:.2f}")

                        last_alertness_compute_time = current_time
            else:
                new_raw_eeg_data = np.full(
                    (4, num_new_samples), np.nan, dtype=EEG_DATA_TYPE)
            
            if lrlr_result is not None:
                print(f"LRLR detection result: {test}, count: {count}")

            # Combine EEG, EOG, and raw EEG data by stacking vertically: (8+4=12, num_new_samples)
            all_data_to_write = np.vstack(
                [new_eeg_data, new_eog_data, new_raw_eeg_data])

            current_block_timestamp = time.time()  # Timestamp for this specific data block
            data_block_timestamps.append(current_block_timestamp)
            data_block_sample_counts.append(num_new_samples)

            # Write to .dat file
            data_file_handle.write(
                all_data_to_write.astype(EEG_DATA_TYPE).tobytes())
            # Update samples_written_count to the new total number of samples processed (per channel)
            samples_written_count = current_total_samples_in_buffer

        # --- Metadata Collection (timestamps and session duration) ---
        metadata_timestamps.append(current_time)
        metadata_session_dur.append(session_duration_seconds)

        # --- Periodic Metadata Save ---
        if not initial_metadata_saved or (current_time - last_metadata_save_time >= METADATA_SAVE_INTERVAL_SECONDS):
            try:
                save_payload = {
                    "session_info": session_info,
                    "timestamps": np.array(metadata_timestamps),
                    "session_duration_log": np.array(metadata_session_dur),
                    "data_block_timestamps": np.array(data_block_timestamps),
                    "data_block_sample_counts": np.array(data_block_sample_counts)
                }
                if cli_args.log_events and target_event_transitions:
                    save_payload["target_event_transitions"] = np.array(
                        target_event_transitions, dtype=object)

                np.savez_compressed(custom_metadata_filepath, **save_payload)
                last_metadata_save_time = current_time
                initial_metadata_saved = True
            except Exception as e_save:
                pass  # Silently pass for now as per "remove all print lines"

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected. Finalizing...")
    pass  # Silently pass
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if data_file_handle is not None:
        data_file_handle.close()

    # Final metadata save
    try:
        if metadata_timestamps:
            final_save_payload = {
                "session_info": session_info,
                "timestamps": np.array(metadata_timestamps),
                "session_duration_log": np.array(metadata_session_dur),
                "data_block_timestamps": np.array(data_block_timestamps),
                "data_block_sample_counts": np.array(data_block_sample_counts)
            }
            if cli_args.log_events and target_event_transitions:
                final_save_payload["target_event_transitions"] = np.array(
                    target_event_transitions, dtype=object)

            np.savez_compressed(custom_metadata_filepath, **final_save_payload)
    except Exception as e_save_final:
        print(f"Error during final metadata save: {e_save_final}")
        pass

    print(
        f"Custom streaming session ended. Data saved in: {session_data_path}")
