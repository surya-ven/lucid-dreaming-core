import sys
import select
import argparse
from frenztoolkit import Streamer
import time
import numpy as np
import os
from datetime import datetime
from alertness_detection import compute_alertness_score
from test_LRLR_detection import detect_lrlr_in_window

# --- Configuration ---
PRODUCT_KEY = "RUtYA4W3kpXi0i9C7VZCQJY5_GRhm4XL2rKp6cviwQI="  # Your actual product key
DEVICE_ID = "FRENZI40"  # Your actual device ID
BASE_RECORDING_FOLDER = "./recorded_data"
METADATA_SAVE_INTERVAL_SECONDS = 60  # Save metadata every 60 seconds
EEG_DATA_TYPE = np.float32  # Data type for saving EEG samples
# EEG_filt(4) + EOG_filt(4)
NUM_COMBINED_COLUMNS = 8  # This now represents the number of channels
FS = 126.0  # Sampling Frequency in Hz (Observed, NEEDS VERIFICATION)

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
        "EOG_Filt_1", "EOG_Filt_2", "EOG_Filt_3", "EOG_Filt_4"
    ]
}

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

            if (filtered_eeg_buffer.shape[1] > 2000):
                # Alertness detection
                data = filtered_eeg_buffer.reshape(-1, 4)
                eeg_data = data[:, :4].T
                _, latest_alertness_score = compute_alertness_score(
                    eeg_data, sfreq=FS)
                # print(
                #     f"Latest alertness score: {latest_alertness_score:.2f}"
                # )

                # LRLR Detection
                eog_data = filtered_eog_buffer.reshape(-1, 4)
                test, count = detect_lrlr_in_window(
                    eog_data, FS, seconds=10, threshold=1, test=False)
                # print(f"LRLR Detection - Test: {test}, Count: {count}")

            # Slice the new samples from the EOG buffer: (4, num_new_samples)
            if filtered_eog_buffer is not None and filtered_eog_buffer.ndim == 2 and \
               filtered_eog_buffer.shape[0] == 4 and filtered_eog_buffer.shape[1] >= current_total_samples_in_buffer:
                new_eog_data = filtered_eog_buffer[:,
                                                   samples_written_count:current_total_samples_in_buffer]
            else:
                # If EOG data is missing or inconsistent for the new sample range, fill with NaNs
                new_eog_data = np.full(
                    (4, num_new_samples), np.nan, dtype=EEG_DATA_TYPE)

            # Combine EEG and EOG data by stacking vertically: (8, num_new_samples)
            all_data_to_write = np.vstack([new_eeg_data, new_eog_data])

            current_block_timestamp = time.time()  # Timestamp for this specific data block
            data_block_timestamps.append(current_block_timestamp)
            data_block_sample_counts.append(num_new_samples)

            # print(f"Shape of data to write to .dat file: {all_data_to_write.shape}") # Should be (8, num_new_samples)

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
