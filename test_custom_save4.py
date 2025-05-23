import sys
import select
import argparse
from frenztoolkit import Streamer
import time
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
PRODUCT_KEY = "RUtYA4W3kpXi0i9C7VZCQJY5_GRhm4XL2rKp6cviwQI="  # Your actual product key
DEVICE_ID = "FRENZI40"  # Your actual device ID
BASE_RECORDING_FOLDER = "./recorded_data"
METADATA_SAVE_INTERVAL_SECONDS = 60  # Save metadata every 60 seconds
EEG_DATA_TYPE = np.float32  # Data type for saving EEG samples
# EEG_filt(4) + EOG_filt(4) + RAW_EEG_SELECTED(4)
NUM_COMBINED_COLUMNS = 12 # 4 filtered EEG, 4 filtered EOG, 4 raw EEG (cols 0,1,3,4)
FS = 125.0  # Sampling Frequency in Hz (Assumed, verify if critical)

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
session_timestamp_str = session_start_time_obj.strftime("%Y%m%d_%H%M%S_%f")
session_data_path = os.path.join(BASE_RECORDING_FOLDER, session_timestamp_str)
os.makedirs(session_data_path, exist_ok=True)

custom_data_filepath = os.path.join(session_data_path, "custom_combined_data.dat")
custom_metadata_filepath = os.path.join(session_data_path, "custom_metadata.npz")

print(f"Custom streaming session starting. Data will be saved in: {session_data_path}")

# --- Initialize Streamer ---
streamer = Streamer(
    device_id=DEVICE_ID,
    product_key=PRODUCT_KEY,
    data_folder=BASE_RECORDING_FOLDER # Streamer might use this for its own logs/cache
)

# --- In-memory data accumulators ---
samples_written_count = 0
metadata_timestamps = [] # Timestamps for each main loop iteration (for session_duration_log)
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
    "expected_columns": NUM_COMBINED_COLUMNS,
    "data_shape_on_save": "channels_first", # Indicates (channels, samples_per_write)
    "sampling_frequency_hz": FS,
    "column_names": [
        "EEG_Filt_1", "EEG_Filt_2", "EEG_Filt_3", "EEG_Filt_4",
        "EOG_Filt_1", "EOG_Filt_2", "EOG_Filt_3", "EOG_Filt_4",
        "RAW_EEG_1", "RAW_EEG_2", "RAW_EEG_3", "RAW_EEG_4" # Corresponding to original cols 0,1,3,4
    ]
}

data_file_handle = None
last_metadata_save_time = time.time()

try:
    streamer.start()
    data_file_handle = open(custom_data_filepath, 'ab') # Open file in append binary mode
    initial_metadata_saved = False

    if cli_args.log_events:
        print("\nEvent logging enabled. Press 'S' then Enter to START an event, 'E' then Enter to END an event.")
        print("You can do this multiple times during the session.\n")

    while True:
        current_time = time.time()
        session_duration_seconds = streamer.session_dur

        # Example: run for a predefined duration, e.g., 60 minutes
        if session_duration_seconds > 60 * 60:
            print("Session duration limit reached.")
            break

        if cli_args.log_events:
            # Non-blocking check for keyboard input
            if select.select([sys.stdin], [], [], 0.01)[0]:
                user_input = sys.stdin.readline().strip().upper()
                if user_input == 'S':
                    if not target_event_active:
                        target_event_active = True
                        ts = time.time() # Use current time for event timestamp
                        target_event_transitions.append((ts, True))
                        print(f"--- EVENT STARTED at {datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')} ---")
                elif user_input == 'E':
                    if target_event_active:
                        target_event_active = False
                        ts = time.time()
                        target_event_transitions.append((ts, False))
                        print(f"--- EVENT ENDED at {datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')} ---")

        # --- Data Handling ---
        filtered_eeg_buffer = streamer.DATA["FILTERED"]["EEG"] # Expected: (4, N_total_samples)
        filtered_eog_buffer = streamer.DATA["FILTERED"]["EOG"] # Expected: (4, N_total_samples)
        raw_eeg_buffer = streamer.DATA["RAW"]["EEG"]           # Expected: (N_total_samples, 6)

        current_total_samples_in_buffer = 0
        if filtered_eeg_buffer is not None and filtered_eeg_buffer.ndim == 2 and filtered_eeg_buffer.shape[0] == 4:
            current_total_samples_in_buffer = filtered_eeg_buffer.shape[1]
        else:
            # Fallback if filtered EEG buffer is not as expected
            current_total_samples_in_buffer = samples_written_count

        num_new_samples = current_total_samples_in_buffer - samples_written_count

        if num_new_samples > 0:
            # Slice new filtered EEG data
            new_eeg_data = filtered_eeg_buffer[:, samples_written_count:current_total_samples_in_buffer]

            # Slice new filtered EOG data
            if filtered_eog_buffer is not None and filtered_eog_buffer.ndim == 2 and \
               filtered_eog_buffer.shape[0] == 4 and filtered_eog_buffer.shape[1] >= current_total_samples_in_buffer:
                new_eog_data = filtered_eog_buffer[:, samples_written_count:current_total_samples_in_buffer]
            else:
                # Fill with NaNs if EOG data is missing or malformed for this block
                new_eog_data = np.full((4, num_new_samples), np.nan, dtype=EEG_DATA_TYPE)

            # Slice new raw EEG data (specific columns)
            if raw_eeg_buffer is not None and raw_eeg_buffer.ndim == 2 and raw_eeg_buffer.shape[1] >= 5 and \
               raw_eeg_buffer.shape[0] >= current_total_samples_in_buffer: # Ensure enough rows
                # Select the relevant segment of rows, then specific columns
                # Raw buffer is (total_samples, 6), we need the last num_new_samples
                new_raw_eeg_data_segment = raw_eeg_buffer[samples_written_count:current_total_samples_in_buffer, :]
                new_raw_eeg_data = new_raw_eeg_data_segment[:, [0, 1, 3, 4]].T # Transpose to (4, num_new_samples)
            else:
                # Fill with NaNs if raw EEG data is missing or malformed
                new_raw_eeg_data = np.full((4, num_new_samples), np.nan, dtype=EEG_DATA_TYPE)

            # Combine all data: (12, num_new_samples)
            all_data_to_write = np.vstack([new_eeg_data, new_eog_data, new_raw_eeg_data])

            # Record block metadata
            current_block_timestamp = time.time()
            data_block_timestamps.append(current_block_timestamp)
            data_block_sample_counts.append(num_new_samples)

            # Write to .dat file
            data_file_handle.write(all_data_to_write.astype(EEG_DATA_TYPE).tobytes())
            samples_written_count = current_total_samples_in_buffer

        # --- Metadata Collection ---
        metadata_timestamps.append(current_time)
        metadata_session_dur.append(session_duration_seconds)

        # --- Periodic Metadata Save ---
        if not initial_metadata_saved or (current_time - last_metadata_save_time >= METADATA_SAVE_INTERVAL_SECONDS):
            try:
                save_payload = {
                    "session_info": session_info,
                    "loop_timestamps": np.array(metadata_timestamps),
                    "session_duration_log_seconds": np.array(metadata_session_dur),
                    "data_block_write_timestamps": np.array(data_block_timestamps),
                    "data_block_sample_counts": np.array(data_block_sample_counts)
                }
                if cli_args.log_events and target_event_transitions:
                    save_payload["target_event_transitions"] = np.array(target_event_transitions, dtype=object)

                np.savez_compressed(custom_metadata_filepath, **save_payload)
                last_metadata_save_time = current_time
                initial_metadata_saved = True
                # print(f"Metadata saved at {datetime.now()}")
            except Exception as e_save:
                print(f"Error saving metadata: {e_save}")
        
        time.sleep(0.05) # Small delay to prevent busy-waiting

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected. Finalizing...")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if streamer:
        streamer.stop()
    if data_file_handle is not None and not data_file_handle.closed:
        data_file_handle.close()

    # Final metadata save
    try:
        if metadata_timestamps: # Ensure there's something to save
            final_save_payload = {
                "session_info": session_info,
                "loop_timestamps": np.array(metadata_timestamps),
                "session_duration_log_seconds": np.array(metadata_session_dur),
                "data_block_write_timestamps": np.array(data_block_timestamps),
                "data_block_sample_counts": np.array(data_block_sample_counts)
            }
            if cli_args.log_events and target_event_transitions:
                final_save_payload["target_event_transitions"] = np.array(target_event_transitions, dtype=object)
            
            np.savez_compressed(custom_metadata_filepath, **final_save_payload)
            print("Final metadata saved.")
    except Exception as e_save_final:
        print(f"Error during final metadata save: {e_save_final}")

    print(f"Custom streaming session ended. Data saved in: {session_data_path}")
