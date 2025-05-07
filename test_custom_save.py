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

# --- Session Setup ---
session_start_time_obj = datetime.now()
session_timestamp_str = session_start_time_obj.strftime(
    "%Y%m%d_%H%M%S_%f")  # Added microseconds for more uniqueness
session_data_path = os.path.join(BASE_RECORDING_FOLDER, session_timestamp_str)
os.makedirs(session_data_path, exist_ok=True)
print(f"Custom session data will be saved in: {session_data_path}")

custom_eeg_filepath = os.path.join(session_data_path, "custom_eeg.dat")
custom_metadata_filepath = os.path.join(
    session_data_path, "custom_metadata.npz")

# --- Initialize Streamer ---
# Note: streamer's data_folder is still set, its internal saving will also occur if it works.
# Our custom saving is independent.
streamer = Streamer(
    device_id=DEVICE_ID,
    product_key=PRODUCT_KEY,
    # Frenztoolkit will create its own subfolder here
    data_folder=BASE_RECORDING_FOLDER
)

# --- In-memory data accumulators ---
eeg_samples_written = 0
metadata_timestamps = []
metadata_session_dur = []
metadata_posture = []
metadata_poas = []
metadata_sleep_stage = []

# Initial session info for metadata
session_info = {
    "product_key": PRODUCT_KEY,
    "device_id": DEVICE_ID,
    "session_start_iso": session_start_time_obj.isoformat(),
    "custom_eeg_datatype": str(EEG_DATA_TYPE),
    "eeg_expected_columns": 6  # Based on observed raw data
}

eeg_file_handle = None
last_metadata_save_time = time.time()

try:
    print("Starting streaming session...")
    streamer.start()
    print(f"Streamer started. Target session duration: {10*60} seconds.")

    initial_metadata_saved = False

    while True:
        current_time = time.time()
        session_duration_seconds = streamer.session_dur

        if session_duration_seconds > 10 * 60:  # Example: run for 10 minutes
            print("Desired session duration reached.")
            break

        # --- EEG Data Handling ---
        # raw_eeg_data shape is (num_samples_accumulated, num_raw_columns=6)
        raw_eeg_data_buffer = streamer.DATA["RAW"]["EEG"]

        if raw_eeg_data_buffer is not None and raw_eeg_data_buffer.ndim == 2 and raw_eeg_data_buffer.shape[1] == 6:
            current_total_samples = raw_eeg_data_buffer.shape[0]
            num_new_samples = current_total_samples - eeg_samples_written

            if num_new_samples > 0:
                new_eeg_to_write = raw_eeg_data_buffer[eeg_samples_written:current_total_samples, :]

                if eeg_file_handle is None:
                    print(
                        f"Opening custom EEG file for writing: {custom_eeg_filepath}")
                    eeg_file_handle = open(
                        custom_eeg_filepath, 'ab')  # Append binary

                eeg_file_handle.write(
                    new_eeg_to_write.astype(EEG_DATA_TYPE).tobytes())
                eeg_samples_written = current_total_samples
                # print(f"Written {num_new_samples} new EEG samples. Total written: {eeg_samples_written}")

        # --- Metadata Collection ---
        posture = streamer.SCORES.get("posture")
        poas = streamer.SCORES.get("poas")
        sleep_stage = streamer.SCORES.get("sleep_stage")

        metadata_timestamps.append(current_time)
        metadata_session_dur.append(session_duration_seconds)
        metadata_posture.append(
            posture if posture is not None else np.nan)  # Store NaN if None
        metadata_poas.append(poas if poas is not None else np.nan)
        metadata_sleep_stage.append(
            sleep_stage if sleep_stage is not None else np.nan)

        # --- Periodic Metadata Save ---
        if not initial_metadata_saved or (current_time - last_metadata_save_time >= METADATA_SAVE_INTERVAL_SECONDS):
            try:
                np.savez_compressed(
                    custom_metadata_filepath,
                    session_info=session_info,  # Save as a dictionary object
                    timestamps=np.array(metadata_timestamps),
                    session_duration_log=np.array(metadata_session_dur),
                    # dtype=object for potential Nones/mixed types
                    posture_log=np.array(metadata_posture, dtype=object),
                    poas_log=np.array(metadata_poas, dtype=object),
                    sleep_stage_log=np.array(
                        metadata_sleep_stage, dtype=object)
                )
                print(
                    f"Custom metadata saved to {custom_metadata_filepath} at {session_duration_seconds:.2f}s")
                last_metadata_save_time = current_time
                initial_metadata_saved = True
            except Exception as e_save:
                print(f"Error saving custom metadata: {e_save}")

        # --- Console Output (similar to original script) ---
        if raw_eeg_data_buffer is not None:
            print(
                f"AT TIME: {session_duration_seconds:.2f}s | EEG Buffer Shape: {raw_eeg_data_buffer.shape} | Samples Written: {eeg_samples_written}")
            # print("EEG data (first 10 of buffer):", raw_eeg_data_buffer[:10, :]) # Can be very verbose
        else:
            print(
                f"AT TIME: {session_duration_seconds:.2f}s | EEG Buffer: None")

        print(
            f"Latest POSTURE: {posture} | POAS: {poas} | Sleep Stage: {sleep_stage}\n")

        time.sleep(5)  # Original sleep interval

except KeyboardInterrupt:
    print("Session stopped by user (KeyboardInterrupt).")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Finalizing custom data saving...")
    if eeg_file_handle is not None:
        print(f"Closing custom EEG file: {custom_eeg_filepath}")
        eeg_file_handle.close()

    # Final metadata save
    try:
        if metadata_timestamps:  # Only save if there's something to save
            np.savez_compressed(
                custom_metadata_filepath,
                session_info=session_info,
                timestamps=np.array(metadata_timestamps),
                session_duration_log=np.array(metadata_session_dur),
                posture_log=np.array(metadata_posture, dtype=object),
                poas_log=np.array(metadata_poas, dtype=object),
                sleep_stage_log=np.array(metadata_sleep_stage, dtype=object)
            )
            print(f"Final custom metadata saved to {custom_metadata_filepath}")
    except Exception as e_save_final:
        print(f"Error during final metadata save: {e_save_final}")

    print("Stopping streamer...")
    if 'streamer' in locals() and streamer is not None and streamer.is_streaming:
        streamer.stop()  # This will trigger Frenztoolkit's own saving mechanism
        print("Streamer stopped.")
    else:
        print("Streamer was not active or already stopped.")
    print(f"Custom session data was saved in: {session_data_path}")
