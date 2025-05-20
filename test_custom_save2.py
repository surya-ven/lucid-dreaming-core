import sys
import select
import argparse
from frenztoolkit import Streamer
import time
import numpy as np
import os
import pandas as pd
from datetime import datetime

# For plotting
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt  # Added imports

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
parser.add_argument(
    "--plot-live",
    nargs='?',
    const=True,
    default=False,
    help="Enable live plotting of Raw EEG, Filtered EEG, and EOG signals (latest 20s). Optionally specify a channel number (1-4) to plot only that channel."
)
cli_args = parser.parse_args()

# Determine which channel to plot (if any)
plot_live_channel = None
if cli_args.plot_live is True:
    plot_live_enabled = True
elif cli_args.plot_live is not False:
    try:
        ch = int(cli_args.plot_live)
        if 1 <= ch <= 4:
            plot_live_channel = ch - 1  # zero-based index
            plot_live_enabled = True
        else:
            plot_live_enabled = False
    except Exception:
        plot_live_enabled = False
else:
    plot_live_enabled = False

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

# For live plotting
plot_initialized = False
PLOT_WINDOW_DURATION_S = 20  # Display last 20 seconds
plot_timestamps = []
plot_data_raw_eeg = [[] for _ in range(4)]
plot_data_filt_eeg = [[] for _ in range(4)]
plot_data_filt_eog = [[] for _ in range(4)]
ax_raw_eeg, ax_filt_eeg, ax_filt_eog = None, None, None

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

# --- Helper function to process EOG for plotting ---


def process_eog_for_plotting(data, srate, lowcut=0.5, highcut=3.0, artifact_threshold=100, mft_kernel_size=5):
    """
    Processes EOG data for clearer live plotting, focusing on eye movement frequencies.
    Applies artifact rejection, median filtering, and bandpass filtering.
    """
    processed_data = np.copy(data)
    if processed_data.ndim == 1:  # Handle single channel data
        # Convert to (samples, 1)
        processed_data = processed_data[:, np.newaxis]

    min_len_for_butter = 3 * (4 + 1)  # (order + 1) * 3 for filtfilt
    if processed_data.shape[0] < max(mft_kernel_size, min_len_for_butter):
        return processed_data

    n_channels = processed_data.shape[1]

    # 1. Artifact removal/rejection with thresholding
    for channel in range(n_channels):
        channel_data = processed_data[:, channel]
        artifacts = np.abs(channel_data) > artifact_threshold

        if np.any(artifacts):
            artifact_indices = np.where(artifacts)[0]
            diff_indices = np.diff(artifact_indices)
            split_points = np.where(diff_indices > 1)[0] + 1
            segments = np.split(artifact_indices, split_points)

            for segment in segments:
                if len(segment) > 0:
                    start_idx = segment[0]
                    end_idx = segment[-1]

                    pre_value = 0
                    if start_idx == 0:
                        non_artifact_after_segment = np.where(
                            ~artifacts[end_idx+1:])[0]
                        if len(non_artifact_after_segment) > 0:
                            pre_value = channel_data[end_idx +
                                                     1 + non_artifact_after_segment[0]]
                        else:
                            pre_value = channel_data[end_idx] if end_idx < len(
                                channel_data) else 0
                    else:
                        pre_value = channel_data[start_idx-1]

                    post_value = 0
                    if end_idx == len(channel_data) - 1:
                        non_artifact_before_segment = np.where(
                            ~artifacts[:start_idx])[0]
                        if len(non_artifact_before_segment) > 0:
                            post_value = channel_data[non_artifact_before_segment[-1]]
                        else:
                            post_value = channel_data[start_idx] if start_idx > 0 else 0
                    else:
                        post_value = channel_data[end_idx+1]

                    segment_length = len(segment)
                    if segment_length > 0:
                        x_interp = np.array([start_idx - 1, end_idx + 1])
                        y_interp = np.array([pre_value, post_value])
                        x_segment = segment
                        if x_interp[0] < x_segment[0] and x_interp[1] > x_segment[-1]:
                            interp_values = np.interp(
                                x_segment, x_interp, y_interp)
                            processed_data[segment, channel] = interp_values

    # 2. Median filter
    actual_mft_kernel_size = mft_kernel_size
    if actual_mft_kernel_size % 2 == 0:
        actual_mft_kernel_size += 1

    if processed_data.shape[0] >= actual_mft_kernel_size:
        try:
            processed_data = medfilt(
                processed_data, kernel_size=(actual_mft_kernel_size, 1))
        except ValueError:
            pass

    # 3. Zero-phase band-pass filter
    nyquist = srate / 2.0
    actual_highcut = min(highcut, nyquist - 0.01)
    actual_lowcut = max(0.01, lowcut)

    if actual_highcut > actual_lowcut and processed_data.shape[0] > min_len_for_butter:
        try:
            b, a = butter(4, [actual_lowcut, actual_highcut],
                          btype='band', fs=srate)
            processed_data = filtfilt(b, a, processed_data, axis=0)
        except ValueError as e:
            pass
    return processed_data


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

    if plot_live_enabled:
        plt.ion()
        fig, (ax_raw_eeg, ax_filt_eeg, ax_filt_eog) = plt.subplots(
            3, 1, sharex=True, figsize=(12, 10))
        ax_raw_eeg.set_title("Raw EEG Signals")
        ax_filt_eeg.set_title("Filtered EOG Signals")  # Corrected Title
        ax_filt_eog.set_title(
            "Filtered EOG Signals (Processed for Eye Movements)")

        ax_raw_eeg.set_ylabel("Amplitude")
        ax_filt_eeg.set_ylabel("Amplitude")  # Corrected Label
        ax_filt_eog.set_ylabel("Amplitude")
        ax_filt_eog.set_xlabel("Session Time (s)")
        plot_initialized = True

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
        # streamer.DATA["RAW"]["EEG"] is expected to be (N_total_accumulated_samples, 6)
        filtered_eeg_buffer = streamer.DATA["FILTERED"]["EEG"]
        filtered_eog_buffer = streamer.DATA["FILTERED"]["EOG"]
        raw_eeg_buffer = streamer.DATA["RAW"]["EEG"]

        # Determine the total number of samples currently available in the buffer (per channel)
        current_total_samples_in_buffer = 0
        if filtered_eeg_buffer is not None and filtered_eeg_buffer.ndim == 2 and filtered_eeg_buffer.shape[0] == 4:
            current_total_samples_in_buffer = filtered_eeg_buffer.shape[1]
        else:
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
            else:
                new_eog_data = np.full(
                    (4, num_new_samples), np.nan, dtype=EEG_DATA_TYPE)

            # Slice the new samples from the RAW EEG buffer: (num_new_samples, 6)
            # Save columns [0, 1, 3, 4] as (4, num_new_samples)
            if raw_eeg_buffer is not None and raw_eeg_buffer.ndim == 2 and raw_eeg_buffer.shape[1] >= 5:
                new_raw_eeg_data = raw_eeg_buffer[-num_new_samples:,
                                                  [0, 1, 3, 4]].T
            else:
                new_raw_eeg_data = np.full(
                    (4, num_new_samples), np.nan, dtype=EEG_DATA_TYPE)

            # Combine EEG, EOG, and raw EEG data by stacking vertically: (8+4=12, num_new_samples)
            all_data_to_write = np.vstack(
                [new_eeg_data, new_eog_data, new_raw_eeg_data])

            current_block_timestamp = time.time()  # Timestamp for this specific data block
            data_block_timestamps.append(current_block_timestamp)
            data_block_sample_counts.append(num_new_samples)

            # Write to .dat file
            data_file_handle.write(
                all_data_to_write.astype(EEG_DATA_TYPE).tobytes())
            samples_written_count = current_total_samples_in_buffer

            # --- Populate data for plotting ---
            if plot_live_enabled and plot_initialized:
                block_end_time = session_duration_seconds
                block_start_time = block_end_time - \
                    (num_new_samples - 1) / \
                    float(FS) if num_new_samples > 0 else block_end_time
                new_timestamps = np.linspace(
                    block_start_time, block_end_time, num_new_samples)

                plot_timestamps.extend(new_timestamps)
                for i in range(4):  # Assuming 4 channels for each
                    if new_raw_eeg_data.shape[0] == 4 and new_raw_eeg_data.shape[1] == num_new_samples:
                        plot_data_raw_eeg[i].extend(new_raw_eeg_data[i, :])
                    else:
                        plot_data_raw_eeg[i].extend([np.nan] * num_new_samples)

                    if new_eeg_data.shape[0] == 4 and new_eeg_data.shape[1] == num_new_samples:
                        plot_data_filt_eeg[i].extend(new_eeg_data[i, :])
                    else:
                        plot_data_filt_eeg[i].extend(
                            [np.nan] * num_new_samples)

                    if new_eog_data.shape[0] == 4 and new_eog_data.shape[1] == num_new_samples:
                        plot_data_filt_eog[i].extend(new_eog_data[i, :])
                    else:
                        plot_data_filt_eog[i].extend(
                            [np.nan] * num_new_samples)

                if plot_timestamps:
                    min_plot_time = plot_timestamps[-1] - \
                        PLOT_WINDOW_DURATION_S
                    start_idx = 0
                    for idx, ts in enumerate(plot_timestamps):
                        if ts >= min_plot_time:
                            start_idx = idx
                            break

                    plot_timestamps = plot_timestamps[start_idx:]
                    for i in range(4):
                        plot_data_raw_eeg[i] = plot_data_raw_eeg[i][start_idx:]
                        plot_data_filt_eeg[i] = plot_data_filt_eeg[i][start_idx:]
                        plot_data_filt_eog[i] = plot_data_filt_eog[i][start_idx:]

        # --- Live Plotting ---
        if plot_live_enabled and plot_initialized and plot_timestamps:
            ax_raw_eeg.cla()
            ax_filt_eeg.cla()
            ax_filt_eog.cla()

            ax_raw_eeg.set_title("Raw EEG Signals")
            ax_filt_eeg.set_title("Filtered EOG Signals")  # Corrected Title
            ax_filt_eog.set_title(
                "Filtered EOG Signals (Processed for Eye Movements)")

            ax_raw_eeg.set_ylabel("Amplitude")
            ax_filt_eeg.set_ylabel("Amplitude")  # Corrected Label
            ax_filt_eog.set_ylabel("Amplitude")
            ax_filt_eog.set_xlabel("Session Time (s)")

            channel_colors = [f'C{i}' for i in range(4)]
            channels_to_plot = [
                plot_live_channel] if plot_live_channel is not None else range(4)

            eog_data_for_processing = []
            valid_eog_data_present = True
            if len(plot_timestamps) > 0:
                for i in range(4):
                    if len(plot_data_filt_eog[i]) != len(plot_timestamps):
                        valid_eog_data_present = False
                        break
                if valid_eog_data_present:
                    for i in range(4):
                        eog_data_for_processing.append(plot_data_filt_eog[i])

                if not eog_data_for_processing or len(eog_data_for_processing) != 4:
                    valid_eog_data_present = False
            else:
                valid_eog_data_present = False

            processed_eog_plot_data = None
            if valid_eog_data_present:
                eog_window_np = np.array(eog_data_for_processing).T
                if eog_window_np.shape[0] > 0:
                    processed_eog_plot_data = process_eog_for_plotting(
                        eog_window_np,
                        FS,
                        lowcut=0.5,
                        highcut=3.0,
                        artifact_threshold=100,
                        # Ensure kernel is at least 3
                        mft_kernel_size=max(3, int(FS * 0.04))
                    )

            for i in channels_to_plot:
                if len(plot_timestamps) == len(plot_data_raw_eeg[i]):
                    ax_raw_eeg.plot(
                        plot_timestamps, plot_data_raw_eeg[i], label=f'Raw EEG Ch {i+1}', color=channel_colors[i], alpha=0.7)

                if len(plot_timestamps) == len(plot_data_filt_eog[i]):
                    ax_filt_eeg.plot(
                        plot_timestamps, plot_data_filt_eog[i], label=f'Filt EOG Ch {i+1}', color=channel_colors[i], alpha=0.7)

                plot_label_eog = f'EOG Ch {i+1}'
                line_style_eog = '-'

                if processed_eog_plot_data is not None and \
                   processed_eog_plot_data.shape[0] == len(plot_timestamps) and \
                   processed_eog_plot_data.shape[1] == 4:
                    ax_filt_eog.plot(
                        plot_timestamps, processed_eog_plot_data[:, i], label=f'{plot_label_eog} (Proc.)', color=channel_colors[i], linestyle=line_style_eog)
                elif len(plot_timestamps) == len(plot_data_filt_eog[i]):
                    ax_filt_eog.plot(
                        plot_timestamps, plot_data_filt_eog[i], label=f'{plot_label_eog} (Orig.)', color=channel_colors[i], linestyle=':')

            if plot_live_channel is None:
                ax_raw_eeg.legend(loc='upper right', fontsize='small')
                # Corrected legend for Filtered EOG
                ax_filt_eeg.legend(loc='upper right', fontsize='small')
                ax_filt_eog.legend(loc='upper right', fontsize='small')
            else:
                ch_num = plot_live_channel + 1
                ax_raw_eeg.set_title(f"Raw EEG Signal - Channel {ch_num}")
                # Corrected title for Filtered EOG
                ax_filt_eeg.set_title(
                    f"Filtered EOG Signal - Channel {ch_num}")
                ax_filt_eog.set_title(
                    f"Processed EOG Signal - Channel {ch_num}")

            if plot_timestamps:
                current_xlim_start = plot_timestamps[0]
                current_xlim_end = plot_timestamps[-1]
                if current_xlim_start >= current_xlim_end:
                    current_xlim_end = current_xlim_start + 1

                ax_raw_eeg.set_xlim(current_xlim_start, current_xlim_end)

            plt.tight_layout()
            plt.pause(0.01)

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
                pass

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected. Finalizing...")
    pass
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

    if plot_live_enabled:
        plt.ioff()
        plt.show()
    print(
        f"Custom streaming session ended. Data saved in: {session_data_path}")
