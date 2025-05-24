import sys
import select
import argparse
from frenztoolkit import Streamer
import time
import numpy as np
import os
import pandas as pd
from datetime import datetime
from test_LRLR_detection import detect_lrlr_window_from_lstm, LSTM_SAMPLE_LENGTH
import tensorflow as tf

# For plotting
import matplotlib.pyplot as plt

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
METRICS_REPORT_INTERVAL_SECONDS = 30  # Interval for reporting collected metrics

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
    action="store_true",
    help="Enable live plotting of LRLR detection (latest 20s, starts after 30s)."
)
parser.add_argument(
    "--plot-metrics",
    action="store_true",
    help="Enable live plotting of system performance metrics."
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

last_LRLR_compute_time = 0
lrlr_result = None

# For live plotting
lrlr_plot_times = []  # seconds since session start
lrlr_plot_values = []  # 1 if LRLR detected, 0 otherwise
plot_initialized = False
fig_event_plot = None
ax_lrlr_event_plot = None

# --- Metrics Plotting Data (if enabled) ---
metrics_plot_times = []
# Loop performance
metrics_plot_avg_loop_duration = []
metrics_plot_max_loop_duration = []
# LRLR inference
metrics_plot_avg_lrlr_inf_time = []
metrics_plot_max_lrlr_inf_time = []
# Data throughput
metrics_plot_effective_data_rate = []
metrics_plot_data_completeness = []
# SQC Avg Scores (one list per channel)
metrics_plot_avg_sqc_lf = []
metrics_plot_avg_sqc_otel = []
metrics_plot_avg_sqc_rf = []
metrics_plot_avg_sqc_oter = []
# SQC % Good (one list per channel)
metrics_plot_perc_good_sqc_lf = []
metrics_plot_perc_good_sqc_otel = []
metrics_plot_perc_good_sqc_rf = []
metrics_plot_perc_good_sqc_oter = []

metrics_plot_initialized = False
fig_metrics = None
axes_metrics = None

# --- Metrics Accumulators (for periodic reporting) ---
last_metrics_report_time = 0  # Will be set properly after streamer starts
# Variables for accumulating metrics over one reporting interval
interval_loop_count = 0
interval_total_loop_duration = 0.0
interval_max_loop_duration = 0.0
interval_total_lrlr_inference_time = 0.0
interval_lrlr_inference_count = 0
interval_max_lrlr_inference_time = 0.0
interval_total_new_samples_processed = 0
interval_sqc_scores_sum = np.zeros(4, dtype=float)
interval_sqc_scores_good_count = np.zeros(4, dtype=int)
interval_sqc_readings_count = 0
num_sqc_reports_processed_overall = 0  # Tracks total SQC history processed

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

data_file_handle = None
last_metadata_save_time = time.time()

try:
    streamer.start()
    last_metrics_report_time = time.time()  # Initialize here after start
    # Open file immediately after streamer starts
    data_file_handle = open(custom_data_filepath, 'ab')
    initial_metadata_saved = False

    if cli_args.log_events:
        print("\nEvent logging enabled. Press 'S' then Enter to START an event, 'E' then Enter to END an event.")
        print("You can do this multiple times during the session.\n")

    if cli_args.plot_live:
        plt.ion()
        fig_event_plot, ax_lrlr_event_plot = plt.subplots()
        plot_initialized = True

    if cli_args.plot_metrics:
        plt.ion()  # Ensure interactive mode is on for multiple figures
        fig_metrics, axes_metrics = plt.subplots(
            3, 2, figsize=(18, 12), sharex=True)  # Share X axis (time)
        metrics_plot_initialized = True
        fig_metrics.suptitle(
            "System Performance Metrics Over Time", fontsize=16)

    while True:
        loop_start_time = time.time()  # Metric
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
                        lrlr_inf_start_time = time.time()  # Metric
                        # Detect LRLR in the new EOG data
                        lrlr_result = detect_lrlr_window_from_lstm(
                            filtered_eog_buffer.T, srate=118, detection_threshold=0.51)
                        lrlr_inf_end_time = time.time()  # Metric
                        if lrlr_result is not None:
                            current_lrlr_inf_time = lrlr_inf_end_time - lrlr_inf_start_time  # Metric
                            interval_total_lrlr_inference_time += current_lrlr_inf_time  # Metric
                            interval_max_lrlr_inference_time = max(
                                interval_max_lrlr_inference_time, current_lrlr_inf_time)  # Metric
                            interval_lrlr_inference_count += 1  # Metric
                            test, count = lrlr_result
                            # For plotting: mark LRLR event (binary or count)
                            if cli_args.plot_live:
                                lrlr_plot_times.append(
                                    session_duration_seconds)
                                lrlr_plot_values.append(1 if test else 0)

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
            interval_total_new_samples_processed += num_new_samples  # Metric

        # --- SQC Metrics Processing ---
        all_sqc_history = streamer.SCORES.get("array__sqc_scores")
        if all_sqc_history:
            new_sqc_reports = all_sqc_history[num_sqc_reports_processed_overall:]
            for report_values in new_sqc_reports:
                if report_values is not None and isinstance(report_values, (list, np.ndarray)) and len(report_values) == 4:
                    interval_sqc_scores_sum += np.array(
                        report_values, dtype=float)
                    interval_sqc_scores_good_count += (
                        np.array(report_values) == 1).astype(int)
                    interval_sqc_readings_count += 1
            num_sqc_reports_processed_overall = len(all_sqc_history)

        # --- Live Plotting ---
        if cli_args.plot_live and session_duration_seconds > 30:
            # Only plot the latest 20 seconds
            plot_window = 20
            min_time = session_duration_seconds - plot_window
            # Filter LRLR data
            lrlr_times = np.array(lrlr_plot_times)
            lrlr_vals = np.array(lrlr_plot_values)
            mask_lrlr = lrlr_times >= min_time

            if plot_initialized:
                ax_lrlr_event_plot.cla()

            # Plot LRLR
            if np.any(mask_lrlr) and ax_lrlr_event_plot:
                ax_lrlr_event_plot.plot(
                    lrlr_times[mask_lrlr], lrlr_vals[mask_lrlr], 'r.-', label="LRLR Detected")
                ax_lrlr_event_plot.set_ylabel("LRLR Detected", color='r')
                ax_lrlr_event_plot.set_xlabel("Time (s)")
                ax_lrlr_event_plot.set_ylim(-0.1, 1.1)
                ax_lrlr_event_plot.legend(loc='upper left')

            if fig_event_plot:
                plt.figure(fig_event_plot.number)
                plt.title("Live LRLR Detection (last 20s)")
                plt.tight_layout()
                plt.pause(0.01)

        # --- Metadata Collection (timestamps and session duration) ---
        metadata_timestamps.append(current_time)
        metadata_session_dur.append(session_duration_seconds)

        # --- Periodic Metrics Reporting ---
        if (current_time - last_metrics_report_time) >= METRICS_REPORT_INTERVAL_SECONDS:
            actual_interval_duration = current_time - last_metrics_report_time
            print(
                f"\n--- Metrics Report (Interval: {actual_interval_duration:.2f}s) ---")

            # Initialize metrics for the current interval to default values (0 or NaN)
            avg_loop_duration_ms_val = 0
            max_loop_duration_ms_val = 0
            avg_lrlr_inf_time_ms_val = 0
            max_lrlr_inf_time_ms_val = 0
            effective_data_rate_sps_val = 0
            data_completeness_pct_val = np.nan  # Use NaN if not calculable
            avg_sqc_vals = [np.nan] * 4
            perc_good_sqc_vals = [np.nan] * 4

            if interval_loop_count > 0:
                avg_loop_duration_ms_val = (
                    interval_total_loop_duration / interval_loop_count) * 1000
                max_loop_duration_ms_val = interval_max_loop_duration * 1000
                loops_per_sec = interval_loop_count / actual_interval_duration
                print(
                    f"  Loop Performance: Avg: {avg_loop_duration_ms_val:.2f}ms, Max: {max_loop_duration_ms_val:.2f}ms, Rate: {loops_per_sec:.2f} Loops/s ({interval_loop_count} loops)")
            else:
                print("  Loop Performance: No loops in interval.")

            if interval_lrlr_inference_count > 0:
                avg_lrlr_inf_time_ms_val = (
                    interval_total_lrlr_inference_time / interval_lrlr_inference_count) * 1000
                max_lrlr_inf_time_ms_val = interval_max_lrlr_inference_time * 1000
                lrlr_inf_per_sec = interval_lrlr_inference_count / actual_interval_duration
                print(
                    f"  LRLR Model: Avg Inf: {avg_lrlr_inf_time_ms_val:.2f}ms, Max Inf: {max_lrlr_inf_time_ms_val:.2f}ms, Rate: {lrlr_inf_per_sec:.2f} Inf/s ({interval_lrlr_inference_count} inferences)")
            else:
                print("  LRLR Model: No inferences in interval.")

            if actual_interval_duration > 0:
                effective_data_rate_sps_val = interval_total_new_samples_processed / \
                    actual_interval_duration
                print(
                    f"  Data Throughput: {interval_total_new_samples_processed} samples in interval. Effective Rate: {effective_data_rate_sps_val:.2f} Samples/s")
                expected_samples = FS * actual_interval_duration
                if expected_samples > 0:  # Avoid division by zero if FS or duration is zero
                    data_completeness_pct_val = (
                        interval_total_new_samples_processed / expected_samples) * 100
                    print(
                        f"    Data Completeness (vs {FS} Hz): {data_completeness_pct_val:.2f}% (expected {expected_samples:.0f} samples)")
                else:
                    print(
                        "    Data Completeness: Cannot calculate (expected samples is zero).")
            else:
                print(
                    f"  Data Throughput: {interval_total_new_samples_processed} samples in interval. (Interval duration zero or negative)")

            sqc_channels = ["LF", "OTEL", "RF", "OTER"]
            if interval_sqc_readings_count > 0:
                avg_sqc_calc = interval_sqc_scores_sum / interval_sqc_readings_count
                perc_good_sqc_calc = (
                    interval_sqc_scores_good_count / interval_sqc_readings_count) * 100
                print(
                    f"  Signal Quality (SQC) ({interval_sqc_readings_count} readings):")
                for i in range(4):
                    avg_sqc_vals[i] = avg_sqc_calc[i]
                    perc_good_sqc_vals[i] = perc_good_sqc_calc[i]
                    print(
                        f"    {sqc_channels[i]}: Avg Score: {avg_sqc_vals[i]:.2f}, % Good: {perc_good_sqc_vals[i]:.1f}%")
            else:
                print("  Signal Quality (SQC): No new readings in interval.")

            print("--- End Metrics Report ---")

            # Append data for plotting if enabled
            if cli_args.plot_metrics:
                # Use session_duration at time of report
                metrics_plot_times.append(session_duration_seconds)
                metrics_plot_avg_loop_duration.append(avg_loop_duration_ms_val)
                metrics_plot_max_loop_duration.append(max_loop_duration_ms_val)
                metrics_plot_avg_lrlr_inf_time.append(avg_lrlr_inf_time_ms_val)
                metrics_plot_max_lrlr_inf_time.append(max_lrlr_inf_time_ms_val)
                metrics_plot_effective_data_rate.append(
                    effective_data_rate_sps_val)
                metrics_plot_data_completeness.append(
                    data_completeness_pct_val)

                metrics_plot_avg_sqc_lf.append(avg_sqc_vals[0])
                metrics_plot_avg_sqc_otel.append(avg_sqc_vals[1])
                metrics_plot_avg_sqc_rf.append(avg_sqc_vals[2])
                metrics_plot_avg_sqc_oter.append(avg_sqc_vals[3])

                metrics_plot_perc_good_sqc_lf.append(perc_good_sqc_vals[0])
                metrics_plot_perc_good_sqc_otel.append(perc_good_sqc_vals[1])
                metrics_plot_perc_good_sqc_rf.append(perc_good_sqc_vals[2])
                metrics_plot_perc_good_sqc_oter.append(perc_good_sqc_vals[3])

                # Update metrics plots
                plt.figure(fig_metrics.number)  # Select the correct figure

                # Subplot 1: Loop Performance
                ax = axes_metrics[0, 0]
                ax.cla()
                ax.plot(metrics_plot_times, metrics_plot_avg_loop_duration,
                        label="Avg Loop Time (ms)", marker='.')
                ax.plot(metrics_plot_times, metrics_plot_max_loop_duration,
                        label="Max Loop Time (ms)", marker='.')
                ax.set_title("Loop Performance")
                ax.set_ylabel("Time (ms)")
                ax.legend(loc='upper left')
                ax.grid(True)

                # Subplot 2: Alertness Model Inference
                ax = axes_metrics[0, 1]
                ax.cla()
                ax.set_title("Alertness Model Inference Time (Removed)")
                ax.set_ylabel("Time (ms)")
                ax.grid(True)

                # Subplot 3: LRLR Model Inference
                ax = axes_metrics[1, 0]
                ax.cla()
                ax.plot(metrics_plot_times, metrics_plot_avg_lrlr_inf_time,
                        label="Avg LRLR Inf. (ms)", marker='.')
                ax.plot(metrics_plot_times, metrics_plot_max_lrlr_inf_time,
                        label="Max LRLR Inf. (ms)", marker='.')
                ax.set_title("LRLR Model Inference Time")
                ax.set_ylabel("Time (ms)")
                ax.set_xlabel("Session Time (s)")
                ax.legend(loc='upper left')
                ax.grid(True)

                # Subplot 4: Data Throughput & Completeness
                ax = axes_metrics[1, 1]
                ax.cla()
                ax.plot(metrics_plot_times, metrics_plot_effective_data_rate,
                        label="Effective Data Rate (Samples/s)", marker='.')
                ax.set_ylabel("Samples/s", color='tab:blue')
                ax.tick_params(axis='y', labelcolor='tab:blue')
                ax.legend(loc='upper left')
                ax.grid(True, axis='y', color='tab:blue',
                        linestyle='--', alpha=0.7)

                ax_twin = ax.twinx()
                ax_twin.plot(metrics_plot_times, metrics_plot_data_completeness,
                             label="Data Completeness (%)", color='tab:red', marker='.')
                ax_twin.set_ylabel("Completeness (%)", color='tab:red')
                ax_twin.tick_params(axis='y', labelcolor='tab:red')
                ax_twin.legend(loc='upper right')
                ax.set_title("Data Throughput & Completeness")
                ax.set_xlabel("Session Time (s)")

                # Subplot 5: Average SQC Scores
                ax = axes_metrics[2, 0]
                ax.cla()
                ax.plot(metrics_plot_times, metrics_plot_avg_sqc_lf,
                        label="Avg SQC LF", marker='.')
                ax.plot(metrics_plot_times, metrics_plot_avg_sqc_otel,
                        label="Avg SQC OTEL", marker='.')
                ax.plot(metrics_plot_times, metrics_plot_avg_sqc_rf,
                        label="Avg SQC RF", marker='.')
                ax.plot(metrics_plot_times, metrics_plot_avg_sqc_oter,
                        label="Avg SQC OTER", marker='.')
                ax.set_title("Average SQC Scores (0=Not Good, 1=Good)")
                ax.set_ylabel("Average Score")
                ax.set_xlabel("Session Time (s)")
                ax.legend(loc='upper left', fontsize='small')
                ax.grid(True)
                ax.set_ylim(-0.1, 1.1)

                # Subplot 6: % Good SQC Scores
                ax = axes_metrics[2, 1]
                ax.cla()
                ax.plot(metrics_plot_times, metrics_plot_perc_good_sqc_lf,
                        label="% Good SQC LF", marker='.')
                ax.plot(metrics_plot_times, metrics_plot_perc_good_sqc_otel,
                        label="% Good SQC OTEL", marker='.')
                ax.plot(metrics_plot_times, metrics_plot_perc_good_sqc_rf,
                        label="% Good SQC RF", marker='.')
                ax.plot(metrics_plot_times, metrics_plot_perc_good_sqc_oter,
                        label="% Good SQC OTER", marker='.')
                ax.set_title("Percentage Good SQC Scores")
                ax.set_ylabel("Percentage (%)")
                ax.set_xlabel("Session Time (s)")
                ax.legend(loc='upper left', fontsize='small')
                ax.grid(True)
                ax.set_ylim(-5, 105)

                # Adjust layout to make space for suptitle
                fig_metrics.tight_layout(rect=[0, 0, 1, 0.96])
                plt.pause(0.01)

            # Reset accumulators for the next interval
            interval_loop_count = 0
            interval_total_loop_duration = 0.0
            interval_max_loop_duration = 0.0
            interval_total_lrlr_inference_time = 0.0
            interval_lrlr_inference_count = 0
            interval_max_lrlr_inference_time = 0.0
            interval_total_new_samples_processed = 0
            interval_sqc_scores_sum.fill(0)
            interval_sqc_scores_good_count.fill(0)
            interval_sqc_readings_count = 0

            last_metrics_report_time = current_time

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

        # --- Loop Duration Update ---
        loop_end_time = time.time()
        current_loop_duration = loop_end_time - loop_start_time
        interval_total_loop_duration += current_loop_duration
        interval_max_loop_duration = max(
            interval_max_loop_duration, current_loop_duration)
        interval_loop_count += 1

        # Small sleep to be polite to the CPU, especially if data comes slowly or select timeout is short.
        # The select() call already has a 0.01s timeout, so this might be redundant but harmless.
        time.sleep(0.001)

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

    if cli_args.plot_live and fig_event_plot:
        plt.figure(fig_event_plot.number)
        plt.ioff()
        plt.show()

    if cli_args.plot_metrics and fig_metrics:  # Ensure fig_metrics was created
        plt.figure(fig_metrics.number)  # Select the correct figure
        plt.ioff()
        plt.show()

    print(
        f"Custom streaming session ended. Data saved in: {session_data_path}")
