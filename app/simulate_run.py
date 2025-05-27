import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import mne

# --- Configuration (adapted from app/main.py and conversation) ---
PRODUCT_KEY_SIM = "SIMULATED_RUtYA4W3kpXi0i9C7VZCQJY5_GRhm4XL2rKp6cviwQI="
DEVICE_ID_PREFIX_SIM = "SIM_FRENZI40"
FS = 125.0  # Sampling frequency
WINDOW_SIZE_S = 5.0  # Processing window size in seconds
EEG_DATA_TYPE = np.float32
NUM_COMBINED_COLUMNS = 12  # Target number of channels for eeg_eog_data.dat

EEG_EOG_DATA_FILENAME = "eeg_eog_data.dat"
AUX_SENSOR_DATA_FILENAME = "aux_sensor_data.dat"  # Will be empty for simulation
METADATA_FILENAME = "session_metadata.npz"

REM_SLEEP_STAGE_NAME = "REM"
# Sleep stage mapping
SLEEP_STAGE_MAP = {
    "Wake": 0, "W": 0, "wake": 0, "WK": 0,
    "N1": 1, "NREM1": 1, "Light": 1, "L": 1, "light": 1, "NREM1": 1,
    "N2": 1, "NREM2": 1,  # Also mapped to Light sleep
    "N3": 2, "NREM3": 2, "Deep": 2, "D": 2, "deep": 2, "SWS": 2, "DEEP": 2,
    "REM": 3, "R": 3, "rem": 3,
    "Unknown": 4, "U": 4, "unknown": 4, "A": 4, "Artifact": 4, "ART": 4, "MOVEMENT": 4, "Movement": 4,
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4  # Allow numeric inputs too
}
REM_SLEEP_STAGE_VALUE = SLEEP_STAGE_MAP[REM_SLEEP_STAGE_NAME]
DEFAULT_UNKNOWN_STAGE_VALUE = SLEEP_STAGE_MAP["Unknown"]

SLEEP_STAGE_INTERVAL_S = 30.0  # Assumed duration of each stage listed in CSV

MAX_SUCCESSIVE_REM_CUES = 2
# Min interval between cues in a REM sequence (must be >= WINDOW_SIZE_S)
AUDIO_CUE_INTERVAL_S = 5.0

# --- Logging Helpers ---


def _ensure_dir_exists(path_obj: Path):
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(
            f"Critical error creating directory {path_obj}: {e}", file=sys.stderr)
        raise


def _log_to_file(message: str, log_file_path: Path):
    try:
        with open(log_file_path, "a") as f:
            f.write(message + "\n")
    except Exception as log_e:
        print(
            f"Failed to write to log {log_file_path}: {log_e}", file=sys.stderr)


def log_session_message(full_message: str, session_log_path: Optional[Path], is_error: bool):
    print(full_message, file=sys.stderr if is_error else sys.stdout)
    if session_log_path:
        _log_to_file(full_message, session_log_path)

# --- Main Simulation Function ---


def simulate_session_from_files(dataset_dir: Path, output_base_dir: Path) -> Optional[Path]:
    session_id_level_path: Optional[Path] = None
    output_session_folder_for_dataset: Optional[Path] = None
    # Ensure valid device ID chars
    device_id_sim = f"{DEVICE_ID_PREFIX_SIM}_{dataset_dir.name[:15].replace(' ', '_')}"
    # Define with a broader scope
    simulation_run_log_file: Optional[Path] = None

    try:
        # 1. Setup session output directory
        session_timestamp_fs = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_id_sim = f"SIM_{dataset_dir.name.replace(' ', '_')}_{session_timestamp_fs}"
        session_id_level_path = output_base_dir / session_id_sim
        _ensure_dir_exists(session_id_level_path)

        # Centralized log file for this simulation attempt (dataset level)
        simulation_run_log_file = session_id_level_path / "simulation_run.log"

        def current_log(message: str, is_error: bool = False):
            timestamp = datetime.now().isoformat()
            prefix = "ERROR" if is_error else "INFO"
            log_session_message(
                f"{timestamp} - {prefix}: {message}", simulation_run_log_file, is_error)

        current_log(f"Starting simulation for dataset: {dataset_dir.name}")
        current_log(f"Session ID: {session_id_sim}")
        current_log(f"Device ID: {device_id_sim}")

        # 2. Locate and load data files
        try:
            edf_file = next(dataset_dir.glob('*.edf'))
            # Try to find a CSV that is not _SQC.CSV (case-insensitive)
            stage_csv_files = [
                f for f in dataset_dir.glob('*.csv')
                if not f.name.upper().endswith('_SQC.CSV')
            ]
            if not stage_csv_files:  # If no non-SQC CSV, try any CSV as a fallback
                current_log(
                    "No non-SQC CSV found, trying any CSV file.", is_error=False)
                stage_csv_files = list(dataset_dir.glob('*.csv'))
                if not stage_csv_files:
                    raise FileNotFoundError("Sleep stage CSV file not found.")
            stage_csv_file = stage_csv_files[0]  # Pick the first one found
            if len(stage_csv_files) > 1:
                current_log(
                    f"Multiple CSV files found, using: {stage_csv_file}. Others: {[str(f) for f in stage_csv_files[1:]]}", is_error=False)

        except (StopIteration, FileNotFoundError) as e:
            current_log(
                f"Could not find required data files in {dataset_dir}: {e}", is_error=True)
            # Create minimal error metadata if possible, even if files are missing
            placeholder_start_unix = datetime.now(timezone.utc).timestamp()
            output_session_folder_for_dataset = session_id_level_path / \
                str(int(placeholder_start_unix))
            _ensure_dir_exists(output_session_folder_for_dataset)
            error_metadata_path = output_session_folder_for_dataset / METADATA_FILENAME
            np.savez_compressed(error_metadata_path, error_message=f"Data file missing: {e}",
                                product_key=PRODUCT_KEY_SIM, device_id=device_id_sim,
                                session_id=session_id_sim,
                                simulated_data_source_folder=str(dataset_dir))
            return output_session_folder_for_dataset

        current_log(f"Found EDF file: {edf_file}")
        current_log(f"Found Stage CSV file: {stage_csv_file}")

        # Load EDF
        try:
            raw_edf = mne.io.read_raw_edf(
                edf_file, preload=True, verbose=False)
        except Exception as e:
            current_log(
                f"Error loading EDF file {edf_file}: {e}", is_error=True)
            placeholder_start_unix = datetime.now(timezone.utc).timestamp()
            output_session_folder_for_dataset = session_id_level_path / \
                str(int(placeholder_start_unix))
            _ensure_dir_exists(output_session_folder_for_dataset)
            error_metadata_path = output_session_folder_for_dataset / METADATA_FILENAME
            np.savez_compressed(error_metadata_path, error_message=f"EDF load error: {e}",
                                product_key=PRODUCT_KEY_SIM, device_id=device_id_sim,
                                session_id=session_id_sim,
                                simulated_data_source_folder=str(dataset_dir))
            return output_session_folder_for_dataset

        eeg_all_channels_data_raw = raw_edf.get_data()  # (channels, samples)
        num_raw_channels, num_raw_samples = eeg_all_channels_data_raw.shape
        edf_sampling_frequency = raw_edf.info['sfreq']

        if abs(edf_sampling_frequency - FS) > 1e-3:
            current_log(
                f"Warning: EDF sampling frequency {edf_sampling_frequency}Hz differs from target {FS}Hz. Data will be used as is for chunking logic, but output metadata will state {FS}Hz.", is_error=False)
            # This means num_samples_per_window might not perfectly align with WINDOW_SIZE_S if based on FS.
            # We will use edf_sampling_frequency for slicing EDF, but num_samples_per_window (based on FS) for output file structure.

        if raw_edf.info['meas_date'] is not None:
            eeg_start_time_unix = raw_edf.info['meas_date'].timestamp()
        else:
            current_log(
                f"EDF file {edf_file} lacks measurement date. Using file modification time as an approximation.", is_error=True)
            try:
                eeg_start_time_unix = edf_file.stat().st_mtime
            except Exception as stat_e:
                current_log(
                    f"Could not get modification time for {edf_file}: {stat_e}. Using current time.", is_error=True)
                eeg_start_time_unix = datetime.now(timezone.utc).timestamp()

        output_session_folder_for_dataset = session_id_level_path / \
            str(int(eeg_start_time_unix))
        _ensure_dir_exists(output_session_folder_for_dataset)
        current_log(
            f"Output will be saved to: {output_session_folder_for_dataset}")

        # Load and process sleep stage data from CSV
        numerical_stages_from_csv: List[int] = []
        try:
            df_stages = pd.read_csv(stage_csv_file)
            stage_col_found = None
            # More comprehensive list of possible stage column names
            possible_stage_cols = [
                'stage', 'Stage', 'STAGE', 'Hypnogram', 'Annotation', 'Sleep stage', 'Sleep Stage',
                'sleep_stage', 'sleep stage', 'hypnogram', 'annotation', 'stages', 'STAGES',
                'Epochs/Stage'  # From a sample file
            ]
            for col in df_stages.columns:  # Iterate through actual columns for case-insensitive partial match
                for p_col in possible_stage_cols:
                    if p_col.lower() in col.lower():
                        stage_col_found = col
                        break
                if stage_col_found:
                    break

            if not stage_col_found:  # Fallback: try the first column if only one, or second if first is timestamp-like
                current_log(
                    f"No standard stage column found. Trying heuristics...", is_error=False)
                if len(df_stages.columns) == 1:
                    stage_col_found = df_stages.columns[0]
                elif len(df_stages.columns) > 1:
                    first_col_name = df_stages.columns[0].lower()
                    if 'time' in first_col_name or 'epoch' in first_col_name or 'index' in first_col_name:
                        if len(df_stages.columns) > 1:
                            # Assume second column is stages
                            stage_col_found = df_stages.columns[1]
                        else:  # Only one column, and it looks like time. Cannot determine stages.
                            current_log(
                                f"Only one column found, and it appears to be a time/index column ('{df_stages.columns[0]}'). Cannot determine stage column.", is_error=True)
                    else:  # First column doesn't look like time, assume it's stages
                        stage_col_found = df_stages.columns[0]

            if not stage_col_found:
                current_log(
                    f"Could not identify sleep stage column in {stage_csv_file}. Columns: {list(df_stages.columns)}", is_error=True)
            else:
                current_log(
                    f"Using stage column: '{stage_col_found}' from {stage_csv_file}")
                raw_stages = df_stages[stage_col_found].tolist()
                numerical_stages_from_csv = [SLEEP_STAGE_MAP.get(s, SLEEP_STAGE_MAP.get(
                    str(s), DEFAULT_UNKNOWN_STAGE_VALUE)) for s in raw_stages]
                current_log(
                    f"Loaded {len(numerical_stages_from_csv)} stages from CSV. First 5 raw: {raw_stages[:5]}, mapped: {numerical_stages_from_csv[:5]}")

        except Exception as e:
            current_log(
                f"Error processing stage CSV {stage_csv_file}: {e}", is_error=True)

        if not numerical_stages_from_csv:
            current_log(
                "No numerical sleep stages extracted. Simulation will use UNKNOWN for all stages.", is_error=True)
            max_possible_stages = int(
                np.ceil((num_raw_samples / edf_sampling_frequency) / SLEEP_STAGE_INTERVAL_S))
            # Ensure at least one stage
            numerical_stages_from_csv = [
                DEFAULT_UNKNOWN_STAGE_VALUE] * max(1, max_possible_stages)

        # 3. Prepare data files and metadata lists
        eeg_output_filepath = output_session_folder_for_dataset / EEG_EOG_DATA_FILENAME

        all_chunk_timestamps_unix = []
        all_sleep_stages_for_chunks = []
        all_audio_cue_timestamps_unix = []
        all_audio_cue_initiation_timestamps_unix = []

        in_rem_segment = False
        rem_consecutive_cues_this_segment = 0
        last_cue_time_this_segment_unix = -float('inf')

        # --- Main data processing loop ---
        # num_samples_per_output_window is based on target FS
        num_samples_per_output_window = int(FS * WINDOW_SIZE_S)
        # num_samples_per_edf_slice is based on EDF's actual FS
        num_samples_per_edf_slice = int(edf_sampling_frequency * WINDOW_SIZE_S)

        total_windows = num_raw_samples // num_samples_per_edf_slice

        current_log(
            f"EDF has {num_raw_samples} samples at {edf_sampling_frequency}Hz, duration approx {num_raw_samples/edf_sampling_frequency:.2f}s.")
        current_log(
            f"Processing {total_windows} windows. Each window represents {WINDOW_SIZE_S}s of data.")
        current_log(
            f"EDF data per window: {num_samples_per_edf_slice} samples. Output data per window: {num_samples_per_output_window} samples (target {FS}Hz).")

        total_samples_written_eeg_eog_per_channel_cumulative = 0

        with open(eeg_output_filepath, 'wb') as eeg_file_handle:
            for i in range(total_windows):
                current_chunk_relative_start_s = i * WINDOW_SIZE_S
                current_chunk_start_time_unix = eeg_start_time_unix + current_chunk_relative_start_s
                all_chunk_timestamps_unix.append(current_chunk_start_time_unix)

                start_sample_in_edf = i * num_samples_per_edf_slice
                end_sample_in_edf = start_sample_in_edf + num_samples_per_edf_slice

                eeg_chunk_raw_edf_fs = eeg_all_channels_data_raw[:,
                                                                 start_sample_in_edf:end_sample_in_edf]

                # Prepare chunk for output file (NUM_COMBINED_COLUMNS channels, num_samples_per_output_window samples)
                # This simplified simulation will take the first NUM_COMBINED_COLUMNS channels (or pad)
                # and then take the first num_samples_per_output_window samples from the slice (or pad).
                # This means if edf_sampling_frequency != FS, there's an implicit resampling/truncation.

                output_chunk_data = np.zeros(
                    (NUM_COMBINED_COLUMNS, num_samples_per_output_window), dtype=EEG_DATA_TYPE)

                chans_to_copy = min(num_raw_channels, NUM_COMBINED_COLUMNS)
                samps_to_copy_from_edf_slice = min(
                    eeg_chunk_raw_edf_fs.shape[1], num_samples_per_output_window)

                output_chunk_data[:chans_to_copy, :samps_to_copy_from_edf_slice] = \
                    eeg_chunk_raw_edf_fs[:chans_to_copy, :samps_to_copy_from_edf_slice].astype(
                        EEG_DATA_TYPE)

                eeg_file_handle.write(output_chunk_data.tobytes())
                total_samples_written_eeg_eog_per_channel_cumulative += num_samples_per_output_window

                # Sleep Stage Determination
                stage_index = int(
                    current_chunk_relative_start_s // SLEEP_STAGE_INTERVAL_S)
                current_sleep_stage = DEFAULT_UNKNOWN_STAGE_VALUE
                if stage_index < len(numerical_stages_from_csv):
                    current_sleep_stage = numerical_stages_from_csv[stage_index]
                all_sleep_stages_for_chunks.append(current_sleep_stage)

                # REM Cue Logic
                if current_sleep_stage == REM_SLEEP_STAGE_VALUE:
                    if not in_rem_segment:
                        in_rem_segment = True
                        rem_consecutive_cues_this_segment = 0
                        last_cue_time_this_segment_unix = -float('inf')
                        all_audio_cue_initiation_timestamps_unix.append(
                            current_chunk_start_time_unix)
                        current_log(
                            f"REM segment started at {current_chunk_start_time_unix:.2f} (chunk {i})")

                    time_since_last_cue = current_chunk_start_time_unix - \
                        last_cue_time_this_segment_unix
                    if rem_consecutive_cues_this_segment < MAX_SUCCESSIVE_REM_CUES and \
                       time_since_last_cue >= AUDIO_CUE_INTERVAL_S:
                        all_audio_cue_timestamps_unix.append(
                            current_chunk_start_time_unix)
                        rem_consecutive_cues_this_segment += 1
                        last_cue_time_this_segment_unix = current_chunk_start_time_unix
                        current_log(
                            f"Audio cue {rem_consecutive_cues_this_segment}/{MAX_SUCCESSIVE_REM_CUES} triggered at {current_chunk_start_time_unix:.2f}")
                else:
                    if in_rem_segment:
                        current_log(
                            f"REM segment ended after time {current_chunk_start_time_unix:.2f} (chunk {i})")
                    in_rem_segment = False

        # Create empty aux sensor data file
        aux_output_filepath = output_session_folder_for_dataset / AUX_SENSOR_DATA_FILENAME
        with open(aux_output_filepath, 'wb') as aux_file_handle:
            pass

        # 4. Save Metadata
        if not all_chunk_timestamps_unix:
            current_log(
                "No EEG/EOG data was processed or written. Skipping metadata saving.", is_error=True)
            # Save minimal error metadata
            error_metadata_path = output_session_folder_for_dataset / METADATA_FILENAME
            np.savez_compressed(error_metadata_path, error_message="No data processed/written",
                                product_key=PRODUCT_KEY_SIM, device_id=device_id_sim,
                                session_id=session_id_sim,
                                simulated_data_source_folder=str(dataset_dir))
            return output_session_folder_for_dataset

        session_start_actual_unix = all_chunk_timestamps_unix[0]
        session_end_actual_unix = all_chunk_timestamps_unix[-1] + WINDOW_SIZE_S
        session_duration_actual_s = session_end_actual_unix - session_start_actual_unix

        num_processed_chunks = len(all_chunk_timestamps_unix)

        session_info_simulated = {
            'product_key': PRODUCT_KEY_SIM,
            'device_id': device_id_sim,
            'session_id': session_id_sim,
            'session_start_iso_str': datetime.fromtimestamp(session_start_actual_unix, tz=timezone.utc).isoformat(),
            'session_duration_s': session_duration_actual_s,
            'sampling_frequency_hz': FS,
            'eeg_num_channels': NUM_COMBINED_COLUMNS,
            'eeg_data_type': str(EEG_DATA_TYPE),
            'total_eeg_eog_samples_written_per_channel': total_samples_written_eeg_eog_per_channel_cumulative,
            'simulated_data_source_folder': str(dataset_dir),
            'frenz_app_version': 'simulated_v0.3_redo',
            'frenz_firmware_version': 'simulated_fw_v0.3_redo',
            'frenz_serial_number': f"SIM_SN_{dataset_dir.name[:10]}",
        }

        scores_to_save = {
            "array__sleep_stage": np.array(all_sleep_stages_for_chunks, dtype=np.int8),
            "array__poas": np.zeros(num_processed_chunks, dtype=np.float32),
            "array__posture": np.zeros(num_processed_chunks, dtype=np.int8),
            "array__focus_score": np.zeros(num_processed_chunks, dtype=np.float32),
            "array__sqc_scores": np.full((num_processed_chunks, NUM_COMBINED_COLUMNS), -1, dtype=np.int8)
        }

        final_metadata_to_save = dict(session_info_simulated)
        final_metadata_to_save["scores"] = scores_to_save
        final_metadata_to_save["metadata_eeg_eog_timestamps"] = np.array(
            all_chunk_timestamps_unix, dtype=np.float64)
        final_metadata_to_save["metadata_eeg_eog_sample_counts"] = np.full(
            num_processed_chunks, num_samples_per_output_window, dtype=np.int32)
        final_metadata_to_save["metadata_aux_timestamps"] = np.array(
            [], dtype=np.float64)
        final_metadata_to_save["metadata_aux_sample_counts"] = np.array(
            [], dtype=np.int32)
        final_metadata_to_save["audio_cue_timestamps"] = np.array(
            all_audio_cue_timestamps_unix, dtype=np.float64)
        final_metadata_to_save["audio_cue_sequence_initiation_timestamps"] = np.array(
            all_audio_cue_initiation_timestamps_unix, dtype=np.float64)

        metadata_filepath = output_session_folder_for_dataset / METADATA_FILENAME
        np.savez_compressed(metadata_filepath, **final_metadata_to_save)
        current_log(
            f"Simulated session metadata successfully saved to {metadata_filepath}")

        return output_session_folder_for_dataset

    except Exception as e:
        tb_str = traceback.format_exc()
        # Use the simulation_run_log_file if it was initialized
        log_path_for_critical_error = simulation_run_log_file if simulation_run_log_file else None

        critical_err_msg = f"CRITICAL ERROR during simulation for {dataset_dir.name}: {e}\n{tb_str}"
        print(critical_err_msg, file=sys.stderr)
        if log_path_for_critical_error:
            _log_to_file(datetime.now().isoformat() + " - CRITICAL: " +
                         critical_err_msg, log_path_for_critical_error)

        # Try to save error metadata
        # Determine device_id and session_id safely for error metadata
        safe_device_id = device_id_sim if 'device_id_sim' in locals(
        ) else f"{DEVICE_ID_PREFIX_SIM}_ERROR"
        safe_session_id = session_id_sim if 'session_id_sim' in locals(
        ) else f"SIM_ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        if output_session_folder_for_dataset:
            error_metadata_path = output_session_folder_for_dataset / METADATA_FILENAME
        elif session_id_level_path:
            placeholder_start_unix = datetime.now(timezone.utc).timestamp()
            error_session_folder = session_id_level_path / \
                str(int(placeholder_start_unix))
            _ensure_dir_exists(error_session_folder)
            error_metadata_path = error_session_folder / METADATA_FILENAME
        else:
            # Fallback: try to create a basic error log in a predefined error directory if all else fails
            # This case should ideally not be reached if session_id_level_path is always created first.
            error_output_base = output_base_dir if 'output_base_dir' in locals() else Path(".") / \
                "simulation_errors"
            _ensure_dir_exists(error_output_base)
            error_metadata_path = error_output_base / \
                f"{safe_session_id}_CRITICAL_ERROR_{METADATA_FILENAME}"

        if error_metadata_path:
            try:
                np.savez_compressed(error_metadata_path,
                                    error_message=str(e), traceback=tb_str,
                                    product_key=PRODUCT_KEY_SIM, device_id=safe_device_id,
                                    session_id=safe_session_id,
                                    simulated_data_source_folder=str(dataset_dir))
                # Log this attempt to the central log if possible, otherwise print
                err_meta_msg = f"Critical error metadata saved to {error_metadata_path}"
                if log_path_for_critical_error:
                    _log_to_file(datetime.now().isoformat(
                    ) + " - INFO: " + err_meta_msg, log_path_for_critical_error)
                else:
                    print(err_meta_msg, file=sys.stderr)

            except Exception as meta_e:
                err_meta_fail_msg = f"Failed to save critical error metadata: {meta_e}"
                if log_path_for_critical_error:
                    _log_to_file(datetime.now().isoformat(
                    ) + " - ERROR: " + err_meta_fail_msg, log_path_for_critical_error)
                else:
                    print(err_meta_fail_msg, file=sys.stderr)

        # Return path if available, for inspection
        return output_session_folder_for_dataset


if __name__ == '__main__':
    print("Simulate_run.py executed as main script.")
    print("This script is intended to be imported and used by another script or notebook (e.g., run_simulation_analysis.ipynb).")
    print("Example usage within a calling script:")
    print("from pathlib import Path")
    print("from app.simulate_run import simulate_session_from_files # Assuming simulate_run.py is in 'app' directory")
    print("# Adjust import if simulate_run.py is in the same directory as the notebook:")
    print("# from simulate_run import simulate_session_from_files")
    print("dataset_p = Path('path/to/your/sample_data/dataset_folder')")
    print("output_p = Path('path/to/your/simulation_output')")
    print("result_path = simulate_session_from_files(dataset_p, output_p)")
    print(
        "if result_path: print(f'Simulation output (or error metadata) in: {result_path}')")
    print("else: print('Simulation failed catastrophically before output folder could be determined.')")

    if len(sys.argv) > 2:
        data_dir_arg = Path(sys.argv[1])
        out_dir_arg = Path(sys.argv[2])

        # Create output directory if it doesn't exist
        try:
            out_dir_arg.mkdir(parents=True, exist_ok=True)
            print(f"Ensured output directory exists: {out_dir_arg}")
        except Exception as e:
            print(
                f"Error creating output directory {out_dir_arg}: {e}", file=sys.stderr)
            sys.exit(1)

        if data_dir_arg.is_dir() and out_dir_arg.is_dir():
            print(
                f"Attempting test simulation with data from subdirectories in: {data_dir_arg}, output to: {out_dir_arg}")

            datasets_found = [d for d in data_dir_arg.iterdir() if d.is_dir()]
            if not datasets_found:
                print(
                    f"No subdirectories (datasets) found in {data_dir_arg} to run a test.")
            else:
                first_dataset = datasets_found[0]
                print(f"Using first found dataset for test: {first_dataset}")
                result = simulate_session_from_files(
                    first_dataset, out_dir_arg)
                if result:
                    print(f"Test simulation finished. Output/log in: {result}")
                else:
                    print(
                        f"Test simulation failed. Check logs in {out_dir_arg} or console output.")
        else:
            if not data_dir_arg.is_dir():
                print(
                    f"Provided data_dir is not a valid directory: {data_dir_arg}", file=sys.stderr)
            if not out_dir_arg.is_dir():
                print(
                    f"Provided out_dir is not a valid directory (or could not be created): {out_dir_arg}", file=sys.stderr)
