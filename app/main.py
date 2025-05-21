# Conceptual structure for FastAPI app (main.py)
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from datetime import datetime
from frenztoolkit import Streamer, Scanner
import numpy as np
import os
import time
import asyncio
from scipy.signal import butter, filtfilt, detrend
from typing import Optional, List, Dict
import sys
import traceback
import subprocess
import json
import argparse
from playsound3 import playsound
import tempfile
import soundfile as sf
from pydantic import BaseModel, Field
import io
import shutil

# --- Configuration ---
PRODUCT_KEY = "RUtYA4W3kpXi0i9C7VZCQJY5_GRhm4XL2rKp6cviwQI="
DEVICE_ID = "FRENZI40"
BASE_RECORDING_FOLDER = "./recorded_data"
FS = 125.0
WINDOW_SIZE_S = 5
EEG_DATA_TYPE = np.float32
NUM_COMBINED_COLUMNS = 12
LEFT_EOG_CH = 0
RIGHT_EOG_CH = 2

# Signal Quality Check Configuration
SQC_CHECK_INTERVAL_S = 5.0
SQC_CONSECUTIVE_BAD_LIMIT = 3

# Frenzband Discovery
FRENZ_SCAN_CACHE_DURATION_S = 10.0

# Caffeinate Configuration
_IS_CAFFEINATED_ENV_VAR = "LUCID_DREAMING_CORE_CAFFEINATED"

# Command-line argument flags
LOG_FILTERED_DATA = False
session_max_audio_volume: float = 0.8

# --- New File Configuration ---
EEG_EOG_DATA_FILENAME = "eeg_eog_data.dat"
AUX_SENSOR_DATA_FILENAME = "aux_sensor_data.dat"
METADATA_FILENAME = "session_metadata.npz"

# Global state
streamer_instance: Optional[Streamer] = None
session_active = False
session_data_path: Optional[str] = None
current_status = "Idle"
session_start_time_obj: Optional[datetime] = None
session_info_global: dict = {}

# File handles
eeg_eog_data_file_handle: Optional[io.BufferedWriter] = None
aux_sensor_data_file_handle: Optional[io.BufferedWriter] = None

is_frenz_band_available: Optional[bool] = None
last_frenz_scan_time: float = 0.0

# Metadata lists
metadata_eeg_eog_timestamps: List[float] = []
metadata_eeg_eog_sample_counts: List[int] = []
metadata_aux_timestamps: List[float] = []
metadata_aux_sample_counts: List[int] = []

# Counters for samples written
samples_written_eeg_eog = 0


# --- Caffeinate Helper ---
def ensure_caffeinated():
    if sys.platform == "darwin":
        if os.environ.get(_IS_CAFFEINATED_ENV_VAR) != "1":
            print(
                "INFO: Script not running under 'caffeinate'. Re-launching to prevent sleep...", file=sys.stdout)
            log_session_info("Attempting to re-launch under caffeinate.", None)
            env = os.environ.copy()
            env[_IS_CAFFEINATED_ENV_VAR] = "1"
            args = ["/usr/bin/caffeinate", "-i", sys.executable] + sys.argv
            try:
                os.execve(args[0], args, env)
            except OSError as e:
                err_msg = f"ERROR: Failed to re-launch with caffeinate: {e}. Please run manually: {' '.join(args)}"
                print(err_msg, file=sys.stderr)
                log_session_error(f"Caffeinate re-launch failed: {e}", None)


# --- Helper for Logging ---
def log_session_error(message: str, session_p: Optional[str]):
    timestamp = datetime.now().isoformat()
    full_message = f"{timestamp} - ERROR: {message}"
    print(full_message, file=sys.stderr)

    if session_p:
        if not os.path.isdir(session_p):
            print(f"{timestamp} - WARNING: Session path '{session_p}' not valid. Cannot write to session_errors.log.", file=sys.stderr)
            return

        log_file_path = os.path.join(session_p, "session_errors.log")
        try:
            with open(log_file_path, "a") as f:
                f.write(full_message + "\n")
        except Exception as log_e:
            print(
                f"{timestamp} - CRITICAL: Failed to write to session log '{log_file_path}': {log_e}", file=sys.stderr)


def log_session_info(message: str, session_p: Optional[str]):
    timestamp = datetime.now().isoformat()
    full_message = f"{timestamp} - INFO: {message}"
    print(full_message, file=sys.stdout)

    if session_p:
        log_file_path = os.path.join(session_p, "session_info.log")
        try:
            with open(log_file_path, "a") as f:
                f.write(full_message + "\n")
        except Exception as log_e:
            log_session_error(
                f"Failed to write to info log '{log_file_path}': {log_e}", session_p)


def _ensure_dir_exists(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        log_session_error(
            f"Critical error creating directory {path}: {e}", None)
        raise


async def _save_final_metadata(current_session_data_path: Optional[str], current_session_info: Dict):
    global streamer_instance, metadata_eeg_eog_timestamps, metadata_eeg_eog_sample_counts
    global metadata_aux_timestamps, metadata_aux_sample_counts

    if not current_session_data_path or not current_session_info:
        log_session_info(
            "Skipping final metadata save: no session path or info.", current_session_data_path)
        return

    metadata_filepath = os.path.join(
        current_session_data_path, METADATA_FILENAME)
    log_session_info(
        f"Attempting to save final metadata to {metadata_filepath}", current_session_data_path)

    final_metadata_to_save = dict(current_session_info)

    final_metadata_to_save["metadata_eeg_eog_timestamps"] = np.array(
        metadata_eeg_eog_timestamps, dtype=np.float64)
    final_metadata_to_save["metadata_eeg_eog_sample_counts"] = np.array(
        metadata_eeg_eog_sample_counts, dtype=np.int32)
    final_metadata_to_save["metadata_aux_timestamps"] = np.array(
        metadata_aux_timestamps, dtype=np.float64)
    final_metadata_to_save["metadata_aux_sample_counts"] = np.array(
        metadata_aux_sample_counts, dtype=np.int32)

    scores_to_save = {}
    score_keys = ["array__sleep_stage", "array__poas",
                  "array__posture", "array__focus_score", "array__sqc_scores"]

    streamer_scores = None
    if streamer_instance:
        try:
            streamer_scores = streamer_instance.SCORES
        except Exception as e:
            log_session_error(
                f"Error accessing streamer_instance.SCORES: {e}", current_session_data_path)
            streamer_scores = None

    if streamer_scores:
        for key in score_keys:
            try:
                score_value = streamer_scores.get(key)
                if isinstance(score_value, list):
                    scores_to_save[key] = np.array(score_value)
                elif score_value is not None:
                    scores_to_save[key] = score_value
                else:
                    scores_to_save[key] = np.array([])
            except Exception as e:
                log_session_error(
                    f"Error processing score {key}: {e}", current_session_data_path)
                scores_to_save[key] = np.array([])
    else:
        log_session_info(
            "Streamer instance or its SCORES not available for metadata saving. Saving empty scores.", current_session_data_path)
        for key in score_keys:
            scores_to_save[key] = np.array([])

    final_metadata_to_save["scores"] = scores_to_save

    try:
        np.savez(metadata_filepath, **final_metadata_to_save)
        log_session_info(
            f"Final metadata successfully saved to {metadata_filepath}", current_session_data_path)
    except Exception as e:
        log_session_error(
            f"CRITICAL ERROR: Could not save final .npz metadata to '{metadata_filepath}': {e}\n{traceback.format_exc()}", current_session_data_path)


async def real_time_processing_loop():
    global session_active, current_status, streamer_instance, session_data_path, session_info_global
    global eeg_eog_data_file_handle, aux_sensor_data_file_handle
    global metadata_eeg_eog_timestamps, metadata_eeg_eog_sample_counts
    global metadata_aux_timestamps, metadata_aux_sample_counts
    global samples_written_eeg_eog
    global LOG_FILTERED_DATA

    loop_properly_initialized = False
    current_status = "Real-time processing loop started."
    log_session_info(current_status, session_data_path)

    if not eeg_eog_data_file_handle or eeg_eog_data_file_handle.closed or \
       not aux_sensor_data_file_handle or aux_sensor_data_file_handle.closed:
        err_msg = "Data file handles are not open at the start of processing loop."
        log_session_error(err_msg, session_data_path)
        current_status = f"ERROR: {err_msg}"
        session_active = False
        return

    loop_properly_initialized = True

    try:
        while session_active:
            if streamer_instance is None:
                current_status = "Error: Streamer not available."
                log_session_error(current_status, session_data_path)
                session_active = False
                break

            now = time.time()

            f_eeg_all = streamer_instance.DATA["FILTERED"]["EEG"]

            if f_eeg_all is None or f_eeg_all.ndim != 2 or f_eeg_all.shape[0] != 4 or f_eeg_all.shape[1] == 0:
                await asyncio.sleep(0.1)
                continue

            current_total_filt_eeg_samples = f_eeg_all.shape[1]

            if current_total_filt_eeg_samples <= samples_written_eeg_eog:
                await asyncio.sleep(0.1)
                continue

            new_n_samples = current_total_filt_eeg_samples - samples_written_eeg_eog
            new_filt_eeg = f_eeg_all[:, -new_n_samples:]

            r_eeg_all = streamer_instance.DATA["RAW"]["EEG"]
            if r_eeg_all is not None and r_eeg_all.ndim == 2 and r_eeg_all.shape[1] == 6 and r_eeg_all.shape[0] >= new_n_samples:
                new_raw_eeg_T = r_eeg_all[-new_n_samples:, :].T
            else:
                new_raw_eeg_T = np.full(
                    (6, new_n_samples), np.nan, dtype=EEG_DATA_TYPE)

            f_eog_all = streamer_instance.DATA["FILTERED"]["EOG"]
            if f_eog_all is not None and f_eog_all.ndim == 2 and f_eog_all.shape[0] == 4 and f_eog_all.shape[1] >= new_n_samples:
                new_filt_eog = f_eog_all[:, -new_n_samples:]
            else:
                new_filt_eog = np.full(
                    (4, new_n_samples), np.nan, dtype=EEG_DATA_TYPE)

            eeg_eog_block = np.vstack(
                [new_raw_eeg_T, new_filt_eeg, new_filt_eog])
            if eeg_eog_data_file_handle and not eeg_eog_data_file_handle.closed:
                eeg_eog_data_file_handle.write(
                    eeg_eog_block.astype(EEG_DATA_TYPE).tobytes())
            metadata_eeg_eog_timestamps.append(now)
            metadata_eeg_eog_sample_counts.append(new_n_samples)

            f_emg_all = streamer_instance.DATA["FILTERED"]["EMG"]
            if f_emg_all is not None and f_emg_all.ndim == 2 and f_emg_all.shape[0] == 4 and f_emg_all.shape[1] >= new_n_samples:
                new_filt_emg = f_emg_all[:, -new_n_samples:]
            else:
                new_filt_emg = np.full(
                    (4, new_n_samples), np.nan, dtype=EEG_DATA_TYPE)

            r_imu_all = streamer_instance.DATA["RAW"]["IMU"]
            if r_imu_all is not None and r_imu_all.ndim == 2 and r_imu_all.shape[1] == 3 and r_imu_all.shape[0] >= new_n_samples:
                new_raw_imu_T = r_imu_all[-new_n_samples:, :].T
            else:
                new_raw_imu_T = np.full(
                    (3, new_n_samples), np.nan, dtype=EEG_DATA_TYPE)

            r_ppg_all = streamer_instance.DATA["RAW"]["PPG"]
            if r_ppg_all is not None and r_ppg_all.ndim == 2 and r_ppg_all.shape[1] == 3 and r_ppg_all.shape[0] >= new_n_samples:
                new_raw_ppg_T = r_ppg_all[-new_n_samples:, :].T
            else:
                new_raw_ppg_T = np.full(
                    (3, new_n_samples), np.nan, dtype=EEG_DATA_TYPE)

            aux_block = np.vstack([new_filt_emg, new_raw_imu_T, new_raw_ppg_T])
            if aux_sensor_data_file_handle and not aux_sensor_data_file_handle.closed:
                aux_sensor_data_file_handle.write(
                    aux_block.astype(EEG_DATA_TYPE).tobytes())
            metadata_aux_timestamps.append(now)
            metadata_aux_sample_counts.append(new_n_samples)

            samples_written_eeg_eog += new_n_samples

            if LOG_FILTERED_DATA and new_n_samples > 0:
                num_samples_to_log = min(5, new_n_samples)
                log_session_info(
                    f"-- Filtered EEG (last {num_samples_to_log} samples) --", session_data_path)
                for i in range(new_filt_eeg.shape[0]):
                    log_session_info(
                        f"EEG Ch {i+1}: {new_filt_eeg[i, -num_samples_to_log:]}", session_data_path)
                log_session_info(
                    f"-- Filtered EOG (last {num_samples_to_log} samples) --", session_data_path)
                for i in range(new_filt_eog.shape[0]):
                    log_session_info(
                        f"EOG Ch {i+1}: {new_filt_eog[i, -num_samples_to_log:]}", session_data_path)
                if f_emg_all is not None:
                    log_session_info(
                        f"-- Filtered EMG (last {num_samples_to_log} samples) --", session_data_path)
                    for i in range(new_filt_emg.shape[0]):
                        log_session_info(
                            f"EMG Ch {i+1}: {new_filt_emg[i, -num_samples_to_log:]}", session_data_path)

            current_status = f"Processing... Session Time: {streamer_instance.session_dur:.2f}s. Samples written this block: {new_n_samples}"
            await asyncio.sleep(0.1)

        current_status = "Session ended. Processing loop finalizing."
        log_session_info(current_status, session_data_path)

    except asyncio.CancelledError:
        current_status = "Processing loop cancelled."
        log_session_info(current_status, session_data_path)
    except Exception as e_loop:
        error_msg = f"ERROR in real-time processing loop: {e_loop}\n{traceback.format_exc()}"
        log_session_error(error_msg, session_data_path)
        current_status = f"ERROR: Loop failed: {e_loop}"
        session_active = False
    finally:
        log_session_info(
            f"Entering finally block of processing loop. Initial session_active: {session_active}, current_status: '{current_status}'", session_data_path)

        if eeg_eog_data_file_handle and not eeg_eog_data_file_handle.closed:
            eeg_eog_data_file_handle.close()
            log_session_info(
                f"{EEG_EOG_DATA_FILENAME} closed.", session_data_path)
        eeg_eog_data_file_handle = None

        if aux_sensor_data_file_handle and not aux_sensor_data_file_handle.closed:
            aux_sensor_data_file_handle.close()
            log_session_info(
                f"{AUX_SENSOR_DATA_FILENAME} closed.", session_data_path)
        aux_sensor_data_file_handle = None

        if loop_properly_initialized:
            log_session_info(
                "Loop was initialized, attempting to save final metadata from finally block.", session_data_path)
            await _save_final_metadata(session_data_path, session_info_global)
        else:
            log_session_info(
                "Loop was not properly initialized. Skipping metadata save from finally block.", session_data_path)
            if session_data_path:
                fallback_metadata_path = os.path.join(
                    session_data_path, METADATA_FILENAME)
                try:
                    fallback_info = {"error": "Session loop did not initialize or failed early.",
                                     "session_start_iso": session_info_global.get("session_start_iso", datetime.now().isoformat())}
                    np.savez(fallback_metadata_path, **fallback_info)
                    log_session_info(
                        f"Fallback/error metadata saved to {fallback_metadata_path}", session_data_path)
                except Exception as e_fallback:
                    log_session_error(
                        f"Could not save fallback metadata: {e_fallback}", session_data_path)

        log_session_info(
            f"Exiting finally block of processing loop. Final session_active: {session_active}, final current_status: '{current_status}'", session_data_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global streamer_instance, current_status, BASE_RECORDING_FOLDER, is_frenz_band_available, last_frenz_scan_time
    global eeg_eog_data_file_handle, aux_sensor_data_file_handle, session_data_path, session_info_global

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(BASE_RECORDING_FOLDER):
        BASE_RECORDING_FOLDER = os.path.join(script_dir, BASE_RECORDING_FOLDER)
    os.makedirs(BASE_RECORDING_FOLDER, exist_ok=True)

    current_status = "Service Initialized. Checking Frenzband availability..."
    log_session_info(
        "Service lifespan start: Initializing and performing initial Frenzband scan.", None)
    try:
        scanner = Scanner()
        available_devices = await run_in_threadpool(scanner.scan)
        if DEVICE_ID in available_devices:
            is_frenz_band_available = True
            current_status = f"Service Initialized. Frenzband {DEVICE_ID} found. Idle."
            log_session_info(
                f"Initial Frenzband scan: {DEVICE_ID} found.", None)
        else:
            is_frenz_band_available = False
            current_status = f"Service Initialized. Frenzband {DEVICE_ID} not found. Idle."
            log_session_info(
                f"Initial Frenzband scan: {DEVICE_ID} not found. Devices: {available_devices}", None)
    except Exception as e:
        is_frenz_band_available = False
        current_status = "Service Initialized. Error during initial Frenzband scan. Idle."
        log_session_error(f"Error during initial Frenzband scan: {e}", None)
    last_frenz_scan_time = time.time()

    yield

    log_session_info("Service lifespan end: Shutting down.", None)
    global session_active
    if session_active:
        log_session_info(
            "Session was active during service shutdown. Attempting to stop and save.", session_data_path)
        session_active = False

        if session_data_path and session_info_global:
            log_session_info(
                "Lifespan: calling _save_final_metadata due to active session on shutdown.", session_data_path)
            await _save_final_metadata(session_data_path, session_info_global)

        if streamer_instance:
            try:
                await run_in_threadpool(streamer_instance.stop)
                log_session_info(
                    "Streamer stopped during service shutdown.", session_data_path)
            except Exception as e:
                log_session_error(
                    f"Error stopping streamer during service shutdown: {e}", session_data_path)
            streamer_instance = None

        if eeg_eog_data_file_handle and not eeg_eog_data_file_handle.closed:
            eeg_eog_data_file_handle.close()
            log_session_info(
                f"{EEG_EOG_DATA_FILENAME} closed during service shutdown.", session_data_path)
        eeg_eog_data_file_handle = None

        if aux_sensor_data_file_handle and not aux_sensor_data_file_handle.closed:
            aux_sensor_data_file_handle.close()
            log_session_info(
                f"{AUX_SENSOR_DATA_FILENAME} closed during service shutdown.", session_data_path)
        aux_sensor_data_file_handle = None

    log_session_info("Service shutdown complete.", None)


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=FileResponse)
async def read_index():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_html_path = os.path.join(script_dir, "index.html")
    return index_html_path


@app.get("/frenz/check_availability")
async def check_frenz_availability():
    global is_frenz_band_available, last_frenz_scan_time, current_status

    if time.time() - last_frenz_scan_time < FRENZ_SCAN_CACHE_DURATION_S:
        status_message = f"Frenzband {DEVICE_ID} available: {is_frenz_band_available} (cached)."
        return {"device_id": DEVICE_ID, "available": is_frenz_band_available, "status": status_message, "timestamp": last_frenz_scan_time}

    log_session_info(f"Frenz check: Scanning for {DEVICE_ID}...", None)
    current_status = f"Scanning for Frenzband {DEVICE_ID}..."
    try:
        scanner = Scanner()
        available_devices = await run_in_threadpool(scanner.scan)
        last_frenz_scan_time = time.time()
        if DEVICE_ID in available_devices:
            is_frenz_band_available = True
            status_message = f"Frenzband {DEVICE_ID} found."
            log_session_info(
                f"Frenz check: {DEVICE_ID} found. Devices: {available_devices}", None)
        else:
            is_frenz_band_available = False
            status_message = f"Frenzband {DEVICE_ID} not found. Ensure it is on and blinking blue. Found: {available_devices}"
            log_session_info(
                f"Frenz check: {DEVICE_ID} not found. Devices: {available_devices}", None)
        current_status = status_message
        return {"device_id": DEVICE_ID, "available": is_frenz_band_available, "status": status_message, "timestamp": last_frenz_scan_time}
    except Exception as e:
        is_frenz_band_available = False
        last_frenz_scan_time = time.time()
        error_message = f"Error scanning for Frenzband: {e}"
        log_session_error(error_message, None)
        current_status = error_message
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/session/start")
async def start_session(background_tasks: BackgroundTasks):
    global session_active, current_status, streamer_instance, session_data_path, session_start_time_obj, session_info_global
    global is_frenz_band_available, DEVICE_ID, PRODUCT_KEY, FS, EEG_DATA_TYPE
    global eeg_eog_data_file_handle, aux_sensor_data_file_handle
    global metadata_eeg_eog_timestamps, metadata_eeg_eog_sample_counts
    global metadata_aux_timestamps, metadata_aux_sample_counts
    global samples_written_eeg_eog

    current_scan_time = time.time()
    if not is_frenz_band_available or (current_scan_time - last_frenz_scan_time > FRENZ_SCAN_CACHE_DURATION_S / 2):
        log_session_info(
            "Session start: Re-validating Frenzband availability...", None)
        try:
            scan_result = await check_frenz_availability()
            if not scan_result.get("available"):
                error_msg = f"Frenzband {DEVICE_ID} not found or not available. Please ensure it is turned on (blinking blue) and try again."
                log_session_error(error_msg, None)
                raise HTTPException(status_code=400, detail=error_msg)
        except HTTPException as http_exc:
            if http_exc.status_code == 500:
                error_msg = f"Error occurred while scanning for Frenzband {DEVICE_ID}. Cannot start session. Details: {http_exc.detail}"
                log_session_error(error_msg, None)
                raise HTTPException(status_code=500, detail=error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error while checking Frenzband {DEVICE_ID} availability: {e}. Cannot start session."
            log_session_error(error_msg, None)
            raise HTTPException(status_code=500, detail=error_msg)

    if not is_frenz_band_available:
        error_msg = f"Frenzband {DEVICE_ID} is not available. Session cannot start."
        log_session_error(error_msg, None)
        raise HTTPException(status_code=400, detail=error_msg)

    if session_active:
        log_session_error(
            "Attempt to start a session when one is already active.", None)
        raise HTTPException(status_code=400, detail="Session already active.")

    session_start_time_obj = datetime.now()
    session_timestamp_folder_str = session_start_time_obj.strftime(
        "%Y%m%d_%H%M%S_%f")

    temp_session_data_path = os.path.join(
        BASE_RECORDING_FOLDER, session_timestamp_folder_str)

    try:
        _ensure_dir_exists(temp_session_data_path)
        session_data_path = temp_session_data_path
        log_session_info(
            f"Session data path created: {session_data_path}", session_data_path)
    except OSError as e:
        error_message = f"Fatal: Could not create session directory '{temp_session_data_path}': {e}"
        log_session_error(error_message, None)
        current_status = error_message
        raise HTTPException(status_code=500, detail=error_message)

    samples_written_eeg_eog = 0
    metadata_eeg_eog_timestamps.clear()
    metadata_eeg_eog_sample_counts.clear()
    metadata_aux_timestamps.clear()
    metadata_aux_sample_counts.clear()

    session_info_global = {
        "product_key": PRODUCT_KEY,
        "device_id": DEVICE_ID,
        "session_start_iso": session_start_time_obj.isoformat(),
        "sampling_frequency_hz": FS,
        "eeg_eog_data_info": {
            "filename": EEG_EOG_DATA_FILENAME,
            "data_type": EEG_DATA_TYPE.__name__,
            "channel_names": [
                "RAW_EEG_LF", "RAW_EEG_OTEL", "RAW_EEG_REF1", "RAW_EEG_RF", "RAW_EEG_OTER", "RAW_EEG_REF2",
                "FILT_EEG_LF", "FILT_EEG_OTEL", "FILT_EEG_RF", "FILT_EEG_OTER",
                "FILT_EOG_CH1", "FILT_EOG_CH2", "FILT_EOG_CH3", "FILT_EOG_CH4"
            ],
            "num_channels": 14,
            "shape_on_save": "channels_first"
        },
        "aux_sensor_data_info": {
            "filename": AUX_SENSOR_DATA_FILENAME,
            "data_type": EEG_DATA_TYPE.__name__,
            "channel_names": [
                "FILT_EMG_CH1", "FILT_EMG_CH2", "FILT_EMG_CH3", "FILT_EMG_CH4",
                "RAW_IMU_X", "RAW_IMU_Y", "RAW_IMU_Z",
                "RAW_PPG_GREEN", "RAW_PPG_RED", "RAW_PPG_IR"
            ],
            "num_channels": 10,
            "shape_on_save": "channels_first"
        },
    }

    eeg_eog_data_filepath = os.path.join(
        session_data_path, EEG_EOG_DATA_FILENAME)
    aux_sensor_data_filepath = os.path.join(
        session_data_path, AUX_SENSOR_DATA_FILENAME)
    metadata_filepath = os.path.join(session_data_path, METADATA_FILENAME)

    try:
        eeg_eog_data_file_handle = open(eeg_eog_data_filepath, 'ab')
        aux_sensor_data_file_handle = open(aux_sensor_data_filepath, 'ab')
        log_session_info(
            f"{EEG_EOG_DATA_FILENAME} and {AUX_SENSOR_DATA_FILENAME} opened.", session_data_path)
    except IOError as e:
        error_message = f"Fatal: Could not open data files: {e}"
        log_session_error(error_message, session_data_path)
        current_status = error_message
        if os.path.exists(session_data_path):
            shutil.rmtree(session_data_path)
        session_data_path = None
        raise HTTPException(status_code=500, detail=error_message)

    try:
        np.savez(metadata_filepath, **session_info_global)
        log_session_info(
            f"Initial metadata saved to {metadata_filepath}", session_data_path)
    except Exception as e:
        warn_msg = f"Warning: Could not save initial .npz metadata to '{metadata_filepath}': {e}\n{traceback.format_exc()}"
        log_session_error(warn_msg, session_data_path)

    try:
        streamer_instance = Streamer(
            device_id=DEVICE_ID, product_key=PRODUCT_KEY, data_folder=session_data_path)
        log_session_info("Streamer instance created.", session_data_path)
        await run_in_threadpool(streamer_instance.start)
        log_session_info(
            "Streamer started successfully via thread pool.", session_data_path)
    except Exception as e:
        error_message = f"Error initializing or starting FRENZ streamer: {e}\n{traceback.format_exc()}"
        log_session_error(error_message, session_data_path)
        current_status = f"Error starting streamer: {e}"
        if eeg_eog_data_file_handle:
            eeg_eog_data_file_handle.close()
        if aux_sensor_data_file_handle:
            aux_sensor_data_file_handle.close()
        raise HTTPException(
            status_code=500, detail=f"Could not start FRENZ streamer: {e}")

    session_active = True
    current_status = "Session Starting..."
    log_session_info(
        "Session starting, background task scheduled.", session_data_path)
    background_tasks.add_task(real_time_processing_loop)
    return {"status": "success", "message": "Session started.", "session_id": session_timestamp_folder_str, "data_path": session_data_path}


@app.post("/session/stop")
async def stop_session():
    global session_active, current_status, streamer_instance, session_data_path, session_info_global
    global eeg_eog_data_file_handle, aux_sensor_data_file_handle

    if not session_active and not streamer_instance:
        log_session_info(
            "Stop session called but no session appears to be active.", session_data_path)
        return {"status": "no_active_session", "message": "No active session was found to stop."}

    log_session_info("Stop session requested.", session_data_path)

    session_active = False
    current_status = "Session stopping..."

    await asyncio.sleep(0.2)

    if session_data_path and session_info_global:
        log_session_info(
            "Stop_session: calling _save_final_metadata.", session_data_path)
        await _save_final_metadata(session_data_path, session_info_global)
    else:
        log_session_info(
            "Stop_session: session_data_path or session_info_global not available, cannot save final metadata here.", session_data_path)

    if streamer_instance:
        try:
            await run_in_threadpool(streamer_instance.stop)
            log_session_info(
                "Streamer stopped successfully via thread pool.", session_data_path)
        except Exception as e:
            err_msg = f"Error stopping streamer: {e}\n{traceback.format_exc()}"
            log_session_error(err_msg, session_data_path)
        streamer_instance = None

    if eeg_eog_data_file_handle and not eeg_eog_data_file_handle.closed:
        eeg_eog_data_file_handle.close()
        log_session_info(f"{EEG_EOG_DATA_FILENAME} closed.", session_data_path)
    eeg_eog_data_file_handle = None

    if aux_sensor_data_file_handle and not aux_sensor_data_file_handle.closed:
        aux_sensor_data_file_handle.close()
        log_session_info(
            f"{AUX_SENSOR_DATA_FILENAME} closed.", session_data_path)
    aux_sensor_data_file_handle = None

    final_message = f"Session stopped. Data saved in {session_data_path if session_data_path else 'N/A'}."
    current_status = "Session stopped."
    log_session_info(final_message, session_data_path)

    return {"status": "success", "message": final_message}


@app.get("/session/status")
async def get_status():
    global current_status, session_active, session_data_path, is_frenz_band_available, DEVICE_ID
    return {
        "status": current_status,
        "session_active": session_active,
        "current_session_path": session_data_path,
        "device_id": DEVICE_ID,
        "is_frenz_band_available": is_frenz_band_available
    }


@app.get("/sessions/past")
async def list_past_sessions():
    global BASE_RECORDING_FOLDER
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_base_recording_folder = BASE_RECORDING_FOLDER
    if not os.path.isabs(abs_base_recording_folder):
        abs_base_recording_folder = os.path.join(
            script_dir, abs_base_recording_folder)

    if not os.path.exists(abs_base_recording_folder) or not os.path.isdir(abs_base_recording_folder):
        log_session_info(
            f"Past sessions folder '{abs_base_recording_folder}' not found.", None)
        return {"sessions": [], "message": "Recording folder not found."}

    past_sessions = []
    try:
        session_folders = sorted(
            [d for d in os.listdir(abs_base_recording_folder) if os.path.isdir(
                os.path.join(abs_base_recording_folder, d))],
            reverse=True
        )

        for item_name in session_folders:
            item_path = os.path.join(abs_base_recording_folder, item_name)
            session_display_info = {
                "session_id": item_name,
                "data_path": item_path,
                "start_time_iso": None,
                "metadata_exists": False,
                "has_errors": False,
                "error_log_path": None
            }

            metadata_path = os.path.join(item_path, METADATA_FILENAME)
            if os.path.exists(metadata_path):
                try:
                    with np.load(metadata_path, allow_pickle=True) as metadata:
                        start_iso = metadata.get("session_start_iso")
                        if isinstance(start_iso, np.ndarray):
                            session_display_info["start_time_iso"] = str(
                                start_iso.item()) if start_iso.size == 1 else str(start_iso)
                        elif start_iso is not None:
                            session_display_info["start_time_iso"] = str(
                                start_iso)

                        session_display_info["metadata_exists"] = True
                except Exception as e:
                    log_session_error(
                        f"Error reading metadata file {metadata_path}: {e}", item_path)
                    session_display_info["start_time_iso"] = "Error reading metadata"

            if not session_display_info["start_time_iso"]:
                try:
                    dt_obj = datetime.strptime(item_name.split(
                        '_')[0] + item_name.split('_')[1], "%Y%m%d%H%M%S")
                    session_display_info["start_time_iso"] = dt_obj.isoformat()
                except ValueError:
                    session_display_info["start_time_iso"] = "Unknown (folder name format error)"

            error_log_filepath = os.path.join(item_path, "session_errors.log")
            if os.path.exists(error_log_filepath) and os.path.getsize(error_log_filepath) > 0:
                session_display_info["has_errors"] = True
                session_display_info["error_log_path"] = error_log_filepath

            past_sessions.append(session_display_info)
        return {"sessions": past_sessions}
    except Exception as e:
        err_msg = f"Error listing past sessions: {e}\n{traceback.format_exc()}"
        log_session_error(err_msg, None)
        raise HTTPException(
            status_code=500, detail=f"Error listing past sessions: {e}")


class VolumeRequest(BaseModel):
    volume: float = Field(..., ge=0.0, le=1.0,
                          description="Volume level, from 0.0 to 1.0")



# Lower the maximum allowed volume for safety
MAX_SAFE_AUDIO_VOLUME = 0.2  # Reduce this value as needed (was 1.0, now 0.3)

@app.post("/audio/set_max_volume")
async def set_max_volume(request: VolumeRequest):
    global session_max_audio_volume
    # Clamp the requested volume to the new lower max
    safe_volume = min(request.volume, MAX_SAFE_AUDIO_VOLUME)
    session_max_audio_volume = safe_volume
    log_session_info(
        f"Maximum audio cue volume set to {session_max_audio_volume} (user requested {request.volume}, capped at {MAX_SAFE_AUDIO_VOLUME})", None)
    return {"status": "success", "message": f"Max audio cue volume set to {session_max_audio_volume} (max allowed: {MAX_SAFE_AUDIO_VOLUME})"}



@app.post("/audio/play_sample_cue")
async def play_sample_cue(request: VolumeRequest):
    # Always cap the volume to the safe max
    volume = min(request.volume, MAX_SAFE_AUDIO_VOLUME)
    sample_rate = 22050
    duration = 0.5
    num_samples = int(sample_rate * duration)

    noise = (np.random.rand(num_samples) * 2 - 1)

    audio_data = (noise * volume * 32767).astype(np.int16)

    temp_file_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            temp_file_name = tmp_f.name

        sf.write(temp_file_name, audio_data, sample_rate, subtype='PCM_16')

        await run_in_threadpool(playsound, temp_file_name, block=True)

        return {"status": "success", "message": f"Sample cue played at volume {volume} (max allowed: {MAX_SAFE_AUDIO_VOLUME})"}
    except Exception as e:
        error_msg = f"Error playing sample cue: {e}\n{traceback.format_exc()}"
        log_session_error(error_msg, None)
        raise HTTPException(
            status_code=500, detail=f"Error playing sample cue: {e}")
    finally:
        if temp_file_name and os.path.exists(temp_file_name):
            try:
                os.remove(temp_file_name)
            except Exception as e_del:
                log_session_info(
                    f"WARNING: Could not delete temporary audio file {temp_file_name}: {e_del}", None)


try:
    import numpy as np
except ImportError:
    print("Numpy is not installed. Please install it: pip install numpy")

try:
    from scipy.signal import butter, filtfilt, medfilt, detrend
except ImportError:
    print("Scipy is not installed. Please install it: pip install scipy")

try:
    from frenztoolkit import Streamer
except ImportError as e:
    log_session_error(
        f"CRITICAL IMPORT ERROR: frenztoolkit not found: {e}", None)
    print("CRITICAL: frenztoolkit is not installed. The application will not function.", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lucid Dreaming Core FastAPI Server")
    parser.add_argument(
        "--log-filtered-data",
        action="store_true",
        help="Log the latest 5 rows of FILTERED EEG/EOG data to the terminal during an active session."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for uvicorn server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for uvicorn server (default: 8000)"
    )
    args = parser.parse_args()

    LOG_FILTERED_DATA = args.log_filtered_data

    ensure_caffeinated()

    import uvicorn
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_base_folder = BASE_RECORDING_FOLDER
    if not os.path.isabs(BASE_RECORDING_FOLDER):
        resolved_base_folder = os.path.join(script_dir, BASE_RECORDING_FOLDER)
    os.makedirs(resolved_base_folder, exist_ok=True)
    print(f"Data will be recorded in: {resolved_base_folder}")
    if LOG_FILTERED_DATA:
        print("[INFO] Filtered EEG/EOG logging is ENABLED. The latest 5 samples will be printed during active sessions.")

    uvicorn.run(app, host=args.host, port=args.port)
