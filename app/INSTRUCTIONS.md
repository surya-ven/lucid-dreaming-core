Overall Goal:
Complete the implementation of a local sleep data processing system. This involves fleshing out a FastAPI backend (main.py) for real-time biosignal processing and potentially event-based audio cue delivery, and ensuring its interaction with a simple HTML/JavaScript frontend (index.html). The core logic for data acquisition, initial saving, and some signal processing examples will be adapted and expanded from test_custom_save3.py.

File Key:
main.py: The FastAPI backend application skeleton you will be completing.
index.html: The HTML/JavaScript frontend skeleton you will be ensuring integrates with main.py.
test_custom_save3.py: An existing Python script demonstrating FRENZ Brainband data streaming, basic processing, and saving. You will adapt and integrate its functionalities.
Phase 1: Backend Setup and FRENZ Toolkit Integration (in main.py)

Refer to test_custom_save3.py for FRENZ Toolkit usage patterns and data handling.
Configuration & Globals:
In main.py, define and initialize global variables and configuration constants.
Source values like PRODUCT_KEY, DEVICE_ID, BASE_RECORDING_FOLDER, FS (sampling frequency, e.g., 125.0), EEG_DATA_TYPE (e.g., np.float32) directly from test_custom_save3.py.
Define WINDOW_SIZE_S = 5 for 5-second processing windows.
Ensure session_data_path (for session-specific data) is created under BASE_RECORDING_FOLDER using a timestamp, similar to session_timestamp_str logic in test_custom_save3.py.
The session_info dictionary in test_custom_save3.py provides a good example of metadata to save; plan to incorporate this.
Streamer Initialization and Data Acquisition Loop:
In the start_session endpoint of main.py, initialize the Streamer instance from frenztoolkit using DEVICE_ID, PRODUCT_KEY, and the session-specific session_data_path for data_folder (note: test_custom_save3.py uses BASE_RECORDING_FOLDER for the streamer's data_folder; decide if session-specific path is better here or stick to base). Start the streamer.
Adapt the main data acquisition loop from test_custom_save3.py (while True: ...) into the real_time_processing_loop background task in main.py.
Continuously fetch feeg = streamer.DATA["FILTERED"]["EEG"], feog = streamer.DATA["FILTERED"]["EOG"], and reeg = streamer.DATA["RAW"]["EEG"] as shown in test_custom_save3.py. Consider fetching IMU data if the streamer.DATA dictionary provides it and it seems relevant for artifact detection.
Implement the logic for determining new_n (number of new samples based on samples_written_count) from test_custom_save3.py.
Prepare new_eeg, new_eog, and new_raw data blocks from the fetched data, mirroring test_custom_save3.py. Pay attention to the channel selection for new_raw (e.g., reeg[-new_n:, [0, 1, 3, 4]].T).
Initial Data Saving (Adapting test_custom_save3.py):
Within real_time_processing_loop, create custom_data_filepath (e.g., os.path.join(session_data_path, "custom_combined_data.dat")).
Open this file in append binary ('ab') mode when the processing loop starts.
Write the combined data block (e.g., block = np.vstack([new_eeg, new_eog, new_raw])) to this file, ensuring it's cast to EEG_DATA_TYPE.tobytes() as done in test_custom_save3.py. The column structure for this file should implicitly follow NUM_COMBINED_COLUMNS and column_names from test_custom_save3.py.
Ensure the file handle is properly closed when the loop terminates (e.g., session_active becomes False).
Phase 2: Biosignal Processing (in main.py)

Create modular functions for these steps and call them within real_time_processing_loop after acquiring new data segments.
Pre-processing:
Create a function preprocess_data(raw_eeg_window, raw_eog_window, raw_emg_window_if_available, raw_imu_window_if_available) that takes a window of raw data.
Implement robust pre-processing steps. Consider common techniques like detrending, baseline correction, and filtering (bandpass/notch).
For EOG-specific cleaning, refer to the process_eog_for_plotting function in test_custom_save3.py for examples of artifact handling (thresholding outliers, median filtering, bandpass filtering). Adapt these or similar techniques for general pre-processing.
The goal is to clean signals for feature extraction. Output cleaned windows of EEG, EOG, etc.
Windowing:
Ensure that the data processing pipeline operates on fixed-length, overlapping windows (e.g., 5 seconds, WINDOW_SIZE_S) of the incoming cleaned data. The new_n logic from test_custom_save3.py processes available chunks; ensure this is adapted or followed by logic that groups data into appropriate windows for feature extraction.
Feature Extraction:
Create a function extract_features(processed_eeg_window, processed_eog_window, processed_emg_window_if_available) that takes a window of pre-processed data.
Extract features relevant for sleep analysis. Examples:
EEG: Spectral power in different bands (e.g., delta, theta, alpha, beta), ratios of these bands.
EOG: Metrics related to eye movements (e.g., number of movements, amplitude, velocity if calculable). The horizontal EOG (new_horiz = new_eog[LEFT_EOG_CH] - new_eog[RIGHT_EOG_CH]) from test_custom_save3.py is a good starting point for EOG features.
Return a dictionary of extracted features.
Phase 3: Event Detection & Logic (in main.py)

Implement functions to detect events based on extracted features.
Sleep Stage / REM Detection (Placeholder/Basic):
Create a function detect_sleep_events(features, frenz_toolkit_scores).
Utilize streamer.SCORES["sleep_stage"] from the FRENZ toolkit (this needs to be accessed from the streamer_instance within the loop) as a primary indicator of sleep stage or REM sleep.
Based on these scores or extracted features, determine if a REM sleep period is ongoing.
This function should return information about the current sleep state, particularly identifying REM.
LRLR Eye Movement Detection (Placeholder/Basic):
Create a function detect_lrlr_movements(eog_features, is_in_rem_sleep)
If is_in_rem_sleep is true, attempt to identify distinct Left-Right-Left-Right (LRLR) eye movement patterns from EOG features.
This might start as a rule-based approach (e.g., thresholding EOG deflection magnitude and checking sequence logic).
Return whether an LRLR pattern was detected.
Audio Cue Logic (Event-Based):
Create a function manage_audio_cues(is_rem_detected, lrlr_detected_optional).
If REM is detected:
Implement a randomization logic (e.g., 50/50 chance) to decide between an "Active" cue condition and a "Sham" (no cue) condition for the REM episode.
Active Condition: Use a Python audio library (e.g., playsound or python-sounddevice, ensure it's added to requirements) to play a pre-selected audio cue file. Log the fact that a cue was played, its type/name, and the timestamp.
Sham Condition: Do not play a cue, but log that this REM period was a sham condition, along with the timestamp.
(Optional) Consider if audio cues should be triggered by LRLR detection during REM.
Phase 4: Enhanced Data Storage and Session Management (in main.py)

Expand on the initial data saving.
Metadata and Event Logging:
Create a custom_metadata_filepath (e.g., os.path.join(session_data_path, "session_log.jsonl") or .csv).
Throughout the real_time_processing_loop, log key events and processed information to this file. Each log entry should be timestamped.
Information to log:
Sleep stage indications (e.g., from streamer.SCORES).
REM detection events (start/end times).
Active/Sham condition assignments.
Audio cue playback events (type of cue, timing).
LRLR detection events.
Any errors or significant system status changes.
(Consider) Periodically save the session_info dictionary from test_custom_save3.py, perhaps updated with ongoing session stats if METADATA_SAVE_INTERVAL_SECONDS logic is adapted.
Session Data Packaging:
In the stop_session endpoint (and ensure it's called if the loop terminates unexpectedly):
Finalize all open data/log files.
Bundle all files in the session_data_path (the .dat file, the metadata/log file(s)) into a single zip file (e.g., session_timestamp_str.zip).
This zip file path should be returned by the endpoint to be displayed on the frontend.
Phase 5: API Endpoints and Frontend Integration (in main.py and index.html)

Refine API endpoints in main.py and ensure index.html JavaScript interacts correctly.
API Endpoint Robustness (main.py):
POST /session/start:
Ensure it correctly initializes and starts the Streamer and the real_time_processing_loop background task.
Handle potential errors during startup (e.g., device not found) and return appropriate JSON responses.
POST /session/stop:
Reliably signal real_time_processing_loop to stop.
Stop the Streamer.
Ensure data packaging is completed before returning the response containing the zip file name/path.
GET /session/status:
Return informative status messages reflecting the current state of the backend (e.g., "Idle", "Connecting", "Streaming & Processing", "REM Detected", "Cueing Active", "Error: ...", "Packaging data").
Frontend JavaScript (index.html):
Review and enhance the JavaScript functions in index.html.
Ensure Workspace calls correctly target the API endpoints.
Update the statusDiv with messages from /session/status.
When /session/stop is successful, clearly display the name of the zip file that the participant needs to upload and instructions to do so.
Manage button states (disabled/enabled) effectively based on the session state.
Static File Serving for Frontend (Optional, main.py):
If index.html should be served directly by the FastAPI app, add the necessary static files configuration (e.g., app.mount("/static", StaticFiles(directory="static_folder_name"), name="static") and place index.html in that static_folder_name).
Phase 6: Error Handling and General Refinements
Error Handling (main.py):
Implement try-except blocks around major operations: FRENZ toolkit interactions, file I/O, critical processing steps.
Log any exceptions encountered to your metadata/log file and update the system status accessible via /session/status.
If the FRENZ device disconnects or the stream fails, the system should attempt to handle this gracefully, log the error, update status, and potentially try to stop the session cleanly.
Code Cleanup and Modularity:
Ensure main.py is well-organized with functions for distinct tasks (e.g., initialize_streamer, process_data_window, log_event, package_session_data).
Remove any redundant code adapted from test_custom_save3.py if it's fully superseded by new logic in main.py. For instance, the live plotting from test_custom_save3.py should be entirely removed.
Add comments to explain complex logic or important decisions.