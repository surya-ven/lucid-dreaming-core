import asyncio
import time
import numpy as np
import os
import shutil
import unittest
from unittest.mock import patch, MagicMock

# Ensure the 'app' module can be found.
# This assumes 'test_lrlr_in_main.py' is in the root 'frenztoolkit' directory.
from app import main as app_main

# Define a temporary directory for test artifacts
TEST_SESSION_DIR = os.path.join(os.path.dirname(
    __file__), "temp_test_session_data_lrlr")


class TestLRLRIntegration(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Ensure the test directory is clean
        if os.path.exists(TEST_SESSION_DIR):
            shutil.rmtree(TEST_SESSION_DIR)
        os.makedirs(TEST_SESSION_DIR)

        # Override app_main's global session_data_path for tests
        self.original_session_data_path = app_main.session_data_path
        app_main.session_data_path = TEST_SESSION_DIR

        # Reset relevant global states from app_main before each test
        app_main.LRLR_COMPONENT_LOADED = True  # Assume components are loaded for test
        app_main.lrlr_detection_active = False
        app_main.last_lrlr_detection_time = 0.0
        app_main.metadata_lrlr_detections = []

        # Ensure EEG_DATA_TYPE is set (it's a global in app_main, usually float32)
        # This should match the definition in app_main.py
        app_main.EEG_DATA_TYPE = np.float32
        app_main.LRLR_EOG_CHANNELS = 4  # Should match app_main.py

        app_main.eog_data_buffer_for_lrlr = np.empty(
            (0, app_main.LRLR_EOG_CHANNELS), dtype=app_main.EEG_DATA_TYPE)

        app_main.is_in_rem_cycle = False
        app_main.current_sleep_stage = 0  # Non-REM
        app_main.session_active = True  # Simulate active session for the loop

        # Mock essential parts of session_info_global
        # FS is critical for some internal logic if it were used (e.g. buffer trimming by time)
        app_main.session_info_global = {
            "device_id": "test_device",
            "fs": 250,  # Sample rate
            "eeg_eog_data_info": {"num_channels": 8, "channel_names": [f"CH{i}" for i in range(8)]},
            "shape_on_save": "channel-major",
            "start_time_unix": time.time(),
            "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        }
        # Used by _save_final_metadata
        app_main.current_session_info = app_main.session_info_global

        # REM_SLEEP_STAGE_VALUE is defined as 4 in app_main.py
        app_main.REM_SLEEP_STAGE_VALUE = 4

        # Override model sample length and detection interval for faster tests
        self.original_lrlr_model_sample_length = app_main.LRLR_MODEL_SAMPLE_LENGTH
        self.original_lrlr_detection_interval_s = app_main.LRLR_DETECTION_INTERVAL_S
        app_main.LRLR_MODEL_SAMPLE_LENGTH = 10
        app_main.LRLR_DETECTION_INTERVAL_S = 1.0

        # Mock app.state for the rem_cue_task check in full real_time_processing_loop
        # Our simulate_data_processing_cycle bypasses this, but good practice if we were calling more broadly
        # If app_main.app (FastAPI instance) isn't directly available
        if not hasattr(app_main, 'app'):
            app_main.app = MagicMock()
        app_main.app.state = MagicMock()
        app_main.app.state.rem_cue_task = MagicMock()
        app_main.app.state.rem_cue_task.done = MagicMock(return_value=True)

    def tearDown(self):
        app_main.session_active = False  # Stop simulated loop
        if os.path.exists(TEST_SESSION_DIR):
            shutil.rmtree(TEST_SESSION_DIR)

        # Restore original values
        app_main.session_data_path = self.original_session_data_path
        app_main.LRLR_MODEL_SAMPLE_LENGTH = self.original_lrlr_model_sample_length
        app_main.LRLR_DETECTION_INTERVAL_S = self.original_lrlr_detection_interval_s

    async def simulate_data_processing_cycle(self, data_block_eeg_eog, sleep_stage):
        """
        Simulates one cycle of the data processing logic relevant to LRLR
        from app_main.real_time_processing_loop.
        """
        app_main.current_sleep_stage = sleep_stage

        # --- EOG Data Buffering (adapted from app_main.real_time_processing_loop) ---
        if app_main.LRLR_COMPONENT_LOADED and app_main.get_lrlr is not None and app_main.eog_data_buffer_for_lrlr is not None:
            if data_block_eeg_eog is not None and data_block_eeg_eog.size > 0:
                num_total_channels = data_block_eog_eog.shape[0]
                if num_total_channels >= app_main.LRLR_EOG_CHANNELS:
                    new_eog_data = data_block_eeg_eog[-app_main.LRLR_EOG_CHANNELS:, :].T
                    if new_eog_data.dtype != app_main.eog_data_buffer_for_lrlr.dtype:
                        new_eog_data = new_eog_data.astype(
                            app_main.eog_data_buffer_for_lrlr.dtype)
                    app_main.eog_data_buffer_for_lrlr = np.vstack(
                        (app_main.eog_data_buffer_for_lrlr, new_eog_data))

                    max_buffer_len = app_main.LRLR_MODEL_SAMPLE_LENGTH * 2
                    if app_main.eog_data_buffer_for_lrlr.shape[0] > max_buffer_len:
                        app_main.eog_data_buffer_for_lrlr = app_main.eog_data_buffer_for_lrlr[-max_buffer_len:, :]
                # else: app_main logs error (omitted for test simplicity)

        # --- REM Cycle Logic (adapted for LRLR activation) ---
        if app_main.current_sleep_stage == app_main.REM_SLEEP_STAGE_VALUE:
            if not app_main.is_in_rem_cycle:
                app_main.is_in_rem_cycle = True
                # This simulates the effect of fire_rem_audio_cues_sequence() on LRLR state
                if app_main.LRLR_COMPONENT_LOADED and app_main.get_lrlr is not None:
                    app_main.lrlr_detection_active = True
                    app_main.last_lrlr_detection_time = time.time()
        else:  # Not in REM
            if app_main.is_in_rem_cycle:
                app_main.is_in_rem_cycle = False
                if app_main.LRLR_COMPONENT_LOADED and app_main.get_lrlr is not None:
                    if app_main.lrlr_detection_active:  # Log in app_main
                        pass
                    app_main.lrlr_detection_active = False
                    app_main.last_lrlr_detection_time = 0.0

        # --- LRLR Detection Logic (adapted) ---
        if app_main.LRLR_COMPONENT_LOADED and app_main.get_lrlr is not None and \
           app_main.lrlr_detection_active and app_main.eog_data_buffer_for_lrlr is not None:

            current_time_for_lrlr_check = time.time()
            # Ensure last_lrlr_detection_time is not 0 if active, otherwise interval check is always true
            if app_main.last_lrlr_detection_time == 0.0 and app_main.lrlr_detection_active:
                app_main.last_lrlr_detection_time = current_time_for_lrlr_check - \
                    app_main.LRLR_DETECTION_INTERVAL_S  # Ensure first check can pass

            if (current_time_for_lrlr_check - app_main.last_lrlr_detection_time >= app_main.LRLR_DETECTION_INTERVAL_S):
                if app_main.eog_data_buffer_for_lrlr.shape[0] >= app_main.LRLR_MODEL_SAMPLE_LENGTH:
                    eog_for_model = app_main.eog_data_buffer_for_lrlr[-app_main.LRLR_MODEL_SAMPLE_LENGTH:, :]
                    eog_for_model_contiguous = np.ascontiguousarray(
                        eog_for_model, dtype=np.float32)

                    # run_in_threadpool and get_lrlr are mocked in the test method
                    is_lrlr, score = await app_main.run_in_threadpool(app_main.get_lrlr, eog_for_model_contiguous)

                    detection_ts = time.time()  # Timestamp of when detection result is available
                    app_main.metadata_lrlr_detections.append(
                        (detection_ts, bool(is_lrlr), float(score)))
                    app_main.last_lrlr_detection_time = current_time_for_lrlr_check
                # else: app_main logs "not enough EOG data" (omitted for test simplicity)

    @patch('app.main.run_in_threadpool')
    # Mock the actual get_lrlr function in app_main's scope
    @patch('app.main.get_lrlr')
    async def test_lrlr_detection_flow(self, mock_get_lrlr_in_main, mock_run_in_threadpool):
        # Configure mocks
        async def side_effect_run_in_threadpool(func_to_run, *args, **kwargs):
            # This simulates run_in_threadpool calling the (mocked) get_lrlr
            # func_to_run will be mock_get_lrlr_in_main here
            return func_to_run(*args, **kwargs)
        mock_run_in_threadpool.side_effect = side_effect_run_in_threadpool

        mock_get_lrlr_in_main.side_effect = [
            (True, 0.85),  # First call result
            (False, 0.15)  # Second call result
        ]

        # --- Test Scenario ---
        self.assertFalse(app_main.lrlr_detection_active,
                         "LRLR should be inactive initially")
        self.assertEqual(len(app_main.metadata_lrlr_detections),
                         0, "LRLR metadata should be empty initially")

        dummy_eeg_data_block = np.random.rand(8, 5).astype(
            app_main.EEG_DATA_TYPE)  # 8 channels, 5 samples

        # Non-REM
        await self.simulate_data_processing_cycle(dummy_eeg_data_block, sleep_stage=0)
        self.assertFalse(app_main.lrlr_detection_active,
                         "LRLR should be inactive after non-REM data")
        self.assertEqual(
            app_main.eog_data_buffer_for_lrlr.shape[0], 5, "EOG buffer should have 5 samples")

        # Enter REM
        await self.simulate_data_processing_cycle(None, sleep_stage=app_main.REM_SLEEP_STAGE_VALUE)
        self.assertTrue(app_main.is_in_rem_cycle, "Should be in REM cycle")
        self.assertTrue(app_main.lrlr_detection_active,
                        "LRLR should activate upon entering REM")
        initial_lrlr_activation_time = app_main.last_lrlr_detection_time
        self.assertNotEqual(initial_lrlr_activation_time,
                            0.0, "last_lrlr_detection_time should be set")

        # Add 5 more samples
        await self.simulate_data_processing_cycle(dummy_eeg_data_block, sleep_stage=app_main.REM_SLEEP_STAGE_VALUE)
        self.assertEqual(
            app_main.eog_data_buffer_for_lrlr.shape[0], 10, "EOG buffer should have 10 samples")

        # Wait for LRLR detection interval and trigger detection
        await asyncio.sleep(app_main.LRLR_DETECTION_INTERVAL_S + 0.1)
        # Process to trigger
        await self.simulate_data_processing_cycle(None, sleep_stage=app_main.REM_SLEEP_STAGE_VALUE)

        self.assertEqual(mock_get_lrlr_in_main.call_count,
                         1, "get_lrlr should be called once")
        self.assertEqual(len(app_main.metadata_lrlr_detections),
                         1, "One LRLR detection should be recorded")
        self.assertTrue(
            app_main.metadata_lrlr_detections[0][1], "First detection should be True (is_lrlr)")
        self.assertAlmostEqual(
            app_main.metadata_lrlr_detections[0][2], 0.85, places=2, msg="First score mismatch")
        self.assertNotEqual(app_main.last_lrlr_detection_time,
                            initial_lrlr_activation_time, "last_lrlr_detection_time should update")

        # Add more data, trigger second detection
        time_before_second_detection_check = app_main.last_lrlr_detection_time
        # Buffer more (total 15)
        await self.simulate_data_processing_cycle(dummy_eeg_data_block, sleep_stage=app_main.REM_SLEEP_STAGE_VALUE)
        self.assertEqual(
            app_main.eog_data_buffer_for_lrlr.shape[0], 15, "EOG buffer should have 15 samples")

        await asyncio.sleep(app_main.LRLR_DETECTION_INTERVAL_S + 0.1)
        # Process to trigger
        await self.simulate_data_processing_cycle(None, sleep_stage=app_main.REM_SLEEP_STAGE_VALUE)

        self.assertEqual(mock_get_lrlr_in_main.call_count, 2,
                         "get_lrlr should be called twice")
        self.assertEqual(len(app_main.metadata_lrlr_detections),
                         2, "Two LRLR detections should be recorded")
        self.assertFalse(
            app_main.metadata_lrlr_detections[1][1], "Second detection should be False (is_lrlr)")
        self.assertAlmostEqual(
            app_main.metadata_lrlr_detections[1][2], 0.15, places=2, msg="Second score mismatch")
        self.assertNotEqual(app_main.last_lrlr_detection_time,
                            time_before_second_detection_check, "last_lrlr_detection_time should update again")

        # Exit REM
        await self.simulate_data_processing_cycle(None, sleep_stage=0)
        self.assertFalse(app_main.is_in_rem_cycle, "Should exit REM cycle")
        self.assertFalse(app_main.lrlr_detection_active,
                         "LRLR should deactivate upon exiting REM")

        # Save metadata
        await app_main._save_final_metadata(app_main.session_data_path, app_main.current_session_info)

        metadata_file = os.path.join(
            app_main.session_data_path, "session_metadata.npz")
        self.assertTrue(os.path.exists(metadata_file),
                        "Metadata file should be created")

        loaded_metadata = np.load(metadata_file, allow_pickle=True)
        self.assertIn("lrlr_detections", loaded_metadata,
                      "lrlr_detections key should be in metadata")
        lrlr_data_from_file = loaded_metadata["lrlr_detections"]
        self.assertEqual(len(lrlr_data_from_file), 2,
                         "LRLR data from file has incorrect length")

        self.assertTrue(
            lrlr_data_from_file[0]['is_lrlr'], "File: First detection is_lrlr mismatch")
        self.assertAlmostEqual(
            lrlr_data_from_file[0]['score'], 0.85, places=2, msg="File: First score mismatch")
        self.assertFalse(
            lrlr_data_from_file[1]['is_lrlr'], "File: Second detection is_lrlr mismatch")
        self.assertAlmostEqual(
            lrlr_data_from_file[1]['score'], 0.15, places=2, msg="File: Second score mismatch")


if __name__ == '__main__':
    unittest.main()
