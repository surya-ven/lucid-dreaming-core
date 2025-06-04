import sys
import select
import argparse
from frenztoolkit import Streamer
import time
import numpy as np
import os
from datetime import datetime

# For plotting
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt
from dl_alertness_detection import predict_alertness_ema

# --- Configuration ---
PRODUCT_KEY = "RUtYA4W3kpXi0i9C7VZCQJY5_GRhm4XL2rKp6cviwQI="
DEVICE_ID = "FRENZI40"
BASE_RECORDING_FOLDER = "./recorded_data"
METADATA_SAVE_INTERVAL_SECONDS = 60
EEG_DATA_TYPE = np.float32
NUM_COMBINED_COLUMNS = 12
FS = 125.0

epoch_sec = 30
seq_len = 5
input_len = epoch_sec * 125  # 3750
needed_len = seq_len * input_len  # 18750

LEFT_EOG_CH = 0  # Outer-canthus left
RIGHT_EOG_CH = 2  # Outer-canthus right

parser = argparse.ArgumentParser(
    description="FrenzToolkit Eye Movement Streamer")
parser.add_argument("--log-events", action="store_true")
parser.add_argument("--plot-live", nargs='?', const=True, default=False)
cli_args = parser.parse_args()

last_alert_time = 0

plot_live_channel = None
if cli_args.plot_live is True:
    plot_live_enabled = True
elif cli_args.plot_live is not False:
    try:
        ch = int(cli_args.plot_live)
        if 1 <= ch <= 4:
            plot_live_channel = ch - 1
            plot_live_enabled = True
        else:
            plot_live_enabled = False
    except Exception:
        plot_live_enabled = False
else:
    plot_live_enabled = False

session_start_time_obj = datetime.now()
session_timestamp_str = session_start_time_obj.strftime("%Y%m%d_%H%M%S_%f")
session_data_path = os.path.join(BASE_RECORDING_FOLDER, session_timestamp_str)
os.makedirs(session_data_path, exist_ok=True)

custom_data_filepath = os.path.join(
    session_data_path, "custom_combined_data.dat")
custom_metadata_filepath = os.path.join(
    session_data_path, "custom_metadata.npz")

plot_initialized = False
PLOT_WINDOW_DURATION_S = 20
plot_timestamps = []
plot_data_raw_eeg = [[] for _ in range(4)]
plot_data_filt_eog = [[] for _ in range(4)]
horiz_buf = []



streamer = Streamer(device_id=DEVICE_ID, product_key=PRODUCT_KEY,
                    data_folder=BASE_RECORDING_FOLDER)

samples_written_count = 0
metadata_timestamps = []
metadata_session_dur = []
data_block_timestamps = []
data_block_sample_counts = []
target_event_active = False
target_event_transitions = []

session_info = {
    "product_key": PRODUCT_KEY,
    "device_id": DEVICE_ID,
    "session_start_iso": session_start_time_obj.isoformat(),
    "custom_data_type": EEG_DATA_TYPE.__name__,
    "expected_columns": NUM_COMBINED_COLUMNS,
    "data_shape_on_save": "channels_first",
    "column_names": [
        "EEG_Filt_1", "EEG_Filt_2", "EEG_Filt_3", "EEG_Filt_4",
        "EOG_Filt_1", "EOG_Filt_2", "EOG_Filt_3", "EOG_Filt_4",
        "RAW_EEG_1", "RAW_EEG_2", "RAW_EEG_3", "RAW_EEG_4"
    ]
}


def process_eog_for_plotting(data, srate, lowcut=0.2, highcut=4.0,
                             artifact_threshold=150, mft_kernel_size=5):
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.shape[0] < 15:
        return data

    for ch in range(data.shape[1]):
        cd = data[:, ch]
        artifacts = np.abs(cd) > artifact_threshold
        if artifacts.any():
            idx = np.where(artifacts)[0]
            splits = np.where(np.diff(idx) > 1)[0] + 1
            segs = np.split(idx, splits)
            for seg in segs:
                if len(seg) == 0:
                    continue
                start, end = seg[0], seg[-1]
                pre = cd[start-1] if start > 0 else cd[end +
                                                       1] if end+1 < len(cd) else 0
                post = cd[end+1] if end + \
                    1 < len(cd) else cd[start-1] if start > 0 else 0
                data[seg, ch] = np.linspace(pre, post, len(seg))

    k = mft_kernel_size + (1 - mft_kernel_size % 2)
    try:
        data = medfilt(data, kernel_size=(k, 1))
    except ValueError:
        pass

    b, a = butter(4, [lowcut, highcut], btype='band', fs=srate)
    data = filtfilt(b, a, data, axis=0)
    return data


data_file_handle = None
last_metadata_save_time = time.time()

try:
    streamer.start()
    data_file_handle = open(custom_data_filepath, 'ab')
    if plot_live_enabled:
        plt.ion()
        fig, (ax_raw_eeg, ax_filt_eeg, ax_filt_eog) = plt.subplots(
            3, 1, sharex=True, figsize=(12, 10))
        plot_initialized = True

    while True:
        now = time.time()
        session_dur = streamer.session_dur
        if session_dur > 600:
            break

        feeg = streamer.DATA["FILTERED"]["EEG"]
        
        # print(feeg.shape)
        current_time = time.time()
        # compute alertness
        if (current_time - last_alert_time) >= 3 and feeg.shape[0] > 0:

            channel_2_data = feeg[2, :]   # shape (N,)

            last_alert_time = current_time
            if len(channel_2_data) >= needed_len:
                eeg_raw_for_pred = channel_2_data[-int(needed_len):] 
                score_ewm = predict_alertness_ema(eeg_raw_for_pred)
                print("alertness EMA socre:", score_ewm)
            else:
                print(f"Only {len(channel_2_data)}, need {needed_len} to predict")
            
        feog = streamer.DATA["FILTERED"]["EOG"]
        reeg = streamer.DATA["RAW"]["EEG"]

        if feeg is None or feog is None or feeg.ndim != 2 or feeg.shape[1] < 1:
            continue

        total = feeg.shape[1]
        new_n = total - samples_written_count
        if new_n <= 0:
            continue

        new_eeg = feeg[:, -new_n:]
        new_eog = feog[:, -new_n:]
        new_raw = reeg[-new_n:, [0, 1, 3, 4]
                       ].T if reeg is not None else np.full((4, new_n), np.nan, EEG_DATA_TYPE)
        new_horiz = new_eog[LEFT_EOG_CH] - new_eog[RIGHT_EOG_CH]

        block = np.vstack([new_eeg, new_eog, new_raw])
        data_file_handle.write(block.astype(EEG_DATA_TYPE).tobytes())
        samples_written_count += new_n
        data_block_timestamps.append(now)
        data_block_sample_counts.append(new_n)

        if plot_live_enabled and plot_initialized:
            ts = np.linspace(session_dur - (new_n - 1) /
                             FS, session_dur, new_n)
            plot_timestamps.extend(ts)
            for ch in range(4):
                plot_data_raw_eeg[ch].extend(new_raw[ch])
                plot_data_filt_eog[ch].extend(new_eog[ch])
            horiz_buf.extend(new_horiz)

            while plot_timestamps and plot_timestamps[-1] - plot_timestamps[0] > PLOT_WINDOW_DURATION_S:
                plot_timestamps.pop(0)
                for buf in (plot_data_raw_eeg, plot_data_filt_eog):
                    buf[:] = [b[1:] for b in buf]
                horiz_buf.pop(0)

            ax_raw_eeg.cla()
            ax_filt_eeg.cla()
            ax_filt_eog.cla()

            for ch in range(4):
                ax_raw_eeg.plot(plot_timestamps,
                                plot_data_raw_eeg[ch], alpha=0.5)
                ax_filt_eeg.plot(
                    plot_timestamps, plot_data_filt_eog[ch], alpha=0.3)

            clean = process_eog_for_plotting(np.array(horiz_buf), FS).flatten()
            ax_filt_eog.plot(
                plot_timestamps, clean[-len(plot_timestamps):], 'k', lw=1.5, label='H-Eye')

            ax_filt_eog.legend(fontsize='x-small')
            plt.tight_layout()
            plt.pause(0.01)

except KeyboardInterrupt:
    pass
finally:
    if data_file_handle:
        data_file_handle.close()
    if plot_live_enabled:
        plt.ioff()
        plt.show()
    print(f"Session ended. Data saved to {session_data_path}")
