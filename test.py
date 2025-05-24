from frenztoolkit import Streamer
import time
import matplotlib.pyplot as plt

PRODUCT_KEY = "RUtYA4W3kpXi0i9C7VZCQJY5_GRhm4XL2rKp6cviwQI="
DEVICE_ID = "FRENZI40"

# Initialize the streamer with your device ID and product key
streamer = Streamer(
    device_id=DEVICE_ID,
    product_key=PRODUCT_KEY,
    data_folder="./recorded_data"
)

# Start the streaming session
streamer.start()

while True:
    if streamer.session_dur > 10*60:
        break

    eeg = streamer.DATA["RAW"]["EEG"]
    print("EEG shape:", eeg.shape)
    print("EEG data:", eeg[:5, :])

    posture = streamer.SCORES.get("posture")
    poas = streamer.SCORES.get("poas")
    sleep_stage = streamer.SCORES.get("sleep_stage")

    print("AT TIME: ", streamer.session_dur)
    print("EEG shape:", eeg.shape)
    print("Latest POSTURE:", posture)
    print("Latest POAS:", poas)
    print("Latest Sleep Stage:", sleep_stage)
    print("\n")

    time.sleep(5)

# Stop the session and save data to disk
streamer.stop()


