# **Frenz streaming toolkit**

The Frenz Streaming Toolkit enables scientists to stream data from the Frenz Brainband to their laptop or PC, facilitating the development of custom applications. It provides access to distinct brain signals, offering a valuable alternative to PSG, the gold standard in brain signal recording. [[1](https://doi.org/10.1038/s41598-023-43975-1)].

NOTE: you can see some example programs here: [Example programs](https://www.notion.so/Example-programs-1d1fc59c5f898048a753f6e2755495b3?pvs=21)

---

### **I. Getting started**

You can install the Frenz Streaming Toolkit on your PC using pip:

`*pip install frenztoolkit*`

Alternatively, you can download the package and build it from source as:

`*pip install frenztoolkit-x.x.x-py3-none-any.whl*`

# x.x.x is version of frenztoolkit (ex: 0.2.0)

**Product Key Requirement**

A valid product key is required to use the toolkit. Please contact our sales department to obtain your product key.

**System Requirements**

Before using the toolkit, ensure you have the following:

-   Python version 3.9
-   A Frenz Brainband
-   A laptop (Mac/Windows/Linux) with Bluetooth and internet connectivity
-   A product key (contact Earable’s sales department if you don’t have one)
-   Create Python 3.9 environment:

# Check if Python 3.9 is installed

_python3.9 --version_

# Create new virtual environment

_python3.9 -m venv vir_name_

# Environment activation:

# Windows OS

_vir_name\Scripts\activate_

# macOS/Linux

_source vir_name/bin/activate_

### **II. Connecting to Your Device**

To connect your Frenz Brainband, you first need to identify its Bluetooth ID. The toolkit provides a Scanner utility to help you retrieve this ID.

**Scanning for Available Devices**

`*from frenztoolkit import Scanner*`

`*scanner = Scanner()*`

`*print(scanner.scan())*`

**Example Output:**

["DEVICE_1", "DEVICE_2", "DEVICE_3"]

This function returns a list of available Frenz Brainbands that are not currently connected to any phone or laptop.

### **III. Start your session**

Once you have the band’s Bluetooth ID, you can start your session.

Refer to the following code snippets for guidance on how to:

-   Connect to a Frenz Brainband
-   Start a session
-   Access real-time data
-   Stop a session

_import time_

_from frenztoolkit import Streamer_

_PRODUCT_KEY = "YOUR_PRODUCT_KEY"_

_DEVICE_ID = "YOUR_BRAIN_BAND_BLUETOOTH_ID"_ # ex: FRENZG15

# START A SESSION

# Config your streamer

_streamer = Streamer(_

    *device_id = DEVICE_ID,*

_product_key = PRODUCT_KEY,_

_data_folder = "./" # The folder stored your session data when it be completed_

_)_

# Start a session

_streamer.start()_

# ACCESS DATA

# When the session successfully started,

# you can access the raw brain signals.

# This is a array shape [N, 6] where stored

# data of 6 channels from the band with ordered

# LF, OTEL, REF1, RF, OTER, REF2; and the data are continuously (REF1, REF2 not use)

# added to the END of the array.

_streamer.DATA["RAW"]["EEG"]_

# You also can access to the filtered signals, which have

# applied power-line noise, proprirate band-pass filter to

# remove noises from the signals.

# The filtered signals are much better for humans to read directly.

# This is a array shape [4, N] where stored

# unit: uV (micro Volt)

# channel: LF, OTEL, RF, OTER

# The data are continuously added to the END of the array.

_streamer.DATA["FILTERED"]["EEG"]_

_streamer.DATA["FILTERED"]["EOG"]_

_streamer.DATA["FILTERED"]["EMG"]_

# Similarly, you can access to the IMU signals,

# which is recorded from the IMU sensor.

# The data have a shape of [M, 3], stored 3 channels of

# data accordingly: x, y, z;

# where x, y, z: accelerometers

# The data are continuously added to the END of the array.

_streamer.DATA["RAW"]["IMU"]_

# You can access to the PPG signals:

# which have shape [K, 3], stored data of 3 channel:

# GREEN, RED, INFRARED

# The data are continuously added to the END of the array.

_streamer.DATA["RAW"]["PPG"]_

# You also can access to the calculated scores

# from our machine learning models:

# CURRENT STATES: you can access to the latest scores by:

# Sleep stage

# Calculated for every 30 seconds

# The data are continuously added to the END of the array.

# value < 0: undefined, value = 0: awake, value = 1: light, value = 2: deep, value = 3: REM

_streamer.SCORES.get("sleep_stage")_

# POAS: probability of falling asleep

# Calculated for every 30 seconds

# Value: from 0-1

# The data are continuously added to the END of the array.

_streamer.SCORES.get("poas")_

# Posture

# Calculated for every 5 seconds

# The data are continuously added to the END of the array.

_streamer.SCORES.get("posture")_

# FOCUS

# Calculated for every 2 seconds

# Value: from 1-100

# The data are continuously added to the END of the array.

_streamer.SCORES.get("focus_score")_

# Signal quality

# Shape: (LF, OTEL, RF, OTER), calculated for every 5 seconds

# The data are continuously added to the END of the array.

# value 0 - not good; value 1 - good

_streamer.SCORES.get("sqc_scores")_

# HISTORICAL STATES: you also can access to the historical scores

# by adding `array__` to the scores

# POAS: probability of falling asleep

# List of POAS

# Value: from 0-1

# The data are continuously added to the END of the array.

_streamer.SCORES.get("array\_\_poas")_

# Posture:

# List of posture

# The data are continuously added to the END of the array.

_streamer.SCORES.get("array\_\_posture")_

# Sleep stage

# List of Sleep Stage

# The data are continuously added to the END of the array.

_streamer.SCORES.get("array\_\_sleep_stage")_

# Signal quality

# List of Signal Quality

# The data are continuously added to the END of the array.

_streamer.SCORES.get("array\_\_sqc_scores")_

# FOCUS

# List of focus score

# Value: from 1-100

# The data are continuously added to the END of the array.

_streamer.SCORES.get("array\_\_focus_score")_

# To be updated: We still have many more scores.

# STOP YOUR SESSION

# Limit your session by duration

_while True:_

_if streamer.session_dur > 10000:_

_streamer.stop()_

_break_

# YOUR CAN DO SOMETHING WITH THE REAL TIME STREAMING DATA HERE

_time.sleep(5)_

# Or press[CTRL + Q] for the window or [CMD+Q] For Mac to quit the session.

# When the session stopped, raw signals and scores will be stored in the

# data_folder folder.
