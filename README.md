# CS189 - Applied AI for Wearable Neurotech

## Note
The bulk of our experiments and scripts are under the  `experiments` folder, `app` contains our main system and real-time application code, and `test_algorithm.ipynb` contains code for testing the system offline via simulation.

## Requirements

Please install uv.
```bash
brew install uv
```
If the brew installation is giving you errors, you can find other installation methods for uv [here](https://github.com/astral-sh/uv).

Once uv is installed, run the following command in the project root to create the virtual environment and install dependencies:

```bash
uv sync
```


## 1. Pre-recorded testing/simulation of algorithm (FRENZ's data)
- NOTE: you would need to place the 20 sleep stages under a `sample_data/` folder
-   Open `test_algorithm.ipynb` to see the results from a previous run in the cells.
- **Please ensure** that the environment is using the `.venv` environment that uv created.
-   To change the session being used, please change `EXPERIMENT_ID` to the folder name of the session you want to test
-   You can run all cells to double-check and see the simulation of our logic in action.

## 2. Live run

This is the main application code used for actual data collection and where the algorithm's logic resides.
-   Please update the value `DEVICE_ID = "FRENZI40"` in `app/main.py` to the correct device ID which you're using

-   Please also connect your Brainband to your computer's audio via Bluetooth.
-   To run the application, execute the following command in the project root:
    ```bash
    uv run app/main.py
    ```
    **PLEASE NOTE**: it will take a few seconds after you run this command for the first time for everything to be cached (by uv), after that it will start running the app — so please don't worry if it does nothing for a minute
-   After running the command, open `localhost:8000` in your browser. The exact link will also be displayed in the terminal.
-   Wait for the application to detect the Brainband.

-   Click on "start session" in the web interface.

### Logging

-   A log file will be created under `app/recorded_data/<sessionID>/session_info.log`.
-   Sleep stage is logged every second.
-   Alertness score is logged every 3 seconds (this logging starts after 150 seconds into the session).
-   LRLR detections will be logged every 3 seconds during REM stage

### Audio Cue Trigger Conditions

The application will trigger an audio cue if all the following conditions are met:

1. The current sleep stage is REM.
2. The alertness score is lower than 0.6. (we're currently using 0.5 within app/main.py just to let more audio cues in for testing)
3. The last audio cue was triggered at least 180 seconds ago.
