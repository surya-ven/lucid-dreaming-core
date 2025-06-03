import mne
import yasa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments




def load_process_eeg_datafile(session_folder_path):
    channel_eeg = 6

    # Load the recorded session from your experiment folder
    memmap_eeg_data = np.memmap(session_folder_path, dtype=np.float64, mode='r')
    l_eeg = int(len(memmap_eeg_data)//channel_eeg)

    # EEG DATA
    # Reshape the data into a 2D array
    # l_eeg is the number of samples in the eeg data
    # channel_eeg is the number of channels in the eeg data
    eeg_data = np.array(memmap_eeg_data)[:l_eeg*channel_eeg].reshape((l_eeg, channel_eeg))
    raw_data = eeg_data[:, [0, 1, 3, 4]].T * 1e-8

    channel_names = ['LF-FpZ', 'OTE_L-FpZ', 'RF-FpZ', 'OTE_R-FpZ']
    sfreq = 125 

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(channel_names)
    )

    raw = mne.io.RawArray(raw_data, info)
    raw.pick_channels(channel_names)

    raw.filter(0.5,40)
    raw.notch_filter(60)
    return raw
    


# Load the EDF file
raw_edf_filepath = "provided_data/night_01.edf"
raw_dat_filepath = "recorded_data/liu_sleep/1748599265.051497/eeg.dat"

# raw = mne.io.read_raw_edf(raw_edf_filepath, preload=True, verbose=False)
raw = load_process_eeg_datafile(raw_dat_filepath)


print('Available channels:', raw.ch_names)
print('Sampling frequency:', raw.info['sfreq'])

# Channel mapping - your channels roughly correspond to:
# 'LF–FpZ' -> Fp1 (left frontal)
# 'OTE_L–FpZ' -> T3 (left temporal)
# 'RF–FpZ' -> Fp2 (right frontal)
# 'OTE_R–FpZ' -> T4 (right temporal)

# Rename channels to more standard names for YASA
channel_mapping = {
    'LF-FpZ': 'Fp1',
    'OTE_L-FpZ': 'T3', 
    'RF-FpZ': 'Fp2',
    'OTE_R-FpZ': 'T4'
}

raw.rename_channels(channel_mapping)
print('Renamed channels:', raw.ch_names)

# Perform sleep staging using the best available EEG channel
# Using Fp2 as it's often a good central-frontal derivation for sleep staging
sls = yasa.SleepStaging(raw, eeg_name="Fp2")
print('Sleep staging model initialized')

# Predict sleep stages
y_pred = sls.predict()
print('Sleep stages predicted')

# Print predicted hypnogram
print('Predicted sleep stages:')
print(y_pred)

# Convert to Hypnogram object for plotting and statistics (YASA 0.6.5 compatibility)
hypno = yasa.Hypnogram(y_pred, freq="30s")

# Plot the predicted hypnogram
fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True, dpi=100)
ax = hypno.plot_hypnogram(fill_color="lightblue", ax=ax)
ax.set_title('Predicted Sleep Stages - night_01.edf')
plt.savefig('night_01_hypnogram.png', dpi=150, bbox_inches='tight')
print('Hypnogram plot saved as night_01_hypnogram.png')
plt.close()

# Calculate sleep statistics
sleep_stats = hypno.sleep_statistics()
print('\nSleep Statistics:')
print(sleep_stats)

# Get prediction probabilities and confidence
proba = sls.predict_proba()
confidence = proba.max(1)

# Create a summary dataframe
df_results = hypno.as_int().to_frame(name='Stage')
df_results['Confidence'] = confidence
print(df_results.head(10))

# Export results to CSV
output_file = "night_01_sleep_stages.csv"
df_results.to_csv(output_file)
print(f'\nResults exported to: {output_file}')

# Plot prediction probabilities
fig, ax = plt.subplots(figsize=(12, 6))
sls.plot_predict_proba()
plt.title('Sleep Stage Prediction Probabilities - night_01.edf')
plt.savefig('night_01_probabilities.png', dpi=150, bbox_inches='tight')
print('Prediction probabilities plot saved as night_01_probabilities.png')
plt.close()

print(f'\nOverall summary:')
print(f'Total epochs: {len(y_pred)}')
print(f'Recording duration: {len(y_pred) * 30 / 3600:.1f} hours')  # 30 seconds per epoch
print(f'Average confidence: {confidence.mean():.3f}')