from datetime import datetime

import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch
from mne.utils import use_log_level
import numpy as np
from lightgbm import LGBMRegressor
from mne.time_frequency import psd_array_welch
import numpy as np
import mne
import joblib
from mne.preprocessing import read_ica

def preprocess_alertness_data(data, ica_model):
    # raw_data = data.reshape(-1,4).T
    raw_data = data
    raw_data = raw_data
    sfreq = 125
    ch_names = ['LF-FpZ', 'OTE_L-FpZ', 'RF-FpZ', 'OTE_R-FpZ']
    ch_types = ['eeg'] * 4
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(raw_data, info)

    data = raw.get_data()
    data_clean = np.nan_to_num(data, nan=0.0)
    raw._data = data_clean

    raw.filter(l_freq=0.5, h_freq=40)
    raw.notch_filter(freqs=60)
    ica_model.exclude = [0,1]
    ica_model.apply(raw)
    return raw

def calculate_ML_based_alertness_score(data, ica_model, lgb_model):
    win_sec = 3
    sfreq = 125
    processed_data = preprocess_alertness_data(data, ica_model)
    segment = processed_data.get_data()[:, -win_sec * sfreq :]

    alertness_score = predict_alertness_from_segment(segment, 125, lgb_model)
    return alertness_score


def predict_alertness_from_segment(segment, sfreq, model: LGBMRegressor):
    psd, f = psd_array_welch(segment, sfreq=sfreq, fmin=0.5, fmax=40, n_fft=segment.shape[-1])

    band = lambda lo, hi: psd[:, (f >= lo) & (f < hi)].mean(axis=1).mean()

    alpha, theta = band(8, 12), band(4, 8)
    beta,  gamma = band(13, 30), band(30, 40)
    delta = band(0.5, 4)

    features = np.array([
        alpha, beta, theta, gamma, delta,
        alpha/theta, beta/theta, beta/alpha,
        (beta + gamma)/(alpha + theta),
        theta/(alpha + beta)
    ]).reshape(1, -1)

    alertness_score = model.predict(features)[0]
    return alertness_score

def save_alertness_score(df, score):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = {
        "Timestamp": timestamp_str,
        "AlertnessScore": score
    }
    df["AlertnessScore_EMA"] = df["AlertnessScore"].ewm(span=20, adjust=False).mean()

def compute_alertness_score(data, sfreq=125, win_sec=3, step_sec=1, n_fft=256):
    """
    Compute alertness features and score from EEG data using sliding window PSD analysis.

    Parameters:
    - data: EEG array with shape (n_channels, n_times)
    - sfreq: Sampling frequency (Hz)
    - win_sec: Window size in seconds for each PSD segment
    - step_sec: Step size between windows (sliding window)
    - n_fft: Number of FFT points used in Welch method

    Returns:
    - df: DataFrame containing time-based features and alertness score
    - latest_score: Most recent alertness score (float)
    """
    n_win_samples = int(win_sec * sfreq)
    n_step_samples = int(step_sec * sfreq)
    features_over_time = []

    for start in range(0, data.shape[1] - n_win_samples, n_step_samples):
        segment = data[:, start:start + n_win_samples]

        with use_log_level("ERROR"):
            psds, freqs = psd_array_welch(segment, sfreq=sfreq, fmin=1, fmax=40, n_fft=n_fft)

        theta = psds[:, (freqs >= 4) & (freqs <= 8)].mean(axis=1)
        alpha = psds[:, (freqs >= 8) & (freqs <= 12)].mean(axis=1)
        beta  = psds[:, (freqs >= 13) & (freqs <= 30)].mean(axis=1)

        alpha_theta_ratio = alpha / (theta)
        beta_theta_ratio  = beta / (theta)

        features_over_time.append({
            "start_time": start / sfreq,
            "theta": theta.mean(),
            "alpha": alpha.mean(),
            "beta": beta.mean(),
            "alpha_theta_ratio": alpha_theta_ratio.mean(),
            "beta_theta_ratio": beta_theta_ratio.mean()
        })

    # Convert to DataFrame
    df = pd.DataFrame(features_over_time)

    df['alpha_theta_smooth'] = df['alpha_theta_ratio'].rolling(window=3).mean()
    df['beta_theta_smooth']  = df['beta_theta_ratio'].rolling(window=3).mean()

    df['alpha_theta_score'] = df['alpha_theta_smooth'].clip(0.2, 0.6)
    df['alpha_theta_score'] = (df['alpha_theta_score'] - 0.2) / (0.6 - 0.2)

    df['beta_theta_score'] = df['beta_theta_smooth'].clip(0.2, 0.6)
    df['beta_theta_score'] = (df['beta_theta_score'] - 0.2) / (0.6 - 0.2)

    # Combine both scores
    df['alertness_score'] = (df['alpha_theta_score'] + df['beta_theta_score']) / 2

    # Final EMA score
    df['alertness_score_ema'] = df['alertness_score'].ewm(span=100, adjust=True).mean()

    # Get the latest alertness score
    latest_score = df['alertness_score_ema'].iloc[-1] if not df.empty else np.nan

    return df, latest_score
