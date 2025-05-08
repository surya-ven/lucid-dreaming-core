import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_welch
from mne.utils import use_log_level

def compute_alertness_score(data, sfreq=125, win_sec=10, step_sec=1, n_fft=256):
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

        alpha_theta_ratio = alpha / (theta + 1e-6)
        beta_theta_ratio  = beta / (theta + 1e-6)

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

    # Smooth the alpha/theta and beta/theta ratios with a moving average
    df['alpha_theta_smooth'] = df['alpha_theta_ratio'].rolling(window=3).mean()
    df['beta_theta_smooth']  = df['beta_theta_ratio'].rolling(window=3).mean()

    # Define alertness condition: both smoothed ratios above thresholds
    df['alert'] = (df['alpha_theta_smooth'] > 0.4) & (df['beta_theta_smooth'] > 0.3)

    # Compute rolling average of alert flag to get a continuous score
    df['alertness_score'] = df['alert'].rolling(window=20, min_periods=1).mean()

    # Get the latest alertness score
    latest_score = df['alertness_score'].iloc[-1] if not df.empty else np.nan

    return df, latest_score
