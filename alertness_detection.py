import json
from datetime import datetime
import pandas as pd
from mne.utils import use_log_level
from mne.time_frequency import psd_array_welch
import numpy as np
import mne
import joblib
from tensorflow.keras.models import load_model

scaler = joblib.load('models/RandomState42_MLP_scaler.pkl')
dp_model = load_model('models/alertness_mlp_regressor.h5')
orp_lookup = json.load(open('models/orp_lookup.json'))
qs = {k: np.array(v) for k, v in orp_lookup['quantiles'].items()}
b2o = {int(k): v for k, v in orp_lookup['bin_orp'].items()}

df_alert = pd.DataFrame(columns=['timestamp', 'alertness_raw'])

win_sec = 15
step_sec = 5
segment_length = 30
sfreq = 125
n_win, n_step = int(win_sec * sfreq), int(step_sec * sfreq)

def preprocess_alertness_data(data):
    # raw_data = data.reshape(-1,4).T
    raw_data = data
    raw_data = raw_data
    ch_names = ['LF-FpZ', 'OTE_L-FpZ', 'RF-FpZ', 'OTE_R-FpZ']
    ch_types = ['eeg'] * 4
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(raw_data, info)

    raw.pick_channels(['RF-FpZ'])
    raw.filter(l_freq=0.5, h_freq=40)
    raw.notch_filter(freqs=60)

    return raw

def calculate_DL_based_alertness_score(data):
    segment = data[:, -segment_length * sfreq:]

    segment = preprocess_alertness_data(segment).get_data()

    return predict_alertness_from_segment(segment)


def predict_alertness_from_segment(data):
    raw_feats, label_list, stage_list = [], [], []

    for st in range(0, data.shape[1] - n_win, n_step):
        seg = data[:, st:st + n_win]
        psd, f = psd_array_welch(seg, sfreq, fmin=1, fmax=40, n_fft=n_win)
        idx = lambda lo, hi: np.logical_and(f >= lo, f < hi)

        delta = psd[:, idx(1, 4)].mean()
        theta = psd[:, idx(4, 8)].mean()
        alpha = psd[:, idx(8, 13)].mean()
        beta  = psd[:, idx(13, 30)].mean()
        gamma = psd[:, idx(30, 40)].mean()

        total = delta + theta + alpha + beta
        rel_delta = delta / total
        rel_theta = theta / total
        rel_alpha = alpha / total

        alpha_ratio = alpha / (alpha + theta + beta)
        theta_alpha_ratio = theta / alpha
        delta_theta_ratio = delta / theta
        beta_delta_ratio = beta / delta


        psd_flat = psd.flatten()
        psd_norm = psd_flat / psd_flat.sum()
        spec_entropy = -(psd_norm * np.log(psd_norm)).sum()

        sig = seg.flatten()
        var0 = np.var(sig)
        var1 = np.var(np.diff(sig))
        var2 = np.var(np.diff(np.diff(sig)))
        hj_mobility = np.sqrt(var1 / var0)
        hj_complexity = np.sqrt(var2 / var1)

        cumsum_psd = np.cumsum(psd_flat)
        sef95 = np.interp(cumsum_psd[-1] * 0.95, cumsum_psd, np.repeat(f, psd.shape[0]))

        n_3s = int(sfreq * 3)
        orp_values = []
        for k in range(0, seg.shape[1] - n_3s + 1, n_3s):
            sub_seg = seg[:, k:k+n_3s]
            sub_psd, freqs = psd_array_welch(sub_seg, sfreq=sfreq, fmin=1, fmax=40, n_fft=n_3s)
            bands = four_band_power(sub_psd, freqs)
            ranks = [digitize_rank(bands[j], qs[list(qs)[j]]) for j in range(4)]
            bin_id = make_bin_id(ranks)
            orp_val = b2o.get(bin_id, 1.25)
            orp_values.append(orp_val)
        orp_mean = np.mean(orp_values)
        orp_std = np.std(orp_values)
        orp_range = np.ptp(orp_values)
        orp_slop  = np.diff(orp_values).mean()
        orp_cv = orp_std / (orp_mean + 1e-12)

        raw_feats.append([
            # delta, theta, alpha, beta, gamma,
            rel_delta, rel_theta, rel_alpha,
            alpha_ratio, theta_alpha_ratio, delta_theta_ratio,
            beta_delta_ratio,
            spec_entropy, hj_mobility, hj_complexity, sef95,
            orp_mean, orp_std, orp_range, orp_slop, orp_cv
        ])

    feature_names = [
            # 'delta', 'theta', 'alpha', 'beta', 'gamma',
            'rel_delta_power', 'rel_theta_power', 'rel_alpha_power',
            'alpha_ratio', 'theta_alpha_ratio', 'delta_theta_ratio',
            'beta_delta_ratio',
            'spectral_entropy', 'hjorth_mobility', 'hjorth_complexity', 'sef95',
            'orp_mean', 'orp_std', 'orp_range', 'orp_slope', 'orp_cv'
    ]

    df_feat = pd.DataFrame(raw_feats, columns=feature_names)
    for col in feature_names:
        df_feat[col] = df_feat[col].ewm(span=3, adjust=False).mean()

    X = scaler.transform(df_feat[feature_names])

    df_feat['alert_pred'] = dp_model.predict(X, batch_size = 2048)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alertness = df_feat['alert_pred'].iloc[-1]

    df_alert.loc[len(df_alert)] = [timestamp_str, alertness]

    df_alert["alertness_ema"] = df_alert["alertness_raw"].ewm(span=20, adjust=False).mean()

    ema_alertness = df_alert["alertness_ema"].iloc[-1]

    return alertness, ema_alertness, ema_alertness < 0.35

def save_alertness_score(df, score):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = {
        "Timestamp": timestamp_str,
        "AlertnessScore": score
    }

    df["alertness_ema"] = df["alertness_raw"].ewm(span=20, adjust=False).mean()

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

def digitize_rank(value, thresholds):
    return np.searchsorted(thresholds, value)

def make_bin_id(ranks):
    return ranks[0]*1000 + ranks[1]*100 + ranks[2]*10 + ranks[3]

def four_band_power(psd, freqs):
    return np.array([
        psd[:, np.logical_and(freqs >= 0.3,  freqs < 2.3)].mean(),
        psd[:, np.logical_and(freqs >= 2.7,  freqs < 6.3)].mean(),
        psd[:, np.logical_and(freqs >= 7.3, freqs < 14.0)].mean(),
        psd[:, np.logical_and(freqs >= 14.3, freqs < 35.0)].mean(),
    ])