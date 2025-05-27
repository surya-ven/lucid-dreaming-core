import pandas as pd
import numpy as np

import re
import mne
from tqdm import tqdm
from mne.time_frequency import psd_array_welch






sleep_stage = '/recorded_data/n1sleepdata/sleep_stage.csv'
signaledf = '/recorded_data/n1sleepdata/signal.edf'
signaldat = '/recorded_data/n1sleepdata/signal.dat'


sleep_data = pd.read_csv(sleep_stage)




def get_soft_label(sleep_series, center_idx, radius=100, sigma=4.0):
    weights = []
    scores = []

    for offset in range(-radius, radius + 1):
        idx = center_idx + offset
        if 0 <= idx < len(sleep_series):
            stage = sleep_series.iloc[idx]
            if stage in stage_to_score:
                score = stage_to_score[stage]
                weight = np.exp(-0.5 * (offset / sigma)**2)
                scores.append(score * weight)
                weights.append(weight)

    if weights:
        return np.sum(scores) / np.sum(weights)
    else:
        return np.nan

def extract_edf_file_to_npy_with_orp(file_path, orp_lookup_path):
    match = re.search(r'night_\d+', file_path)
    night_id = match.group() if match else ''
    print(f'start process data for {night_id}')

    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.pick_channels(['RF-FpZ'])
    raw.filter(l_freq=0.5, h_freq=40)
    raw.notch_filter(freqs=60)

    sleep_file_path = file_path.replace('.edf', '_label.csv')
    sleep_data = pd.read_csv(sleep_file_path)
    sleep_series = sleep_data['Sleep stage'].reset_index(drop=True)

    sfreq = raw.info['sfreq']
    data = raw.get_data() 
    print(f"sfreq: {sfreq}")

    win_sec, step_sec = 15, 5
    n_win, n_step = int(win_sec * sfreq), int(step_sec * sfreq)

    raw_feats, label_list, stage_list = [], [], []

    for st in tqdm(range(0, data.shape[1] - n_win, n_step), desc="Extracting windows"):
        seg = data[:, st:st + n_win]
        psd, f = psd_array_welch(seg, sfreq, fmin=1, fmax=40, n_fft=n_win)
        idx = lambda lo, hi: np.logical_and(f >= lo, f < hi)

        epoch_idx = int((st / sfreq) // 30)
        if epoch_idx >= len(sleep_series):
            break


        delta = psd[:, idx(1, 4)].mean()
        theta = psd[:, idx(4, 8)].mean()
        alpha = psd[:, idx(8, 13)].mean()
        beta = psd[:, idx(13, 30)].mean()
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

        arousal_intensity = beta / (delta)


# === ORP 特征 ===
# 将当前 15 秒段划分为 3 秒 epoch（共 5 个）
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
        orp_slop  = np.diff(orp_values).mean()
        orp_cv = orp_std / (orp_mean + 1e-12)



        raw_feats.append([
            # delta, theta, alpha, beta, gamma,
            rel_delta, rel_theta, rel_alpha, 
            alpha_ratio, theta_alpha_ratio, delta_theta_ratio,
            beta_delta_ratio,
            spec_entropy, hj_mobility, hj_complexity, sef95,
            orp_mean, orp_std, orp_range, orp_slop, orp_cv
        ])


        stage = sleep_series.iloc[epoch_idx]
        label = np.clip(get_soft_label(sleep_series, epoch_idx, radius=5), 0, 1)
        stage_list.append(stage)
        label_list.append(label)

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

    df_feat['label'] = label_list
    df_feat['stage'] = stage_list
    print(f"Done preprocessing for {night_id}")
    return df_feat