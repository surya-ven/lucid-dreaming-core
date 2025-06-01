import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mne
from datetime import datetime
import os

class DeepSleepNetBinary(nn.Module):
    def __init__(self, input_len=3000, cnn_feat_dim=128, lstm_hidden=128, seq_len=20):
        super().__init__()
        # Multi-scale CNN branch 1 (large kernel)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=0),  # (batch, 64, ?)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Multi-scale CNN branch 2 (small kernel)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=0),  # (batch, 64, ?)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        # 计算flatten后输出的特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 1, input_len)
            feat1 = self.cnn1(dummy).view(1, -1).shape[1]
            feat2 = self.cnn2(dummy).view(1, -1).shape[1]
        self.cnn_out_dim = feat1 + feat2

        # 全连接用于 residual shortcut，输出改为256
        self.fc_residual = nn.Sequential(
            nn.Linear(self.cnn_out_dim, 256),  # <--- 改这里
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        # Output layer: binary classification
        self.final_fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, 3000)
        b, t, n = x.shape
        x = x.view(b * t, 1, n)  # (batch*seq, 1, 3000)
        # Multi-scale CNN
        x1 = self.cnn1(x).view(b * t, -1)
        x2 = self.cnn2(x).view(b * t, -1)
        cnn_feat = torch.cat([x1, x2], dim=1)  # (batch*seq, feat)
        # Residual shortcut
        res_feat = self.fc_residual(cnn_feat)  # (batch*seq, 256)
        # Reshape for LSTM
        cnn_feat_seq = cnn_feat.view(b, t, -1)  # (batch, seq_len, feat)
        # LSTM
        lstm_out, _ = self.lstm(cnn_feat_seq)
        # 用最后一个时刻的 LSTM 输出
        lstm_last = lstm_out[:, -1, :]  # (batch, hidden*2=256)
        # Add residual (res_feat 对应最后一帧)
        res_feat_seq = res_feat.view(b, t, -1)[:, -1, :]  # (batch, 256)
        fused = lstm_last + res_feat_seq  # (batch, 256)
        # Output
        logits = self.final_fc(fused).squeeze(1)  # (batch,)
        return logits

class DeepSleepNetREMBinary(nn.Module):
    def __init__(self, input_len=3000, cnn_feat_dim=128, lstm_hidden=128, seq_len=20):
        super().__init__()
        # Multi-scale CNN branch 1 (large kernel)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=0),  # (batch, 64, ?)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Multi-scale CNN branch 2 (small kernel)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=0),  # (batch, 64, ?)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        # 计算flatten后输出的特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 1, input_len)
            feat1 = self.cnn1(dummy).view(1, -1).shape[1]
            feat2 = self.cnn2(dummy).view(1, -1).shape[1]
        self.cnn_out_dim = feat1 + feat2

        # 全连接用于 residual shortcut，输出改为256
        self.fc_residual = nn.Sequential(
            nn.Linear(self.cnn_out_dim, 256),  # <--- 改这里
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        # Output layer: binary classification
        self.final_fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, 3000)
        b, t, n = x.shape
        x = x.view(b * t, 1, n)  # (batch*seq, 1, 3000)
        # Multi-scale CNN
        x1 = self.cnn1(x).view(b * t, -1)
        x2 = self.cnn2(x).view(b * t, -1)
        cnn_feat = torch.cat([x1, x2], dim=1)  # (batch*seq, feat)
        # Residual shortcut
        res_feat = self.fc_residual(cnn_feat)  # (batch*seq, 256)
        # Reshape for LSTM
        cnn_feat_seq = cnn_feat.view(b, t, -1)  # (batch, seq_len, feat)
        # LSTM
        lstm_out, _ = self.lstm(cnn_feat_seq)
        # 用最后一个时刻的 LSTM 输出
        lstm_last = lstm_out[:, -1, :]  # (batch, hidden*2=256)
        # Add residual (res_feat 对应最后一帧)
        res_feat_seq = res_feat.view(b, t, -1)[:, -1, :]  # (batch, 256)
        fused = lstm_last + res_feat_seq  # (batch, 256)
        # Output
        logits = self.final_fc(fused).squeeze(1)  # (batch,)
        return logits



fs = 100
seq_len = 5
epoch_sec = 30
input_len = epoch_sec * fs

# Get the absolute path to the current file's directory (lucid-dreaming-core)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_weight_path = os.path.join(BASE_DIR, 'models', 'alertness_torch_weights_salt.pth')
model = DeepSleepNetBinary(input_len=3000, seq_len=seq_len)
model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
model.eval()

rem_model_weight_path = os.path.join(BASE_DIR, 'models', 'REM_binary_model_weights.pth')
rem_seq_len = 5
rem_model = DeepSleepNetREMBinary(input_len=3000, seq_len=rem_seq_len)
rem_model.load_state_dict(torch.load(rem_model_weight_path, map_location='cpu'))
rem_model.eval()

# Set mne logging level to ERROR
mne.set_log_level('ERROR')

def predict_is_rem(eeg_raw, ema_span = 5):

    channel_names = ['LF-FpZ', 'OTE_L-FpZ', 'RF-FpZ', 'OTE_R-FpZ']
    sfreq = 125 

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(channel_names)
    )

    raw = mne.io.RawArray(eeg_raw, info)
    raw.pick_channels(['RF-FpZ'])

    raw.filter(0.1, 45)
    raw = raw.resample(100)

    eeg_raw = raw.to_data_frame()['RF-FpZ']
    
    chunks = [eeg_raw[i*input_len : (i+1)*input_len] for i in range(rem_seq_len)]
    
    chunks_zscore = []
    for arr in chunks:
        arr = np.asarray(arr)
        m = arr.mean()
        s = arr.std() + 1e-8
        arr_norm = (arr - m) / s
        chunks_zscore.append(arr_norm)
    segs = np.stack(chunks_zscore, axis=0)     # (seq_len, input_len)
    segs = segs[None, :, :]          
    x_tensor = torch.from_numpy(segs.astype(np.float32))
    with torch.no_grad():
        out = rem_model(x_tensor)
        rem_possibility = round(torch.sigmoid(out).item(), 4)
    return rem_possibility

def predict_alertness_ema(eeg_raw, ema_span=100):

    if not hasattr(predict_alertness_ema, "alertness_history"):
        predict_alertness_ema.alertness_history = []

    channel_names = ['LF-FpZ', 'OTE_L-FpZ', 'RF-FpZ', 'OTE_R-FpZ']
    sfreq = 125 

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(channel_names)
    )

    raw = mne.io.RawArray(eeg_raw, info)
    raw.pick_channels(['RF-FpZ'])

    raw.filter(0.35, 40)
    raw = raw.resample(100)

    eeg_raw = raw.to_data_frame()['RF-FpZ']

    chunks = [eeg_raw[i*input_len : (i+1)*input_len] for i in range(seq_len)]

    chunks_zscore = []
    for arr in chunks:
        arr = np.asarray(arr)
        m = arr.mean()
        s = arr.std() + 1e-8
        arr_norm = (arr - m) / s
        chunks_zscore.append(arr_norm)
    segs = np.stack(chunks_zscore, axis=0)     # (seq_len, input_len)
    segs = segs[None, : :]          
    x_tensor = torch.from_numpy(segs.astype(np.float32))
    with torch.no_grad():
        out = model(x_tensor)
        alertness_score = round(1 - torch.sigmoid(out).item(), 4)
    predict_alertness_ema.alertness_history.append(alertness_score)
    series = pd.Series(predict_alertness_ema.alertness_history)
    ewm_score = series.ewm(span=ema_span, adjust=False).mean().iloc[-1]
    return ewm_score


