import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

class DeepSleepNetBinarySmall(nn.Module):
    def __init__(self, input_len=3750, n_channels=1, lstm_hidden=64, seq_len=5):
        super().__init__()
        # CNN branch 1: 大卷积核（宽度减半）
        self.cnn1 = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=400, stride=50, padding=0),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2)
        )
        # CNN branch 2: 小卷积核（宽度减半）
        self.cnn2 = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=50, stride=6, padding=0),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(8),
            nn.Conv1d(16, 32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4)
        )
        # 自动计算特征维度
        with torch.no_grad():
            dummy = torch.randn(1, n_channels, input_len)
            feat1 = self.cnn1(dummy).view(1, -1).shape[1]
            feat2 = self.cnn2(dummy).view(1, -1).shape[1]
        self.cnn_out_dim = feat1 + feat2
        # 保证 residual 输出和LSTM输出一致（128）
        self.fc_residual = nn.Sequential(
            nn.Linear(self.cnn_out_dim, 128), nn.BatchNorm1d(128), nn.ReLU()
        )
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True, dropout=0.3, bidirectional=True
        )
        self.final_fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        # x: (batch, seq, n_channels, epoch_len)
        b, t, c, n = x.shape
        x = x.view(b * t, c, n)
        x1 = self.cnn1(x).view(b * t, -1)
        x2 = self.cnn2(x).view(b * t, -1)
        cnn_feat = torch.cat([x1, x2], dim=1)  # (b*t, feat)
        res_feat = self.fc_residual(cnn_feat)  # (b*t, 128)
        cnn_feat_seq = cnn_feat.view(b, t, -1)  # (batch, seq, feat)
        lstm_out, _ = self.lstm(cnn_feat_seq)
        lstm_last = lstm_out[:, -1, :]          # (batch, 128)
        res_feat_seq = res_feat.view(b, t, -1)[:, -1, :]  # (batch, 128)
        fused = lstm_last + res_feat_seq        # (batch, 128)
        logits = self.final_fc(fused).squeeze(1)  # (batch,)
        return logits



# 2. 全局参数写死
fs = 125
seq_len = 5
epoch_sec = 30
input_len = epoch_sec * fs    # 3750
model_weight_path = '/Users/lejieliu/Documents/CS189/lucid-dreaming-core/models/alertness_torch_weights.pth'

# 3. 加载模型
model = DeepSleepNetBinarySmall(input_len=input_len, n_channels=1, lstm_hidden=64, seq_len=seq_len)
model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
model.eval()

# 4. 预测函数：只传入一个刚好够长度的数据段
def predict_alertness_ema(eeg_raw, ema_span=20):
    # 静态变量维护历史
    if not hasattr(predict_alertness_ema, "alertness_history"):
        predict_alertness_ema.alertness_history = []
    # 分chunk
    chunks = [eeg_raw[i*input_len : (i+1)*input_len] for i in range(seq_len)]
    # 每个epoch做z-score
    chunks_zscore = []
    for arr in chunks:
        arr = np.asarray(arr)
        m = arr.mean()
        s = arr.std() + 1e-8
        arr_norm = (arr - m) / s
        chunks_zscore.append(arr_norm)
    segs = np.stack(chunks_zscore, axis=0)     # (seq_len, input_len)
    segs = segs[None, :, None, :]              # (1, seq_len, 1, input_len)
    x_tensor = torch.from_numpy(segs.astype(np.float32))
    with torch.no_grad():
        out = model(x_tensor)
        alertness_score = torch.sigmoid(out).item()
    predict_alertness_ema.alertness_history.append(alertness_score)
    series = pd.Series(predict_alertness_ema.alertness_history)
    ewm_score = series.ewm(span=ema_span, adjust=True).mean().iloc[-1]
    return ewm_score
