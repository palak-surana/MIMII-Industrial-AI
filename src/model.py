"""Phase 4: CNN-LSTM. Input = log-Mel window (n_mels, T): the Mel bands are the Conv1D channels,
convolution runs over time, the LSTM models the temporal behaviour. Output = logit (sigmoid -> P(faulty))."""
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, n_mels=64, c1=64, c2=128, hidden=64, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, c1, 3, padding=1), nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, 3, padding=1), nn.BatchNorm1d(c2), nn.ReLU())
        self.lstm = nn.LSTM(c2, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (B, n_mels, T)
        h = self.conv(x).transpose(1, 2)        # (B, T', c2)
        _, (hn, _) = self.lstm(h)
        return self.fc(self.drop(hn[-1])).squeeze(-1)   # logits (B,)
