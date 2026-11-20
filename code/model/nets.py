"""
Models (PyTorch). NOT run in the authoring environment (no space to install torch):
test on your machine / Colab first.

  concat      CNN -> BiLSTM on channel-concatenated sensors   (baseline)
  independent per-sensor temporal conv -> TEMPORAL self-attention inside each sensor (no cross-sensor mixing) -> BiLSTM
  relational  per-sensor temporal conv -> cross-sensor self-attention -> BiLSTM (Novelty A)

Both return (rul_pred (B,), sensor_importance (B,S,T) or None).
sensor_importance is the attention mass each sensor receives, used by the Stage-2
physics loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Concat(nn.Module):
    def __init__(self, S, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(S, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                                  # x (B,T,S)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B,T,64)
        h, _ = self.lstm(h)
        return self.head(h[:, -1]).squeeze(-1), None


class Relational(nn.Module):
    def __init__(self, S, d=16, hidden=64):
        super().__init__()
        self.S, self.d = S, d
        self.temporal = nn.Sequential(nn.Conv1d(1, d, 3, padding=1), nn.ReLU())   # shared across sensors
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.lstm = nn.LSTM(S * d, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                                  # x (B,T,S)
        B, T, S = x.shape
        z = self.temporal(x.permute(0, 2, 1).reshape(B * S, 1, T))        # (B*S,d,T)
        z = z.reshape(B, S, self.d, T).permute(0, 3, 1, 2)               # (B,T,S,d)
        A = torch.softmax(self.q(z) @ self.k(z).transpose(-1, -2) / self.d ** 0.5, dim=-1)  # (B,T,S,S)
        z = self.norm(z + A @ self.v(z))                                  # residual + norm
        h, _ = self.lstm(z.reshape(B, T, S * self.d))
        imp = A.mean(dim=2).permute(0, 2, 1)                              # (B,S,T) mass each sensor receives
        return self.head(h[:, -1]).squeeze(-1), imp


class Independent(nn.Module):
    """Independent per-channel attention baseline: each sensor attends over its own time steps only."""
    def __init__(self, S, d=16, hidden=64):
        super().__init__()
        self.S, self.d = S, d
        self.temporal = nn.Sequential(nn.Conv1d(1, d, 3, padding=1), nn.ReLU())   # shared across sensors
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.lstm = nn.LSTM(S * d, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                                  # x (B,T,S)
        B, T, S = x.shape
        z = self.temporal(x.permute(0, 2, 1).reshape(B * S, 1, T))        # (B*S,d,T)
        z = z.reshape(B, S, self.d, T).permute(0, 1, 3, 2)               # (B,S,T,d)
        A = torch.softmax(self.q(z) @ self.k(z).transpose(-1, -2) / self.d ** 0.5, dim=-1)  # (B,S,T,T) over time
        z = self.norm(z + A @ self.v(z))                                  # (B,S,T,d)
        h, _ = self.lstm(z.permute(0, 2, 1, 3).reshape(B, T, S * self.d))
        return self.head(h[:, -1]).squeeze(-1), None


def physics_loss(imp, ref_sensor_mask):
    """1 - fraction of attention mass on reference sensors. imp (B,S,T), mask (S,) bool tensor."""
    mass = imp.sum(dim=-1)                                 # (B,S)
    mass = mass / (mass.sum(dim=1, keepdim=True) + 1e-8)
    return (1.0 - mass[:, ref_sensor_mask].sum(dim=1)).mean()
