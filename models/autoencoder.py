import torch
import torch.nn as nn
from config import CONFIG


class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        D = CONFIG["input_dim"]
        H = CONFIG["hidden_dim"]
        L = CONFIG["latent_dim"]
        S = CONFIG["seq_len"]
        self.seq_len = S
        self.enc_lstm = nn.LSTM(D, H, num_layers=2, batch_first=True, dropout=0.1)
        self.enc_fc   = nn.Linear(H, L)
        self.dec_fc   = nn.Linear(L, H)
        self.dec_lstm = nn.LSTM(H, D, num_layers=2, batch_first=True, dropout=0.1)

    def forward(self, x):
        _, (h, _) = self.enc_lstm(x)
        z = self.enc_fc(h[-1])
        d = self.dec_fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.dec_lstm(d)
        return out


def recon_error(x, x_hat):
    return torch.mean((x - x_hat) ** 2, dim=(1, 2))
