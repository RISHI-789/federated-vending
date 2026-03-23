import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import CONFIG


class SequenceDataset(Dataset):
    def __init__(self, X, y, normal_only=False):
        seq_len = CONFIG["seq_len"]
        seqs, lbls = [], []
        for i in range(len(X) - seq_len + 1):
            lbl = int(y[i + seq_len - 1])
            if normal_only and lbl == 1:
                continue
            seqs.append(X[i:i+seq_len])
            lbls.append(lbl)
        self.seqs = np.array(seqs, dtype=np.float32)
        self.lbls = np.array(lbls, dtype=np.int64)

    def __len__(self):  return len(self.seqs)
    def __getitem__(self, i):
        return torch.tensor(self.seqs[i]), torch.tensor(self.lbls[i])


def make_loaders(node):
    bs = CONFIG["batch_size"]
    tr = DataLoader(SequenceDataset(node["X_train"],node["y_train"],normal_only=True),
                    batch_size=bs, shuffle=True,  drop_last=True)
    ev = DataLoader(SequenceDataset(node["X_test"], node["y_test"],normal_only=False),
                    batch_size=bs, shuffle=False, drop_last=False)
    return tr, ev
