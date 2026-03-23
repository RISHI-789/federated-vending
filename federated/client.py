import time
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from models.autoencoder import LSTMAutoencoder, recon_error
from models.threshold   import AdaptiveThreshold
from config import CONFIG


class VendingClient(fl.client.NumPyClient):
    def __init__(self, node_id, train_loader, eval_loader):
        self.node_id      = node_id
        self.train_loader = train_loader
        self.eval_loader  = eval_loader
        self.model        = LSTMAutoencoder()
        self.opt          = torch.optim.Adam(self.model.parameters(),
                                             lr=CONFIG["learning_rate"])
        self.criterion    = nn.MSELoss()
        self.threshold    = AdaptiveThreshold()

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        keys  = list(self.model.state_dict().keys())
        state = OrderedDict({k: torch.tensor(v) for k,v in zip(keys,params)})
        self.model.load_state_dict(state, strict=True)

    def fit(self, params, config):
        self.set_parameters(params)
        self.model.train()
        total_loss = 0; n = 0
        for _ in range(CONFIG["local_epochs"]):
            for x, _ in self.train_loader:
                self.opt.zero_grad()
                loss = self.criterion(self.model(x), x)
                loss.backward()
                self.opt.step()
                total_loss += loss.item(); n += 1
        avg = total_loss / max(n,1)
        return self.get_parameters({}), len(self.train_loader.dataset), {"loss": avg}

    def evaluate(self, params, config):
        self.set_parameters(params)
        self.model.eval()
        errors, labels, lats = [], [], []

        with torch.no_grad():
            # Fit threshold on normal samples
            norm_errs = []
            for x, lbl in self.eval_loader:
                e = recon_error(x, self.model(x)).numpy()
                norm_errs.extend(e[lbl.numpy()==0].tolist())
            if norm_errs:
                self.threshold.fit(np.array(norm_errs))
            else:
                self.threshold.threshold = 0.1

            for x, lbl in self.eval_loader:
                t0 = time.perf_counter()
                e  = recon_error(x, self.model(x)).numpy()
                lats.append((time.perf_counter()-t0)*1000/len(x))
                errors.extend(e.tolist())
                labels.extend(lbl.numpy().tolist())

        preds = self.threshold.predict(np.array(errors))
        labels = np.array(labels)

        try:    auc = float(roc_auc_score(labels, errors))
        except: auc = 0.0

        metrics = {
            "node_id":    self.node_id,
            "f1":         round(float(f1_score(labels,preds,zero_division=0)),4),
            "precision":  round(float(precision_score(labels,preds,zero_division=0)),4),
            "recall":     round(float(recall_score(labels,preds,zero_division=0)),4),
            "auc_roc":    round(auc,4),
            "latency_ms": round(float(np.mean(lats)),3),
            "recon_error":round(float(np.mean(errors)),6),
            "threshold":  round(float(self.threshold.threshold),6),
            "normal_errors":  [round(e,6) for e in norm_errs[:200]],
            "anomaly_errors": [round(e,6) for e in
                               np.array(errors)[labels==1][:200].tolist()],
        }
        return float(np.mean(errors)), len(self.eval_loader.dataset), metrics
