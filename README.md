# 🔐 Federated Vending Machine — Anomaly Detection

> Privacy-Preserving Anomaly Detection for Payment Security in Vending Machines Using Federated Deep Autoencoders

**Paper by:** K. Rishi J — Jain (Deemed-to-be) University, Bangalore, India

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/federated-vending-anomaly/blob/main/fedvending_colab.ipynb)

---

## ▶ Run in Google Colab (Recommended)

1. Click the **Open in Colab** badge above
2. In Cell 1, replace `YOUR_USERNAME` with your GitHub username
3. Run all cells top to bottom (`Runtime → Run all`)
4. No dataset downloads needed — synthetic data used automatically

---

## What the Notebook Demonstrates

| Cell | Output | Paper Claim |
|------|--------|-------------|
| Cell 3 | Federated simulation runs | FedAvg + DP works |
| Cell 4 | 7-panel results dashboard | F1, Loss, Latency, AUC-ROC |
| Cell 5 | Per-node metrics table | Non-IID nodes handled |
| Cell 6 | DP privacy-accuracy tradeoff | Privacy mechanism proven |
| Cell 7 | Live transaction inference | Real-time detection (<50ms) |
| Cell 8 | Download all results as ZIP | Paper figures ready |

---

## (Optional) Use Real Datasets

Upload any of these in **Cell 2** of the notebook for real results:

| Dataset | Download | Node |
|---------|----------|------|
| `creditcard.csv` | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Node A — Office |
| `KDDTrain+.txt` | [UNB](https://www.unb.ca/cic/datasets/nsl.html) | Node B — Campus |
| `UNSW_NB15_training-set.csv` | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | Node C — Transport |

Missing files automatically use synthetic proxy data.

---

## Project Structure

```
├── fedvending_colab.ipynb   ← Main notebook (run this)
├── config.py                ← All hyperparameters
├── data/
│   ├── raw/                 ← Place downloaded datasets here
│   ├── partition.py         ← Non-IID node partitioning
│   └── dataset.py           ← Sliding window sequences
├── models/
│   ├── autoencoder.py       ← LSTM autoencoder
│   └── threshold.py         ← Adaptive anomaly threshold
├── federated/
│   ├── client.py            ← Flower federated client
│   ├── strategy.py          ← FedAvg + differential privacy
│   └── runner.py            ← Simulation orchestrator
└── requirements.txt
```

---

## Key Hyperparameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_rounds` | 10 | Federated rounds |
| `num_clients` | 4 | Vending machine nodes |
| `dp_sigma` | 0.005 | DP Gaussian noise std |
| `hidden_dim` | 64 | LSTM hidden size |
| `latent_dim` | 16 | Autoencoder bottleneck |
| `seq_len` | 10 | Transaction sequence window |
