CONFIG = {
    "seq_len":        10,
    "input_dim":      28,
    "hidden_dim":     64,
    "latent_dim":     16,
    "local_epochs":   3,
    "learning_rate":  1e-3,
    "batch_size":     64,
    "num_rounds":     10,
    "num_clients":    4,
    "fraction_fit":   1.0,
    "dp_sigma":       0.005,
    "threshold_pct":  95,
}

NODE_NAMES = {
    0: "Node A — Office (Credit Card Fraud)",
    1: "Node B — Campus (NSL-KDD)",
    2: "Node C — Transport (UNSW-NB15)",
    3: "Node D — Retail (Mixed)",
}
