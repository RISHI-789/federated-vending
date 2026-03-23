"""
Partitions real or synthetic data into 4 non-IID federated nodes.

Place real datasets in data/raw/:
  creditcard.csv             -> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  KDDTrain+.txt              -> https://www.unb.ca/cic/datasets/nsl.html
  UNSW_NB15_training-set.csv -> https://research.unsw.edu.au/projects/unsw-nb15-dataset

Missing files fall back to synthetic data automatically.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def _synthetic(n=8000, feats=28, anom_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n_norm = int(n * (1 - anom_rate))
    n_anom = n - n_norm
    X = np.vstack([
        rng.normal(0, 1,   (n_norm, feats)).astype(np.float32),
        rng.normal(3, 1.5, (n_anom, feats)).astype(np.float32),
    ])
    y = np.array([0]*n_norm + [1]*n_anom)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def _load_creditcard():
    p = os.path.join(RAW_DIR, "creditcard.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    X  = df[[f"V{i}" for i in range(1,29)]].values.astype(np.float32)
    y  = df["Class"].values
    return X, y


def _load_nslkdd():
    p = os.path.join(RAW_DIR, "KDDTrain+.txt")
    if not os.path.exists(p):
        return None
    cols = [f"f{i}" for i in range(41)] + ["label","difficulty"]
    df   = pd.read_csv(p, header=None, names=cols)
    for c in ["f1","f2","f3"]:
        df[c] = pd.Categorical(df[c]).codes
    X = df[[f"f{i}" for i in range(28)]].fillna(0).values.astype(np.float32)
    y = (df["label"] != "normal").astype(int).values
    return X, y


def _load_unswnb15():
    p = os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv")
    if not os.path.exists(p):
        return None
    df   = pd.read_csv(p)
    drop = ["id","attack_cat","proto","service","state"]
    df   = df.drop(columns=[c for c in drop if c in df.columns])
    lbl  = "label" if "label" in df.columns else df.columns[-1]
    y    = df[lbl].values
    cols = [c for c in df.columns if c != lbl][:28]
    X    = df[cols].fillna(0).values.astype(np.float32)
    return X, y


def _scale_split(X, y, split=0.8, seed=0):
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n    = int(len(X)*split)
    sc   = StandardScaler()
    Xtr  = sc.fit_transform(X[:n])
    Xte  = sc.transform(X[n:])
    return Xtr.astype(np.float32), y[:n], Xte.astype(np.float32), y[n:]


def get_node_partitions():
    cc   = _load_creditcard()
    kdd  = _load_nslkdd()
    unsw = _load_unswnb15()

    nodes = {}

    # Node 0 — Credit Card Fraud (Office)
    if cc:
        X, y = cc
        Xtr,ytr,Xte,yte = _scale_split(X,y,seed=0)
        nodes[0] = dict(X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte,
                        name="Node A — Office",source="Credit Card Fraud (Kaggle)",
                        using_synthetic=False)
    else:
        X,y = _synthetic(8000,28,0.002,seed=0)
        Xtr,ytr,Xte,yte = _scale_split(X,y,seed=0)
        nodes[0] = dict(X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte,
                        name="Node A — Office",source="Synthetic (Credit Card proxy)",
                        using_synthetic=True)

    # Node 1 — NSL-KDD (Campus)
    if kdd:
        X,y = kdd
        rng = np.random.default_rng(1)
        idx = rng.permutation(min(20000,len(X)))
        X,y = X[idx],y[idx]
        Xtr,ytr,Xte,yte = _scale_split(X,y,seed=1)
        nodes[1] = dict(X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte,
                        name="Node B — Campus",source="NSL-KDD (UNB)",
                        using_synthetic=False)
    else:
        X,y = _synthetic(8000,28,0.30,seed=1)
        Xtr,ytr,Xte,yte = _scale_split(X,y,seed=1)
        nodes[1] = dict(X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte,
                        name="Node B — Campus",source="Synthetic (NSL-KDD proxy)",
                        using_synthetic=True)

    # Node 2 — UNSW-NB15 (Transport)
    if unsw:
        X,y = unsw
        rng = np.random.default_rng(2)
        idx = rng.permutation(min(20000,len(X)))
        X,y = X[idx],y[idx]
        Xtr,ytr,Xte,yte = _scale_split(X,y,seed=2)
        nodes[2] = dict(X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte,
                        name="Node C — Transport",source="UNSW-NB15 (UNSW)",
                        using_synthetic=False)
    else:
        X,y = _synthetic(8000,28,0.20,seed=2)
        Xtr,ytr,Xte,yte = _scale_split(X,y,seed=2)
        nodes[2] = dict(X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte,
                        name="Node C — Transport",source="Synthetic (UNSW proxy)",
                        using_synthetic=True)

    # Node 3 — Mixed (Retail)
    parts = [d for d in [cc,kdd,unsw] if d is not None]
    if len(parts) >= 2:
        rng = np.random.default_rng(3)
        Xs,ys = [],[]
        for d in parts[:2]:
            i = rng.permutation(min(4000,len(d[0])))
            Xs.append(d[0][i]); ys.append(d[1][i])
        X = np.vstack(Xs); y = np.concatenate(ys)
        src = "Mixed (CC + NSL-KDD)"; syn = False
    else:
        X,y = _synthetic(8000,28,0.05,seed=3)
        src = "Synthetic (Mixed proxy)"; syn = True
    Xtr,ytr,Xte,yte = _scale_split(X,y,seed=3)
    nodes[3] = dict(X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte,
                    name="Node D — Retail",source=src,using_synthetic=syn)

    return nodes
