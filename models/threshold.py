import numpy as np
from config import CONFIG


class AdaptiveThreshold:
    def __init__(self):
        self.threshold = None

    def fit(self, normal_errors):
        self.threshold = float(np.percentile(normal_errors, CONFIG["threshold_pct"]))

    def predict(self, errors):
        return (np.array(errors) > self.threshold).astype(int)
