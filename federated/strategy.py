import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


class FedAvgDP(FedAvg):
    """FedAvg + Gaussian differential privacy noise."""

    def __init__(self, dp_sigma=0.005, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_sigma     = dp_sigma
        self.round_log    = []   # [{round, avg_loss, avg_f1, node_results}]

    def aggregate_fit(self, rnd, results, failures):
        agg = super().aggregate_fit(rnd, results, failures)
        if agg is None: return None
        params, metrics = agg
        # Inject DP noise
        noisy = [a + np.random.normal(0, self.dp_sigma, a.shape)
                 for a in parameters_to_ndarrays(params)]
        losses = [r.metrics.get("loss",0) for _,r in results]
        self.round_log.append({
            "round":    rnd,
            "avg_loss": round(float(np.mean(losses)),6),
        })
        return ndarrays_to_parameters(noisy), metrics

    def aggregate_evaluate(self, rnd, results, failures):
        agg = super().aggregate_evaluate(rnd, results, failures)
        if agg is None: return None
        loss, metrics = agg
        node_results = [r.metrics for _,r in results]
        avg_f1 = round(float(np.mean([n.get("f1",0) for n in node_results])),4)
        # Attach to matching round log entry
        for entry in reversed(self.round_log):
            if entry["round"] == rnd:
                entry["avg_f1"]       = avg_f1
                entry["node_results"] = node_results
                break
        return loss, metrics
