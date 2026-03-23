import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from data.partition   import get_node_partitions
from data.dataset     import make_loaders
from federated.client   import VendingClient
from federated.strategy import FedAvgDP
from config import CONFIG


def run_simulation(dp_sigma=None, num_rounds=None, verbose=True):
    dp_sigma   = dp_sigma   or CONFIG["dp_sigma"]
    num_rounds = num_rounds or CONFIG["num_rounds"]

    if verbose: print("Loading datasets...")
    nodes   = get_node_partitions()
    loaders = {i: make_loaders(n) for i,n in nodes.items()}

    # Print dataset summary
    if verbose:
        print(f"\n{'Node':<30} {'Source':<35} {'Train':>7} {'Test':>6} {'Anom%':>6}")
        print("-"*85)
        for i,n in nodes.items():
            ar = round(n["y_test"].mean()*100,2)
            syn = " [synthetic]" if n["using_synthetic"] else ""
            print(f"{n['name']:<30} {n['source']+syn:<35} "
                  f"{len(n['X_train']):>7} {len(n['X_test']):>6} {ar:>5}%")
        print()

    def client_fn(cid):
        nid = int(cid)
        tr, ev = loaders[nid]
        return VendingClient(nid, tr, ev).to_client()

    strategy = FedAvgDP(
        dp_sigma=dp_sigma,
        fraction_fit=1.0,
        min_fit_clients=CONFIG["num_clients"],
        min_evaluate_clients=CONFIG["num_clients"],
        min_available_clients=CONFIG["num_clients"],
    )

    if verbose:
        print(f"Starting simulation: {num_rounds} rounds | "
              f"{CONFIG['num_clients']} nodes | DP σ={dp_sigma}\n")

    import logging
    logging.getLogger("flwr").setLevel(logging.WARNING)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=CONFIG["num_clients"],
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus":1,"num_gpus":0.0},
    )

    return strategy.round_log, nodes
