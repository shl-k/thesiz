"""
Evaluation script for models trained on SimpleDispatchEnv.
It loads one or more PPO models, runs a single episode,
and prints & saves stats including calls responded, response times,
and a list of completed deliveries (logged when an ambulance reaches the hospital).
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# Import your environment and simulator
from envs.simple_dispatch import SimpleDispatchEnv
from src.simulator.simulator import AmbulanceSimulator


MODEL_PATHS = [
    "models/dispatch_M3/dispatch_M3",
    # Add additional model paths if desired
]
GRAPH_PATH    = "data/processed/princeton_graph.gpickle"
CALLS_PATH    = "data/processed/synthetic_calls.csv"
CACHE_PATH    = "data/matrices/path_cache.pkl"
IDX_PATH      = "data/matrices/node_id_to_idx.json"
NODE_PATH     = "data/matrices/idx_to_node_id.json"
LATLON_PATH   = "data/matrices/node_to_lat_lon.json"

NUM_AMBULANCES   = 3
BASE_NODE        = 241
HOSP_NODE        = 1293
CALL_TIMEOUT_MEAN = 491
CALL_TIMEOUT_STD  = 10

RESULTS_DIR = "results/simple_dispatch_eval"

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_model(model_path: str) -> None:
    model_name = Path(model_path).stem
    print(f"\n=== Evaluating {model_name} ===")


    graph      = pickle.load(open(GRAPH_PATH, "rb"))
    calls      = pd.read_csv(CALLS_PATH)
    path_cache = pickle.load(open(CACHE_PATH, "rb"))
    node_to_idx= json.load(open(IDX_PATH, "r"))
    idx_to_node= json.load(open(NODE_PATH, "r"))
    node_to_idx = {int(k): v for k, v in node_to_idx.items()}
    idx_to_node = {int(k): int(v) for k, v in idx_to_node.items()}


    sim = AmbulanceSimulator(
        graph=graph,
        call_data=calls,
        num_ambulances=NUM_AMBULANCES,
        base_location=BASE_NODE,
        hospital_node=HOSP_NODE,
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        call_timeout_mean=CALL_TIMEOUT_MEAN,
        call_timeout_std=CALL_TIMEOUT_STD,
        manual_mode=True,   # PPO will decide dispatch actions
        verbose=False
    )
    env = DummyVecEnv([lambda: SimpleDispatchEnv(sim, node_to_lat_lon_file=LATLON_PATH, verbose=False)])

    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize stats from {vecnorm_path}")


    model = PPO.load(model_path)
    print(f"Loaded PPO model from {model_path}")

    obs = env.reset()
    done = False
    decision_times = []  # To measure model decision speed

    while not done:
        t0 = time.time()
        action, _ = model.predict(obs, deterministic=True)
        t1 = time.time()
        decision_times.append(t1 - t0)
        obs, _, done, _ = env.step(action)

    print("\n===== Simulation Statistics =====")
    sim._print_statistics()

    total_calls = sim.total_calls
    responded   = sim.calls_responded
    rts         = np.array(sim.response_times)
    deliveries  = sim.get_completed_deliveries()
    num_deliveries = len(deliveries)  # NEW: count the deliveries

    avg_decision_time_ms = float(np.mean(decision_times) * 1000) if decision_times else 0.0
    if len(rts) > 0:
        avg_rt_min = float(np.mean(rts) / 60.0)
        std_rt_min = float(np.std(rts) / 60.0)
        min_rt_min = float(np.min(rts) / 60.0)
        max_rt_min = float(np.max(rts) / 60.0)
    else:
        avg_rt_min = std_rt_min = min_rt_min = max_rt_min = None

    print(f"\n--- RESULTS for {model_name} ---")
    print(f"Calls responded: {responded} / {total_calls}")
    if avg_rt_min is not None:
        print(f"Avg response time: {avg_rt_min:.2f} min (Std: {std_rt_min:.2f}, Min: {min_rt_min:.2f}, Max: {max_rt_min:.2f})")
    else:
        print("No response times recorded (0 calls responded).")
    print(f"Avg model decision time: {avg_decision_time_ms:.3f} ms")
    print(f"Total deliveries: {num_deliveries}")  # NEW: print total deliveries
    print("\n=== Completed Deliveries ===")
    if deliveries:
        for d in deliveries:
            print(f"  Call {d['call_id']} delivered by Ambulance {d['ambulance_id']} at t={d['delivery_time']:.1f}s")
    else:
        print("  (No deliveries.)")

    out = {
        "model_name": model_name,
        "total_calls": total_calls,
        "calls_responded": responded,
        "avg_response_time_min": avg_rt_min,
        "std_response_time_min": std_rt_min,
        "min_response_time_min": min_rt_min,
        "max_response_time_min": max_rt_min,
        "avg_decision_time_ms": avg_decision_time_ms,
        "num_deliveries": num_deliveries, 
        "deliveries": deliveries,
    }



def main():
    for mp in MODEL_PATHS:
        evaluate_model(mp)


if __name__ == "__main__":
    main()
