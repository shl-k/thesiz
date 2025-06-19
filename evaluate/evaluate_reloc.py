"""
Evaluation script for models trained on DispatchRelocEnv.
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
from stable_baselines3.common.evaluation import evaluate_policy

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from envs.dispatch_relocation_env import DispatchRelocEnv
from src.simulator.simulator import AmbulanceSimulator

MODEL_PATHS = [
    "models/reloc_M3/reloc_M3",
]
GRAPH_PATH = "data/processed/princeton_graph.gpickle"
CALLS_PATH = "data/processed/synthetic_calls.csv"
CACHE_PATH = "data/matrices/path_cache.pkl"
IDX_PATH   = "data/matrices/node_id_to_idx.json"
NODE_PATH  = "data/matrices/idx_to_node_id.json"
LATLON_PATH= "data/matrices/node_to_lat_lon.json"

NUM_AMBULANCES = 3
BASE_NODE = 241
HOSP_NODE = 1293
CALL_TIMEOUT_MEAN = 491
CALL_TIMEOUT_STD  = 10

def evaluate_model(model_path: str) -> None:
    model_name = Path(model_path).stem
    print(f"\n=== Evaluating {model_name} ===")

    # Load data
    graph      = pickle.load(open(GRAPH_PATH, "rb"))
    calls      = pd.read_csv(CALLS_PATH)
    path_cache = pickle.load(open(CACHE_PATH, "rb"))
    node_to_idx= json.load(open(IDX_PATH, "r"))
    idx_to_node= json.load(open(NODE_PATH, "r"))

    node_to_idx = {int(k): v for k, v in node_to_idx.items()}
    idx_to_node = {int(k): int(v) for k, v in idx_to_node.items()}

    # Setup simulator
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
        manual_mode=True,
        verbose=False
    )

    # Setup environment
    env = DummyVecEnv([lambda: DispatchRelocEnv(
        simulator=sim,
        n_clusters=8,
        node_to_lat_lon_file=LATLON_PATH,
        verbose=False
    )])

    # Load normalization stats if available
    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    # Load model
    model = PPO.load(model_path)

    # Run evaluation
    print("\nRunning evaluation...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    
    # Get final statistics
    env_stats = env.envs[0].get_stats()
    total_calls = env_stats['total_calls']
    responded = env_stats['calls_responded']
    completion_rate = env_stats['completion_rate']
    avg_response_time = env_stats['avg_response_time']

    # Print results
    print("\n=== Key Metrics ===")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Completion Rate: {completion_rate:.1%}")
    print(f"Calls Responded: {responded}/{total_calls}")
    if avg_response_time is not None:
        print(f"Avg Response Time: {avg_response_time/60:.1f} minutes")

def main():
    for mp in MODEL_PATHS:
        evaluate_model(mp)

if __name__ == "__main__":
    main()
