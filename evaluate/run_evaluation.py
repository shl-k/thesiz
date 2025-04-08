from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import pandas as pd
import pickle
import json
import os
import sys
import numpy as np
# Add project root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator
from src.rl.simple_dispatch import SimpleDispatchEnv




def print_statistics(simulator: AmbulanceSimulator):
    """Print detailed simulation statistics using the simulator's built-in method."""
    simulator.print_statistics()

def main():
    # === Load trained model ===
    model_path = "models/dispatch_1_ambulance_3M_v1.zip"
    model = PPO.load(model_path)
    print(f"✅ Loaded model from: {model_path}")

    # === Load graph and data ===
    graph = pickle.load(open("data/processed/princeton_graph.gpickle", "rb"))
    call_data = pd.read_csv("data/processed/synthetic_calls.csv")
    path_cache = pd.read_pickle("data/matrices/path_cache.pkl")
    node_to_idx = json.load(open("data/matrices/node_id_to_idx.json"))
    idx_to_node = json.load(open("data/matrices/idx_to_node_id.json"))

    # === Set up simulator ===
    simulator = AmbulanceSimulator(
        graph=graph,
        call_data=call_data,
        num_ambulances=1,
        base_location=241,
        hospital_node=1293,
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        manual_mode=True,     # Let the RL agent control dispatching
        verbose=True          # Print output
    )

    # Initialize and run the simulation
    simulator.initialize()

    # === Wrap in RL environment for evaluation ===
    env = DummyVecEnv([lambda: SimpleDispatchEnv(simulator)])

    # === Evaluate ===
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print("\n🎯 Evaluation complete!")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    print_statistics(simulator)

if __name__ == "__main__":
    main()
