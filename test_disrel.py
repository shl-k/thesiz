"""
Quick smoke test for DispatchRelocEnv.
We create a small environment with random actions for a few episodes
just to confirm that nothing is broken (no errors, the environment runs).
"""

import pickle
import json
import pandas as pd
import numpy as np


from stable_baselines3.common.env_checker import check_env
from src.simulator.simulator import AmbulanceSimulator
from envs.dispatch_relocation_env import DispatchRelocEnv

def main():
    # 1) Load minimal data
    # or you can load real data if you prefer
    graph_path = "data/processed/princeton_graph.gpickle"
    calls_path = "data/processed/synthetic_calls.csv"
    path_cache_path = "data/matrices/path_cache.pkl"
    node2idx_path = "data/matrices/node_id_to_idx.json"
    idx2node_path = "data/matrices/idx_to_node_id.json"

    graph = pickle.load(open(graph_path, "rb"))
    calls = pd.read_csv(calls_path)
    path_cache = pickle.load(open(path_cache_path, "rb"))
    node2idx = json.load(open(node2idx_path))
    idx2node = json.load(open(idx2node_path))

    # 2) Create simulator
    simulator = AmbulanceSimulator(
        graph=graph,
        call_data=calls.sample(min(100, len(calls))),
        num_ambulances=2,
        base_location=241,
        hospital_node=1293,
        path_cache=path_cache,
        node_to_idx={int(k): v for k,v in node2idx.items()},
        idx_to_node={int(k): int(v) for k,v in idx2node.items()},
        manual_mode=True,  # Let our environment handle dispatch
        verbose=False
    )

    # 3) Create our environment
    env = DispatchRelocEnv(
        simulator=simulator,
        n_clusters=5,  # fewer clusters for faster testing
        node_to_lat_lon_file="data/matrices/node_to_lat_lon.json",
        verbose=True,
    )

    # 4) Optional: run SB3's check_env for basic sanity
    # This won't handle the queue logic thoroughly, but good for shape checks:
    check_env(env, warn=True)

    # 5) Run a small number of episodes with random actions
    n_episodes = 2
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = env.action_space.sample()  # random
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if done or truncated:
                break
        print(f"Episode {ep+1}/{n_episodes} ended after {steps} steps. Total reward: {total_reward:.2f}")

    print("\nSmoke test complete. If no errors were raised, environment likely works.")

if __name__ == "__main__":
    main()
