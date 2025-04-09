"""
Evaluate a *trained* PPO model on the DispatchRelocEnv.
Runs one episode using evaluate_policy.
"""

import os, sys, json, pickle, argparse, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator
from envs.dispatch_relocation_env import DispatchRelocEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/dispatch_relocation/dispatch_relocation_1_ambulance_3M_v1.zip")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--verbose", action="store_true", help="Print simulation details")
    args = parser.parse_args()

    # ---------------- Load data ----------------
    graph = pickle.load(open("data/processed/princeton_graph.gpickle", "rb"))
    calls = pd.read_csv("data/processed/synthetic_calls.csv")
    path_cache = pickle.load(open("data/matrices/path_cache.pkl", "rb"))
    node_to_idx = {int(k): v for k, v in json.load(open("data/matrices/node_id_to_idx.json")).items()}
    idx_to_node = {int(k): int(v) for k, v in json.load(open("data/matrices/idx_to_node_id.json")).items()}

    # ---------------- Simulator ----------------
    sim = AmbulanceSimulator(
        graph=graph,
        call_data=calls,
        num_ambulances=1,
        base_location=241,
        hospital_node=1293,
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        call_timeout_mean=600,
        call_timeout_std=60,
        verbose=args.verbose,
        manual_mode=True
    )

    # ---------------- Environment ----------------
    env_core = DispatchRelocEnv(
        simulator=sim,
        n_clusters=8,
        node_to_lat_lon_file="data/matrices/node_to_lat_lon.json",
        verbose=args.verbose
    )
    vec_env = DummyVecEnv([lambda: env_core])

    # ---------------- VecNormalize (optional) ----------------
    vecnorm_path = args.model_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"✅ Loaded VecNormalize stats from {vecnorm_path}")

    # ---------------- Load model ----------------
    model = PPO.load(args.model_path)
    print(f"✅ Loaded PPO model from {args.model_path}")

    # ---------------- Evaluate ----------------
    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=args.episodes,
        deterministic=True,
        render=False
    )

    print(f"\n🎯 Evaluation complete (episodes = {args.episodes})")
    print(f"Mean episode reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Optional: print simulator stats from last episode
    print("\n📊 Simulator stats (from last run):")
    sim._print_statistics()

if __name__ == "__main__":
    main()
