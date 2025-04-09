"""
Evaluate a *trained* RL model with stable_baselines3's evaluate_policy function.
This runs multiple episodes (default 5) in sequence.

We use the environment EXACTLY as in training: i.e. wrap with VecNormalize again,
load the saved running stats, etc.
"""

import sys, os, pickle, json, pandas as pd, numpy as np, argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to import simulator + env
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import StaticRelocationPolicy
from envs.simple_dispatch import SimpleDispatchEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/dispatch_1ambulance.zip")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print simulation details?")
    args = parser.parse_args()

    # 1) Load environment data (same as training)
    G          = pickle.load(open("data/processed/princeton_graph.gpickle","rb"))
    calls      = pd.read_csv("data/processed/synthetic_calls.csv")
    path_cache = pickle.load(open("data/matrices/path_cache.pkl","rb"))
    node_to_index   = {int(k): v for k,v in json.load(open("data/matrices/node_id_to_idx.json")).items()}
    index_to_node   = {int(k): int(v) for k,v in json.load(open("data/matrices/idx_to_node_id.json")).items()}

    # 2) Create simulator with static relocation & manual_mode=True
    sim = AmbulanceSimulator(
        graph=G,
        call_data=calls,
        num_ambulances=1,
        base_location=241,
        hospital_node=1293,
        path_cache=path_cache,
        node_to_idx=node_to_index,
        idx_to_node=index_to_node,
        relocation_policy=StaticRelocationPolicy(G, base_location=241),
        manual_mode=True,
        verbose=args.verbose
    )

    # 3) Wrap in dispatch env
    env_core = SimpleDispatchEnv(sim, verbose=args.verbose)
    vec_env = DummyVecEnv([lambda: env_core])

    # 4) Attempt to load saved VecNormalize stats:
    vecnorm_path = args.model_path.replace(".zip","_vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"✅ Loaded VecNormalize stats from {vecnorm_path}")

    # 5) Load model
    model = PPO.load(args.model_path)
    print(f"✅ Loaded model from: {args.model_path}")

    # 6) Evaluate using SB3's evaluate_policy:.
    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=args.episodes,
        deterministic=True,
        render=False  # We already have verbose logs from the simulator if needed
    )
    print(f"\n🎯 Evaluate complete over {args.episodes} episodes!")
    print(f"Mean episode reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Note: Because we re-set the environment after each episode,
    # the simulator stats (calls responded, etc.) are from the final run only.
    # To see them, do:
    print("\n==== Simulator stats after final episode ====")
    sim.print_statistics()

if __name__ == "__main__":
    main()
