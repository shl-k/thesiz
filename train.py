# train.py
import os
import networkx as nx
import pandas as pd
import pickle
# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit  # <-- Import the TimeLimit wrapper

# Your environment
from src.rl.AmbulanceEnv import AmbulanceEnv

def main():
    # 1) Load graph 
    with open('data/processed/princeton_graph.gpickle', 'rb') as f:
        G = pickle.load(f)
    
    # Critical nodes (these should be consistent with princeton_data_prep.py)
    pfars_node = 241  # PFARS HQ node
    hospital_node = 1293  # Hospital node

    call_data_path = 'data/processed/synthetic_calls.csv'

    # 2) Provide JSON node mapping, or load from files
    idx_to_node_path = "data/matrices/idx_to_node_id.json"
    node_to_idx_path = "data/matrices/node_id_to_idx.json"

    # 3) Create the environment
    env = AmbulanceEnv(
        graph=G,
        call_data_path=call_data_path,
        num_ambulances=4,
        base_location=pfars_node,
        hospital_node=hospital_node,
        idx_to_node_path=idx_to_node_path,
        node_to_idx_path=node_to_idx_path,
        verbose=False
    )
    # Wrap your environment with a TimeLimit to force termination after max_episode_steps:
    max_episode_steps = 1000  # adjust this number as needed
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # 4) Wrap the environment in a Monitor to log episode rewards
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    monitored_env = Monitor(env, filename=os.path.join(log_dir, "ambulance_monitor.csv"))

    # 5) Create a vectorized environment (here we use a single environment)
    vec_env = DummyVecEnv([lambda: monitored_env])

    # 6) Setup an evaluation callback (optional)
    eval_env = DummyVecEnv([lambda: Monitor(AmbulanceEnv(
        graph=G,
        call_data_path=call_data_path,
        num_ambulances=4,
        base_location=pfars_node,
        hospital_node=hospital_node,
        idx_to_node_path=idx_to_node_path,
        node_to_idx_path=node_to_idx_path,
        verbose=False
    ), filename=os.path.join(log_dir, "eval_monitor.csv"))])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=500,  # Evaluate every 500 steps
        n_eval_episodes=2,
        deterministic=True,
        render=False
    )

    # 7) Create the PPO model – you can force the device to "cuda" if desired
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        n_steps=256,
        batch_size=64,
        learning_rate=2.5e-4,
        gamma=0.99,
        device="cpu"  # force GPU usage if available
    )

    # 8) Train the model (adjust total_timesteps as needed)
    run_name = "run_with_timelimit"
    model.learn(total_timesteps=10_000, callback=eval_callback, tb_log_name=run_name)

    # 9) Evaluate final policy in a short rollout
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    print(f"Final policy test episode reward: {total_reward}")
    print("Training complete! Check the logs/ folder for monitor files and best model.")

if __name__ == "__main__":
    main()
