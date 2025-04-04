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

# Your environment
from src.rl.AmbulanceEnv import AmbulanceEnv

def main():
    #load graph 
    with open('data/processed/princeton_graph.gpickle', 'rb') as f:
        G = pickle.load(f)
    
    # Critical nodes (these should be consistent from princeton_data_prep.py)
    pfars_node = 241  # PFARS HQ
    hospital_node = 1293  # Hospital

    call_data_path = 'data/processed/synthetic_calls.csv'

    # 3) Provide JSON node mapping, or load from actual files
    idx_to_node_path = "data/matrices/idx_to_node_id.json"
    node_to_idx_path = "data/matrices/node_id_to_idx.json"

    # 4) Create the environment
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

    # 5) Wrap the environment in a Monitor, which logs episode rewards to a file
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    monitored_env = Monitor(env, filename=os.path.join(log_dir, "ambulance_monitor.csv"))

    # 6) If you want multi-env or just single, you can do:
    vec_env = DummyVecEnv([lambda: monitored_env])

    # 7) (Optional) Setup an evaluation callback
    #    We'll create a separate environment for evaluation
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

    # 8) Create the PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        n_steps=256,         # <-- Much shorter rollout interval
        batch_size=64,       # Optional but keeps it balanced
        learning_rate=2.5e-4,
        gamma=0.99
    )

    # 9) Train
    #    Provide callback=eval_callback if you want periodic evaluation logs
    run_name = "hehe"
    model.learn(total_timesteps=1_000, callback=eval_callback, tb_log_name=run_name)

    # 10) (Optional) Evaluate final policy
    #     Let's do a quick deterministic rollout
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # model.predict returns (action, state), we only need action
        action, _states = model.predict(obs, deterministic=True)
        # Unpack the Gymnasium step return format
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    print(f"Final policy test episode reward: {total_reward}")

    # 11) Print final logs or handle cleanup
    print("Training complete! Check logs/ folder for monitor files and best model.")
    # The user can parse the CSV logs or load them into e.g. Pandas for further analysis

if __name__ == "__main__":
    main()
