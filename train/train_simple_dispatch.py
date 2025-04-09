"""
Basic training script for the SimpleDispatchEnv.
Uses Stable Baselines3 to train a PPO agent.

Action space: 
- 0 to num_ambulances-1: dispatch the ambulance with that index
- num_ambulances: do not dispatch any ambulance (no-op)
"""

import os, sys, pandas as pd, json, pickle
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor 
from typing import Dict, List, Optional
from datetime import datetime

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from envs.simple_dispatch import SimpleDispatchEnv
from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import StaticRelocationPolicy

CSV_REWARD_LOG_FILE = "logs/simple_dispatch/reward_log.csv"

class CSVRewardLogger(BaseCallback):
    """
    After each episode, log the reward to a CSV file.
    """
    def __init__(self, filename: str = CSV_REWARD_LOG_FILE, verbose: int = 0):
        super().__init__(verbose)
        self.filename = filename
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Headers
        with open(filename, "w") as f:
            f.write("episode,reward\n")

    def _on_step(self) -> bool:
        """
        Called at each environment step.
        We'll accumulate reward until `done=True`, then log one line per episode.
        """
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                self.current_episode_reward += self.locals["rewards"][i]
                if done:
                    # Episode ended
                    episode_index = len(self.episode_rewards) + 1
                    self.episode_rewards.append(self.current_episode_reward)
                    with open(self.filename, "a") as f:
                        f.write(f"{episode_index},{self.current_episode_reward:.3f}\n")
                    # Reset counters
                    self.current_episode_reward = 0.0
        return True

def main():
    # Training parameters
    TOTAL_TIMESTEPS = 3000000  
    SAVE_FREQ = 1000000
    LOG_DIR = Path("logs/simple_dispatch")
    MODEL_DIR = Path("models/simple_dispatch")
    MODEL_NAME = "dispatch_1_ambulance_3M_v1"  # Single ambulance version

    # Load real data
    G          = pickle.load(open("data/processed/princeton_graph.gpickle", "rb"))
    calls      = pd.read_csv("data/processed/synthetic_calls.csv")
    path_cache = pickle.load(open("data/matrices/path_cache.pkl", "rb"))
    node_to_index   = {int(k): v for k, v in json.load(open("data/matrices/node_id_to_idx.json")).items()}
    index_to_node   = {int(k): int(v) for k, v in json.load(open("data/matrices/idx_to_node_id.json")).items()}
    base_node = 241  # PFARS location
    hospital_node = 1293  # Princeton hospital location

        
    # Create the simulator
    simulator = AmbulanceSimulator(
        graph=G,
        call_data=calls,
        num_ambulances=1,  # Single ambulance
        base_location=base_node,
        hospital_node=hospital_node,
        call_timeout_mean=600,  # 10 minute timeout
        call_timeout_std=60,   # 1 minute std
        relocation_policy=StaticRelocationPolicy(G, base_location=base_node),
        verbose=False,  
        path_cache=path_cache,
        node_to_idx=node_to_index,
        idx_to_node=index_to_node,
        manual_mode=True  # Let the RL agent make all dispatch decisions
    )

    simple_dispatch_env = SimpleDispatchEnv(simulator)
    monitored_env = Monitor(simple_dispatch_env, LOG_DIR / "monitor_logs") 
    vec_env  = DummyVecEnv([lambda: simple_dispatch_env])
    vec_env  = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO("MultiInputPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=str(LOG_DIR),
                learning_rate=2e-4,
                n_steps=2048,
                batch_size=512,
                gamma=0.99,
                ent_coef=0.015,
                clip_range=0.2,
                n_epochs=10,
                vf_coef=0.6
                )


    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, 
                                            save_path=str(MODEL_DIR), 
                                            name_prefix=MODEL_NAME)
    
    reward_logger = CSVRewardLogger()
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=[checkpoint_callback, reward_logger]
        )

    # Save the trained model and environment
    model_path = MODEL_DIR / f"{MODEL_NAME}"
    vec_env.save(model_path)
    model.save(model_path)
    vec_env.close()

    # Save training information and hyperparameters
    training_info = {
        # Model hyperparameters
        "hyperparameters": {
            "learning_rate": 2e-4,
            "n_steps": 2048,
            "batch_size": 512,
            "gamma": 0.99,
            "ent_coef": 0.015,
            "clip_range": 0.2,
            "n_epochs": 10,
            "vf_coef": 0.6,
            "num_ambulances": 1,  # Document that this is a 1-ambulance model
            "total_timesteps": TOTAL_TIMESTEPS,
        },
        
        # Environment configuration
        "environment": {
            "num_ambulances": 1,  # Explicitly document ambulance count
            "base_location": base_node,  # PFARS location
            "hospital_location": hospital_node,  # Princeton hospital
            "call_timeout_mean": 600,  # 10 minute timeout
            "call_timeout_std": 60,    # 1 minute std
        },
        
        # Call data statistics
        "call_data_stats": {
            "total_calls": len(calls),
            "days_covered": int(calls["day"].max()) if "day" in calls.columns else 1,
            "unique_origin_nodes": len(calls["origin_node"].unique()),
            "unique_destination_nodes": len(calls["destination_node"].unique()) if "destination_node" in calls.columns else 0,
        },
        
        # Training metadata
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "model_name": MODEL_NAME,
            "final_model_path": str(model_path),
            "description": "Single ambulance configuration to test performance with reduced fleet size"
        }
    }

    info_path = MODEL_DIR / f"{MODEL_NAME}_info.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"Training info and hyperparameters saved to {info_path}")

if __name__ == "__main__":
    main()
