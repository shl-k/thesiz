"""
Train PPO on the DispatchRelocEnv that merges queue-based dispatch + relocation.
We handle:
 - Graph, calls, path data loading
 - Simulator + Environment
 - PPO with SB3
 - Tensorboard logging
 - Reward logging to CSV
 - Checkpointing every X timesteps
 - Saving final model and metadata
Also auto-handles OpenMP library conflict via environment flag.
"""

import os, sys, json, pickle, pandas as pd
from datetime import datetime
from pathlib import Path

# Patch OpenMP crash warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator
from envs.dispatch_relocation_env import DispatchRelocEnv

CSV_REWARD_LOG_FILE = "logs/dispatch_relocation/dispatch_relocation_1_ambulance_3M_v1.csv"

class CSVRewardLogger(BaseCallback):
    def __init__(self, filename: str = CSV_REWARD_LOG_FILE, verbose: int = 0):
        super().__init__(verbose)
        self.filename = filename
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        with open(filename, "w") as f:
            f.write("episode,reward\n")

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                self.current_episode_reward += self.locals["rewards"][i]
                if done:
                    episode_index = len(self.episode_rewards) + 1
                    self.episode_rewards.append(self.current_episode_reward)
                    with open(self.filename, "a") as f:
                        f.write(f"{episode_index},{self.current_episode_reward:.3f}\n")
                    self.current_episode_reward = 0.0
        return True

def main():
    TOTAL_TIMESTEPS = 3_000_000
    SAVE_FREQ = 1000000
    LOG_DIR = Path("logs/dispatch_relocation")
    MODEL_DIR = Path("models/dispatch_relocation")
    MODEL_NAME = "dispatch_relocation_1_ambulance_3M_v1"

    LOG_DIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # Load data
    graph = pickle.load(open("data/processed/princeton_graph.gpickle", "rb"))
    calls = pd.read_csv("data/processed/synthetic_calls.csv")
    path_cache = pickle.load(open("data/matrices/path_cache.pkl", "rb"))
    node_to_idx = {int(k): v for k, v in json.load(open("data/matrices/node_id_to_idx.json")).items()}
    idx_to_node = {int(k): int(v) for k, v in json.load(open("data/matrices/idx_to_node_id.json")).items()}

    # Simulator
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
        verbose=False,
        manual_mode=True
    )

    # Env
    core_env = DispatchRelocEnv(
        simulator=sim,
        n_clusters=8,
        node_to_lat_lon_file="data/matrices/node_to_lat_lon.json",
        verbose=False
    )

    monitored_env = Monitor(core_env, str(LOG_DIR / "monitor_logs"))
    vec_env = DummyVecEnv([lambda: monitored_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
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

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=str(MODEL_DIR),
        name_prefix=MODEL_NAME
    )
    csv_logger = CSVRewardLogger()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, csv_logger]
    )

    vec_env.save(MODEL_DIR / f"{MODEL_NAME}_vecnorm.pkl")
    model.save(MODEL_DIR / f"{MODEL_NAME}.zip")
    vec_env.close()

    metadata = {
        "hyperparams": {
            "learning_rate": 2e-4,
            "n_steps": 2048,
            "batch_size": 512,
            "gamma": 0.99,
            "ent_coef": 0.015,
            "clip_range": 0.2,
            "n_epochs": 10,
            "vf_coef": 0.6,
            "total_timesteps": TOTAL_TIMESTEPS
        },
        "env_settings": {
            "num_ambulances": 1,
            "n_clusters": 8,
            "timeout": 600
        },
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    }

    with open(MODEL_DIR / f"{MODEL_NAME}_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Training complete. Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
