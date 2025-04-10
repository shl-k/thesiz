# ========== CONFIG ==========
MODEL_NAME = "reloc_M4"
NUM_AMBULANCES = 4
TOTAL_TIMESTEPS = 1000_000
CALLS_FILE = "data/processed/synthetic_calls.csv"
CALL_TIMEOUT_MEAN = 491
CALL_TIMEOUT_STD = 10
BASE_NODE = 241
HOSPITAL_NODE = 1293

# ========== DIRECTORIES ==========
from pathlib import Path
LOG_DIR = Path(f"logs/{MODEL_NAME}")
MODEL_DIR = Path(f"models/{MODEL_NAME}")
REWARD_LOG_FILE = LOG_DIR / f"{MODEL_NAME}_reward_log.csv"


LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ========== IMPORTS ==========
import os, sys, json, pickle, pandas as pd
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator
from envs.dispatch_relocation_env import DispatchRelocEnv

# ========== LEARNING PARAMS ==========
LEARNING_PARAMS = dict(
    learning_rate = 1e-4,
    n_steps       = 4096,
    batch_size    = 512,
    gamma         = 0.99,
    ent_coef      = 0.015,
    clip_range    = 0.2,
    n_epochs      = 10,
    vf_coef       = 0.6,
)

# ========== REWARD LOGGER ==========
class CSVRewardLogger(BaseCallback):
    def __init__(self, filename: Path, verbose: int = 0):
        super().__init__(verbose)
        self.filename = filename
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        with open(filename, "w") as f:
            f.write("episode,reward\n")

    def _on_step(self) -> bool:
        if self.locals.get("dones") is None:
            return True

        for i, done in enumerate(self.locals["dones"]):
            self.current_episode_reward += self.locals["rewards"][i]
            if done:
                ep_index = len(self.episode_rewards) + 1
                self.episode_rewards.append(self.current_episode_reward)
                with open(self.filename, "a") as f:
                    f.write(f"{ep_index},{self.current_episode_reward:.3f}\n")
                self.current_episode_reward = 0.0
        return True

# ========== MAIN ==========
def main():
    DATA_DIR   = Path("data/processed")
    MATRIX_DIR = Path("data/matrices")

    # Load data
    graph = pickle.load(open(DATA_DIR / "princeton_graph.gpickle", "rb"))
    calls = pd.read_csv(CALLS_FILE)
    path_cache = pickle.load(open(MATRIX_DIR / "path_cache.pkl", "rb"))
    node_to_idx = {int(k): v for k, v in json.load(open(MATRIX_DIR / "node_id_to_idx.json")).items()}
    idx_to_node = {int(k): int(v) for k, v in json.load(open(MATRIX_DIR / "idx_to_node_id.json")).items()}

    sim = AmbulanceSimulator(
        graph=graph,
        call_data=calls,
        num_ambulances=NUM_AMBULANCES,
        base_location=BASE_NODE,
        hospital_node=HOSPITAL_NODE,
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        call_timeout_mean=CALL_TIMEOUT_MEAN,
        call_timeout_std=CALL_TIMEOUT_STD,
        verbose=False,
        manual_mode=True
    )

    env = DummyVecEnv([
        lambda: Monitor(DispatchRelocEnv(
            simulator=sim,
            n_clusters=8,
            node_to_lat_lon_file="data/matrices/node_to_lat_lon.json",
            verbose=False
        ))
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(LOG_DIR), **LEARNING_PARAMS)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            CheckpointCallback(save_freq=TOTAL_TIMESTEPS, save_path=str(MODEL_DIR), name_prefix=MODEL_NAME),
            CSVRewardLogger(REWARD_LOG_FILE)
        ]
    )

    env.save(MODEL_DIR / f"{MODEL_NAME}_vecnorm.pkl")
    model.save(MODEL_DIR / f"{MODEL_NAME}.zip")

    metadata = {
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "timesteps": TOTAL_TIMESTEPS,
        "num_ambulances": NUM_AMBULANCES,
        "calls_file": CALLS_FILE,
        "learning_params": LEARNING_PARAMS,
        "reward_structure": {
            "dispatch_reward": "15 - travel_time (min)",
            "missed_dispatch_penalty": -15,
            "scene_arrival_bonus": 15,
            "hospital_arrival_bonus": 30,
            "correct_no_dispatch_bonus": 5,
            "incorrect_no_dispatch_penalty": -30,
            "timeout_penalty": -30,
            "relocation_reward": 15,
            "no_relocate_penalty": -10,
            "invalid_action_penalty": -50,
            "response_time_improvement_bonus": "(pre_avg - post_avg) / 60.0"
        }
    }

    with open(LOG_DIR / f"{MODEL_NAME}_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Training complete!")

    

if __name__ == "__main__":
    main()
