"""
Train a PPO agent on the SimpleDispatchEnv.
"""

import os, sys, json, pickle, pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies   import StaticRelocationPolicy
from envs.simple_dispatch import SimpleDispatchEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# ---------------------------------------------------------------------
#  Paths & constants
# ---------------------------------------------------------------------
DATA_DIR   = Path("data/processed")
MATRIX_DIR = Path("data/matrices")
LOG_DIR    = Path("logs/simple_dispatch")
MODEL_DIR  = Path("models/simple_dispatch")

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CSV_REWARD_LOG = LOG_DIR / "reward_log.csv"
MODEL_PREFIX   = "dispatch_agent"

TOTAL_TIMESTEPS   = 300_000
NUM_AMBULANCES    = 1
CALL_TIMEOUT_MEAN = 491
CALL_TIMEOUT_STD  = 10
BASE_NODE         = 241
HOSPITAL_NODE     = 1293

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

# ---------------------------------------------------------------------
#  Reward logger
# ---------------------------------------------------------------------
class CSVRewardLogger(BaseCallback):
    def __init__(self, filename: str, verbose: int = 0):
        super().__init__(verbose)
        self.filename = filename
        self.episode_rewards = []
        self.current_episode_reward = 0.0

        with open(filename, "w") as f:
            f.write("episode,reward,calls_responded\n")

    def _on_step(self) -> bool:
        if self.locals.get("dones") is None:
            return True

        for i, done in enumerate(self.locals["dones"]):
            self.current_episode_reward += self.locals["rewards"][i]
            if done:
                # Unwrap: DummyVecEnv → Monitor → SimpleDispatchEnv
                env = self.training_env.envs[i]
                raw_env = env.env if hasattr(env, "env") else env.unwrapped

                ep_index = len(self.episode_rewards) + 1
                self.episode_rewards.append(self.current_episode_reward)
                calls_responded = raw_env.episode_calls_responded

                with open(self.filename, "a") as f:
                    f.write(f"{ep_index},{self.current_episode_reward:.3f},{calls_responded}\n")

                self.current_episode_reward = 0.0

        return True

# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main():
    # ----- data -------------------------------------------------------
    graph        = pickle.load(open(DATA_DIR / "princeton_graph.gpickle", "rb"))
    calls        = pd.read_csv(DATA_DIR / "synthetic_calls.csv")
    path_cache   = pickle.load(open(MATRIX_DIR / "path_cache.pkl", "rb"))
    node_to_idx  = {int(k): v for k, v in json.load(open(MATRIX_DIR / "node_id_to_idx.json")).items()}
    idx_to_node  = {int(k): int(v) for k, v in json.load(open(MATRIX_DIR / "idx_to_node_id.json")).items()}


    sim = AmbulanceSimulator(
        graph               = graph,
        call_data           = calls,
        num_ambulances      = NUM_AMBULANCES,
        base_location       = BASE_NODE,
        hospital_node       = HOSPITAL_NODE,
        call_timeout_mean   = CALL_TIMEOUT_MEAN,
        call_timeout_std    = CALL_TIMEOUT_STD,
        relocation_policy   = StaticRelocationPolicy(graph, base_location=BASE_NODE),
        path_cache          = path_cache,
        node_to_idx         = node_to_idx,
        idx_to_node         = idx_to_node,
        verbose             = False,
        manual_mode         = True,
    )

    # ----- environment ------------------------------------------------

    core_env   = SimpleDispatchEnv(sim, max_steps=100_000, verbose=False)
    monitor    = Monitor(core_env)
    vec_env    = DummyVecEnv([lambda: monitor])
    vec_env    = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ----- PPO --------------------------------------------------------
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose        = 1,
        tensorboard_log= str(LOG_DIR),
        **LEARNING_PARAMS,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq = TOTAL_TIMESTEPS,
        save_path = str(MODEL_DIR),
        name_prefix = MODEL_PREFIX,
    )
    reward_logger = CSVRewardLogger(CSV_REWARD_LOG)

    print(f"🚀 Training for {TOTAL_TIMESTEPS:,} timesteps …")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [checkpoint_cb, reward_logger],
    )

    # ----- save -------------------------------------------------------
    vec_env.save(MODEL_DIR / f"{MODEL_PREFIX}_vecnorm.pkl")
    model.save(MODEL_DIR / f"{MODEL_PREFIX}.zip")

    with open(MODEL_DIR / f"{MODEL_PREFIX}_info.json", "w") as f:
        json.dump(
            {
                "hyperparameters": LEARNING_PARAMS,
                "timestamp"      : datetime.now().isoformat(),
                "timesteps"      : TOTAL_TIMESTEPS,
                "num_ambulances" : NUM_AMBULANCES,
                "call_timeout_mean" : CALL_TIMEOUT_MEAN,
                "call_timeout_std"  : CALL_TIMEOUT_STD, 
                "base_node"         : BASE_NODE,
                "hospital_node"     : HOSPITAL_NODE,       
                "call_data"         : DATA_DIR / "synthetic_calls.csv"
            },
            f,
            indent=2,
        )

    print("✅ Training complete!")

if __name__ == "__main__":
    main()
