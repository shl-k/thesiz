"""
Training script for the SimpleRelocationEnv.
Uses Stable Baselines3 to train a SAC agent for continuous action space (lat/lon coordinates).

Action space: 
- 2D continuous space representing lat/lon coordinates for relocation
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Dict, List, Optional
from datetime import datetime

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.rl.simple_relocation import SimpleRelocationEnv
from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import StaticRelocationPolicy

# Training parameters
TOTAL_TIMESTEPS = 1000000
SAVE_FREQ = 10000  # How often to save checkpoints (set to 0 to disable)
LOG_DIR = Path("logs/simple_relocation")
MODEL_DIR = Path("models")
MODEL_NAME = "simple_relocation_v1"

# Data paths
GRAPH_FILE = "data/processed/princeton_graph.gpickle"
CALLS_FILE = "data/processed/synthetic_calls.csv"
PATH_CACHE_FILE = "data/matrices/path_cache.pkl"
NODE_TO_IDX_FILE = "data/matrices/node_id_to_idx.json"
IDX_TO_NODE_FILE = "data/matrices/idx_to_node_id.json"
LAT_LON_FILE = "data/matrices/lat_lon_mapping.json"

def load_real_data():
    """Load the real Princeton data instead of test data."""
    import pickle
    import json
    
    # Load graph
    with open(GRAPH_FILE, 'rb') as f:
        G = pickle.load(f)
        
    # Load calls
    calls = pd.read_csv(CALLS_FILE)
    
    # Load path cache
    with open(PATH_CACHE_FILE, 'rb') as f:
        path_cache = pickle.load(f)
    
    # Load node mappings
    with open(NODE_TO_IDX_FILE, 'r') as f:
        node_to_idx = json.load(f)
        # Convert string keys to integers
        node_to_idx = {int(k): v for k, v in node_to_idx.items()}
        
    with open(IDX_TO_NODE_FILE, 'r') as f:
        idx_to_node = json.load(f)
        # Convert string keys to integers
        idx_to_node = {int(k): int(v) for k, v in idx_to_node.items()}
    
    return G, calls, path_cache, node_to_idx, idx_to_node

def create_env():
    """Create and initialize the environment."""
    # Load real data
    G, calls, path_cache, node_to_idx, idx_to_node = load_real_data()
    
    # Set up a dummy relocation policy (the RL agent will override this for newly freed ambulances)
    # This is only used for other relocation scenarios not handled by the RL agent
    relocation_policy = StaticRelocationPolicy(G, base_location=241)  # PFARS location
    
    # Find base and hospital nodes from the graph
    base_node = 241  # PFARS location
    hospital_node = 1293  # Princeton hospital location
    
    # Create the simulator with manual mode enabled so our RL agent can make relocation decisions
    simulator = AmbulanceSimulator(
        graph=G,
        call_data=calls,
        num_ambulances=3,
        base_location=base_node,
        hospital_node=hospital_node,
        call_timeout_mean=600,  # 10 minute timeout
        call_timeout_std=60,   # 1 minute std
        relocation_policy=relocation_policy,
        verbose=False,  # Set to False for training
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        manual_mode=True  # Let the RL agent make all decisions
    )
    
    # Create the environment
    env = SimpleRelocationEnv(
        simulator=simulator,
        lat_lon_file=LAT_LON_FILE,
        closest_n_nodes=10,  # Consider 10 closest nodes for relocation
        verbose=False,
        max_steps=1000000,  # Set to 1 million to never cut off episodes
        reward_scale=1.0
    )
    
    return env

def main():
    """Run the training."""
    # Create log and model directories if they don't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create and vectorize environment
    print("Creating environment...")
    vec_env = make_vec_env(
        create_env,
        n_envs=1,
    )
    
    # Create the agent (using SAC for continuous action space)
    print("Creating agent...")
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        action_noise=None
    )
    
    # Set up checkpoint callback if saving frequency > 0
    callbacks = []
    if SAVE_FREQ > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=SAVE_FREQ,
            save_path=str(MODEL_DIR),
            name_prefix=MODEL_NAME
        )
        callbacks.append(checkpoint_callback)
        print(f"Will save checkpoints every {SAVE_FREQ} steps")
    
    # Train the agent
    print("Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=callbacks if callbacks else None
    )
    
    # Save the final model
    final_model_path = MODEL_DIR / f"{MODEL_NAME}_final"
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Load call data to get statistics for logging
    calls_df = pd.read_csv(CALLS_FILE)
    
    # Save hyperparameters and data info for reference
    import json
    import hashlib
    
    # Generate a hash of the calls file to uniquely identify it
    with open(CALLS_FILE, 'rb') as f:
        calls_file_hash = hashlib.md5(f.read()).hexdigest()
    
    training_info = {
        # Model hyperparameters
        "hyperparameters": {
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
            "max_steps": 1000000,
            "reward_scale": 1.0,
            "num_ambulances": 3,
            "total_timesteps": TOTAL_TIMESTEPS,
        },
        
        # Data files used
        "data_files": {
            "graph_file": GRAPH_FILE,
            "calls_file": CALLS_FILE,
            "path_cache_file": PATH_CACHE_FILE,
            "node_to_idx_file": NODE_TO_IDX_FILE,
            "idx_to_node_file": IDX_TO_NODE_FILE,
            "lat_lon_file": LAT_LON_FILE,
            "calls_file_md5": calls_file_hash
        },
        
        # Call data statistics
        "call_data_stats": {
            "total_calls": len(calls_df),
            "days_covered": int(calls_df["day"].max()) if "day" in calls_df.columns else 1,
            "unique_origin_nodes": len(calls_df["origin_node"].unique()),
            "unique_destination_nodes": len(calls_df["destination_node"].unique()) if "destination_node" in calls_df.columns else 0,
        },
        
        # Training metadata
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "model_name": MODEL_NAME,
            "final_model_path": str(final_model_path)
        }
    }
    
    info_path = MODEL_DIR / f"{MODEL_NAME}_info.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"Training info and hyperparameters saved to {info_path}")
    
    # Close environment
    vec_env.close()

if __name__ == "__main__":
    main() 