"""
Evaluation script for the RL relocation model.
This script loads a trained model and runs it on a test dataset to evaluate performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import SAC
from typing import Dict, List, Optional
from datetime import datetime
import argparse
import json

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.rl.simple_relocation import SimpleRelocationEnv
from src.simulator.simulator import AmbulanceSimulator, EventType
from src.simulator.policies import StaticRelocationPolicy, RLDispatchPolicy
from src.simulator.ambulance import AmbulanceStatus

# Evaluation parameters
OUTPUT_DIR = Path("outputs/relocation_eval")
FIGURE_DIR = Path("outputs/figures")

# Data paths
GRAPH_FILE = "data/processed/princeton_graph.gpickle"
CALLS_FILE = "data/processed/synthetic_test_calls.csv"
PATH_CACHE_FILE = "data/matrices/path_cache.pkl"
NODE_TO_IDX_FILE = "data/matrices/node_id_to_idx.json"
IDX_TO_NODE_FILE = "data/matrices/idx_to_node_id.json"
LAT_LON_TO_NODE_FILE = "data/matrices/lat_lon_to_node.json"
NODE_TO_LAT_LON_FILE = "data/matrices/node_to_lat_lon.json"

def load_data():
    """Load the test data."""
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
        
    # Load lat/lon mapping
    with open(NODE_TO_LAT_LON_FILE, 'r') as f:
        node_to_lat_lon = json.load(f)
        # Convert string keys to tuples
        node_to_lat_lon = {tuple(map(float, k.strip('()').split(','))): v for k, v in node_to_lat_lon.items()}
    
    return G, calls, path_cache, node_to_idx, idx_to_node, node_to_lat_lon

class RLRelocationPolicy:
    """
    Custom relocation policy that uses a trained RL model to make relocation decisions.
    """
    def __init__(self, model, lat_lon_to_node, node_to_lat_lon, path_cache):
        self.model = model
        self.lat_lon_to_node = lat_lon_to_node  # Store the lat/lon to node mapping
        self.node_to_lat_lon = node_to_lat_lon  # Store the node to lat/lon mapping
        self.path_cache = path_cache
        
        # For tracking when we last made a relocation decision
        self.last_relocate_time = 0
        self.relocate_cooldown = 300  # 5 minute cooldown between relocations
        
        # Statistics
        self.relocation_decisions = 0
        self.relocation_destinations = []
        
    def get_nearest_node(self, lat, lon):
        """Find the nearest node to the given coordinates using path cache."""
        # Convert lat/lon to node ID using existing mapping
        rounded_lat, rounded_lon = round(lat, 6), round(lon, 6)
        if (rounded_lat, rounded_lon) in self.lat_lon_to_node:
            return self.lat_lon_to_node[(rounded_lat, rounded_lon)]
        
        # If exact coordinates not found, find the nearest node using coordinates
        # First get coordinates for all nodes
        node_distances = []
        for node_id, (node_lat, node_lon) in self.node_to_lat_lon.items():
            # Calculate straight-line distance
            dist = ((node_lat - lat) ** 2 + (node_lon - lon) ** 2) ** 0.5
            node_distances.append((node_id, dist))
        
        # Sort by distance and take top 10 nodes
        node_distances.sort(key=lambda x: x[1])
        nearest_nodes = [node_id for node_id, _ in node_distances[:10]]
        
        # Among these nodes, find the one with minimum travel time from PFARS
        base_node = 241  # PFARS location
        min_travel_time = float('inf')
        nearest_node = None
        
        for node_id in nearest_nodes:
            if node_id in self.path_cache[base_node]:
                travel_time = self.path_cache[base_node][node_id]['travel_time']
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    nearest_node = node_id
        
        return nearest_node if nearest_node is not None else 241  # Fallback to base
    
    def _build_observation(self, all_ambulances, completed_ambulance, current_time):
        """Build the observation for the model."""
        # Ambulance observations
        ambulances_obs = []
        for amb in all_ambulances:
            # Get coordinates
            if amb.location in self.node_to_lat_lon:
                lat, lon = self.node_to_lat_lon[amb.location]
            else:
                lat, lon = 0.0, 0.0
                
            status_value = amb.status.value
            
            # Calculate busy time
            if hasattr(amb, 'busy_until'):
                busy_time = max(0, amb.busy_until - current_time)
            else:
                busy_time = 0.0
                
            ambulances_obs.append([lat, lon, status_value, busy_time])
            
        # Completed ambulance
        if completed_ambulance.location in self.node_to_lat_lon:
            lat, lon = self.node_to_lat_lon[completed_ambulance.location]
        else:
            lat, lon = 0.0, 0.0
            
        completed_amb_obs = [lat, lon, float(completed_ambulance.id)]
        
        # Time of day (normalized)
        time_of_day = (current_time % (24 * 3600)) / (24 * 3600)
        
        return {
            'ambulances': np.array(ambulances_obs, dtype=np.float32),
            'completed_ambulance': np.array(completed_amb_obs, dtype=np.float32),
            'time_of_day': np.array([time_of_day], dtype=np.float32)
        }
    
    def select_relocation_target(self, ambulance, ambulances=None, current_time=None):
        """
        Decide where to relocate the ambulance using the trained RL model.
        
        Args:
            ambulance: Ambulance that needs relocation
            ambulances: List of all ambulances (for context)
            current_time: Current simulation time
            
        Returns:
            int: Node ID of the relocation target
        """
        # Default to base if no model or not enough context
        if self.model is None or ambulances is None or current_time is None:
            return 241  # Default PFARS base
            
        # Check cooldown
        if current_time - self.last_relocate_time < self.relocate_cooldown:
            return 241  # Default to base during cooldown
            
        try:
            # Build observation for the model
            obs = self._build_observation(ambulances, ambulance, current_time)
            
            # Get action from model (lat/lon coordinates)
            action, _ = self.model.predict(obs, deterministic=True)
            lat, lon = action
            
            # Convert to nearest node
            target_node = self.get_nearest_node(lat, lon)
            
            # Update statistics
            self.relocation_decisions += 1
            self.relocation_destinations.append(target_node)
            self.last_relocate_time = current_time
            
            # Print decision
            print(f"RL relocation decision: Moving ambulance {ambulance.id} to node {target_node} ({lat:.4f}, {lon:.4f})")
            
            return target_node
            
        except Exception as e:
            import traceback
            print(f"Error in relocation decision: {e}")
            traceback.print_exc()
            return 241  # Default to base in case of error
    
    def relocate_ambulances(self, available_ambulances, busy_ambulances, current_time=None, all_ambulances=None):
        """
        Process all ambulances that need relocation.
        Returns a dictionary mapping ambulance IDs to target locations.
        
        Args:
            available_ambulances: List of available ambulances
            busy_ambulances: List of busy ambulances
            current_time: Current simulation time
            all_ambulances: List of all ambulances (for context)
            
        Returns:
            Dict[int, int]: Mapping from ambulance ID to target node
        """
        relocations = {}
        
        # Only relocate ambulances that have just completed service
        for amb in available_ambulances:
            amb_obj = next((a for a in all_ambulances if a.id == amb["id"]), None)
            if amb_obj:
                target = self.select_relocation_target(
                    amb_obj, 
                    ambulances=all_ambulances, 
                    current_time=current_time
                )
                relocations[amb["id"]] = target
                
        return relocations

def run_evaluation(model_path, num_episodes=5, visualize=True, verbose=True):
    """
    Run evaluation of the RL relocation model.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to run
        visualize: Whether to generate visualizations
        verbose: Whether to print detailed output
    """
    # Create output directories if they don't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    G, calls, path_cache, node_to_idx, idx_to_node, node_to_lat_lon = load_data()
    
    # Load lat/lon to node mapping
    with open(LAT_LON_TO_NODE_FILE, 'r') as f:
        lat_lon_to_node = json.load(f)
        # Convert string keys to tuples
        lat_lon_to_node = {tuple(map(float, k.strip('()').split(','))): v for k, v in lat_lon_to_node.items()}
    
    # Load the trained model
    model = SAC.load(model_path)
    
    # Find base and hospital nodes from the graph
    base_node = 241  # PFARS location
    hospital_node = 1293  # Princeton hospital location
    
    # Create the custom RL relocation policy
    rl_relocation_policy = RLRelocationPolicy(model, lat_lon_to_node, node_to_lat_lon, path_cache)
    
    # Statistics to collect
    all_episode_stats = []
    
    for episode in range(num_episodes):
        print(f"Running episode {episode+1}/{num_episodes}")
        
        # Create the simulator with our RL relocation policy
        simulator = AmbulanceSimulator(
            graph=G,
            call_data=calls,
            num_ambulances=3,
            base_location=base_node,
            hospital_node=hospital_node,
            call_timeout_mean=600,  # 10 minute timeout
            call_timeout_std=60,   # 1 minute std
            relocation_policy=rl_relocation_policy,
            verbose=verbose,
            path_cache=path_cache,
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
            manual_mode=False  # Let simulator dispatch calls normally
        )
        
        # Run the simulation
        simulator.initialize()
        episode_complete = False
        
        while not episode_complete:
            event_time, event_type, event_data = simulator.step()
            
            # Check if simulation is done
            if event_type is None:
                episode_complete = True
        
        # Collect statistics
        episode_stats = {
            "episode": episode,
            "total_calls": simulator.stats["total_calls"],
            "calls_served": simulator.stats["calls_served"],
            "calls_timed_out": simulator.stats["calls_timed_out"],
            "mean_response_time": simulator.stats["mean_response_time"],
            "median_response_time": simulator.stats["median_response_time"],
            "relocation_decisions": rl_relocation_policy.relocation_decisions,
            "total_simulation_time": simulator.current_time
        }
        
        all_episode_stats.append(episode_stats)
        
        if verbose:
            print(f"Episode {episode+1} stats:")
            for k, v in episode_stats.items():
                print(f"  {k}: {v}")
        
        # Reset relocation policy stats for next episode
        rl_relocation_policy.relocation_decisions = 0
        rl_relocation_policy.relocation_destinations = []
    
    # Compute aggregate statistics
    aggregate_stats = {
        "num_episodes": num_episodes,
        "mean_calls_served": np.mean([s["calls_served"] for s in all_episode_stats]),
        "mean_calls_timed_out": np.mean([s["calls_timed_out"] for s in all_episode_stats]),
        "mean_response_time": np.mean([s["mean_response_time"] for s in all_episode_stats]),
        "mean_relocation_decisions": np.mean([s["relocation_decisions"] for s in all_episode_stats]),
    }
    
    # Print aggregate statistics
    print("\nAggregate Statistics:")
    for k, v in aggregate_stats.items():
        print(f"  {k}: {v}")
    
    # Save statistics to file
    results_df = pd.DataFrame(all_episode_stats)
    results_path = OUTPUT_DIR / f"relocation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Visualizations if enabled
    if visualize:
        # Response time histogram
        plt.figure(figsize=(10, 6))
        plt.hist([s["mean_response_time"] for s in all_episode_stats], bins=10)
        plt.xlabel("Mean Response Time (seconds)")
        plt.ylabel("Count")
        plt.title("Distribution of Mean Response Times Across Episodes")
        plt.savefig(FIGURE_DIR / "relocation_response_times.png")
        
        # Calls served vs timed out
        plt.figure(figsize=(10, 6))
        served = [s["calls_served"] for s in all_episode_stats]
        timed_out = [s["calls_timed_out"] for s in all_episode_stats]
        plt.bar(range(num_episodes), served, label="Served")
        plt.bar(range(num_episodes), timed_out, bottom=served, label="Timed Out")
        plt.xlabel("Episode")
        plt.ylabel("Number of Calls")
        plt.title("Calls Served vs Timed Out by Episode")
        plt.legend()
        plt.savefig(FIGURE_DIR / "relocation_calls_served.png")
    
    return all_episode_stats, aggregate_stats

def main():
    """Entry point for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a trained RL relocation model")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    
    args = parser.parse_args()
    
    # Run evaluation
    run_evaluation(
        model_path=args.model_path,
        num_episodes=args.episodes,
        visualize=args.visualize,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main() 