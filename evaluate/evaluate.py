"""
Script to evaluate ambulance dispatch and relocation policies.
Supports both RL-based and nearest-neighbor dispatch policies.
"""
import os
import sys
import numpy as np
import json
import pickle   
from stable_baselines3 import PPO
import pandas as pd
import argparse

# Add project root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import RLDispatchPolicy, StaticRelocationPolicy, NearestDispatchPolicy

# --- Configuration ----------------------------------------------------------
GRAPH   = "data/processed/princeton_graph.gpickle"
CALLS   = "data/processed/synthetic_calls.csv"
PATHS   = "data/matrices/path_cache.pkl"
IDXMAP  = "data/matrices/node_id_to_idx.json"
IDX2NODE = "data/matrices/idx_to_node_id.json"
PFARS   = 241
HOSP    = 1293
DEFAULT_AMBUL = 3  # Default number of ambulances

# Timeout parameters
CALL_TIMEOUT_MEAN = 491.0  # seconds (73 sec base + 418 sec avg travel time)
CALL_TIMEOUT_STD = 10.0   # seconds

def load_data():
    """Load all required data for simulation."""
    print("Loading Princeton road graph...")
    with open(GRAPH, "rb") as f:
        G = pickle.load(f)
    
    print("Loading path cache...")
    with open(PATHS, "rb") as f:
        path_cache = pickle.load(f)
    
    print("Loading node mappings...")
    with open(IDXMAP, "r") as f:
        node_to_idx = {int(k): v for k, v in json.load(f).items()}
    
    with open(IDX2NODE, "r") as f:
        idx_to_node = {int(k): int(v) for k, v in json.load(f).items()}
    
    print("Loading call data...")
    calls = pd.read_csv(CALLS) if CALLS.endswith(".csv") else \
            pickle.load(open(CALLS, "rb"))
    print(f"Calls loaded: {len(calls)} | max day: {calls.day.max()}")
    
    return G, path_cache, node_to_idx, idx_to_node, calls

def create_policies(G, model_path=None, strict=False, num_ambulances=DEFAULT_AMBUL):
    """Create dispatch and relocation policies."""
    if model_path and model_path.lower() != "none":
        print(f"Loading RL model from {model_path}...")
        model = PPO.load(model_path)
        print("Creating RL dispatch policy...")
        dispatch_policy = RLDispatchPolicy(model, G, IDXMAP, num_ambulances, strict=strict)
        print(f"RL policy created with strict={strict}")
    else:
        print("Using nearest-neighbor dispatch policy...")
        dispatch_policy = NearestDispatchPolicy(G)
    
    relocation_policy = StaticRelocationPolicy(G, PFARS)
    return dispatch_policy, relocation_policy

def run_simulation(simulator):
    """Run the simulation and return results."""
    print("\n===== Running Simulation =====")
    print(f"Number of ambulances: {simulator.num_ambulances}")
    print(f"Mean timeout: {CALL_TIMEOUT_MEAN} seconds ({CALL_TIMEOUT_MEAN/60:.1f} minutes)")
    print(f"Timeout std dev: {CALL_TIMEOUT_STD} seconds")
    
    rl_decisions = 0
    fallback_decisions = 0
    total_calls = 0
    last_decision_time = None  # Track last decision time to prevent duplicates
    
    while True:
        ev = simulator.step()
        if ev is None or ev[0] is None:
            break
            
        # Track RL vs fallback decisions only for RL policies
        if hasattr(simulator.dispatch_policy, 'last_decision_type'):
            current_time = ev[0]  # Get current simulation time
            
            # Only count if this is a new decision (not a duplicate)
            if last_decision_time != current_time:
                if simulator.dispatch_policy.last_decision_type == 'rl':
                    rl_decisions += 1
                    print(f"[Simulator] RL decision made (total: {rl_decisions})")
                    if hasattr(simulator.dispatch_policy, 'invalid_action_count'):
                        print(f"  Invalid actions: {simulator.dispatch_policy.invalid_action_count}")
                        print(f"  Could have handled: {simulator.dispatch_policy.could_have_handled_count}")
                elif simulator.dispatch_policy.last_decision_type == 'fallback':
                    fallback_decisions += 1
                    print(f"[Simulator] Fallback decision made (total: {fallback_decisions})")
                
                total_calls += 1
                last_decision_time = current_time
    
    # Only print RL-specific metrics if we're using an RL policy
    if hasattr(simulator.dispatch_policy, 'last_decision_type'):
        print(f"\n[Simulator] Final decision counts:")
        print(f"  RL decisions: {rl_decisions}")
        print(f"  Fallback decisions: {fallback_decisions}")
        print(f"  Total decisions: {rl_decisions + fallback_decisions}")
        if total_calls > 0:  # Only calculate rate if we have decisions
            print(f"  RL decision rate: {(rl_decisions/total_calls*100):.1f}%")
        
        if hasattr(simulator.dispatch_policy, 'invalid_action_count'):
            print(f"\n[RL Policy] Performance metrics:")
            print(f"  Invalid actions: {simulator.dispatch_policy.invalid_action_count}")
            print(f"  Could have handled: {simulator.dispatch_policy.could_have_handled_count}")
            if total_calls > 0:  # Only calculate rate if we have decisions
                print(f"  Invalid action rate: {(simulator.dispatch_policy.invalid_action_count/total_calls*100):.1f}%")
    
    return simulator

def print_statistics(stats):
    """Print detailed simulation statistics."""
    print("\n===== Final Statistics =====")
    
    # Handle both simulator objects and stats dictionaries
    if isinstance(stats, dict):
        total_calls = stats["total_calls"]
        calls_responded = stats["calls_responded"]
        missed_calls = stats["missed_calls"]
        timed_out_calls = stats["timed_out_calls"]
        response_times = stats["response_times"]
    else:
        total_calls = stats.total_calls
        calls_responded = stats.calls_responded
        missed_calls = stats.missed_calls
        timed_out_calls = stats.timed_out_calls
        response_times = stats.response_times
    
    print(f"Total calls: {total_calls}")
    print(f"Calls responded: {calls_responded}")
    print(f"Missed calls: {missed_calls}")
    print(f"Timed out calls: {timed_out_calls}")
    
    if response_times:
        avg_rt = sum(response_times) / len(response_times)
        print(f"Average response time: {avg_rt/60:.1f} minutes")
        print(f"95th percentile response time: {sorted(response_times)[int(len(response_times)*0.95)]/60:.1f} minutes")
    
    total = calls_responded + missed_calls
    rate = 100 * calls_responded / total if total > 0 else 0
    print(f"Response rate: {rate:.1f}%")

def print_comparison(rl_stats, baseline_stats):
    """Print side-by-side comparison of RL and baseline statistics."""
    print("\n===== Policy Comparison =====")
    print(f"{'Metric':<25} {'RL Policy':<15} {'Baseline':<15}")
    print("-" * 55)
    
    metrics = [
        ("Total Calls", "total_calls"),
        ("Calls Responded", "calls_responded"),
        ("Missed Calls", "missed_calls"),
        ("Timed Out Calls", "timed_out_calls"),
        ("Response Rate (%)", lambda s: 100 * s["calls_responded"] / (s["calls_responded"] + s["missed_calls"])),
        ("Avg Response Time (min)", lambda s: sum(s["response_times"]) / len(s["response_times"]) / 60 if s["response_times"] else 0),
        ("95th %ile Response (min)", lambda s: sorted(s["response_times"])[int(len(s["response_times"])*0.95)] / 60 if s["response_times"] else 0)
    ]
    
    for label, metric in metrics:
        if callable(metric):
            rl_val = metric(rl_stats)
            base_val = metric(baseline_stats)
        else:
            rl_val = rl_stats[metric]
            base_val = baseline_stats[metric]
        
        print(f"{label:<25} {rl_val:<15.1f} {base_val:<15.1f}")

def run_policy(G, path_cache, node_to_idx, idx_to_node, calls, model_path=None, strict=False, policy_name="Policy", verbose=False, num_ambulances=DEFAULT_AMBUL):
    """Run a single policy and return its statistics."""
    print(f"\nRunning {policy_name}...")
    dispatch_policy, relocation_policy = create_policies(G, model_path, strict, num_ambulances)
    
    simulator = AmbulanceSimulator(
        graph=G,
        call_data=calls,
        num_ambulances=num_ambulances,
        base_location=PFARS,
        hospital_node=HOSP,
        verbose=verbose,  # Pass through verbose parameter
        call_timeout_mean=CALL_TIMEOUT_MEAN,
        call_timeout_std=CALL_TIMEOUT_STD,
        dispatch_policy=dispatch_policy,
        relocation_policy=relocation_policy,
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        manual_mode=False
    )
    
    simulator = run_simulation(simulator)
    
    return {
        "total_calls": simulator.total_calls,
        "calls_responded": simulator.calls_responded,
        "missed_calls": simulator.missed_calls,
        "timed_out_calls": simulator.timed_out_calls,
        "response_times": simulator.response_times
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to trained RL model")
    parser.add_argument("--strict", action="store_true", help="Use strict mode for RL policy")
    parser.add_argument("--compare", action="store_true", help="Compare RL policy with baseline")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--num_ambulances", type=int, default=DEFAULT_AMBUL, help="Number of ambulances to use")
    args = parser.parse_args()
    
    # Load data
    G, path_cache, node_to_idx, idx_to_node, calls = load_data()
    
    if args.compare:
        # Run both policies and compare
        rl_stats = run_policy(G, path_cache, node_to_idx, idx_to_node, calls, 
                            model_path=args.model, strict=args.strict, 
                            policy_name="RL Policy", verbose=args.verbose,
                            num_ambulances=args.num_ambulances)
        baseline_stats = run_policy(G, path_cache, node_to_idx, idx_to_node, calls,
                                  model_path=None, strict=False,
                                  policy_name="Baseline", verbose=args.verbose,
                                  num_ambulances=args.num_ambulances)
        print_comparison(rl_stats, baseline_stats)
    else:
        # Run single policy
        stats = run_policy(G, path_cache, node_to_idx, idx_to_node, calls,
                          model_path=args.model, strict=args.strict,
                          policy_name="Policy", verbose=args.verbose,
                          num_ambulances=args.num_ambulances)
        print_statistics(stats)

if __name__ == "__main__":
    main()
