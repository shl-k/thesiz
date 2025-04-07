"""
evaluate_unified.py - Universal evaluation script for ambulance dispatch models

Usage:
  python evaluate_unified.py --model=[model_path] --env=[environment_type] --compare
  
  --model:     Path to the trained model (.zip file)
  --env:       Type of environment ('simple', 'pick0', etc.)
  --compare:   If provided, will also run baseline policies for comparison
  --days:      Number of days to evaluate (default: 7)
  --verbose:   If provided, will show detailed logs during evaluation
"""

import os
import sys
import json
import pickle
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from stable_baselines3 import PPO

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# Import environments and policies
from src.rl.simple_dispatch import SimpleDispatchEnv
from src.simulator.ambulance import AmbulanceStatus
from src.simulator.simulator import AmbulanceSimulator, EventType
from src.simulator.policies import (
    NearestDispatchPolicy, 
    RLDispatchPolicy,
    StaticRelocationPolicy  
)

# Constants and paths
GRAPH_PATH = "data/processed/princeton_graph.gpickle"
CALLS_PATH = "data/processed/synthetic_calls.csv"
NODE_TO_IDX_PATH = "data/matrices/node_id_to_idx.json"
IDX_TO_NODE_PATH = "data/matrices/idx_to_node_id.json"
PATH_CACHE_PATH = "data/matrices/path_cache.pkl"
NODE_TO_LAT_LON_PATH = "data/matrices/node_to_lat_lon.json"
LAT_LON_TO_NODE_PATH = "data/matrices/lat_lon_to_node.json"
PFARS_NODE = 241
HOSPITAL_NODE = 1293
NUM_AMBULANCES = 3
LOGS_DIR = "logs/evaluation"

# Make sure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

def format_time(seconds):
    """Convert seconds to a human-readable format."""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def load_data(days=7):
    """Load graph and calls data."""
    print("Loading Princeton road graph...")
    with open(GRAPH_PATH, 'rb') as f:
        G = pickle.load(f)
    
    print("Loading path cache...")
    with open(PATH_CACHE_PATH, 'rb') as f:
        path_cache = pickle.load(f)
    
    # Load node mappings
    with open(NODE_TO_IDX_PATH, 'r') as f:
        node_to_idx = json.load(f)
        # Convert string keys to integers
        node_to_idx = {int(k): v for k, v in node_to_idx.items()}
    
    with open(IDX_TO_NODE_PATH, 'r') as f:
        idx_to_node = json.load(f)
        # Convert string keys to integers
        idx_to_node = {int(k): int(v) for k, v in idx_to_node.items()}
    
    # Filter calls for specified number of days
    calls_df = pd.read_csv(CALLS_PATH)
    if days > 0:
        calls_df = calls_df[calls_df['day'] <= days]
    
    return G, calls_df, path_cache, node_to_idx, idx_to_node

def create_environment(G, calls_df, path_cache, node_to_idx, idx_to_node, verbose=False):
    """Create the SimpleDispatchEnv environment consistently with how it was trained."""
    # Set up the relocation policy (for returning ambulances to base after hospital)
    relocation_policy = StaticRelocationPolicy(G, base_location=PFARS_NODE)
    
    # Create the simulator
    simulator = AmbulanceSimulator(
        graph=G,
        call_data=calls_df,
        num_ambulances=NUM_AMBULANCES,
        base_location=PFARS_NODE,
        hospital_node=HOSPITAL_NODE,
        call_timeout_mean=600,  # 10 minute timeout
        call_timeout_std=60,    # 1 minute std
        relocation_policy=relocation_policy,
        verbose=verbose,
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        manual_mode=True  # Let the RL agent make all dispatch decisions
    )
    
    # Create the environment with the simulator
    env = SimpleDispatchEnv(
        simulator=simulator,
        lat_lon_file=LAT_LON_TO_NODE_PATH,  # This is only used for RL observations
        verbose=verbose
    )
    
    return env

def evaluate_rl_model(model, env, verbose=False):
    """Evaluate a trained RL model in the given environment."""
    call_count = 0
    dispatched_calls = 0
    missed_calls = 0
    all_response_times = []
    amb_utilization = [0] * NUM_AMBULANCES
    amb_busy_when_selected = [0] * NUM_AMBULANCES
    total_reward = 0.0
    history = []
    
    # Reset environment
    obs, _ = env.reset()
    done = False
    
    print("\n===== Starting RL Model Evaluation =====")
    
    while not done:
        # Get the next event
        next_event = env.simulator.event_queue[0] if env.simulator.event_queue else None
        if next_event:
            _, event_time, event_type, event_data = next_event
            
        # Use the model to predict the best action
        action, _ = model.predict(obs, deterministic=True)
        
        # For call arrivals, check ambulance status BEFORE stepping
        if next_event and event_type == "call_arrival":
            call_count += 1
            call_node = event_data["origin_node"]
            
            # Determine which ambulance was chosen
            chosen_amb = int(action) if isinstance(action, (np.ndarray, list, int, np.int64)) else action
            
            if chosen_amb < NUM_AMBULANCES:
                # Check if ambulance is available
                amb = env.simulator.ambulances[chosen_amb]
                if amb.is_available():
                    dispatched_calls += 1
                    amb_utilization[chosen_amb] += 1
                    
                    if verbose:
                        print(f"\n📞 Call at {format_time(env.simulator.current_time)}:")
                        print(f"   From node {call_node} to {HOSPITAL_NODE}")
                        print(f"   ✅ Dispatched ambulance {chosen_amb}")
                else:
                    missed_calls += 1
                    amb_busy_when_selected[chosen_amb] += 1
                    
                    if verbose:
                        print(f"\n📞 Call at {format_time(env.simulator.current_time)}:")
                        print(f"   From node {call_node} to {HOSPITAL_NODE}")
                        print(f"   ❌ Ambulance {chosen_amb} busy - status: {amb.status.name}")
            else:
                missed_calls += 1
                if verbose:
                    print(f"\n📞 Call at {format_time(env.simulator.current_time)}:")
                    print(f"   From node {call_node} to {HOSPITAL_NODE}")
                    print(f"   ❌ No ambulance dispatched (action: {chosen_amb})")
        
        # Take a step in the environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        # After stepping, record response time for successful dispatches
        if next_event and event_type == "call_arrival" and isinstance(action, (int, np.int64, np.ndarray)) and int(action) < NUM_AMBULANCES:
            amb = env.simulator.ambulances[int(action)]
            # Calculate response time based on the ambulance's busy_until
            if amb.busy_until > env.simulator.current_time:  # If busy_until was updated, a dispatch occurred
                response_time = amb.busy_until - env.simulator.current_time
                all_response_times.append(response_time)
                
                if verbose:
                    print(f"   Est. response time: {response_time/60:.1f} min")
        
        # Record history for analysis
        history.append({
            "time": env.simulator.current_time,
            "time_fmt": format_time(env.simulator.current_time),
            "action": int(action) if isinstance(action, (int, np.int64, np.ndarray)) else action,
            **{f"amb_{i}_status": amb.status.name for i, amb in enumerate(env.simulator.ambulances)},
            **{f"amb_{i}_location": amb.location for i, amb in enumerate(env.simulator.ambulances)},
            "reward": reward
        })
        
        # Update for next iteration
        obs = next_obs
        done = terminated or truncated
    
    # Calculate statistics
    avg_rt = np.mean(all_response_times) if all_response_times else 0
    
    print("\n===== RL Model Evaluation Results =====")
    print(f"Total reward      : {total_reward:.1f}")
    print(f"Total calls       : {call_count}")
    print(f"Dispatched calls  : {dispatched_calls}")
    print(f"Missed calls      : {missed_calls}")
    
    if call_count > 0:
        print(f"Response rate     : {dispatched_calls/call_count*100:.1f}%")
    
    if all_response_times:
        print(f"Avg response time : {avg_rt:.1f}s ({avg_rt/60:.1f} min)")
        print(f"Min response time : {min(all_response_times):.1f}s ({min(all_response_times)/60:.1f} min)")
        print(f"Max response time : {max(all_response_times):.1f}s ({max(all_response_times)/60:.1f} min)")
        print(f"95th percentile   : {np.percentile(all_response_times, 95):.1f}s")
    else:
        print("No response times recorded.")
    
    print("\nAmbulance utilization:")
    for i in range(NUM_AMBULANCES):
        if call_count > 0:
            print(f"  Ambulance {i}: {amb_utilization[i]} calls ({amb_utilization[i]/call_count*100:.1f}%)")
        else:
            print(f"  Ambulance {i}: {amb_utilization[i]} calls")
    
    print("\nBusy ambulances when selected:")
    for i in range(NUM_AMBULANCES):
        print(f"  Ambulance {i}: busy {amb_busy_when_selected[i]} times")
    
    # Save history to CSV
    history_df = pd.DataFrame(history)
    output_path = os.path.join(LOGS_DIR, "rl_evaluation_history.csv")
    history_df.to_csv(output_path, index=False)
    print(f"\nDetailed history saved to {output_path}")
    
    # Return results for comparison
    return {
        "method": "RL Model",
        "reward": total_reward,
        "calls": call_count,
        "dispatched": dispatched_calls,
        "missed": missed_calls,
        "response_rate": dispatched_calls/call_count*100 if call_count > 0 else 0,
        "avg_response_time": avg_rt,
        "p95_response_time": np.percentile(all_response_times, 95) if all_response_times else 0,
        "utilization": amb_utilization,
        "busy_when_selected": amb_busy_when_selected
    }

def run_baseline_simulator(G, calls_df, path_cache, node_to_idx, idx_to_node, verbose=False):
    """Run the simulator with a baseline nearest policy and static relocation."""
    # Create the nearest dispatch policy
    nearest_policy = NearestDispatchPolicy(G)
    
    # Create the static relocation policy (same as used with RL)
    relocation_policy = StaticRelocationPolicy(G, base_location=PFARS_NODE)
    
    # Create the simulator with nearest dispatch
    sim = AmbulanceSimulator(
        graph=G,
        call_data=calls_df,
        num_ambulances=NUM_AMBULANCES,
        base_location=PFARS_NODE,
        hospital_node=HOSPITAL_NODE,
        call_timeout_mean=600,  # 10 minute timeout
        call_timeout_std=60,    # 1 minute std
        dispatch_policy=nearest_policy,
        relocation_policy=relocation_policy,
        verbose=verbose,
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        manual_mode=False  # Use the nearest dispatch policy
    )
    
    # Run simulation
    while True:
        result = sim.step()
        if result is None or result[0] is None:
            break
    
    # Calculate statistics
    total_calls = sim.calls_responded + sim.missed_calls
    avg_rt = np.mean(sim.response_times) if sim.response_times else 0
    p95_rt = np.percentile(sim.response_times, 95) if sim.response_times else 0
    
    print(f"\n===== Nearest Dispatch Results =====")
    print(f"Total calls    : {total_calls}")
    print(f"Calls responded: {sim.calls_responded}")
    print(f"Missed calls   : {sim.missed_calls}")
    
    if total_calls > 0:
        print(f"Response rate  : {sim.calls_responded/total_calls*100:.1f}%")
    
    if sim.response_times:
        print(f"Avg response time: {avg_rt:.1f}s ({avg_rt/60:.1f} min)")
        print(f"95th percentile  : {p95_rt:.1f}s ({p95_rt/60:.1f} min)")
    
    # Return results for comparison
    return {
        "method": "Nearest Dispatch",
        "calls": total_calls,
        "dispatched": sim.calls_responded,
        "missed": sim.missed_calls,
        "response_rate": sim.calls_responded/total_calls*100 if total_calls > 0 else 0,
        "avg_response_time": avg_rt,
        "p95_response_time": p95_rt,
    }

def compare_results(results):
    """Compare and display results from different methods."""
    # Convert results to DataFrame for easier comparison
    df = pd.DataFrame(results)
    
    # Set method as index for better display
    df.set_index("method", inplace=True)
    
    # Format response times to minutes
    df["avg_response_min"] = df["avg_response_time"] / 60
    df["p95_response_min"] = df["p95_response_time"] / 60
    
    # Select and order columns for display
    display_cols = [
        "calls", "dispatched", "missed", "response_rate",
        "avg_response_min", "p95_response_min"
    ]
    
    # Rename columns for better readability
    display_names = {
        "calls": "Total Calls",
        "dispatched": "Dispatched",
        "missed": "Missed",
        "response_rate": "Response Rate (%)",
        "avg_response_min": "Avg Response (min)",
        "p95_response_min": "95th % (min)"
    }
    
    # Create a formatted table
    table = df[display_cols].rename(columns=display_names)
    
    print("\n===== COMPARISON OF METHODS =====")
    print(table.round(1))
    
    # Save comparison to CSV
    output_path = os.path.join(LOGS_DIR, "comparison_results.csv")
    table.to_csv(output_path)
    print(f"\nComparison results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RL models against baselines")
    parser.add_argument("--model", type=str, default="models/simple_dispatch_7_1M_v3.zip", 
                      help="Path to trained model .zip file")
    parser.add_argument("--days", type=int, default=7, help="Number of days to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    parser.add_argument("--strict", action="store_true", help="Use strict mode - RL agent must pick valid ambulances only")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return
    
    # Load data
    G, calls_df, path_cache, node_to_idx, idx_to_node = load_data(args.days)
    print(f"Loaded {len(calls_df)} calls from dataset")
    
    results = []
    
    # Load RL model
    print(f"Loading model from {args.model}")
    model = PPO.load(args.model)
    
    # Create RL dispatch policy
    rl_policy = RLDispatchPolicy(
        model=model,
        graph=G,
        node_to_idx_path=NODE_TO_IDX_PATH,
        num_ambulances=NUM_AMBULANCES,
        strict=args.strict  # Pass the strict flag
    )
    
    # Create nearest dispatch policy
    nearest_policy = NearestDispatchPolicy(G)
    
    # Create static relocation policy (same for both)
    relocation_policy = StaticRelocationPolicy(G, base_location=PFARS_NODE)
    
    # 1. Evaluate RL model using the simulator directly
    rl_sim_results = run_simulator_with_policy(
        G, calls_df, path_cache, node_to_idx, idx_to_node,
        dispatch_policy=rl_policy,
        relocation_policy=relocation_policy,
        tag="RL Model",
        verbose=args.verbose
    )
    results.append(rl_sim_results)
    
    # 2. Evaluate nearest dispatch baseline
    nearest_results = run_simulator_with_policy(
        G, calls_df, path_cache, node_to_idx, idx_to_node,
        dispatch_policy=nearest_policy,
        relocation_policy=relocation_policy,
        tag="Nearest Dispatch",
        verbose=args.verbose
    )
    results.append(nearest_results)
    
    # Compare results
    compare_results(results)

def run_simulator_with_policy(G, calls_df, path_cache, node_to_idx, idx_to_node, 
                              dispatch_policy, relocation_policy, tag, verbose=False):
    """Run the simulator with a specific dispatch policy."""
    # Set verbose flag for policies
    if isinstance(dispatch_policy, RLDispatchPolicy):
        import src.simulator.policies as policies
        policies.verbose_global = verbose
    
    # Create the simulator with the given policy
    sim = AmbulanceSimulator(
        graph=G,
        call_data=calls_df.copy(),  # Use a copy to ensure no modification
        num_ambulances=NUM_AMBULANCES,
        base_location=PFARS_NODE,
        hospital_node=HOSPITAL_NODE,
        call_timeout_mean=600,  # 10 minute timeout
        call_timeout_std=60,    # 1 minute std
        dispatch_policy=dispatch_policy,
        relocation_policy=relocation_policy,
        verbose=verbose,
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        manual_mode=False  # Let the policy handle dispatches
    )
    
    # Run simulation
    print(f"\n===== Starting {tag} Evaluation =====")
    event_count = 0
    call_count = 0
    call_ids_seen = set()  # Track all call IDs seen
    
    # Detailed tracking by call ID
    call_details = {}  # call_id -> dict of details about the call
    hospital_deliveries = set()  # Call IDs where patient was delivered to hospital
    
    while True:
        result = sim.step()
        if result is None or result[0] is None:
            # End of simulation
            break
            
        event_count += 1
        event_time, event_type, event_data = result
        
        # Track call arrivals 
        if event_type == "call_arrival":
            call_id = event_data.get("call_id", "unknown")
            call_ids_seen.add(call_id)
            call_count += 1
            
            # Initialize call details
            call_details[call_id] = {
                "time": event_time,
                "origin_node": event_data.get("origin_node", "unknown"),
                "dispatched": False,
                "ambulance_id": None,
                "arrived_at_scene": False,
                "arrived_at_hospital": False,
                "timed_out": False,
                "available_ambulances": sum(1 for amb in sim.ambulances if amb.is_available())
            }
            
            if verbose:
                print(f"\nCall {call_count}: ID={call_id} at time {format_time(event_time)}")
                print(f"  Available ambulances: {call_details[call_id]['available_ambulances']}/{NUM_AMBULANCES}")
        
        # Track when an ambulance is dispatched to a call
        elif event_type == "amb_dispatched":
            amb_id = event_data.get("amb_id")
            call_id = event_data.get("call_id")
            
            if call_id in call_details:
                call_details[call_id]["dispatched"] = True
                call_details[call_id]["ambulance_id"] = amb_id
                if verbose:
                    print(f"Ambulance {amb_id} dispatched to call {call_id}")
        
        # Track ambulance scene arrivals
        elif event_type == "amb_scene_arrival":
            amb_id = event_data.get("amb_id")
            call_id = event_data.get("call_id")
            
            if call_id in call_details:
                call_details[call_id]["arrived_at_scene"] = True
                # Also mark as dispatched in case we missed the dispatch event
                if not call_details[call_id]["dispatched"]:
                    call_details[call_id]["dispatched"] = True
                    call_details[call_id]["ambulance_id"] = amb_id
                if verbose:
                    print(f"Ambulance {amb_id} arrived at scene for call {call_id}")
        
        # Track ambulance hospital arrivals - this counts as a "successful" call
        elif event_type == "amb_hospital_arrival":
            amb_id = event_data.get("amb_id")
            call_id = event_data.get("call_id")
            
            if call_id is not None and call_id in call_details:
                call_details[call_id]["arrived_at_hospital"] = True
                hospital_deliveries.add(call_id)
                if verbose:
                    print(f"Ambulance {amb_id} delivered patient from call {call_id} to hospital")
            else:
                # FALLBACK: If call_id is not in the event data, try to infer it
                # Look up which call this ambulance was responding to
                ambulance = next((a for a in sim.ambulances if a.id == amb_id), None)
                if ambulance:
                    # Try to find the call in our tracking based on ambulance ID
                    for cid, details in call_details.items():
                        if details.get("ambulance_id") == amb_id and details.get("arrived_at_scene"):
                            details["arrived_at_hospital"] = True
                            hospital_deliveries.add(cid)
                            if verbose:
                                print(f"Ambulance {amb_id} delivered patient from call {cid} to hospital (inferred)")
                            break
                            
                    # If we still can't find it, check the ambulance's current_call field
                    if ambulance.current_call is not None:
                        inferred_call_id = ambulance.current_call.get("call_id")
                        if inferred_call_id and inferred_call_id in call_details:
                            call_details[inferred_call_id]["arrived_at_hospital"] = True
                            hospital_deliveries.add(inferred_call_id)
                            if verbose:
                                print(f"Ambulance {amb_id} delivered patient from call {inferred_call_id} to hospital (from ambulance data)")
        
        # Track call timeouts
        elif event_type == "call_timeout":
            call_id = event_data.get("call_id")
            if call_id in call_details:
                call_details[call_id]["timed_out"] = True
                if verbose:
                    print(f"Call {call_id} timed out at {format_time(event_time)}")
        
        # Track ambulance transfer completions (patient successfully delivered)
        elif event_type == "amb_transfer_complete":
            amb_id = event_data.get("amb_id")
            call_id = event_data.get("call_id")
            
            if call_id is not None and call_id in call_details:
                # Mark as successful hospital transfer
                call_details[call_id]["transfer_complete"] = True
                
                # If for some reason we missed the hospital arrival event
                if not call_details[call_id]["arrived_at_hospital"]:
                    call_details[call_id]["arrived_at_hospital"] = True
                    hospital_deliveries.add(call_id)
                    
                if verbose:
                    print(f"Ambulance {amb_id} completed transfer for call {call_id} at hospital")
    
    # Check data frame for expected call count
    expected_call_count = len(calls_df)
    if call_count != expected_call_count:
        print(f"\nWARNING: Processed {call_count} calls but expected {expected_call_count} calls!")
        if verbose:
            print(f"Call IDs seen: {sorted(call_ids_seen)}")
            # Try to show which calls were missed
            call_ids_df = set(range(1, expected_call_count + 1))
            missing_ids = call_ids_df - call_ids_seen
            if missing_ids:
                print(f"Missing call IDs: {sorted(missing_ids)}")
    
    # Count successful calls (patients delivered to hospital)
    successful_calls = len(hospital_deliveries)
    missed_calls = call_count - successful_calls
        
    # Use the simulator's response times list directly
    response_times = sim.response_times

    # Calculate statistics
    avg_rt = np.mean(response_times) if response_times else 0
    p95_rt = np.percentile(response_times, 95) if response_times else 0
    
    # Calculate detailed statistics for reporting
    not_dispatched = sum(1 for details in call_details.values() if not details["dispatched"] and not details["timed_out"])
    timed_out = sum(1 for details in call_details.values() if details["timed_out"])
    dispatched_but_failed = sum(1 for call_id, details in call_details.items() 
                               if details["dispatched"] and not call_id in hospital_deliveries)
    
    print(f"\n===== {tag} Results =====")
    print(f"Total events processed: {event_count}")
    print(f"Calls in dataset      : {expected_call_count}")
    print(f"Calls processed       : {call_count}")
    print(f"Successful calls      : {successful_calls} (patient delivered to hospital)")
    print(f"Missed calls          : {missed_calls}")
    print(f"  - Not dispatched    : {not_dispatched}")
    print(f"  - Timed out         : {timed_out}")
    print(f"  - Dispatch failed   : {dispatched_but_failed}")
    
    # Verify that our totals add up
    total_accounted = successful_calls + not_dispatched + timed_out + dispatched_but_failed
    if total_accounted != call_count:
        print(f"WARNING: Call accounting mismatch! {total_accounted} accounted for vs {call_count} total")
    
    # Calculate final response rate
    response_rate = (successful_calls / call_count * 100) if call_count > 0 else 0
    print(f"Response rate         : {response_rate:.1f}%")
    
    if response_times:
        print(f"Avg response time: {avg_rt:.1f}s ({avg_rt/60:.1f} min)")
        print(f"Min response time: {min(response_times):.1f}s ({min(response_times)/60:.1f} min)")
        print(f"Max response time: {max(response_times):.1f}s ({max(response_times)/60:.1f} min)")
        print(f"95th percentile  : {p95_rt:.1f}s ({p95_rt/60:.1f} min)")
    else:
        print("No successful responses to calculate response times")
    
    # For RL policy, print statistics about invalid actions
    if isinstance(dispatch_policy, RLDispatchPolicy):
        print("\nRL Policy Statistics:")
        print(f"Invalid actions      : {dispatch_policy.invalid_action_count}")
        print(f"Could have handled   : {dispatch_policy.could_have_handled_count}")
    
    # Add diagnostic information about call status
    if verbose:
        print("\nDetailed call status:")
        failed_calls = []
        for call_id, details in call_details.items():
            if details["dispatched"] and not details.get("arrived_at_hospital", False):
                failed_calls.append(call_id)
                print(f"Call {call_id}: Dispatched but failed to reach hospital")
                print(f"  - Details: {details}")
        
        if failed_calls:
            print(f"\nFailed calls: {sorted(failed_calls)}")
            print(f"Total failed dispatched calls: {len(failed_calls)}")
    
    # Return results for comparison
    return {
        "method": tag,
        "calls": call_count,
        "dispatched": successful_calls,  # Only count successful deliveries
        "missed": missed_calls,
        "response_rate": response_rate,
        "avg_response_time": avg_rt,
        "p95_response_time": p95_rt,
    }

if __name__ == "__main__":
    main() 