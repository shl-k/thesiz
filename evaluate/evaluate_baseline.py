"""
Evaluate the nearest-dispatch baseline (or any other policy)
by letting the simulator run from start to finish with manual_mode=False.
Then we print the final simulator stats.
"""

import sys, os
import pickle, json
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import NearestDispatchPolicy, StaticRelocationPolicy

def main():
    # 1) Load environment data
    G          = pickle.load(open("data/processed/princeton_graph.gpickle","rb"))
    calls      = pd.read_csv("data/processed/synthetic_calls.csv")
    path_cache = pickle.load(open("data/matrices/path_cache.pkl","rb"))
    node2idx   = {int(k): v for k,v in json.load(open("data/matrices/node_id_to_idx.json")).items()}
    idx2node   = {int(k): int(v) for k,v in json.load(open("data/matrices/idx_to_node_id.json")).items()}

    # 2) Build the baseline policies
    dispatch_policy   = NearestDispatchPolicy(G)
    relocation_policy = StaticRelocationPolicy(G, base_location=241)

    # 3) Create simulator, letting it auto-dispatch (manual_mode=False)
    simulator = AmbulanceSimulator(
        graph=G,
        call_data=calls,
        num_ambulances=6,
        base_location=241,
        hospital_node=1293,
        path_cache=path_cache,
        node_to_idx=node2idx,
        idx_to_node=idx2node,
        dispatch_policy=dispatch_policy,
        relocation_policy=relocation_policy,
        manual_mode=False,    # auto-dispatch using nearest
        verbose=True
    )

    # 4) Let the simulator run from start to finish
    simulator.run()

    # 5) Print final stats
    print("\n===== Baseline (Nearest Dispatch) Results =====")
    simulator._print_statistics()
    
    # 6) Check for pending calls at the end of simulation
    pending_calls_count = len(simulator.active_calls)
    if pending_calls_count > 0:
        print(f"\n⚠️ Warning: {pending_calls_count} calls still pending at end of simulation")
        print("Pending call IDs:", list(simulator.active_calls.keys()))
        
        # Print details about each pending call
        print("\nPending calls details:")
        for call_id, call_data in simulator.active_calls.items():
            call_time = simulator.format_time(call_data["time"])
            origin_node = call_data["origin_node"]
            print(f"  Call {call_id}: Arrived at {call_time}, Origin node: {origin_node}")
    else:
        print("\n✅ No pending calls at end of simulation")
    
    
if __name__ == "__main__":
    main()
