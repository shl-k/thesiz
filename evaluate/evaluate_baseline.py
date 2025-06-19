CALLS_FILE = "data/processed/synthetic_calls.csv"
NUM_AMBULANCES = 1
EVAL_OUTPUT_PATH = f"results/{NUM_AMBULANCES}amb/baseline_nearest_eval.json"


import sys, os, pickle, json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import NearestDispatchPolicy, StaticRelocationPolicy


def save_eval_results(sim, model_name, out_path, calls_file=None, extra_metrics=None):
    response_times = np.array(sim.response_times)

    results = {
        "model_name": model_name,
        "num_ambulances": NUM_AMBULANCES,
        "total_calls": sim.total_calls,
        "calls_responded": sim.calls_responded,
        "calls_timed_out": sim.timed_out_calls,
        "calls_missed": sim.missed_calls,
        "calls_rescued": sim.calls_rescued,
        "pct_responded": sim.calls_responded / sim.total_calls,
        "avg_response_time_min": float(np.mean(response_times)) / 60 if len(response_times) else None,
        "response_time_std_min": float(np.std(response_times)) / 60 if len(response_times) else None,
        "min_response_time": float(np.min(response_times)) / 60 if len(response_times) else None,
        "max_response_time": float(np.max(response_times)) / 60 if len(response_times) else None,
        "timeout_causes": sim.timeout_causes,
        "dispatch_counts": sim.ambulance_call_counts,
        "calls_file": calls_file,
    }

    if extra_metrics:
        results.update(extra_metrics)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Evaluation results saved to {out_path}")


def main():
    # Load environment data
    G = pickle.load(open("data/processed/princeton_graph.gpickle","rb"))
    calls = pd.read_csv(CALLS_FILE)
    path_cache = pickle.load(open("data/matrices/path_cache.pkl","rb"))
    node2idx = {int(k): v for k,v in json.load(open("data/matrices/node_id_to_idx.json")).items()}
    idx2node = {int(k): int(v) for k,v in json.load(open("data/matrices/idx_to_node_id.json")).items()}

    dispatch_policy = NearestDispatchPolicy(G)
    relocation_policy = StaticRelocationPolicy(G, base_location=241)

    simulator = AmbulanceSimulator(
        graph=G,
        call_data=calls,
        num_ambulances=NUM_AMBULANCES,
        base_location=241,
        hospital_node=1293,
        path_cache=path_cache,
        node_to_idx=node2idx,
        idx_to_node=idx2node,
        dispatch_policy=dispatch_policy,
        relocation_policy=relocation_policy,
        manual_mode=False,
        verbose=True
    )

    simulator.run()

    print("\n===== Baseline (Nearest Dispatch) Results =====")
    simulator._print_statistics()

    save_eval_results(simulator, model_name="baseline_nearest", out_path=EVAL_OUTPUT_PATH, calls_file=CALLS_FILE)

    # Calculate travel times for each call that was responded to
    travel_times = []
    for call_id, call_info in simulator.call_status.items():
        if isinstance(call_info, dict) and call_info.get("status") == "rescued":
            amb = simulator.ambulances[call_info["ambulance_id"]]
            travel_sec = simulator.path_cache[call_info["dispatch_location"]][call_info["origin_node"]]["travel_time"]
            travel_times.append(travel_sec / 60.0)  # Convert to minutes

    print("\n📊 Response Time Details:")
    print(f"Total calls responded: {simulator.calls_responded}")
    if travel_times:
        avg_rt = np.mean(travel_times)
        std_rt = np.std(travel_times)
        min_rt = np.min(travel_times)
        max_rt = np.max(travel_times)
        print(f"⏱️  Average response time: {avg_rt:.2f} min")
        print(f"📉 Std dev response time: {std_rt:.2f} min")
        print(f"⬇️  Min response time: {min_rt:.2f} min")
        print(f"⬆️  Max response time: {max_rt:.2f} min")
    else:
        print("No response times recorded")


if __name__ == "__main__":
    main()
