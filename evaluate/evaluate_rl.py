"""
evaluate_rl.py – compare RL dispatch to heuristic baseline
Useful for comparing models vs. heuristics.
"""
import json, pickle, pathlib, numpy as np
from stable_baselines3 import PPO
import networkx as nx

from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import RLDispatchPolicy, NearestDispatchPolicy, StaticRelocationPolicy

# ------------------------------------------------------------------
# Paths / constants
# ------------------------------------------------------------------
GRAPH_PATH          = "data/processed/princeton_graph.gpickle"
CALLS_PATH          = "data/processed/synthetic_calls.csv"
NODE_TO_IDX_PATH    = "data/matrices/node_id_to_idx.json"
MODEL_PATH          = "models/ppo_simple_env_final"
PFARS_NODE          = 241
HOSPITAL_NODE       = 1293
NUM_AMBULANCES      = 3

# ------------------------------------------------------------------
# Load shared graph + cache
# ------------------------------------------------------------------
G: nx.DiGraph = pickle.load(open(GRAPH_PATH, "rb"))

# ------------------------------------------------------------------
# Helper to run one simulator instance and collect stats
# ------------------------------------------------------------------
def run_sim(dispatch_policy, tag: str):
    sim = AmbulanceSimulator(
        graph=G,
        call_data_path=CALLS_PATH,
        num_ambulances=NUM_AMBULANCES,
        base_location=PFARS_NODE,
        hospital_node=HOSPITAL_NODE,
        verbose=False,
        dispatch_policy=dispatch_policy,
        relocation_policy=StaticRelocationPolicy(G, PFARS_NODE),
    )
    while sim.step():
        pass

    print(f"\n=== {tag} ===")
    print(f"Calls responded: {sim.calls_responded}")
    print(f"Missed calls  : {sim.missed_calls}")
    if sim.response_times:
        print(f"Avg response‑time: {np.mean(sim.response_times):.1f} s")
        print(f"95th percentile  : {np.percentile(sim.response_times,95):.1f} s")

# ------------------------------------------------------------------
# 1) Baseline – nearest dispatch
# ------------------------------------------------------------------
run_sim(NearestDispatchPolicy(G), "Nearest‑Only")

# ------------------------------------------------------------------
# 2) RL‑based dispatch
# ------------------------------------------------------------------
model = PPO.load(MODEL_PATH, device="cpu")
rl_policy = RLDispatchPolicy(
    model=model,
    graph=G,
    node_to_idx_path=NODE_TO_IDX_PATH,
    num_ambulances=NUM_AMBULANCES,
    strict=False  # <-- disables fallback
)

run_sim(rl_policy, "RL (PPO)")
print(f"Invalid RL actions (strict mode): {rl_policy.invalid_action_count}")
print(f"Of those, could have been handled: {rl_policy.could_have_handled_count}")

# ------------------------------------------------------------------
# 3) Pick0
# ------------------------------------------------------------------
