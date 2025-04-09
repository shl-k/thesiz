from envs.simple_dispatch import SimpleDispatchEnv
from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import NearestDispatchPolicy
import pickle, json
import pandas as pd

# ==== Setup your simulator ====
G = pickle.load(open("data/processed/princeton_graph.gpickle", "rb"))
calls = pd.read_csv("data/processed/synthetic_calls.csv")
path_cache = pickle.load(open("data/matrices/path_cache.pkl", "rb"))
node2idx = {int(k): v for k, v in json.load(open("data/matrices/node_id_to_idx.json")).items()}
idx2node = {int(k): int(v) for k, v in json.load(open("data/matrices/idx_to_node_id.json")).items()}

sim = AmbulanceSimulator(
    graph=G,
    call_data=calls,
    num_ambulances=3,
    base_location=241,
    hospital_node=1293,
    path_cache=path_cache,
    node_to_idx=node2idx,
    idx_to_node=idx2node,
    dispatch_policy=NearestDispatchPolicy(G),
    manual_mode=True,
    verbose=True
)

env = SimpleDispatchEnv(sim)

# ==== Now test the env ====
obs = env.reset()
print("Initial observation:", obs)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Action: {action} → Reward: {reward}")
