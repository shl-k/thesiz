from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.rl.AmbulanceEnv import AmbulanceEnv
import networkx as nx
import pandas as pd
import json
import os

# 1. Build toy graph
G = nx.DiGraph()
G.add_edge(101, 102, travel_time=10)
G.add_edge(102, 103, travel_time=10)
G.add_edge(103, 101, travel_time=10)  # Make sure it's strongly connected
G.add_edge(103, 999, travel_time=15)  # Connect to hospital
G.add_edge(999, 101, travel_time=15)  # Return from hospital

# 2. Generate node mappings
node_list = list(G.nodes)
node_id_to_idx = {node: idx for idx, node in enumerate(node_list)}
idx_to_node_id = {idx: node for idx, node in enumerate(node_list)}

# Convert to strings for JSON
node_id_to_idx_str = {str(k): v for k, v in node_id_to_idx.items()}
idx_to_node_id_str = {str(k): int(v) for k, v in idx_to_node_id.items()}

# Create directory if needed
os.makedirs('data/test', exist_ok=True)

# Save mappings
with open('data/test/node_id_to_idx.json', 'w') as f:
    json.dump(node_id_to_idx_str, f)

with open('data/test/idx_to_node_id.json', 'w') as f:
    json.dump(idx_to_node_id_str, f)

# 3. Generate synthetic call data
calls = []
# Create 10 calls at different times, all going from node 101 to node 103
for i in range(10):
    calls.append({
        'day': 1,
        'second_of_day': i * 3600,  # One call per hour
        'origin_node': 101,
        'destination_node': 103,
        'intensity': 1.0
    })

# Convert to DataFrame and save
calls_df = pd.DataFrame(calls)
calls_df.to_csv('data/test/tiny_calls.csv', index=False)
print(f"✅ Generated {len(calls)} synthetic calls")

# 4. Create environment
env = AmbulanceEnv(
    graph=G,
    call_data_path='data/test/tiny_calls.csv',
    num_ambulances=2,
    base_location=101,
    hospital_node=999,
    idx_to_node_path='data/test/idx_to_node_id.json',
    node_to_idx_path='data/test/node_id_to_idx.json',
    verbose=False
)

# 5. Wrap in vector env
vec_env = make_vec_env(lambda: env, n_envs=1)

# 6. Train PPO
print("🚀 Starting PPO training...")
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000)
print("✅ Done training.")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
model.save("models/ppo_debug")

# 7. Evaluate the trained model
print("\n🔍 Evaluating trained model...")

# Create a fresh environment for evaluation
eval_env = AmbulanceEnv(
    graph=G,
    call_data_path='data/test/tiny_calls.csv',
    num_ambulances=2,
    base_location=101,
    hospital_node=999,
    idx_to_node_path='data/test/idx_to_node_id.json',
    node_to_idx_path='data/test/node_id_to_idx.json',
    verbose=True  # Set to True to see detailed logs
)

# Run the model in the environment
obs, info = eval_env.reset()
done = False
total_reward = 0
step_count = 0

while not done:
    # Get action from model
    action, _states = model.predict(obs, deterministic=True)
    
    # Print action in human-readable format
    dispatch_choice = action[0]
    relocate_targets = action[1:3]  # For 2 ambulances
    
    print(f"\nStep {step_count}:")
    print(f"  Action: Dispatch ambulance {dispatch_choice if dispatch_choice < 2 else 'None'}")
    print(f"  Relocation targets: {relocate_targets}")
    
    # Execute action
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    # Print result
    print(f"  Reward: {reward}")
    print(f"  New observation: {obs}")
    
    total_reward += reward
    step_count += 1
    done = terminated or truncated

print(f"\n📊 Evaluation complete:")
print(f"  Total steps: {step_count}")
print(f"  Total reward: {total_reward}")
