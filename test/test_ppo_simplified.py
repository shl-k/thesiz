from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.rl.AmbulanceEnv import AmbulanceEnv
import networkx as nx
import pandas as pd
import json
import os
import numpy as np

# Create directories if needed
os.makedirs('data/test', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("🚗 Building simple road network...")
# Build a very simple graph - just 3 locations in a triangle plus hospital
G = nx.DiGraph()
G.add_edge(101, 102, travel_time=10)
G.add_edge(102, 103, travel_time=10)
G.add_edge(103, 101, travel_time=10)
G.add_edge(103, 999, travel_time=15)  # Route to hospital
G.add_edge(999, 101, travel_time=15)  # Return from hospital

# Generate node mappings
node_list = list(G.nodes)
node_id_to_idx = {str(node): idx for idx, node in enumerate(node_list)}
idx_to_node_id = {str(idx): int(node) for idx, node in enumerate(node_list)}

# Save mappings
with open('data/test/node_id_to_idx.json', 'w') as f:
    json.dump(node_id_to_idx, f)
with open('data/test/idx_to_node_id.json', 'w') as f:
    json.dump(idx_to_node_id, f)

# Generate just 2 emergency calls, clearly separated in time
calls = [
    {
        'day': 1,
        'second_of_day': 3600,  # 1:00
        'origin_node': 101,
        'destination_node': 103,
        'intensity': 1.0
    },
    {
        'day': 1,
        'second_of_day': 7200,  # 2:00
        'origin_node': 102,
        'destination_node': 103,
        'intensity': 1.0
    }
]

# Save calls
calls_df = pd.DataFrame(calls)
calls_df.to_csv('data/test/simple_calls.csv', index=False)
print(f"✅ Created simple environment with 2 calls")

# Function to create the environment
def make_env():
    return AmbulanceEnv(
        graph=G,
        call_data_path='data/test/simple_calls.csv',
        num_ambulances=2,
        base_location=101,
        hospital_node=999,
        idx_to_node_path='data/test/idx_to_node_id.json',
        node_to_idx_path='data/test/node_id_to_idx.json',
        verbose=False,
        relocation_interval=3600  # Set to 1 hour for simplicity
    )

# Train for a bit longer to improve policy
print("🚀 Training PPO agent...")
vec_env = make_vec_env(make_env, n_envs=1)
model = PPO("MlpPolicy", vec_env, verbose=0)
model.learn(total_timesteps=5000)
model.save("models/simple_ppo")
print("✅ Training complete")

# Run evaluation with detailed logging
print("\n🔍 Evaluating trained model on the 2 emergency calls...")
env = make_env()
obs, info = env.reset()

# Track important stats
event_count = 0
call_responses = []

# Helper function to format time
def format_time(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    return f"{hours:02d}:{minutes:02d}"

# Tracking ambulance status over time
amb_history = []

# Process all events
while not env.done:
    # Get the next event before stepping
    next_event = env.event_queue[0] if env.event_queue else None
    if next_event:
        event_time, _, event_type, event_data = next_event
    
    # Get model's action
    action, _states = model.predict(obs, deterministic=True)
    
    # Execute step
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Save status of ambulances at this timestep
    amb_status = []
    for i, amb in enumerate(env.ambulances):
        amb_status.append({
            'id': i,
            'status': amb.status.name,
            'location': amb.location,
            'time': format_time(env.current_time)
        })
    amb_history.append(amb_status)
    
    # Print important events
    if event_type == "call_arrival":
        event_count += 1
        dispatch_choice = action[0]
        relocated_nodes = action[1:3]
        
        print(f"\n📞 Call {event_count} at {format_time(event_time)}:")
        print(f"   From node {event_data['origin_node']} to {event_data['destination_node']}")
        
        if dispatch_choice < 2:
            print(f"   ✅ Model dispatched ambulance {dispatch_choice}")
            used_amb = env.ambulances[dispatch_choice]
            call_responses.append({
                'call': event_count,
                'time': format_time(event_time),
                'ambulance': dispatch_choice,
                'response_status': 'Dispatched'
            })
        else:
            print(f"   ❌ Model chose NOT to dispatch any ambulance")
            call_responses.append({
                'call': event_count,
                'time': format_time(event_time),
                'ambulance': None,
                'response_status': 'Ignored'
            })
        
        # Show ambulance status at this time
        print(f"   Ambulance status at this time:")
        for i, amb in enumerate(env.ambulances):
            status = amb.status.name
            loc = amb.location
            print(f"   🚑 Ambulance {i}: {status} at node {loc}")
    
    elif event_type == "amb_state_change" and reward != 0:
        amb_id = event_data['amb_id']
        amb = env.ambulances[amb_id]
        print(f"\n🔄 Ambulance {amb_id} state change at {format_time(event_time)}:")
        print(f"   New status: {amb.status.name} at node {amb.location}")
        print(f"   Reward: {reward}")
    
    # Update for next iteration
    obs = next_obs

# Print final summary
print("\n📋 Simulation Summary:")
print(f"   Total events processed: {len(amb_history)}")
print(f"   Emergency calls responded to: {sum(1 for c in call_responses if c['ambulance'] is not None)}/{len(call_responses)}")

# Print ambulance journey (simplified)
print("\n🚑 Ambulance Journey:")
last_status = [None, None]  # Track last status for each ambulance

for i, step in enumerate(amb_history):
    for amb in step:
        amb_id = amb['id']
        status = amb['status']
        location = amb['location']
        time = amb['time']
        
        # Only print when status changes
        if status != last_status[amb_id]:
            print(f"   Amb {amb_id} @ {time}: {status} at node {location}")
            last_status[amb_id] = status

print("\n✅ Evaluation complete") 