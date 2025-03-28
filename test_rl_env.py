import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import json
import time

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import AmbulanceEnv from the correct location
from src.simulator.ambulance_env import AmbulanceEnv

def create_minimal_test_data():
    """
    Create a minimal test dataset with 1 ambulance, 1 day, and 5-10 calls.
    """
    # Create temporary directory for test data
    os.makedirs('test_data', exist_ok=True)
    
    # Create a simple graph
    G = nx.grid_2d_graph(5, 5)  # 5x5 grid
    # Convert node labels to integers
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    # Save graph
    nx.write_gexf(G, 'test_data/graph.gexf')  # Using gexf format instead of gpickle
    
    # Create travel time matrix (uniform 5 minutes between adjacent nodes)
    n_nodes = len(G.nodes())
    travel_time_matrix = np.ones((n_nodes, n_nodes)) * np.inf
    for i in range(n_nodes):
        travel_time_matrix[i, i] = 0
        for j in list(G.neighbors(i)):
            travel_time_matrix[i, j] = 5  # 5 minute travel time
    
    # Use Floyd-Warshall to get shortest paths
    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if travel_time_matrix[i, j] > travel_time_matrix[i, k] + travel_time_matrix[k, j]:
                    travel_time_matrix[i, j] = travel_time_matrix[i, k] + travel_time_matrix[k, j]
    
    # Save travel time matrix
    np.save('test_data/travel_time_matrix.npy', travel_time_matrix)
    
    # Define base locations (1 base)
    base_locations = [12]  # Center of the grid
    with open('test_data/base_locations.json', 'w') as f:
        json.dump(base_locations, f)
    
    # Define hospital node
    hospital_node = 24  # Bottom right
    with open('test_data/hospital_node.json', 'w') as f:
        json.dump(hospital_node, f)
    
    # Create call data (3 calls in 1 day, all in the first hour)
    calls = []
    for i in range(3):
        timestamp = 5 + i * 15  # At 5, 20, and 35 minutes
        node = np.random.randint(0, n_nodes)
        calls.append({
            'timestamp': timestamp,
            'node': node,
            'day': 1,
            'priority': 1,  # Add priority field
            'service_time': 5,  # Add service time field (5 minutes)
            'transport_time': 10  # Add transport time field (10 minutes)
        })
    
    # Save call data
    calls_df = pd.DataFrame(calls)
    calls_df.to_csv('test_data/call_data.csv', index=False)
    
    # Debug the calls directly
    print(f"Created {len(calls_df)} calls for day 1")
    print(calls_df)
    
    return 'test_data'

def test_environment():
    """
    Test the RL environment with a minimal scenario.
    """
    # Create test data
    data_dir = create_minimal_test_data()
    
    # Debug: Print call data
    print("Debugging call data:")
    call_data = pd.read_csv(f'{data_dir}/call_data.csv')
    print(call_data.head())
    
    # Load data
    graph = nx.read_gexf(f'{data_dir}/graph.gexf')  # Updated to match the new format
    travel_time_matrix = np.load(f'{data_dir}/travel_time_matrix.npy')
    with open(f'{data_dir}/base_locations.json', 'r') as f:
        base_locations = json.load(f)
    with open(f'{data_dir}/hospital_node.json', 'r') as f:
        hospital_node = json.load(f)
    
    # Create environment
    env = AmbulanceEnv(
        graph=graph,
        call_data_path=f'{data_dir}/call_data.csv',
        distance_matrix=travel_time_matrix,  # Using travel time as proxy for distance
        travel_time_matrix=travel_time_matrix,
        num_ambulances=1,
        base_locations=base_locations,
        hospital_node=hospital_node,
        scenario_days=1,
        verbose=True  # Enable verbose mode to see detailed logs
    )
    
    # Reset environment to day 1
    print("Resetting environment...")
    state = env.reset()
    print(f"Initial state: {state}")
    
    # Debug: Print ambulance status after reset
    print("Ambulance status after reset:")
    for amb in env.simulator.ambulances:
        print(f"  ID: {amb.id}, Location: {amb.location}, Status: {amb.status}, Available: {amb.is_available(env.simulator.current_time)}")
    
    # Loop through events
    print("\nStepping through environment...")
    step_count = 0
    total_reward = 0
    done = False
    
    while not done:
        step_count += 1
        print(f"\nStep {step_count}")
        print(f"Decision type: {env.decision_type}")
        
        # Get available actions
        available_actions = env.get_available_actions()
        print(f"Available actions: {available_actions}")
        
        # Debug ambulance status
        print("Current ambulance status:")
        for amb in env.simulator.ambulances:
            print(f"  ID: {amb.id}, Location: {amb.location}, Status: {amb.status}, Available: {amb.is_available(env.simulator.current_time)}")
        
        # Choose an action
        if available_actions:
            action = available_actions[0]  # Always take first available action for simplicity
            print(f"Taking action: {action}")
        else:
            action = None
            print("No actions available")
        
        # Take a step
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Print details
        print(f"Reward: {reward}")
        print(f"Total reward: {total_reward}")
        print(f"Current time: {env.simulator.current_time}")
        
        # Print ambulance states
        print("Ambulance states:")
        for amb in env.simulator.ambulances:
            print(f"  ID: {amb.id}, Location: {amb.location}, Status: {amb.status}")
        
        # Print call being processed (if any)
        if env.current_call is not None:
            print(f"Current call: Node {env.current_call['node']}")
        
        # Give some time to read output
        time.sleep(0.5)  # Reduced from 1 second to 0.5 seconds
    
    # Print final statistics
    print("\nSimulation complete")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward}")
    
    # Calculate response times
    if env.simulator.response_times:
        mean_response_time = np.mean(env.simulator.response_times)
        print(f"Mean response time: {mean_response_time:.2f} minutes")
        print(f"Number of calls served: {len(env.simulator.response_times)}")
    else:
        print("No calls were served")
    
    # Clean up test data
    # import shutil
    # shutil.rmtree(data_dir)
    
    return env

if __name__ == "__main__":
    test_environment() 