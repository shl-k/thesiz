import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.simulator.ambulance_env import AmbulanceEnv
from rl_agent import QAgent, SarsaAgent
from train_rl import load_data

def run_rl_simulation(data_dir, agent_file, algorithm='q-learning', verbose=False):
    """
    Run a simulation with a trained RL agent.
    
    Args:
        data_dir: Directory containing data files
        agent_file: Path to saved agent file
        algorithm: RL algorithm ('q-learning' or 'sarsa')
        verbose: Whether to print detailed logs
        
    Returns:
        env: Environment after simulation
        stats: Simulation statistics
    """
    # Load data
    graph, distance_matrix, travel_time_matrix, base_locations, hospital_node, call_data_path = load_data(data_dir)
    
    # Create environment
    env = AmbulanceEnv(
        graph=graph,
        call_data_path=call_data_path,
        distance_matrix=distance_matrix,
        travel_time_matrix=travel_time_matrix,
        num_ambulances=len(base_locations),
        base_locations=base_locations,
        hospital_node=hospital_node,
        scenario_days=7,  # Full week for evaluation
        verbose=verbose
    )
    
    # Create and load agent
    action_size = max(env.num_ambulances, len(env.base_locations))
    
    if algorithm == 'q-learning':
        agent = QAgent(None, action_size)
    elif algorithm == 'sarsa':
        agent = SarsaAgent(None, action_size)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Load agent
    agent.load(agent_file)
    
    # Set exploration to 0 for evaluation
    agent.epsilon = 0.0
    
    # Run simulation
    print("Running simulation with trained agent...")
    
    # Reset environment
    state = env.reset()
    done = False
    total_reward = 0
    
    # Simulation loop
    with tqdm(desc="Simulation") as pbar:
        while not done:
            # Get available actions
            available_actions = env.get_available_actions()
            
            # Select action
            action = agent.select_action(state, available_actions)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update statistics
            total_reward += reward
            
            # Update state
            state = next_state
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix(time=f"{env.simulator.current_time:.1f}",
                           calls=len(env.simulator.calls))
    
    # Calculate statistics
    mean_response_time = np.mean(env.simulator.response_times) if env.simulator.response_times else float('inf')
    
    # Print summary
    print(f"Simulation complete.")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Mean response time: {mean_response_time:.2f} min")
    
    # Calculate response time percentiles
    if env.simulator.response_times:
        response_times = np.array(env.simulator.response_times)
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            percentile_value = np.percentile(response_times, p)
            print(f"{p}th percentile response time: {percentile_value:.2f} min")
    
    # Plot response time histogram
    if env.simulator.response_times:
        plt.figure(figsize=(10, 6))
        plt.hist(env.simulator.response_times, bins=30)
        plt.xlabel('Response Time (min)')
        plt.ylabel('Frequency')
        plt.title('Response Time Distribution')
        plt.grid(True)
        plt.savefig("response_time_distribution.png")
        plt.close()
    
    return env, {'mean_response_time': mean_response_time}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulation with trained RL agent')
    parser.add_argument('--data_dir', default='data', help='Directory containing data files')
    parser.add_argument('--agent_file', required=True, help='Path to saved agent file')
    parser.add_argument('--algorithm', choices=['q-learning', 'sarsa'], default='q-learning',
                      help='RL algorithm used for the agent')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs')
    
    args = parser.parse_args()
    
    run_rl_simulation(
        data_dir=args.data_dir,
        agent_file=args.agent_file,
        algorithm=args.algorithm,
        verbose=args.verbose
    ) 