import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.simulator.ambulance_env import AmbulanceEnv
from src.rl.rl_agent import QAgent, SarsaAgent

def load_data(data_dir):
    """
    Load required data for the simulator.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Loaded data
    """
    # Load graph
    graph_path = os.path.join(data_dir, 'graph.gexf')
    graph = nx.read_gexf(graph_path)
    
    # Load distance matrix
    distance_matrix_path = os.path.join(data_dir, 'distance_matrix.npy')
    distance_matrix = np.load(distance_matrix_path)
    
    # Load base locations
    base_locations_path = os.path.join(data_dir, 'base_locations.json')
    with open(base_locations_path, 'r') as f:
        import json
        base_locations = json.load(f)
    
    # Load hospital node
    hospital_node_path = os.path.join(data_dir, 'hospital_node.json')
    with open(hospital_node_path, 'r') as f:
        import json
        hospital_node = json.load(f)
    
    # Load call data
    call_data_path = os.path.join(data_dir, 'call_data.csv')
    
    return graph, distance_matrix, base_locations, hospital_node, call_data_path

def train_rl(
    graph=None,
    distance_matrix=None,
    base_locations=None,
    hospital_node=None,
    data_dir=None,
    num_episodes=10,
    algorithm='q-learning',
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    coverage_model='dsm',
    coverage_params=None,
    avg_speed=8.33,  # m/s (30 km/h)
    verbose=True
):
    """
    Train RL agent for ambulance dispatch.
    
    Args:
        graph: NetworkX graph of the road network
        distance_matrix: Pre-computed distance matrix
        base_locations: List of base location nodes
        hospital_node: Hospital node ID
        data_dir: Directory containing data files (optional)
        num_episodes: Number of episodes to train
        algorithm: RL algorithm ('q-learning' or 'sarsa')
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        coverage_model: Coverage model to use ('dsm', 'artm', 'ertm', etc.)
        coverage_params: Parameters for the coverage model
        avg_speed: Average speed in m/s (default 8.33 m/s = 30 km/h)
        verbose: Whether to print detailed logs
        
    Returns:
        agent: Trained agent
        env: Environment
        stats: Training statistics
    """
    # If data_dir is provided, load data from files
    if data_dir is not None:
        graph, distance_matrix, base_locations, hospital_node, call_data_path = load_data(data_dir)
    else:
        # Use provided data
        if graph is None or distance_matrix is None or base_locations is None or hospital_node is None:
            raise ValueError("Must provide either data_dir or all required data (graph, distance_matrix, base_locations, hospital_node)")
    
    # Create environment
    env = AmbulanceEnv(
        graph=graph,
        call_data_path=call_data_path,
        distance_matrix=distance_matrix,
        num_ambulances=len(base_locations),
        base_locations=base_locations,
        hospital_node=hospital_node,
        coverage_model=coverage_model,
        coverage_params=coverage_params,
        avg_speed=avg_speed,
        verbose=verbose
    )
    
    # Determine action size
    action_size = max(env.num_ambulances, len(env.base_locations))
    
    # Create agent
    if algorithm == 'q-learning':
        agent = QAgent(
            state_size=None,
            action_size=action_size,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )
    elif algorithm == 'sarsa':
        agent = SarsaAgent(
            state_size=None,
            action_size=action_size,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Statistics
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'mean_response_times': [],
        'calls_served': [],
        'calls_dropped': [],
        'q_values': []  # Track Q-values over time
    }
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset environment
        state = env.reset()
        done = False
        
        # Episode statistics
        episode_steps = 0
        episode_start_time = time.time()
        episode_reward = 0
        
        # Episode loop
        while not done:
            # Get available actions
            available_actions = env.get_available_actions()
            
            # Select action
            action = agent.select_action(state, available_actions)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Get available actions for next state
            available_next_actions = env.get_available_actions()
            
            # Update agent
            if algorithm == 'q-learning':
                agent.update(state, action, reward, next_state, done, available_next_actions)
            elif algorithm == 'sarsa':
                next_action = agent.select_action(next_state, available_next_actions)
                agent.update(state, action, reward, next_state, done, next_action)
            
            # Update state
            state = next_state
            episode_steps += 1
            
            if verbose:
                env.render()
        
        # End of episode
        agent.end_episode()
        episode_time = time.time() - episode_start_time
        
        # Calculate statistics
        mean_response_time = np.mean(env.simulator.response_times) if env.simulator.response_times else float('inf')
        calls_served = len(env.simulator.response_times)
        calls_dropped = len(env.simulator.dropped_calls)
        
        # Sample some Q-values for inspection
        sample_q_values = []
        for state_key in list(agent.q_table.keys())[:5]:  # Look at first 5 states
            if state_key in agent.q_table:
                sample_q_values.append((state_key, agent.q_table[state_key]))
        
        # Add episode statistics
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_steps)
        stats['mean_response_times'].append(mean_response_time)
        stats['calls_served'].append(calls_served)
        stats['calls_dropped'].append(calls_dropped)
        stats['q_values'].append(sample_q_values)
        
        # Print detailed episode summary
        print(f"\nEpisode {episode+1}/{num_episodes} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {episode_steps}")
        print(f"  Mean Response Time: {mean_response_time:.2f} min")
        print(f"  Calls Served: {calls_served}")
        print(f"  Calls Dropped: {calls_dropped}")
        print(f"  Time: {episode_time:.2f} sec")
        
        # Print Q-value samples every 2 episodes
        if (episode + 1) % 2 == 0:
            print("\nSample Q-values:")
            for state_key, q_values in sample_q_values:
                print(f"  State: {state_key}")
                print(f"  Q-values: {q_values}")
    
    # Save final agent
    agent.save(f"trained_{algorithm}_agent.json")
    
    # Plot learning curves
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot rewards
    ax1.plot(stats['episode_rewards'])
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True)
    
    # Plot mean response times
    ax2.plot(stats['mean_response_times'])
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Response Time (min)')
    ax2.set_title('Mean Response Time')
    ax2.grid(True)
    
    # Plot calls served vs dropped
    ax3.plot(stats['calls_served'], label='Served')
    ax3.plot(stats['calls_dropped'], label='Dropped')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Number of Calls')
    ax3.set_title('Calls Served vs Dropped')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"training_curves_{algorithm}.png")
    plt.close()
    
    print(f"\nTraining complete. Final statistics:")
    print(f"  Mean Response Time: {stats['mean_response_times'][-1]:.2f} min")
    print(f"  Calls Served: {stats['calls_served'][-1]}")
    print(f"  Calls Dropped: {stats['calls_dropped'][-1]}")
    
    return agent, env, stats

def evaluate_agent(agent, data_dir, num_episodes=10, verbose=False):
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained agent
        data_dir: Directory containing data files
        num_episodes: Number of episodes to evaluate
        verbose: Whether to print detailed logs
        
    Returns:
        stats: Evaluation statistics
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
        scenario_days=1,  # One day per episode
        verbose=verbose
    )
    
    # Set exploration to 0 for evaluation
    agent.epsilon = 0.0
    
    # Statistics
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'mean_response_times': [],
    }
    
    # Evaluation loop
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        # Reset environment
        state = env.reset()
        done = False
        
        # Episode statistics
        episode_steps = 0
        episode_reward = 0
        
        # Episode loop
        while not done:
            # Get available actions
            available_actions = env.get_available_actions()
            
            # Select action
            action = agent.select_action(state, available_actions)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update statistics
            episode_reward += reward
            
            # Update state
            state = next_state
            
            # Increment step counter
            episode_steps += 1
            
            # Optional: render environment
            if verbose:
                env.render()
        
        # Calculate mean response time
        mean_response_time = np.mean(env.simulator.response_times) if env.simulator.response_times else float('inf')
        
        # Add episode statistics
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_steps)
        stats['mean_response_times'].append(mean_response_time)
        
        # Print episode summary
        print(f"Episode {episode+1}/{num_episodes} - "
             f"Reward: {episode_reward:.2f}, "
             f"Steps: {episode_steps}, "
             f"Mean Response Time: {mean_response_time:.2f} min")
    
    # Calculate aggregate statistics
    mean_reward = np.mean(stats['episode_rewards'])
    mean_response_time = np.mean(stats['mean_response_times'])
    
    print(f"Evaluation complete.")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Mean response time: {mean_response_time:.2f} min")
    
    return stats

def compare_policies(data_dir, num_episodes=10, verbose=False):
    """
    Compare different policies for ambulance dispatch.
    
    Args:
        data_dir: Directory containing data files
        num_episodes: Number of episodes to evaluate
        verbose: Whether to print detailed logs
        
    Returns:
        results: Comparison results
    """
    # Load data
    graph, distance_matrix, travel_time_matrix, base_locations, hospital_node, call_data_path = load_data(data_dir)
    
    # Define policies to compare
    policies = {
        'q-learning': None,  # Will load trained agent
        'sarsa': None,       # Will load trained agent
        'nearest': None,     # Will use built-in policy
        'static': None,      # Will use built-in policy
    }
    
    # Load trained agents
    try:
        # Load Q-learning agent
        q_agent = QAgent(None, max(len(base_locations), len(base_locations)))
        q_agent.load("trained_q-learning_agent.json")
        q_agent.epsilon = 0.0  # No exploration during evaluation
        policies['q-learning'] = q_agent
        
        # Load SARSA agent
        sarsa_agent = SarsaAgent(None, max(len(base_locations), len(base_locations)))
        sarsa_agent.load("trained_sarsa_agent.json")
        sarsa_agent.epsilon = 0.0  # No exploration during evaluation
        policies['sarsa'] = sarsa_agent
    except Exception as e:
        print(f"Warning: Could not load trained agents. {e}")
        print("Skipping RL policies.")
    
    # Results dictionary
    results = {policy: {'mean_response_times': []} for policy in policies.keys()}
    
    # Evaluate each policy
    for policy_name in policies.keys():
        print(f"\nEvaluating policy: {policy_name}")
        
        # Create environment with appropriate policy
        if policy_name in ['q-learning', 'sarsa'] and policies[policy_name] is not None:
            # Use RL agent
            env = AmbulanceEnv(
                graph=graph,
                call_data_path=call_data_path,
                distance_matrix=distance_matrix,
                travel_time_matrix=travel_time_matrix,
                num_ambulances=len(base_locations),
                base_locations=base_locations,
                hospital_node=hospital_node,
                scenario_days=1,
                verbose=verbose
            )
            
            agent = policies[policy_name]
            
            # Evaluation loop
            for episode in tqdm(range(num_episodes), desc=f"Evaluating {policy_name}"):
                # Reset environment
                state = env.reset()
                done = False
                
                # Episode loop
                while not done:
                    # Get available actions
                    available_actions = env.get_available_actions()
                    
                    # Select action
                    action = agent.select_action(state, available_actions)
                    
                    # Take action
                    next_state, reward, done, info = env.step(action)
                    
                    # Update state
                    state = next_state
                
                # Calculate mean response time
                mean_response_time = np.mean(env.simulator.response_times) if env.simulator.response_times else float('inf')
                
                # Add to results
                results[policy_name]['mean_response_times'].append(mean_response_time)
        
        else:
            # Use built-in policy
            from src.simulator.simulator import AmbulanceSimulator
            
            # Create simulator with appropriate policy
            for episode in tqdm(range(num_episodes), desc=f"Evaluating {policy_name}"):
                simulator = AmbulanceSimulator(
                    graph=graph,
                    call_data_path=call_data_path,
                    distance_matrix=distance_matrix,
                    travel_time_matrix=travel_time_matrix,
                    num_ambulances=len(base_locations),
                    base_locations=base_locations,
                    hospital_node=hospital_node,
                    scenario_days=1,
                    dispatch_policy='nearest',  # Always use nearest for dispatch
                    relocation_policy=policy_name,  # Use specified policy for relocation
                    verbose=verbose
                )
                
                # Run simulation
                simulator.run_simulation()
                
                # Calculate mean response time
                mean_response_time = np.mean(simulator.response_times) if simulator.response_times else float('inf')
                
                # Add to results
                results[policy_name]['mean_response_times'].append(mean_response_time)
        
        # Calculate average statistics
        avg_response_time = np.mean(results[policy_name]['mean_response_times']) if results[policy_name]['mean_response_times'] else float('inf')
        
        print(f"{policy_name} - Average Response Time: {avg_response_time:.2f} min")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Prepare data for boxplot
    data = []
    labels = []
    for policy_name, policy_results in results.items():
        if policy_results['mean_response_times']:
            data.append(policy_results['mean_response_times'])
            labels.append(policy_name)
    
    plt.boxplot(data, labels=labels)
    plt.ylabel('Mean Response Time (min)')
    plt.title('Policy Comparison - Response Times')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("policy_comparison.png")
    plt.close()
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate RL agents for ambulance dispatch')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'compare'], default='train', 
                      help='Mode: train, evaluate, or compare policies')
    parser.add_argument('--algorithm', choices=['q-learning', 'sarsa'], default='q-learning',
                      help='RL algorithm to use')
    parser.add_argument('--data_dir', default='data', help='Directory containing data files')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_rl(
            data_dir=args.data_dir,
            num_episodes=args.episodes,
            algorithm=args.algorithm,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            verbose=args.verbose
        )
    
    elif args.mode == 'evaluate':
        # Load trained agent
        action_size = 100  # Placeholder, will be determined from environment
        
        if args.algorithm == 'q-learning':
            agent = QAgent(None, action_size)
        else:
            agent = SarsaAgent(None, action_size)
            
        try:
            agent.load(f"trained_{args.algorithm}_agent.json")
        except:
            print(f"Error: Could not load trained agent from trained_{args.algorithm}_agent.json")
            print("Please train an agent first.")
            sys.exit(1)
        
        evaluate_agent(
            agent=agent,
            data_dir=args.data_dir,
            num_episodes=args.episodes,
            verbose=args.verbose
        )
    
    elif args.mode == 'compare':
        compare_policies(
            data_dir=args.data_dir,
            num_episodes=args.episodes,
            verbose=args.verbose
        ) 