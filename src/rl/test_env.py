import numpy as np
import networkx as nx

def test_ambulance_env(env, max_steps=50):
    """
    A simple test function that:
      1) Resets the environment
      2) Chooses random actions or a trivial policy
      3) Steps through the environment, printing each transition
    Useful to confirm the environment logic before training.
    """
    obs = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    while not done and step_count < max_steps:
        step_count += 1

        # --- 1) Build a random action or naive policy
        # 'dispatch' is in [0..num_ambulances], so pick random
        dispatch_action = np.random.randint(0, env.num_ambulances+1)

        # 'relocate' is multi-discrete: an array of length num_ambulances
        # For each ambulance, pick a valid node index from our mapping
        relocate_action = []
        # Get a list of valid node indices (as integers)
        node_indices = [int(idx) for idx in env.idx_to_node.keys()]
        # Make sure indices are all valid (within range of our mapping)
        valid_indices = [idx for idx in node_indices if idx < len(node_indices)]
        
        for _ in range(env.num_ambulances):
            # Pick a valid index
            node_idx = int(np.random.choice(valid_indices))
            relocate_action.append(node_idx)
        
        action = {
            "dispatch": dispatch_action,
            "relocate": np.array(relocate_action, dtype=np.int64)
        }

        # --- 2) Step the environment
        try:
            new_obs, reward, done, info = env.step(action)
            total_reward += reward

            # Print debug info
            print(f"Step {step_count}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward}")
            print(f"  Done? {done}")
            print(f"  Next Obs: {new_obs}")
            
            obs = new_obs
        except Exception as e:
            print(f"Error in step {step_count}: {str(e)}")
            print(f"Action was: {action}")
            print(f"Continuing to next step...")
            continue

    print(f"\nTest finished after {step_count} steps.")
    print(f"Total reward: {total_reward}")
