import numpy as np
import random
import pickle
import os
import json
from collections import defaultdict

class QAgent:
    """
    Q-learning agent for ambulance dispatch.
    """
    
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize the agent.
        
        Args:
            state_size: Size of the state space (not used with dictionary-based Q-table)
            action_size: Size of the action space
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table as a defaultdict
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # For tracking learning progress
        self.rewards = []
        self.episode_rewards = []
    
    def state_to_key(self, state):
        """
        Convert state to a hashable key for the Q-table.
        
        Args:
            state: State from the environment
            
        Returns:
            key: Hashable key for the Q-table
        """
        # Unpack state components
        ambulance_states, call_info, time_of_day, decision_type = state
        
        # Create a hashable key
        key = (ambulance_states, call_info, time_of_day, decision_type)
        
        return key
    
    def select_action(self, state, available_actions=None):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            available_actions: List of available actions (optional)
            
        Returns:
            action: Selected action
        """
        # If no actions available, return None
        if not available_actions:
            return None
        
        # Convert state to key
        state_key = self.state_to_key(state)
        
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(available_actions)
        else:
            # Exploit: best action based on Q-values
            q_values = self.q_table[state_key]
            
            # Filter to available actions
            available_q = {}
            for action in available_actions:
                if action < len(q_values):
                    available_q[action] = q_values[action]
                else:
                    # Handle out-of-range actions (if any)
                    available_q[action] = 0.0
            
            if not available_q:
                return random.choice(available_actions)
                
            # Get action with highest Q-value
            return max(available_q, key=available_q.get)
    
    def update(self, state, action, reward, next_state, done, available_next_actions=None):
        """
        Update Q-value using Q-learning.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Whether episode is done
            available_next_actions: Available actions in next state
        """
        # If action is None, no update needed
        if action is None:
            return
        
        # Convert states to keys
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Ensure action is within bounds
        if action >= self.action_size:
            # Resize q_table for this state
            old_q = self.q_table[state_key]
            new_q = np.zeros(max(action + 1, self.action_size))
            new_q[:len(old_q)] = old_q
            self.q_table[state_key] = new_q
            
            # Update action_size
            self.action_size = max(action + 1, self.action_size)
        
        # Get current Q-value
        current_q = self.q_table[state_key][action]
        
        # Calculate target Q-value
        if done:
            # Terminal state
            target_q = reward
        else:
            # Non-terminal state
            if available_next_actions:
                # Get maximum Q-value for available actions
                max_next_q = float('-inf')
                for a in available_next_actions:
                    if a < len(self.q_table[next_state_key]):
                        q_val = self.q_table[next_state_key][a]
                        max_next_q = max(max_next_q, q_val)
                
                if max_next_q == float('-inf'):
                    max_next_q = 0
            else:
                # If no actions available, max Q-value is 0
                max_next_q = 0
            
            # Q-learning update
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state_key][action] = current_q + self.alpha * (target_q - current_q)
        
        # Track reward
        self.rewards.append(reward)
    
    def end_episode(self):
        """
        End an episode and calculate total reward.
        """
        episode_reward = sum(self.rewards)
        self.episode_rewards.append(episode_reward)
        self.rewards = []
        
        return episode_reward
    
    def save(self, filename):
        """
        Save the Q-table to a file.
        
        Args:
            filename: File to save to
        """
        # Convert defaultdict to regular dict for saving
        q_dict = {}
        for k, v in self.q_table.items():
            # Convert state key to string representation
            key_str = str(k)
            q_dict[key_str] = v.tolist()
        
        with open(filename, 'w') as f:
            json.dump(q_dict, f)
    
    def load(self, filename):
        """
        Load the Q-table from a file.
        
        Args:
            filename: File to load from
        """
        with open(filename, 'r') as f:
            q_dict = json.load(f)
        
        # Convert back to defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        
        # Determine max action size
        max_action = self.action_size
        
        for k_str, v in q_dict.items():
            # Evaluate the string key as a tuple
            try:
                key = eval(k_str)
                
                # Convert list to numpy array
                q_values = np.array(v)
                
                # Update max action size
                max_action = max(max_action, len(q_values))
                
                # Store in Q-table
                self.q_table[key] = q_values
            except Exception as e:
                print(f"Error loading key {k_str}: {e}")
        
        # Update action_size
        self.action_size = max_action

class SarsaAgent(QAgent):
    """
    SARSA agent for ambulance dispatch.
    Inherits from QAgent but uses SARSA update rule.
    """
    
    def update(self, state, action, reward, next_state, done, next_action=None):
        """
        Update Q-value using SARSA.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Whether episode is done
            next_action: Next action to be taken
        """
        # If action is None, no update needed
        if action is None:
            return
        
        # Convert states to keys
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Ensure action is within bounds
        if action >= self.action_size:
            # Resize q_table for this state
            old_q = self.q_table[state_key]
            new_q = np.zeros(max(action + 1, self.action_size))
            new_q[:len(old_q)] = old_q
            self.q_table[state_key] = new_q
            
            # Update action_size
            self.action_size = max(action + 1, self.action_size)
        
        # Get current Q-value
        current_q = self.q_table[state_key][action]
        
        # Calculate target Q-value
        if done:
            # Terminal state
            target_q = reward
        else:
            # Non-terminal state
            if next_action is not None and next_action < len(self.q_table[next_state_key]):
                # SARSA update - use actual next action
                next_q = self.q_table[next_state_key][next_action]
            else:
                # If no next action or out of bounds, use 0
                next_q = 0
            
            # SARSA update
            target_q = reward + self.gamma * next_q
        
        # Update Q-value
        self.q_table[state_key][action] = current_q + self.alpha * (target_q - current_q)
        
        # Track reward
        self.rewards.append(reward) 