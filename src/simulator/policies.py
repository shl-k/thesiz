import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
import random
import heapq
import json

# uses minimimum time to scene to dispatch the nearest available ambulance to a call.
class NearestDispatchPolicy:
    """
    Policy that dispatches the nearest available ambulance to a call.
    """
    def __init__(self, distance_matrix, avg_speed=8.33):
        self.distance_matrix = distance_matrix
        self.avg_speed = avg_speed
    
    def get_travel_time(self, from_node: int, to_node: int) -> float:
        """
        Calculate travel time between two nodes.
        
        Args:
            from_node: Starting node ID
            to_node: Ending node ID
            
        Returns:
            Travel time in seconds
        """
        distance = self.distance_matrix[from_node][to_node]
        return distance / self.avg_speed  # Convert to seconds
    
    def select_ambulance(self, call_node: int, available_ambulances: List[Dict], **kwargs) -> Optional[int]:
        """
        Select the nearest available ambulance to dispatch to a call.
        
        Args:
            call_node: Node where the call is located
            available_ambulances: List of available ambulances
            
        Returns:
            ID of the selected ambulance, or None if no ambulance is available
        """
        if not available_ambulances:
            return None
            
        # Find the nearest ambulance
        nearest_amb = None
        min_time = float('inf')
        
        for amb in available_ambulances:
            time_to_scene = self.get_travel_time(amb['location'], call_node)
            if time_to_scene < min_time:
                min_time = time_to_scene
                nearest_amb = amb
        
        return nearest_amb['id'] if nearest_amb else None

# uses static base assignments to relocate ambulances to their assigned ("home") base when idle.  always PFARS HQ
class StaticRelocationPolicy:
    """
    Policy that relocates ambulances back to PFARS HQ when idle.
    """
    def __init__(self, pfars_hq_node: int):
        self.pfars_hq_node = pfars_hq_node
    
    def relocate_ambulances(self, idle_ambulances: List[Dict], busy_ambulances: List[Dict], **kwargs) -> Dict[int, int]:
        """
        Relocate idle ambulances back to PFARS HQ.
        
        Args:
            idle_ambulances: List of idle ambulances
            busy_ambulances: List of busy ambulances
            
        Returns:
            Dictionary mapping ambulance IDs to PFARS HQ location
        """
        relocations = {}
        
        for amb in idle_ambulances:
            # Only relocate if not already at PFARS HQ
            if amb['location'] != self.pfars_hq_node:
                relocations[amb['id']] = self.pfars_hq_node
        
        return relocations

# uses coverage model to relocate ambulances to maximize coverage | need predictive model for demand
# could use SPO+ for below if we had the relevant data to hydrate the model
class CoverageBasedRelocationPolicy:
    """
    Policy that relocates ambulances based on coverage calculations.
    """
    def __init__(self, distance_matrix, base_locations, coverage_model_func, coverage_params=None, avg_speed=8.33):
        self.distance_matrix = distance_matrix
        self.base_locations = base_locations
        self.coverage_model_func = coverage_model_func
        self.coverage_params = coverage_params or {}
        self.avg_speed = avg_speed
    
    def get_travel_time(self, from_node: int, to_node: int) -> float:
        """
        Calculate travel time between two nodes.
        
        Args:
            from_node: Starting node ID
            to_node: Ending node ID
            
        Returns:
            Travel time in seconds
        """
        distance = self.distance_matrix[from_node][to_node]
        return distance / self.avg_speed  # Convert to seconds
    
    def get_relocation_decision(self, idle_amb_ids: List[int], demand_nodes: List[int], 
                              busy_ambulances: List[Dict] = None, current_locations: Dict[int, int] = None) -> Dict[int, int]:
        """
        Get relocation decisions for idle ambulances.
        
        Args:
            idle_amb_ids: List of IDs of idle ambulances
            demand_nodes: List of nodes with demand
            busy_ambulances: List of busy ambulances (optional)
            current_locations: Dictionary mapping ambulance IDs to their current locations
            
        Returns:
            Dictionary mapping ambulance IDs to their new locations
        """
        return self.coverage_model_func(
            self.distance_matrix,
            demand_nodes,
            idle_amb_ids,
            busy_ambulances,
            current_locations,
            self.base_locations,
            **self.coverage_params
        )


class ADPPolicy:
    """
    Approximate Dynamic Programming policy for ambulance dispatch and relocation.
    This implementation uses reinforcement learning techniques (Q-learning) to learn
    an optimal policy for ambulance operations.
    """
    def __init__(self, simulator, value_function: Dict[str, float] = None, 
                alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
                buffer_size: int = 10000):
        """
        Initialize the ADP policy with reinforcement learning parameters.
        
        Args:
            simulator: Reference to the simulator object
            value_function: Dictionary mapping state representations to values (Q-values)
            alpha: Learning rate for TD updates
            gamma: Discount factor for future rewards
            epsilon: Exploration rate for epsilon-greedy policy
            buffer_size: Size of experience replay buffer
        """
        self.simulator = simulator
        self.travel_time_matrix = simulator.travel_time_matrix
        self.base_locations = simulator.base_locations
        self.value_function = value_function or {}  # Dictionary mapping state-action pairs to Q-values
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Experience replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = []
        
        # State tracking for learning
        self.last_state = None
        self.last_action = None
        
    def select_ambulance(self, call_node: int, available_ambulances: List[Dict], 
                       busy_ambulances: List[Dict] = None, **kwargs) -> Optional[int]:
        """
        Select an ambulance to dispatch using the value function and epsilon-greedy policy.
        
        Args:
            call_node: Node where the call is located
            available_ambulances: List of available ambulances
            busy_ambulances: List of busy ambulances
            
        Returns:
            ID of the selected ambulance, or None if no ambulance is available
        """
        if not available_ambulances:
            return None
            
        busy_ambulances = busy_ambulances or []
        current_state = self.encode_state(available_ambulances, busy_ambulances, call_node)
        
        # Epsilon-greedy policy: With probability epsilon, choose a random ambulance
        if random.random() < self.epsilon:
            selected_amb = random.choice(available_ambulances)
            self.last_state = current_state
            self.last_action = ('dispatch', selected_amb['id'], call_node)
            return selected_amb['id']
        
        # Calculate expected value for dispatching each ambulance
        best_amb = None
        best_value = float('-inf')
        
        for amb in available_ambulances:
            # Calculate immediate cost (travel time)
            immediate_cost = -self.travel_time_matrix[amb['location']][call_node]
            
            # Create state-action key
            state_action = f"{current_state}|dispatch:{amb['id']}:{call_node}"
            
            # Get Q-value
            q_value = self.value_function.get(state_action, 0)
            
            # Total value is immediate cost plus Q-value
            total_value = immediate_cost + q_value
            
            if total_value > best_value:
                best_value = total_value
                best_amb = amb
        
        if best_amb:
            self.last_state = current_state
            self.last_action = ('dispatch', best_amb['id'], call_node)
            
        return best_amb['id'] if best_amb else None
    
    def relocate_ambulances(self, idle_ambulances: List[Dict], busy_ambulances: List[Dict], 
                          **kwargs) -> Dict[int, int]:
        """
        Relocate idle ambulances to maximize expected future value using epsilon-greedy policy.
        
        Args:
            idle_ambulances: List of idle ambulances
            busy_ambulances: List of busy ambulances
            
        Returns:
            Dictionary mapping ambulance IDs to their new base locations
        """
        if not idle_ambulances:
            return {}
        
        current_state = self.encode_state(idle_ambulances, busy_ambulances)
        
        # Epsilon-greedy policy: With probability epsilon, choose random relocations
        if random.random() < self.epsilon:
            relocations = {}
            # Randomly relocate up to 1 ambulance
            if idle_ambulances:
                amb = random.choice(idle_ambulances)
                new_base = random.choice(self.base_locations)
                # Only relocate if not already at this base
                if amb['location'] != new_base:
                    relocations[amb['id']] = new_base
                    self.last_state = current_state
                    self.last_action = ('relocate', amb['id'], new_base)
            return relocations
        
        # Try all possible combinations of relocations (simplification: just one ambulance)
        best_relocations = {}
        best_value = float('-inf')
        
        for amb in idle_ambulances:
            for base in self.base_locations:
                # Skip if ambulance is already at this base
                if amb['location'] == base:
                    continue
                
                # Calculate immediate cost (travel time)
                immediate_cost = -self.travel_time_matrix[amb['location']][base]
                
                # Create state-action key
                state_action = f"{current_state}|relocate:{amb['id']}:{base}"
                
                # Get Q-value
                q_value = self.value_function.get(state_action, 0)
                
                # Total value is immediate cost plus Q-value
                total_value = immediate_cost + q_value
                
                if total_value > best_value:
                    best_value = total_value
                    best_relocations = {amb['id']: base}
        
        if best_relocations:
            amb_id = list(best_relocations.keys())[0]
            new_base = best_relocations[amb_id]
            self.last_state = current_state
            self.last_action = ('relocate', amb_id, new_base)
            
        return best_relocations
    
    def encode_state(self, available_ambulances: List[Dict], busy_ambulances: List[Dict] = None, 
                   call_node: int = None) -> str:
        """
        Encode the system state as a string for the value function lookup.
        
        Args:
            available_ambulances: List of available ambulances
            busy_ambulances: List of busy ambulances
            call_node: Node where the call is located (if any)
            
        Returns:
            String representation of the state
        """
        busy_ambulances = busy_ambulances or []
        
        # Create sorted lists of ambulance locations
        available_locs = sorted([amb['location'] for amb in available_ambulances])
        busy_locs = sorted([amb['location'] for amb in busy_ambulances])
        
        # Encode as string
        state = f"A:{','.join(map(str, available_locs))}_B:{','.join(map(str, busy_locs))}"
        if call_node is not None:
            state += f"_C:{call_node}"
        
        return state
    
    def update_value_function(self, state: str, action: Tuple, reward: float, next_state: str, 
                             next_actions: List[Tuple] = None, terminal: bool = False):
        """
        Update the value function (Q-values) using Q-learning.
        
        Args:
            state: Current state
            action: Taken action (tuple of action_type, amb_id, destination)
            reward: Immediate reward
            next_state: Next state
            next_actions: List of possible actions in next state
            terminal: Whether this is a terminal state
        """
        # Encode state-action pair
        action_type, amb_id, destination = action
        state_action = f"{state}|{action_type}:{amb_id}:{destination}"
        
        # Get current Q-value
        current_q = self.value_function.get(state_action, 0)
        
        if terminal:
            # For terminal states, there's no future value
            target = reward
        else:
            # Calculate maximum Q-value for next state-action pairs
            max_next_q = float('-inf')
            
            if next_actions:
                for next_action in next_actions:
                    next_action_type, next_amb_id, next_destination = next_action
                    next_state_action = f"{next_state}|{next_action_type}:{next_amb_id}:{next_destination}"
                    next_q = self.value_function.get(next_state_action, 0)
                    max_next_q = max(max_next_q, next_q)
            
            # If no valid next actions or max_next_q is still -inf
            if max_next_q == float('-inf'):
                max_next_q = 0
                
            # Calculate target Q-value using Q-learning update rule
            target = reward + self.gamma * max_next_q
        
        # Update Q-value using TD learning
        new_q = current_q + self.alpha * (target - current_q)
        
        # Update value function
        self.value_function[state_action] = new_q
        
    def observe_transition(self, reward: float, new_state: str, terminal: bool = False, 
                         possible_actions: List[Tuple] = None):
        """
        Observe a transition and add it to the replay buffer.
        If we have previous state/action, update Q-values.
        
        Args:
            reward: Reward received
            new_state: New state after transition
            terminal: Whether the episode has ended
            possible_actions: List of possible actions in new state
        """
        if self.last_state is not None and self.last_action is not None:
            # Add experience to replay buffer
            experience = (self.last_state, self.last_action, reward, new_state, possible_actions, terminal)
            self.replay_buffer.append(experience)
            
            # Keep buffer at fixed size
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)
            
            # Update Q-value for this transition
            self.update_value_function(self.last_state, self.last_action, reward, new_state, 
                                    possible_actions, terminal)
            
            # Perform experience replay learning
            self.experience_replay()
            
        # Reset last state/action if terminal
        if terminal:
            self.last_state = None
            self.last_action = None
        else:
            self.last_state = new_state
    
    def experience_replay(self, batch_size: int = 32):
        """
        Perform experience replay learning from the buffer.
        
        Args:
            batch_size: Number of experiences to sample for learning
        """
        if len(self.replay_buffer) < batch_size:
            return
            
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        
        # Update Q-values for sampled experiences
        for state, action, reward, next_state, next_actions, terminal in batch:
            self.update_value_function(state, action, reward, next_state, next_actions, terminal)
    
    def save_value_function(self, filepath: str):
        """
        Save the learned value function to a file.
        
        Args:
            filepath: Path to save the value function
        """
        with open(filepath, 'w') as f:
            json.dump(self.value_function, f)
    
    def load_value_function(self, filepath: str):
        """
        Load a saved value function from a file.
        
        Args:
            filepath: Path to load the value function from
        """
        with open(filepath, 'r') as f:
            self.value_function = json.load(f)
    
    def get_possible_actions(self, state: str, available_ambulances: List[Dict], busy_ambulances: List[Dict], 
                          call_node: int = None) -> List[Tuple]:
        """
        Get all possible actions for a given state.
        
        Args:
            state: Current state
            available_ambulances: List of available ambulances
            busy_ambulances: List of busy ambulances
            call_node: Node where the call is located (if any)
            
        Returns:
            List of possible actions
        """
        possible_actions = []
        
        # If there's a call, generate dispatch actions
        if call_node is not None:
            for amb in available_ambulances:
                possible_actions.append(('dispatch', amb['id'], call_node))
        
        # Generate relocation actions
        for amb in available_ambulances:
            for base in self.base_locations:
                if amb['location'] != base:
                    possible_actions.append(('relocate', amb['id'], base))
        
        return possible_actions 