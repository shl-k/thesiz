"""
Simple RL environment for ambulance dispatch.
This is a thin wrapper around the simulator that handles basic dispatch decisions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import sys

# Add project root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator, EventType

class SimpleDispatchEnv(gym.Env):
    """
    A simple RL environment for ambulance dispatch decisions.
    
    This environment:
    1. Presents the agent with observations when a call arrives
    2. Accepts actions to dispatch specific ambulances
    3. Provides rewards based on response times and successful dispatches
    4. Includes time of day in the observation space for time-based decision making
    
    Call timeouts are handled internally by the simulator and do not require
    agent decisions in this version of the environment.
    
    Observation space:
    - ambulances: Position coordinates, status, and busy time for each ambulance
    - call: Position coordinates, and priority of the current call
    - time_of_day: Normalized time of day (0-1 representing 0-24 hours)
    
    Action space:
    - 0 to num_ambulances-1: Dispatch the ambulance with that index
    - num_ambulances: Do not dispatch any ambulance
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        simulator: AmbulanceSimulator,
        node_to_lat_lon_file: str = 'data/matrices/node_to_lat_lon.json',
        verbose: bool = False,
        max_steps: int = 1000000,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the environment.
        
        Args:
            simulator: The ambulance simulator
            lat_lon_file: Path to JSON file with node_id -> [lat, lon] mapping
            verbose: Whether to print verbose output
            max_steps: Maximum number of steps per episode
            render_mode: How to render the environment (currently only 'human' is supported)
        """
        self.simulator = simulator
        self.verbose = verbose
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.simulator.manual_mode = True
        self.num_ambulances = len(simulator.ambulances)
        
        # Load node to lat/lon mapping from file & convert strings to ints
        with open(node_to_lat_lon_file, 'r') as f:
            mapping = json.load(f)  
            self.node_to_lat_lon = {int(k): v for k, v in mapping.items()}
        
        # Set up observation and action spaces
        self.observation_space = spaces.Dict({
            'ambulances': spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(self.num_ambulances, 4),  # lat, lon, status, busy_until
                dtype=np.float32
            ),
            'call': spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(3,),  # lat, lon, priority
                dtype=np.float32
            ),
            'time_of_day': spaces.Box(
                low=0, 
                high=1, 
                shape=(1,),  # normalized time of day (0-1 for 0-24 hours)
                dtype=np.float32
            ),
        })
        
        # Action space: 0 to num_ambulances-1 for dispatch, num_ambulances for no dispatch
        self.action_space = spaces.Discrete(self.num_ambulances + 1)

        self.current_call = None
        self.steps = 0
        self.done = False
        self.pending_calls = []
    
    def get_node_coords(self, node_id):
        """Get the lat/lon coordinates for a node."""
        coords = self.node_to_lat_lon.get(node_id) 
        if coords is None:
            raise ValueError(f"No coordinates found for node {node_id}")
        return coords
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility (optional)
            options: Additional options for reset (not used)
            
        Returns:
            Initial observation and empty info dict
        """
        # Reset simulator
        self.simulator.initialize()
        self.current_call = None
        self.steps = 0
        self.done = False
        self.pending_calls.clear()
        self.advance_until_action()
        
        obs = self._build_observation()
        
        return obs, {}
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
        Returns:
            observation: The new observation
            reward: The reward for the action
            done: Whether the episode is over
            info: Additional information
        """
        assert not self.done, "Gotta reset before the step"
        reward = 0.0
        
        if self.current_call is not None:
            call_id = self.current_call["call_id"]
            if action == self.num_ambulances: #no dispatch
                if any(amb.is_available() for amb in self.simulator.ambulances):
                    reward -= 30 #should have dispatched man!
            else: #dispatch
                ambulance = self.simulator.ambulances[action]
                if ambulance.is_available() and call_id in self.simulator.active_calls:
                    travel_time = self.simulator.path_cache[ambulance.location][self.current_call["origin_node"]]['travel_time']
                    travel_time_minutes = travel_time / 60.0
                    reward += 15 - travel_time_minutes
                    ambulance.dispatch_to_call(self.current_call, self.simulator.current_time)
                    self.simulator._push_event(
                        ambulance.busy_until, 
                        EventType.AMB_SCENE_ARRIVAL,
                        {"amb_id": ambulance.id, "call_id": call_id}
                    )
                else:
                    reward -= 15 #tried to dispatch unavailable ambulance
                self.current_call = None
                self.steps += 1
                if self.steps >= self.max_steps:
                    self.done = True

        self.advance_until_action(reward_accumulated=reward)
        return self._build_observation(), reward, self.done, False, {}
            
    def advance_until_action(self, reward_accumulated: float = 0.0):
        """
        Advance the simulator until the next action is needed.
        
        Args:
            reward_accumulated: The reward accumulated so far
        """
        while not self.done:
            event_time, event_type, event_data = self.simulator.step()
            if event_type is None:
                self.done = True
                break
            
            if event_type == EventType.CALL_ARRIVAL:
                if any(amb.is_available() for amb in self.simulator.ambulances):
                    self.current_call = self.simulator.active_calls[event_data["call_id"]]
                    break
                else:
                    self.pending_calls.append(event_data)
                    
            elif event_type in (EventType.AMB_TRANSFER_COMPLETE, EventType.AMB_RELOCATION_COMPLETE):
                # Ambulance is now available
                # first clean queueu of timed out calls
                self.pending_calls = [calls for calls in self.pending_calls if calls["call_id"] in self.simulator.active_calls]
                if self.pending_calls:
                    next_call = self.pending_calls.pop(0)
                    self.current_call = self.simulator.active_calls[next_call["call_id"]]
                    if self.current_call is not None:
                        return
                    
            #reached scene for somem call
            elif event_type == EventType.AMB_SCENE_ARRIVAL and self.current_call is None:
                reward_accumulated += 15 
                
            #reached hospital for some call
            elif event_type == EventType.AMB_HOSPITAL_ARRIVAL:
                reward_accumulated += 30

            elif event_type == EventType.CALL_TIMEOUT:
                reward_accumulated -= 30
                self.pending_calls = [calls for calls in self.pending_calls if calls["call_id"] != event_data["call_id"]]

    
    def render(self):
        if self.render_mode == "human":
            print(f"t={self.sim.format_time(self.sim.current_time)}  pending={len(self.queue)}  active={len(self.sim.active_calls)}")
            if self.current_call:
                print(f"→ action needed for call {self.current_call['call_id']}")

   
    def _build_observation(self):
        """
        Build the observation for the agent.
        
        Returns:
            Dictionary with ambulance and call information
        """
        # Ambulance information: [lat, lon, status, busy_until]
        ambulances = []
        for amb in self.simulator.ambulances:
            lat, lon = self.get_node_coords(amb.location)
            busy_until = max(0, amb.busy_until - self.simulator.current_time)  # Time until free
            ambulances.append([lat, lon, amb.status.value, busy_until])
        
        # Call information: [lat, lon, priority]
        if self.current_call:
            lat, lon = self.get_node_coords(self.current_call["origin_node"])
            priority = self.current_call.get("intensity", 1.0)
            call = [lat, lon, priority]
        else:
            call = [0, 0, 0]  # No call
        
        time_of_day = (self.simulator.current_time % 86400) / 86400  # Normalize to 0-1 for current day
        
        # Return observation dictionary with numpy arrays of the correct type
        return {
            'ambulances': np.array(ambulances, dtype=np.float32),
            'call': np.array(call, dtype=np.float32),
            'time_of_day': np.array([time_of_day], dtype=np.float32)
        } 