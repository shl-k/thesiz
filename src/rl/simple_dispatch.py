"""
Simple RL environment for ambulance dispatch.
This is a thin wrapper around the simulator that handles basic dispatch decisions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import sys

# Add project root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator, EventType
from src.simulator.ambulance import Ambulance, AmbulanceStatus

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
        lat_lon_file: str = 'data/matrices/node_to_lat_lon.json',
        verbose: bool = False,
        max_steps: int = 1000000,
        negative_reward_no_dispatch: float = -1000,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the environment.
        
        Args:
            simulator: The ambulance simulator
            lat_lon_file: Path to JSON file with node_id -> [lat, lon] mapping
            verbose: Whether to print verbose output
            max_steps: Maximum number of steps per episode
            negative_reward_no_dispatch: Reward for not dispatching an ambulance
            render_mode: How to render the environment (currently only 'human' is supported)
        """
        self.simulator = simulator
        self.verbose = verbose
        self.max_steps = max_steps
        self.negative_reward_no_dispatch = negative_reward_no_dispatch
        self.render_mode = render_mode
        
        # Set simulator to manual mode - disable automatic dispatch
        # This ensures the RL agent (not the simulator's policy) makes all dispatch decisions
        self.simulator.manual_mode = True
        
        # Load node to lat/lon mapping from file
        with open(lat_lon_file, 'r') as f:
            raw_mapping = json.load(f)  # This maps node_id -> [lat, lon]
            # Convert string keys to integers
            self.node_to_lat_lon = {int(k): v for k, v in raw_mapping.items()}
        
        # State tracking
        self.current_call = None
        self.steps = 0
        self.done = False
        
        # Number of ambulances from simulator
        self.num_ambulances = len(simulator.ambulances)
        
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
        # This is 0-indexed for safer array access
        self.action_space = spaces.Discrete(self.num_ambulances + 1)
    
    def get_node_coords(self, node_id):
        """Get the lat/lon coordinates for a node."""
        coords = self.node_to_lat_lon.get(node_id)  # Now node_id is already an integer
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
        
        # Reset state
        self.current_call = None
        self.steps = 0
        self.done = False
        
        # Get initial observation
        obs = self._build_observation()
        
        return obs, {}
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Which ambulance to dispatch:
                   0 to num_ambulances-1: ambulance index to dispatch
                   num_ambulances: no dispatch
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Initialize reward
        reward = 0.0
        
        # Check if action is "no dispatch" (num_ambulances) or a valid ambulance index
        dispatch_action = None if action == self.num_ambulances else action
        
        # Process simulator steps until we get to a call arrival or are done
        while True:
            # Check if we're done
            if self.done or self.steps >= self.max_steps:
                self.done = True
                return self._build_observation(), reward, True, False, {}
            
            # Process next event in simulator
            event_time, event_type, event_data = self.simulator.step()
            
            # Check if simulation is done
            if event_type is None:
                self.done = True
                return self._build_observation(), reward, True, False, {}
            
            # Handle call arrival
            if event_type == EventType.CALL_ARRIVAL:
                # Set current call
                call_id = event_data["call_id"]
                self.current_call = self.simulator.active_calls[call_id]
                
                # Handle dispatch action
                if dispatch_action is not None:
                    # Get the ambulance
                    ambulance = self.simulator.ambulances[dispatch_action]
                    
                    # Check if ambulance is available
                    if ambulance.is_available():
                        # Get call information
                        call_node = self.current_call["origin_node"]
                        
                        # Calculate travel time (for reward)
                        travel_time = self.simulator.path_cache[ambulance.location][call_node]['travel_time']
                        
                        # Manually dispatch the ambulance
                        dispatch_success = self._dispatch_ambulance(self.current_call, dispatch_action)
                        
                        if dispatch_success:
                            # Reward is negative travel time scaled by call priority, but less severe
                            call_priority = self.current_call.get("intensity", 1.0)  # Default priority is 1.0
                            # Scale travel time penalty by 0.5 to reduce its impact
                            travel_time_penalty = travel_time * call_priority * 0.5
                            # Add time-based bonus: exponentially higher rewards for faster response
                            time_bonus = 500 * np.exp(-travel_time / 600)  # 600s (10 min) as reference time
                            reward -= travel_time_penalty
                            reward += 500  # Increased base reward for dispatch (was 100)
                            reward += time_bonus
                            if self.verbose:
                                print(f"Dispatch reward breakdown:")
                                print(f"  Base reward: +500")
                                print(f"  Travel time penalty: -{travel_time_penalty:.1f}")
                                print(f"  Time bonus: +{time_bonus:.1f}")
                                print(f"  Total: {500 - travel_time_penalty + time_bonus:.1f}")
                        else:
                            # Penalty for failed dispatch - scaled by priority but not as harsh
                            call_priority = self.current_call.get("intensity", 1.0)
                            reward -= 300 * call_priority  # Reduced from 500 to 300
                            if self.verbose:
                                print(f"Failed dispatch penalty: -300 * {call_priority:.2f} = {-300 * call_priority:.1f}")
                    else:
                        # Penalty for trying to dispatch unavailable ambulance - same as failed dispatch
                        call_priority = self.current_call.get("intensity", 1.0)
                        reward -= 300 * call_priority  # Reduced from 500 to 300
                        if self.verbose:
                            print(f"Unavailable ambulance penalty: -300 * {call_priority:.2f} = {-300 * call_priority:.1f}")
                else:
                    # No ambulance dispatched - strong penalty to discourage this behavior
                    reward -= 800  # Reduced from 1000 to 800
                    if self.verbose:
                        print(f"No dispatch penalty: -800")
                
                # Increment step counter
                self.steps += 1
                
                # Return updated observation
                return self._build_observation(), reward, False, False, {}
            
            # For call timeouts, just continue processing events
            # The simulator will handle removing the call from active_calls
            elif event_type == EventType.CALL_TIMEOUT:
                if self.verbose:
                    call_id = event_data.get("call_id", "unknown")
                    print(f"Call {call_id} timed out, continuing to next event")
                continue
            
            # Handle ambulance scene arrival
            elif event_type == EventType.AMB_SCENE_ARRIVAL:
                call_id = event_data.get("call_id")
                if call_id == self.current_call["call_id"]:
                    # Track this call for potential hospital arrival reward
                    self.current_call["waiting_for_hospital"] = True
            
            # Handle hospital arrival - this is a successful delivery
            elif event_type == EventType.AMB_HOSPITAL_ARRIVAL:
                call_id = event_data.get("call_id")
                if call_id == self.current_call.get("call_id"):
                    # Large positive reward for successful delivery
                    reward += 5000  # Increased from 3000 to 5000
                    if self.verbose:
                        print(f"Successful delivery to hospital for call {call_id}: +5000")
            # We don't need to handle other event types as the simulator will process them
    
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human":
            # Time information
            total_seconds = int(self.simulator.current_time)
            hours = total_seconds // 3600 % 24
            minutes = (total_seconds % 3600) // 60
            time_str = f"{hours:02d}:{minutes:02d}"
            
            # Ambulance information
            amb_statuses = [amb.status.name for amb in self.simulator.ambulances]
            amb_locations = [amb.location for amb in self.simulator.ambulances]
            
            print(f"\n=== Environment State ===")
            print(f"Time: {time_str} (Day {1 + total_seconds // 86400}), Step: {self.steps}")
            print(f"Ambulances:")
            for i, (status, location) in enumerate(zip(amb_statuses, amb_locations)):
                coords = self.get_node_coords(location)
                print(f"  #{i}: {status} at node {int(location)} ({coords[0]:.4f}, {coords[1]:.4f})")
            
            if self.current_call:
                call_node = self.current_call['origin_node']
                call_coords = self.get_node_coords(call_node)
                print(f"Current call: ID {self.current_call['call_id']}")
                print(f"  Location: node {int(call_node)} ({call_coords[0]:.4f}, {call_coords[1]:.4f})")
                print(f"  Priority: {self.current_call.get('intensity', 1.0):.2f}")
            else:
                print("No current call")
            
            print(f"Statistics: {self.simulator.calls_responded} responded, {self.simulator.missed_calls} missed")
    
    def _dispatch_ambulance(self, call, amb_idx):
        """
        Manually dispatch an ambulance in the simulator.
        
        Args:
            call: The call information dictionary
            amb_idx: Index of the ambulance to dispatch
            
        Returns:
            Whether dispatch was successful
        """
        # Check if call is still active
        if call["call_id"] not in self.simulator.active_calls:
            return False
        
        # Get the ambulance
        ambulance = self.simulator.ambulances[amb_idx]
        
        # Check if ambulance is available
        if not ambulance.is_available():
            return False
        
        # Calculate travel time
        call_node = call["origin_node"]
        travel_time = self.simulator.path_cache[ambulance.location][call_node]['travel_time']
        
        # Dispatch the ambulance
        ambulance.dispatch_to_call(call, self.simulator.current_time)
        
        # Create a dispatch event (instant)
        self.simulator._push_event(
            self.simulator.current_time,
            EventType.AMB_DISPATCHED,
            {
                "amb_id": ambulance.id,
                "call_id": call["call_id"]
            }
        )
        
        # Schedule arrival event
        self.simulator._push_event(
            self.simulator.current_time + travel_time,
            EventType.AMB_SCENE_ARRIVAL,
            {
                "amb_id": ambulance.id,
                "call_id": call["call_id"]
            }
        )
        
        # Update statistics
        self.simulator.response_times.append(travel_time)
        self.simulator.call_response_times[call["call_id"]] = travel_time
        self.simulator.calls_responded += 1
        
        if self.verbose:
            print(f"\nDispatched ambulance {ambulance.id} to call {call['call_id']}")
            print(f"  Travel time: {travel_time:.1f}s ({travel_time/60:.1f} min)")
            print(f"  From node {ambulance.location} to {call_node}")
        
        return True
    
    def _build_observation(self):
        """
        Build the observation for the agent.
        
        Returns:
            Dictionary with ambulance and call information
        """
        # Ambulance information: [lat, lon, status, busy_until]
        ambulances = []
        for amb in self.simulator.ambulances:
            amb_pos = self.get_node_coords(amb.location)
            amb_status = amb.status.value  # Use enum value
            busy_until = amb.busy_until - self.simulator.current_time  # Time until free
            ambulances.append([amb_pos[0], amb_pos[1], amb_status, busy_until])
        
        # Call information: [lat, lon, priority]
        if self.current_call:
            call_pos = self.get_node_coords(self.current_call["origin_node"])
            priority = self.current_call.get("intensity", 1.0)  # Default priority is 1.0
            call = [call_pos[0], call_pos[1], priority]
        else:
            call = [0, 0, 0]  # No call
        
        # Time of day: normalized time of day (0-1 for 0-24 hours)
        seconds_in_day = 24 * 3600  # 24 hours * 60 minutes * 60 seconds
        time_of_day = (self.simulator.current_time % seconds_in_day) / seconds_in_day  # Normalize to 0-1 for current day
        
        # Return observation dictionary with numpy arrays of the correct type
        return {
            'ambulances': np.array(ambulances, dtype=np.float32),
            'call': np.array(call, dtype=np.float32),
            'time_of_day': np.array([time_of_day], dtype=np.float32)
        } 