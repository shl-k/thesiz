import gymnasium as gym
from gymnasium import spaces
import numpy as np
import heapq
import os
import sys
import pandas as pd
import json
import networkx as nx
import time
call_data_path = "data/processed/synthetic_calls.csv"
idx_to_node_path = "data/matrices/idx_to_node_id.json"
node_list_path = "data/matrices/node_id_to_idx.json"

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.simulator.ambulance import Ambulance, AmbulanceStatus

class EventType:
    CALL_ARRIVAL = "call_arrival"
    AMB_STATE_CHANGE = "amb_state_change"
    PERIODIC_RELOC = "periodic_relocation"

class AmbulanceEnv(gym.Env):
    """
    An event-based environment where the agent can:
      - 'dispatch': choose which ambulance to send for a new call
      - 'relocate': for each ambulance, pick a node index to relocate if IDLE
    The environment uses node_id <-> index mappings from external JSONs to interpret these actions.
    """

    def __init__(
        self,
        graph: nx.Graph,
        call_data_path: str,
        num_ambulances: int,
        base_location: int,
        hospital_node: int,
        idx_to_node_path: str,     # e.g. "idx_to_node_id.json"
        node_to_idx_path: str,     # e.g. "node_id_to_idx.json"
        verbose: bool = False,
        relocation_interval: int = 1800  # 30 min
    ):
        super().__init__()
        self.graph = graph
        self.num_ambulances = num_ambulances
        self.base_location = base_location
        self.hospital_node = hospital_node
        self.verbose = verbose
        self.relocation_interval = relocation_interval

        # 1) Load call data
        self.calls_data = self._load_call_data(call_data_path)

        # 2) Load JSON-based node index mappings
        with open(idx_to_node_path, 'r') as f:
            self.idx_to_node = json.load(f)  # { "0": 123, "1": 456, ... }
        with open(node_to_idx_path, 'r') as f:
            self.node_to_idx = json.load(f)  # { "123": 0, "456": 1, ... }
        # We'll interpret these strings as ints
        # i.e. self.idx_to_node[str(i)] => node ID
        # or self.node_to_idx[str(node_id)] => i

        # The total number of nodes
        self.num_nodes = len(self.idx_to_node)

        # Define the action space as a MultiDiscrete:
        # - First value: dispatch choice [0..num_ambulances], where num_ambulances means "don't dispatch"
        # - Next num_ambulances values: relocate targets [0..num_nodes-1] for each ambulance
        # This is a flattened version of our previous Dict action space
        action_space_nums = [self.num_ambulances + 1] + [self.num_nodes] * self.num_ambulances
        self.action_space = spaces.MultiDiscrete(action_space_nums)

        # 4) Observation space:
        # For each ambulance: (location_index, status.value, busy_until)
        # plus current_time => 3*num_ambulances + 1
        obs_dim = 3*self.num_ambulances + 1
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(obs_dim,), dtype=np.float32
        )
        
        # Counter for unique event priorities
        self.event_counter = 0

    def _load_call_data(self, path: str):
        df = pd.read_csv(path)
        df = df.sort_values(["day", "second_of_day"])
        calls = []
        for _, row in df.iterrows():
            t = (row["day"] - 1) * 24 * 3600 + row["second_of_day"]
            calls.append({
                "time": t,
                "origin_node": row["origin_node"],
                "destination_node": row["destination_node"],
                "intensity": row["intensity"]
            })
        return calls

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (not used)
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        # Optional: set random seed
        if seed is not None:
            np.random.seed(seed)
            
        self.current_time = 0
        self.done = False
        self.event_queue = []
        self.event_counter = 0  # Reset counter on environment reset

        # Create ambulances
        self.ambulances = []
        for i in range(self.num_ambulances):
            amb = Ambulance(i, self.base_location, self.graph)
            self.ambulances.append(amb)

        # Push all call arrival events
        for call in self.calls_data:
            self._push_event(call["time"], EventType.CALL_ARRIVAL, call)

        # Schedule periodic relocation events
        if self.calls_data:
            max_time = max(c["time"] for c in self.calls_data)
        else:
            max_time = 24 * 3600
        t = self.relocation_interval
        while t < max_time:
            self._push_event(t, EventType.PERIODIC_RELOC, {})
            t += self.relocation_interval

        obs = self._get_obs()
        info = {}  # Additional info dict required by Gymnasium
        return obs, info
        
    def _push_event(self, time, event_type, data):
        """Helper method to push events with a unique counter for stable ordering"""
        heapq.heappush(self.event_queue, (time, self.event_counter, event_type, data))
        self.event_counter += 1

    def step(self, action):
        """
        One RL step = pop the earliest event and handle it.
        
        Args:
            action: MultiDiscrete action vector where:
                   - action[0] is which ambulance to dispatch (or num_ambulances for none)
                   - action[1:num_ambulances+1] are the relocation targets for each ambulance
        
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: True if episode is done due to natural termination
            truncated: True if episode is done due to truncation (e.g., max steps)
            info: Additional information
        """
        if self.done or not self.event_queue:
            obs = self._get_obs()
            return obs, 0.0, True, False, {}

        # Pop earliest event
        event_time, _, event_type, data = heapq.heappop(self.event_queue)
        self.current_time = event_time

        # Unpack the MultiDiscrete action
        dispatch_choice = action[0]
        relocate_array = action[1:self.num_ambulances+1]

        reward = 0.0

        if event_type == EventType.CALL_ARRIVAL:
            # If dispatch_choice < num_ambulances, we dispatch that ambulance
            if dispatch_choice < self.num_ambulances:
                reward += self._handle_call_arrival(dispatch_choice, data)
            else:
                # No dispatch => missed call
                reward += -1000.0

            # Now apply relocations to IDLE ambulances
            self._apply_reloc_choices(relocate_array)

        elif event_type == EventType.AMB_STATE_CHANGE:
            # An ambulance finished traveling/hospital
            amb_id = data["amb_id"]
            reward += self._handle_ambulance_state_change(amb_id)

            # Apply relocations after this event
            self._apply_reloc_choices(relocate_array)

        elif event_type == EventType.PERIODIC_RELOC:
            # It's the 30-min check => relocate any IDLE ambulance according to agent's choices
            self._apply_reloc_choices(relocate_array)

            # Schedule next periodic relocation
            next_t = self.current_time + self.relocation_interval
            self._push_event(next_t, EventType.PERIODIC_RELOC, {})

        # Done if no more events remain
        terminated = (len(self.event_queue) == 0)
        self.done = terminated
        
        # Truncated would be True if we ended the episode early (e.g., max steps)
        truncated = False
        
        obs = self._get_obs()
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _handle_call_arrival(self, amb_id, call):
        """
        Dispatch chosen ambulance -> negative reward for idle->scene travel.
        If ambulance is busy, treat it as a missed call.
        """
        amb = self.ambulances[amb_id]
        if amb.status not in (AmbulanceStatus.IDLE, AmbulanceStatus.RELOCATING):
            return -1000.0  # can't dispatch => missed

        start_t = self.current_time
        amb.dispatch_to_call(call, start_t)
        travel_time = amb.busy_until - start_t
        r = -float(travel_time)

        # schedule next event
        self._push_event(amb.busy_until, EventType.AMB_STATE_CHANGE, {"amb_id": amb_id})
        
        if self.verbose:
            print(f"[t={start_t}] Dispatch amb {amb_id} => call at {call['origin_node']} (TT={travel_time:.1f}s)")
        return r

    def _handle_ambulance_state_change(self, amb_id):
        """
        Ambulance reached 'busy_until' => call the next state method from ambulance.py.
        e.g. DISPATCHED->ON_SCENE, ON_SCENE->TRANSPORT, etc.
        Negative reward for scene->hospital travel.
        If it finishes hospital -> IDLE (no auto-relocate).
        """
        amb = self.ambulances[amb_id]
        current_t = self.current_time
        r = 0.0

        if amb.status == AmbulanceStatus.DISPATCHED:
            # arrive_at_scene
            amb.arrive_at_scene(current_t)
            self._push_event(amb.busy_until, EventType.AMB_STATE_CHANGE, {"amb_id": amb_id})

        elif amb.status == AmbulanceStatus.ON_SCENE:
            # begin_transport => negative reward for scene->hospital
            amb.begin_transport(self.hospital_node, current_t)
            travel_time = amb.busy_until - current_t
            r += -float(travel_time)
            self._push_event(amb.busy_until, EventType.AMB_STATE_CHANGE, {"amb_id": amb_id})

        elif amb.status == AmbulanceStatus.TRANSPORT:
            # arrive_at_hospital => random hospital time
            amb.arrive_at_hospital(current_t)
            self._push_event(amb.busy_until, EventType.AMB_STATE_CHANGE, {"amb_id": amb_id})

        elif amb.status == AmbulanceStatus.HOSPITAL:
            # done => becomes IDLE. The agent can relocate if it wants
            amb.status = AmbulanceStatus.IDLE
            amb.current_call = None

        elif amb.status == AmbulanceStatus.RELOCATING:
            # done relocating => set IDLE
            amb.location = amb.destination
            amb.status = AmbulanceStatus.IDLE

        # If still busy, schedule the next finishing event
        if amb.busy_until > current_t and amb.status not in (AmbulanceStatus.IDLE,):
            self._push_event(amb.busy_until, EventType.AMB_STATE_CHANGE, {"amb_id": amb_id})

        return r

    def _apply_reloc_choices(self, relocate_array):
        """
        For each ambulance, if IDLE, relocate to the node chosen by the agent's action.
        'relocate_array' is a list/array of length num_ambulances,
        each is an integer [0..(num_nodes-1)] referencing idx_to_node[str(index)].
        """
        for i, amb in enumerate(self.ambulances):
            if amb.status == AmbulanceStatus.IDLE:
                node_idx = relocate_array[i]
                # Convert to string to index into the JSON dict
                node_idx_str = str(node_idx)
                if node_idx_str not in self.idx_to_node:
                    # Invalid index => skip
                    continue
                target_node_id = int(self.idx_to_node[node_idx_str])
                # If already at that location, do nothing
                if amb.location != target_node_id:
                    amb.relocate(target_node_id, self.current_time)
                    # schedule finishing event
                    self._push_event(amb.busy_until, EventType.AMB_STATE_CHANGE, {"amb_id": i})

    def _get_obs(self):
        """
        Flatten:
          - location index of each ambulance, status.value, busy_until
          - plus current_time
        We convert the actual node_id to its index via node_to_idx
        """
        obs = []
        for amb in self.ambulances:
            node_id = amb.location
            # Convert node_id->str, then lookup node_to_idx[str(node_id)]
            node_id_str = str(node_id)
            if node_id_str in self.node_to_idx:
                loc_idx = float(self.node_to_idx[node_id_str])
            else:
                # fallback if node not in dict (unlikely)
                loc_idx = -1.0

            obs.append(loc_idx)                  # location index
            obs.append(float(amb.status.value))  # status as int
            obs.append(float(amb.busy_until))    # next finishing time

        obs.append(float(self.current_time))
        return np.array(obs, dtype=np.float32)
