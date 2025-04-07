import json
from typing import List, Dict, Optional

import networkx as nx
import numpy as np
from stable_baselines3 import PPO

# Global setting for verbose output from policies
verbose_global = False

# ---------------------------------------------------------------------------
#  Baseline Policies
# ---------------------------------------------------------------------------

class NearestDispatchPolicy:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def _travel_time(self, from_node: int, to_node: int) -> float:
        return nx.shortest_path_length(
            self.graph,
            source=int(from_node),
            target=int(to_node),
            weight="travel_time"
        )

    def select_ambulance(self, call_node: int, available_ambulances: List[Dict], all_ambulances: List[Dict] = None, current_time: float = None) -> Optional[int]:
        if not available_ambulances:
            return None
        return min(
            available_ambulances,
            key=lambda amb: self._travel_time(amb["location"], call_node)
        )["id"]


class StaticRelocationPolicy:
    def __init__(self, graph: nx.Graph, base_location: int):
        self.graph = graph
        self.base_location = base_location

    def select_relocation_target(self, ambulance: Dict, **kwargs) -> int:
        """Always relocate to the base location."""
        return self.base_location
        
    def relocate_ambulances(self, available_ambulances: List[Dict], busy_ambulances: List[Dict]) -> Dict[int, int]:
        """
        Process ambulances that need relocation.
        Returns a dictionary mapping ambulance IDs to target location nodes.
        """
        relocations = {}
        for amb in available_ambulances:
            # Only relocate if not already at base
            if amb["location"] != self.base_location:
                relocations[amb["id"]] = self.base_location
                
        return relocations

# ---------------------------------------------------------------------------
#  RL Policy Wrapper
# ---------------------------------------------------------------------------
# for dispatch, we need 
class RLDispatchPolicy:
    def __init__(self, model: PPO, graph: nx.Graph, node_to_idx_path: str, num_ambulances: int, strict: bool = False):
        self.model = model
        self.graph = graph
        with open(node_to_idx_path, "r") as f:
            self.node_to_idx = {int(k): v for k, v in json.load(f).items()}
        self.num_ambulances = num_ambulances
        self.strict = strict
        self.invalid_action_count = 0
        self.fallback = NearestDispatchPolicy(self.graph)
        self.could_have_handled_count = 0
        self.last_action_was_no_dispatch = False  # Track when the policy chooses not to dispatch
        
        # Load lat/lon mapping directly
        with open("data/matrices/lat_lon_mapping.json", "r") as f:
            self.lat_lon_mapping = {int(k): v for k, v in json.load(f).items()}

    def _build_observation(self, all_ambulances, current_time: float, current_call: Dict = None) -> dict:
        """Build observation dictionary for the trained model."""
        # Create ambulance observations: position, status, busy_until
        ambulances_obs = []
        for amb in all_ambulances:
            # Get location coordinates using lat_lon_mapping
            amb_node = int(amb["location"])
            lat, lon = self.lat_lon_mapping.get(amb_node, (0.0, 0.0))  # Default to (0,0) if not found
            
            # Get status
            status = amb["status"]
            
            # Calculate remaining busy time
            if "busy_until" in amb:
                busy_time = max(0, amb["busy_until"] - current_time)
            else:
                busy_time = 0.0
                
            ambulances_obs.append([lat, lon, status, busy_time])
            
        # Call information using lat_lon_mapping
        call_node_int = int(current_call["origin_node"]) if current_call else 0
        lat, lon = self.lat_lon_mapping.get(call_node_int, (0.0, 0.0))  # Default to (0,0) if not found
        
        # Get call priority from the call data
        call_priority = current_call.get('intensity', 1.0) if current_call else 1.0
        
        # Normalized time of day (0-1)
        time_of_day = (current_time % (24 * 3600)) / (24 * 3600)
        
        # Return properly structured observation dictionary
        return {
            'ambulances': np.array(ambulances_obs, dtype=np.float32),
            'call': np.array([lat, lon, call_priority], dtype=np.float32),
            'time_of_day': np.array([time_of_day], dtype=np.float32)
        }

    def select_ambulance(self, available_ambulances: List[Dict], all_ambulances: List[Dict] = None, current_time: float = None, current_call: Dict = None) -> Optional[int]:
        """Select an ambulance to dispatch to a call using the trained RL model."""
        # Reset the no-dispatch tracking
        self.last_action_was_no_dispatch = False
        
        if not available_ambulances:
            if verbose_global:
                print("[RLDispatchPolicy] No available ambulances to dispatch")
            return None
            
        if all_ambulances is None or current_time is None:
            if verbose_global:
                print("[RLDispatchPolicy] Missing context parameters")
            return self._fallback(available_ambulances, current_call)

        # Debug info - available ambulance IDs
        available_ids = [amb["id"] for amb in available_ambulances]
        if verbose_global:
            print(f"[RLDispatchPolicy] Available ambulances: {available_ids}")
        
        # Build observation for the model
        try:
            obs = self._build_observation(all_ambulances, current_time, current_call)
            
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Handle both vector and scalar actions
            action = int(action) if isinstance(action, (np.ndarray, list)) else action
            
            if verbose_global:
                print(f"[RLDispatchPolicy] Model selected action: {action}")
            
            # Check if action is valid (within bounds)
            if action >= self.num_ambulances:
                if verbose_global:
                    print(f"[RLDispatchPolicy] Action {action} is 'no dispatch'")
                self.invalid_action_count += 1
                self.last_action_was_no_dispatch = True  # Track that we chose not to dispatch
                return None if self.strict else self._fallback(available_ambulances, current_call)
            
            # Check if selected ambulance is available
            selected_id = action
            if selected_id not in available_ids:
                if verbose_global:
                    print(f"[RLDispatchPolicy] Invalid action - ambulance {selected_id} is not available")
                    status = next((amb['status'] for amb in all_ambulances if amb['id'] == selected_id), 'unknown')
                    print(f"[RLDispatchPolicy] Status: {status}")
                if available_ambulances:
                    self.could_have_handled_count += 1
                self.invalid_action_count += 1
                return None if self.strict else self._fallback(available_ambulances, current_call)
            
            if verbose_global:
                print(f"[RLDispatchPolicy] Successfully selected ambulance {selected_id}")
            return selected_id
            
        except Exception as e:
            import traceback
            if verbose_global:
                print(f"[RLDispatchPolicy] Error predicting action: {str(e)}")
                traceback.print_exc()
            return self._fallback(available_ambulances, current_call)

    def _fallback(self, available_ambulances: List[Dict], current_call: Dict = None) -> Optional[int]:
        """Fallback to nearest available ambulance if RL model fails."""
        if not current_call:
            return None
        call_node = current_call["origin_node"]
        return self.fallback.select_ambulance(call_node, available_ambulances)
