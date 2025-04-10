"""
DispatchRelocEnv – merges queue-based dispatch + relocation into one environment,
with a single Discrete action space and partial rewards for scene arrival/hospital arrival.

Action space: 
  0..(N-1) => dispatch ambulance i
  N        => no-dispatch
  (N+1)..(N+K-1) => pick cluster for relocation
We also allow partial rewards for scene arrival/hospital arrival
between steps (accumulated in self.partial_reward).

To remove 2D shape warnings and negative-lon errors:
 - We flatten the observation into a 1D Box with shape=(obs_dim,).
 - The Box range is [-9999, 9999].
"""
# Fixed DispatchRelocEnv with pending call queue + improved no-dispatch logic

import os, sys, numpy as np, pandas as pd
from typing import Optional, List, Dict
import gymnasium as gym
from gymnasium import spaces

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator, EventType
from src.simulator.ambulance import AmbulanceStatus
from src.utils.clustering import cluster_nodes, get_best_node_in_cluster, load_node_mappings

class DispatchRelocEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, simulator: AmbulanceSimulator, n_clusters: int = 10, node_to_lat_lon_file: str = "data/matrices/node_to_lat_lon.json", verbose: bool = False, max_steps: int = 100000):
        super().__init__()
        self.sim = simulator
        self.sim.manual_mode = True
        self.verbose = verbose
        self.max_steps = max_steps

        self.n_amb = simulator.num_ambulances
        self.n_clusters = n_clusters
        
        # Action space:
        # - Dispatch ambulance i (0..(N-1))
        # - No-dispatch (N)
        # - Pick cluster for relocation (N+1)..(N+K-1)
        self.action_space = spaces.Discrete(self.n_amb + 1 + self.n_clusters)

        self.pending_calls: List[dict] = []
        self.current_call: Optional[dict] = None
        self.idle_amb_id: Optional[int] = None
        self.dispatch_attempts = set()


        self.steps = 0
        self.done = False
        self.partial_reward = 0.0

        self.node_to_lat_lon = load_node_mappings(node_to_lat_lon_file)[0]

        self.node_to_cluster, self.cluster_centers = cluster_nodes(
            self.node_to_lat_lon,
            self.sim.call_data,
            n_clusters=self.n_clusters
        )

        # Observation space:
        # - Ambulance status (lat, lon, status, busy_sec) for each ambulance
        # - Current call (lat, lon, priority)
        # - Idle ambulance indicator (1D array of size n_amb)
        # - Time of day (tod)
        obs_dim = self.n_amb * 4 + 3 + self.n_amb + 1
        self.observation_space = spaces.Box(low=-9999, high=9999, shape=(obs_dim,), dtype=np.float32)

        self.response_times = []

    def reset(self, seed=None, options=None):
        self.sim.initialize()
        self.pending_calls.clear()
        self.current_call = None
        self.idle_amb_id = None
        self.steps = 0
        self.done = False
        self.response_times = []
        self.partial_reward = 0.0

        self._advance_until_actionable()
        return self._build_obs(), {}

    def step(self, action: int):
        if self.done:
            return self._build_obs(), 0.0, True, False, {}

        reward = self.partial_reward
        self.partial_reward = 0.0

        if self.current_call is not None:
            cid = self.current_call["call_id"]
            if action < self.n_amb:
                amb = self.sim.ambulances[action]
                if amb.is_available() and cid in self.sim.active_calls:
                    travel_sec = self.sim.path_cache[amb.location][self.current_call["origin_node"]]["travel_time"]
                    travel_min = travel_sec / 60.0
                    reward += 15.0 - travel_min
                    self.dispatch_attempts.add(cid)
                    self.response_times.append(travel_min)
                    amb.dispatch_to_call(self.current_call, self.sim.current_time)
                    self.sim._push_event(amb.busy_until, EventType.AMB_SCENE_ARRIVAL, {"amb_id": amb.id, "call_id": cid})
                else:
                    reward -= 15.0 # dispatch unavailable ambulance
            elif action == self.n_amb:
                if any(a.is_available() for a in self.sim.ambulances):
                    reward -= 30  # incorrect no-dispatch
                else:
                    reward += 5   # correct no-dispatch — no ambulance available
            else:
                reward -= 50

            if cid in self.sim.active_calls:
                self.pending_calls.append(self.current_call)
            self.current_call = None

        elif self.idle_amb_id is not None:
            amb_id = self.idle_amb_id
            ambulance = self.sim.ambulances[amb_id]
            pre_avg = self._avg_response_time(ambulance.location)

            if action < self.n_amb:
                reward -= 50
            elif action == self.n_amb:
                reward -= 10
            else:
                cluster_idx = action - (self.n_amb + 1)
                if 0 <= cluster_idx < self.n_clusters:
                    node = get_best_node_in_cluster(
                        cluster_idx,
                        self.node_to_cluster,
                        self.node_to_lat_lon,
                        self.sim.active_calls,
                        self.sim.path_cache,
                        self.sim.current_time
                    )
                    if node is None:
                        node = self.cluster_centers[cluster_idx]
                    ambulance.relocate(node, self.sim.current_time)
                    reward += 15
                else:
                    reward -= 10

            post_avg = self._avg_response_time(ambulance.location)
            reward += (pre_avg - post_avg) / 60.0
            self.idle_amb_id = None

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
            return self._build_obs(), reward, True, False, {}

        self._advance_until_actionable()
        return self._build_obs(), reward, self.done, False, {}

    def _advance_until_actionable(self):
        while not self.done:
            _, event_type, event_data = self.sim.step()
            if event_type is None:
                self.done = True
                break

            if event_type == EventType.CALL_ARRIVAL:
                new_call = self.sim.active_calls.get(event_data["call_id"])
                if new_call:
                    self.pending_calls.append(new_call)

            elif event_type == EventType.CALL_TIMEOUT:
                cid = event_data["call_id"]
                self.pending_calls = [c for c in self.pending_calls if c["call_id"] != cid]
                self.partial_reward -= 30

            elif event_type == EventType.AMB_SCENE_ARRIVAL:
                self.partial_reward += 15
    

            elif event_type == EventType.AMB_HOSPITAL_ARRIVAL:
                self.partial_reward += 30

            elif event_type in (EventType.AMB_TRANSFER_COMPLETE, EventType.AMB_RELOCATION_COMPLETE):
                self.idle_amb_id = None
                for amb in self.sim.ambulances:
                    if amb.status == AmbulanceStatus.IDLE:
                        self.idle_amb_id = amb.id
                        break

            if self.current_call is None and self.pending_calls:
                self.pending_calls = [c for c in self.pending_calls if c["call_id"] in self.sim.active_calls]
                if self.pending_calls and any(a.is_available() for a in self.sim.ambulances):
                    self.current_call = self.pending_calls.pop(0)
                    break
            elif self.idle_amb_id is not None:
                break

    def _build_obs(self) -> np.ndarray:
        amb_part = []
        for amb in self.sim.ambulances:
            lat, lon = self.node_to_lat_lon.get(amb.location, (0.0, 0.0))
            stat_val = float(amb.status.value)
            busy_sec = max(0, amb.busy_until - self.sim.current_time)
            amb_part.extend([lat, lon, stat_val, busy_sec])

        if self.current_call:
            node = self.current_call["origin_node"]
            lat, lon = self.node_to_lat_lon.get(node, (0.0, 0.0))
            priority = float(self.current_call.get("intensity", 1.0))
            call_part = [lat, lon, priority]
        else:
            call_part = [0.0, 0.0, 0.0]

        idle_arr = [0.0] * self.n_amb
        if self.idle_amb_id is not None:
            idle_arr[self.idle_amb_id] = 1.0

        tod = (self.sim.current_time % 86400) / 86400
        obs = np.array(amb_part + call_part + idle_arr + [tod], dtype=np.float32)
        return obs

    def _avg_response_time(self, location: int) -> float:
        calls_dict = self.sim.active_calls
        if not calls_dict:
            return 0.0
        total = 0.0
        for c in calls_dict.values():
            node = c["origin_node"]
            if node in self.sim.path_cache[location]:
                total += self.sim.path_cache[location][node]["travel_time"]
            else:
                total += 600
        return total / len(calls_dict)

    def render(self):
        if self.verbose:
            print(f"Time={self.sim.current_time:.1f}, steps={self.steps}, pending={len(self.pending_calls)}, active={len(self.sim.active_calls)}")
            if self.current_call:
                print(f"** Must dispatch call {self.current_call['call_id']}")
            elif self.idle_amb_id is not None:
                print(f"** Must relocate ambulance #{self.idle_amb_id}")

    def get_stats(self) -> dict:
        total_calls = self.sim.total_calls  # matches simulator stats
        num_responded = len(self.response_times)
        return {
            "total_calls": total_calls,
            "calls_responded": num_responded,
            "completion_rate": num_responded / total_calls,
            "avg_response_time": float(np.mean(self.response_times)) if self.response_times else None
        }
