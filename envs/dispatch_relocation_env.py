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

    def __init__(
        self,
        simulator: AmbulanceSimulator,
        n_clusters: int = 10,
        node_to_lat_lon_file: str = "data/matrices/node_to_lat_lon.json",
        verbose: bool = False,
        max_steps: int = 100000
    ):
        super().__init__()
        self.sim = simulator
        self.sim.manual_mode = True  # we handle dispatch + relocation
        self.verbose = verbose
        self.max_steps = max_steps

        self.n_amb = simulator.num_ambulances
        self.n_clusters = n_clusters
        # total action = n_amb + 1 (no-dispatch) + n_clusters
        self.action_space = spaces.Discrete(self.n_amb + 1 + self.n_clusters)

        # We'll store calls that arrive when no ambulance is idle
        self.pending_calls: List[dict] = []

        # At each RL decision point: either we have a call (self.current_call) or an idle ambulance (self.idle_amb_id).
        self.current_call: Optional[dict] = None
        self.idle_amb_id: Optional[int]   = None

        self.steps = 0
        self.done = False

        # For partial rewards in the simulator "sub-events"
        self.partial_reward = 0.0

        # Node -> lat/lon
        self.node_to_lat_lon = load_node_mappings(node_to_lat_lon_file)[0]

        # Clusters
        self.node_to_cluster, self.cluster_centers = cluster_nodes(
            self.node_to_lat_lon,
            self.sim.call_data,
            n_clusters=self.n_clusters
        )

        # --------------- Flattened observation ---------------
        # We'll produce a 1D array with these pieces:
        #   - For each ambulance: [lat, lon, status, busy_time_left] => n_amb * 4
        #   - For the call: [call_lat, call_lon, priority] => 3
        #   - For idle_ambulance one-hot => n_amb
        #   - time_of_day => 1
        # total = n_amb*4 + 3 + n_amb + 1
        obs_dim = self.n_amb * 4 + 3 + self.n_amb + 1

        # We allow negative lat/lon => big bounding range
        self.observation_space = spaces.Box(
            low=-9999,
            high=9999,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.sim.initialize()
        self.pending_calls.clear()
        self.current_call = None
        self.idle_amb_id = None
        self.steps = 0
        self.done = False
        self.partial_reward = 0.0

        # step sim until we find a scenario that needs an RL action
        self._advance_until_actionable()
        obs = self._build_obs()
        return obs, {}

    def step(self, action: int):
        if self.done:
            # if done, no more changes
            return self._build_obs(), 0.0, True, False, {}

        reward = 0.0

        # incorporate any partial reward from sub-events
        reward += self.partial_reward
        self.partial_reward = 0.0

        # 1) If there's a call => interpret action as dispatch
        if self.current_call is not None:
            if action < self.n_amb:
                # dispatch ambulance #action
                amb = self.sim.ambulances[action]
                cid = self.current_call["call_id"]
                if amb.is_available() and cid in self.sim.active_calls:
                    travel_sec = self.sim.path_cache[amb.location][self.current_call["origin_node"]]["travel_time"]
                    travel_min = travel_sec / 60.0
                    reward += 15.0 - travel_min
                    # do dispatch
                    amb.dispatch_to_call(self.current_call, self.sim.current_time)
                    self.sim._push_event(
                        amb.busy_until,
                        EventType.AMB_SCENE_ARRIVAL,
                        {"amb_id": amb.id, "call_id": cid}
                    )
                else:
                    reward -= 15
            elif action == self.n_amb:
                # no dispatch
                if any(a.is_available() for a in self.sim.ambulances):
                    reward -= 30
            else:
                # invalid in dispatch mode
                reward -= 50

            self.current_call = None

        # 2) If an ambulance is idle => interpret action as relocation
        elif self.idle_amb_id is not None:
            amb_id = self.idle_amb_id
            ambulance = self.sim.ambulances[amb_id]

            # partial reward for difference in average response time
            pre_avg = self._avg_response_time(ambulance.location)

            if action < self.n_amb:
                # invalid
                reward -= 50
            elif action == self.n_amb:
                # no-relocate
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
                    # relocate
                    ambulance.relocate(node, self.sim.current_time)
                    reward += 15
                else:
                    # unreachable if action_space = n_amb +1 +n_clusters
                    reward -= 10

            post_avg = self._avg_response_time(ambulance.location)
            delta = (pre_avg - post_avg) / 60.0
            reward += delta

            self.idle_amb_id = None

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
            return self._build_obs(), reward, True, False, {}

        # process further sim events until next decision
        self._advance_until_actionable()
        obs = self._build_obs()
        return obs, reward, self.done, False, {}

    def _advance_until_actionable(self):
        """
        Step the simulator until:
          - a call arrives (and an amb is idle) => must dispatch
          - or an ambulance completes => must relocate
          - or we run out of events
        We'll accumulate partial rewards (scene arrival = +15, hospital arrival=+30) 
        in self.partial_reward.
        """
        while not self.done:
            _, event_type, event_data = self.sim.step()
            if event_type is None:
                self.done = True
                break

            if event_type == EventType.CALL_ARRIVAL:
                # if any ambulance is idle => we can dispatch now
                if any(a.is_available() for a in self.sim.ambulances):
                    new_call = self.sim.active_calls[event_data["call_id"]]
                    self.current_call = new_call
                    break
                else:
                    self.pending_calls.append(event_data)

            elif event_type == EventType.CALL_TIMEOUT:
                # remove from pending
                cid = event_data["call_id"]
                self.pending_calls = [d for d in self.pending_calls if d["call_id"] != cid]
                self.partial_reward -= 30  # lost call

            elif event_type == EventType.AMB_SCENE_ARRIVAL:
                self.partial_reward += 15

            elif event_type == EventType.AMB_HOSPITAL_ARRIVAL:
                self.partial_reward += 30

            elif event_type in (EventType.AMB_TRANSFER_COMPLETE, EventType.AMB_RELOCATION_COMPLETE):
                # ambulance idle
                # see if we have queued calls
                self.pending_calls = [cd for cd in self.pending_calls
                                      if cd["call_id"] in self.sim.active_calls]
                if self.pending_calls:
                    call_data = self.pending_calls.pop(0)
                    cobj = self.sim.active_calls.get(call_data["call_id"])
                    if cobj is not None:
                        self.current_call = cobj
                        break
                else:
                    # no calls => relocate
                    for amb in self.sim.ambulances:
                        if amb.status == AmbulanceStatus.IDLE:
                            self.idle_amb_id = amb.id
                            break
                    break
            # else keep stepping

    def _build_obs(self) -> np.ndarray:
        """
        Flattened observation:

         1) for each ambulance: [lat, lon, status, busy] => n_amb*4
         2) call => [call_lat, call_lon, priority] => 3
         3) idle_ambulance => length=n_amb
         4) time_of_day => 1
        total dimension = n_amb*4 + 3 + n_amb + 1
        """
        # gather ambulance info
        amb_part = []
        for amb in self.sim.ambulances:
            lat, lon = self.node_to_lat_lon.get(amb.location, (0.0, 0.0))
            stat_val = float(amb.status.value)
            busy_sec = max(0, amb.busy_until - self.sim.current_time)
            amb_part.extend([lat, lon, stat_val, busy_sec])

        # gather call info
        if self.current_call:
            node = self.current_call["origin_node"]
            lat, lon = self.node_to_lat_lon.get(node, (0.0, 0.0))
            priority = float(self.current_call.get("intensity",1.0))
            call_part = [lat, lon, priority]
        else:
            call_part = [0.0, 0.0, 0.0]

        # idle ambulance one-hot
        idle_arr = [0.0]*self.n_amb
        if self.idle_amb_id is not None:
            idle_arr[self.idle_amb_id] = 1.0

        # time_of_day
        tod = (self.sim.current_time % 86400)/86400

        # flatten
        obs = np.array(amb_part + call_part + idle_arr + [tod], dtype=np.float32)
        return obs

    def _avg_response_time(self, location:int)-> float:
        """
        Average travel time (in seconds) from 'location' to all active calls.
        If none, returns 0.
        """
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
