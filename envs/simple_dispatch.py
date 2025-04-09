"""
Simple RL environment for ambulance dispatch decisions.
"""

import os, sys, json
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------
#  Import simulator classes
# ---------------------------------------------------------------------
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator, EventType

# ---------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------
class SimpleDispatchEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        simulator: AmbulanceSimulator,
        node_to_lat_lon_file: str = "data/matrices/node_to_lat_lon.json",
        verbose: bool = False,
        max_steps: int = 100_000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.sim: AmbulanceSimulator = simulator
        self.sim.manual_mode = True          # RL controls dispatch
        self.verbose = verbose
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.num_ambulances = self.sim.num_ambulances

        # ---- node → (lat, lon) ------------------------------------------------
        with open(node_to_lat_lon_file, "r") as f:
            mapping = json.load(f)
        self.node_to_lat_lon: Dict[int, Tuple[float, float]] = {
            int(k): v for k, v in mapping.items()
        }

        # ---- observation & action spaces -------------------------------------
        BIG = 9_999.0          # generous bound for raw lat / lon
        self.observation_space = spaces.Dict(
            {
                "ambulances": spaces.Box(
                    low=np.array([[-BIG, -BIG, 0.0,   0.0]] * self.num_ambulances,
                                 dtype=np.float32),
                    high=np.array([[ BIG,  BIG, 10.0, 1e6]] * self.num_ambulances,
                                  dtype=np.float32),
                    dtype=np.float32,
                ),
                "call": spaces.Box(
                    low=np.array([-BIG, -BIG, 0.0], dtype=np.float32),
                    high=np.array([ BIG,  BIG, 10.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "time_of_day": spaces.Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array([1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        # 0‥N‑1 → dispatch that ambulance, N → no‑dispatch
        self.action_space = spaces.Discrete(self.num_ambulances + 1)

        # ---- episode state ----------------------------------------------------
        self.current_call: Optional[Dict[str, Any]] = None
        self.pending_calls: List[Dict[str, Any]] = []
        self.partial_reward = 0.0
        self.episode_response_times: List[float] = []
        self.episode_calls_responded = 0
        self.steps = 0
        self.done = False

    # ------------------------------------------------------------------
    #  Helper
    # ------------------------------------------------------------------
    def _coords(self, node_id: int) -> Tuple[float, float]:
        latlon = self.node_to_lat_lon.get(node_id)
        if latlon is None:
            raise KeyError(f"No lat/lon for node {node_id}")
        return latlon

    # ------------------------------------------------------------------
    #  Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.sim.initialize()
        self.current_call = None
        self.pending_calls.clear()
        self.partial_reward = 0.0
        self.steps = 0
        self.done = False
        self.episode_response_times.clear()
        self.episode_calls_responded = 0

        self._advance_until_actionable()
        return self._build_obs(), {}

    def step(self, action: int):
        if self.done:
            return self._build_obs(), 0.0, True, False, {}

        reward = self.partial_reward
        self.partial_reward = 0.0

        # ---------- dispatch decision ----------------------------------
        if self.current_call is not None:
            cid = self.current_call["call_id"]
            if action < self.num_ambulances:
                amb = self.sim.ambulances[action]
                if amb.is_available() and cid in self.sim.active_calls:
                    travel_sec = self.sim.path_cache[amb.location][
                        self.current_call["origin_node"]
                    ]["travel_time"]
                    travel_min = travel_sec / 60.0
                    reward += 15.0 - travel_min
                    self.episode_response_times.append(travel_min)
                    self.episode_calls_responded += 1

                    amb.dispatch_to_call(self.current_call, self.sim.current_time)
                    self.sim._push_event(
                        amb.busy_until,
                        EventType.AMB_SCENE_ARRIVAL,
                        {"amb_id": amb.id, "call_id": cid},
                    )
                else:
                    reward -= 100.0
            else:  # no‑dispatch
                if any(a.is_available() for a in self.sim.ambulances):
                    reward -= 100.0
            self.current_call = None

        # ---------- termination check ----------------------------------
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
            return self._build_obs(), reward / 100.0, True, False, {}

        # ---------- advance simulation until next decision -------------
        reward = self._advance_until_actionable(reward) / 100.0
        return self._build_obs(), reward, self.done, False, {}

    # ------------------------------------------------------------------
    #  Simulator driver
    # ------------------------------------------------------------------
    def _advance_until_actionable(self, reward_acc: float = 0.0) -> float:
        while not self.done:
            _, etype, edata = self.sim.step()
            if etype is None:               # no more events
                self.done = True
                break

            if etype == EventType.CALL_ARRIVAL:
                if any(a.is_available() for a in self.sim.ambulances):
                    self.current_call = self.sim.active_calls[edata["call_id"]]
                    break

            elif etype == EventType.CALL_TIMEOUT:
                reward_acc -= 30.0

            elif etype == EventType.AMB_SCENE_ARRIVAL:
                reward_acc += 15.0

            elif etype == EventType.AMB_HOSPITAL_ARRIVAL:
                reward_acc += 30.0

        return reward_acc

    # ------------------------------------------------------------------
    #  Observation builder
    # ------------------------------------------------------------------
    def _build_obs(self) -> Dict[str, np.ndarray]:
        # ambulances
        amb_arr = []
        for amb in self.sim.ambulances:
            lat, lon = self._coords(amb.location)
            busy_left = max(0.0, amb.busy_until - self.sim.current_time)
            amb_arr.append([lat, lon, amb.status.value, busy_left])

        # call
        if self.current_call:
            lat, lon = self._coords(self.current_call["origin_node"])
            priority = float(self.current_call.get("intensity", 1.0))
            call_arr = [lat, lon, priority]
        else:
            call_arr = [0.0, 0.0, 0.0]

        tod = (self.sim.current_time % 86_400) / 86_400.0

        obs = {
            "ambulances": np.asarray(amb_arr, dtype=np.float32),
            "call": np.asarray(call_arr, dtype=np.float32),
            "time_of_day": np.asarray([tod], dtype=np.float32),
        }

        # sanity check
        assert all(np.all(np.isfinite(v)) for v in obs.values())
        return obs

    # ------------------------------------------------------------------
    #  Render (optional)
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            print(
                f"t={self.sim.format_time(self.sim.current_time)} "
                f"active_calls={len(self.sim.active_calls)} "
                f"pending={len(self.pending_calls)}"
            )
            if self.current_call:
                print(f"→ dispatch needed for call {self.current_call['call_id']}")
