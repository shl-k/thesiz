"""
Event‑driven ambulance operations simulator.
"""

import heapq
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import Counter


import numpy as np
import pandas as pd
import networkx as nx

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.simulator.ambulance import Ambulance, AmbulanceStatus


SECONDS_IN_DAY = 24 * 3600


class EventType:
    """String constants for event queue types."""
    CALL_ARRIVAL = "call_arrival"
    CALL_TIMEOUT = "call_timeout"
    AMB_SCENE_ARRIVAL = "amb_scene_arrival"
    AMB_SERVICE_COMPLETE = "amb_service_complete"
    AMB_HOSPITAL_ARRIVAL = "amb_hospital_arrival"
    AMB_TRANSFER_COMPLETE = "amb_transfer_complete"
    AMB_RELOCATION_COMPLETE = "amb_relocation_complete"


class AmbulanceSimulator:
    """Fully event‑driven ambulance operations simulator."""

    def __init__(
        self,
        graph: nx.Graph,
        call_data: pd.DataFrame,
        *,
        num_ambulances: int = 3,
        base_location: int = 241,
        hospital_node: int = 1293,
        dispatch_policy=None,
        relocation_policy=None,
        path_cache: Dict = None,
        node_to_idx: Dict = None,
        idx_to_node: Dict = None,
        manual_mode: bool = False,
        verbose: bool = False,
        call_timeout_mean: float = 491.0,
        call_timeout_std: float = 10.0,
        day_length: int = SECONDS_IN_DAY,
    ) -> None:
        
        # Static inputs
        self.graph = graph
        self.call_data = call_data.sort_values(["day", "second_of_day"]).reset_index(drop=True)
        self.num_ambulances = num_ambulances
        self.base_location = base_location
        self.hospital_node = hospital_node
        self.dispatch_policy = dispatch_policy
        self.relocation_policy = relocation_policy
        self.path_cache = path_cache
        self.node_to_idx = node_to_idx or {}
        self.idx_to_node = idx_to_node or {}
        self.manual_mode = manual_mode
        self.verbose = verbose
        self.call_timeout_mean = call_timeout_mean
        self.call_timeout_std = call_timeout_std
        self.day_length = day_length

        # Derived / runtime state (initialised in .initialize())
        self.current_time = 0.0
        self.event_queue = []
        self.event_counter = 0
        self.cancelled_event_ids = set()

        self.ambulances = []
        self.ambulance_events = {}
        self.ambulance_call_counts = {}

        # Call‑level tracking
        self.total_calls = len(self.call_data)
        self.calls_seen = 0
        self.calls_responded = 0
        self.missed_calls = 0
        self.timed_out_calls = 0

        self.call_status = {}  # call_id → "missed", "timed_out", "responded"
        self.calls_rescued = 0
        self.timeout_causes = {
            "no_dispatch": 0,
            "en_route_but_late": 0,
        }

        self.active_calls = {}
        self.call_timeouts = {}
        self.call_response_times = {}
        self.call_total_times = {}
        self.response_times = []

        # NEW: list to store completed hospital deliveries
        self.completed_deliveries = []

    def initialize(self) -> None:
        """Prepare simulator for a fresh run."""
        self.current_time = 0.0
        self.event_queue = []
        self.event_counter = 0
        self.cancelled_event_ids.clear()

        # Build ambulances
    
        self.ambulances = [
            Ambulance(
                amb_id=i,
                location=self.base_location,
                path_cache=self.path_cache,
                node_to_idx=self.node_to_idx,
                idx_to_node=self.idx_to_node,
            )
            for i in range(self.num_ambulances)
        ]
        self.ambulance_events = {i: [] for i in range(self.num_ambulances)}
        self.ambulance_call_counts = {i: 0 for i in range(self.num_ambulances)}

        # Reset call counters
        self.calls_seen = 0
        self.calls_responded = 0
        self.missed_calls = 0
        self.timed_out_calls = 0
        self.active_calls.clear()
        self.call_timeouts.clear()
        self.call_response_times.clear()
        self.call_total_times.clear()
        self.response_times.clear()

        # NEW: clear the list of completed deliveries each run
        self.completed_deliveries.clear()

        # Schedule call arrivals
        for idx, row in self.call_data.iterrows():
            call_time = (row["day"] - 1) * self.day_length + row["second_of_day"]
            call_id = idx + 1
            call_event = {
                "time": call_time,
                "day": row["day"],
                "origin_node": row["origin_node"],
                "destination_node": row["destination_node"],
                "call_id": call_id,
                "intensity": row.get("intensity", 1.0),
            }
            self._push_event(call_time, EventType.CALL_ARRIVAL, call_event)

        if self.verbose:
            print(f"Scheduled {self.total_calls} calls across {self.call_data['day'].max()} day(s)")

    def _push_event(self, time: float, event_type: str, payload: Dict) -> int:
        event_id = self.event_counter
        heapq.heappush(self.event_queue, (time, event_id, event_type, payload))
        self.event_counter += 1
        if "amb_id" in payload:
            self.ambulance_events[payload["amb_id"]].append(event_id)
        return event_id

    def _cancel_event(self, event_id: int) -> None:
        self.cancelled_event_ids.add(event_id)

    def run(self) -> None:
        self.initialize()
        while self.event_queue:
            self.step()
        if self.verbose:
            self._print_statistics()

    def step(self) -> Tuple[Optional[float], Optional[str], Optional[Dict]]:
        if not self.event_queue:
            return None, None, None

        time, event_id, event_type, data = heapq.heappop(self.event_queue)
        if event_id in self.cancelled_event_ids:
            self.cancelled_event_ids.remove(event_id)
            return self.step()  # fetch next

        self.current_time = time

        handler = {
            EventType.CALL_ARRIVAL: self._ev_call_arrival,
            EventType.CALL_TIMEOUT: self._ev_call_timeout,
            EventType.AMB_SCENE_ARRIVAL: self._ev_scene_arrival,
            EventType.AMB_SERVICE_COMPLETE: self._ev_service_complete,
            EventType.AMB_HOSPITAL_ARRIVAL: self._ev_hospital_arrival,
            EventType.AMB_TRANSFER_COMPLETE: self._ev_transfer_complete,
            EventType.AMB_RELOCATION_COMPLETE: self._ev_relocation_complete,
        }[event_type]

        handler(data)
        return time, event_type, data

    def _ev_call_arrival(self, call: Dict) -> None:
        self.calls_seen += 1
        call_id = call["call_id"]
        self.active_calls[call_id] = call

        if self.manual_mode:
            # Schedule only a timeout; external agent decides dispatch
            timeout_t = self.current_time + np.random.normal(self.call_timeout_mean, self.call_timeout_std)
            event_id = self._push_event(timeout_t, EventType.CALL_TIMEOUT, {"call_id": call_id})
            self.call_timeouts[call_id] = (timeout_t, event_id)
            return

        # Otherwise, automatically dispatch
        dispatched = self._dispatch_ambulance(call)
        if not dispatched:
            self.missed_calls += 1
            self.call_status[call_id] = "missed"
            timeout_t = self.current_time + np.random.normal(self.call_timeout_mean, self.call_timeout_std)
            event_id = self._push_event(timeout_t, EventType.CALL_TIMEOUT, {"call_id": call_id})
            self.call_timeouts[call_id] = (timeout_t, event_id)
            if self.verbose:
                print("⚠️  Missed call – no available ambulances.")

        if self.verbose:
            print(f"📞 Call {call_id} at {self._fmt(self.current_time)} (node {call['origin_node']})")

    def _dispatch_ambulance(self, call: Dict) -> bool:
        origin = call["origin_node"]
        available_units = [amb for amb in self.ambulances if amb.is_available()]
        if not available_units:
            return False

        if self.dispatch_policy:
            selected_id = self.dispatch_policy.select_ambulance(
                available_ambulances=[{"id": amb.id, "location": amb.location} for amb in available_units],
                all_ambulances=[{"id": amb.id, "location": amb.location} for amb in self.ambulances],
                current_time=self.current_time,
                current_call=call,
            )
            amb = next(amb for amb in available_units if amb.id == selected_id)
        else:
            # Default: nearest
            amb = min(available_units, key=lambda a: a.get_response_time(origin))

        amb.dispatch_to_call(call, self.current_time)
        self.ambulance_call_counts[amb.id] += 1

        # Schedule scene arrival
        self._push_event(
            amb.busy_until,
            EventType.AMB_SCENE_ARRIVAL,
            {"amb_id": amb.id, "call_id": call["call_id"]},
        )

        # Schedule timeout for this call
        timeout_t = self.current_time + np.random.normal(self.call_timeout_mean, self.call_timeout_std)
        event_id = self._push_event(timeout_t, EventType.CALL_TIMEOUT, {"call_id": call["call_id"]})
        self.call_timeouts[call["call_id"]] = (timeout_t, event_id)

        if self.verbose:
            travel_time = amb.busy_until - self.current_time
            print(f"Dispatched Ambulance {amb.id} to Call {call['call_id']}")
            print(f"  • Travel time: {travel_time:.1f} sec")

        return True

    def _ev_call_timeout(self, data: Dict) -> None:
        call_id = data["call_id"]
        if call_id not in self.active_calls:
            return

        origin = self.active_calls[call_id]["origin_node"]
        was_en_route = any(
            amb.status == AmbulanceStatus.DISPATCHED and amb.call_id == call_id
            for amb in self.ambulances
        )

        # Abort en‑route ambulance if needed
        for amb in self.ambulances:
            if amb.status == AmbulanceStatus.DISPATCHED and amb.call_id == call_id:
                relocation_target = None
                if self.relocation_policy:
                    relocation_map = self.relocation_policy.relocate_ambulances(
                        [{"id": amb.id, "location": amb.location}],
                        [{"id": a.id, "location": a.location} for a in self.ambulances if a.id != amb.id],
                    )
                    relocation_target = relocation_map.get(amb.id)

                if relocation_target is None:
                    relocation_target = self.base_location

                amb.abort_dispatch(self.current_time, relocation_target)

                # Cancel future events for this ambulance
                for event_id in self.ambulance_events[amb.id]:
                    self._cancel_event(event_id)
                self.ambulance_events[amb.id].clear()

                if amb.status == AmbulanceStatus.RELOCATING:
                    self._push_event(
                        amb.busy_until,
                        EventType.AMB_RELOCATION_COMPLETE,
                        {"amb_id": amb.id},
                    )
                break

        # Mark the call as timed out if not previously 'missed'
        if self.call_status.get(call_id) != "missed":
            self.timed_out_calls += 1

            if not was_en_route and any(a.is_available() for a in self.ambulances):
                self.call_status[call_id] = "ignored"
                if self.verbose:
                    print(f"😐 Call {call_id} timed out — idle ambulance existed but no dispatch was made (IGNORED)")
            else:
                self.call_status[call_id] = "timed_out"
                if was_en_route:
                    self.timeout_causes["en_route_but_late"] += 1
                    if self.verbose:
                        print(f"❌ Call {call_id} timed out — ambulance was en route but too slow")
                else:
                    self.timeout_causes["no_dispatch"] += 1
                    if self.verbose:
                        print(f"❌ Call {call_id} timed out — no ambulance dispatched in time")

        self.active_calls.pop(call_id, None)
        self.call_timeouts.pop(call_id, None)

        if self.verbose:
            print(f"👺 Call {call_id} timed‑out at {self._fmt(self.current_time)}")

    def _ev_scene_arrival(self, data: Dict) -> None:
        amb_id = data["amb_id"]
        call_id = data["call_id"]
        amb = self.ambulances[amb_id]

        if amb.status != AmbulanceStatus.DISPATCHED:
            return

        if call_id not in self.active_calls:
            # Timed out while en route, so we might relocate
            relocation_target = None
            if self.relocation_policy:
                relocation_map = self.relocation_policy.relocate_ambulances(
                    [{"id": amb.id, "location": amb.location}],
                    [{"id": a.id, "location": a.location} for a in self.ambulances if a.id != amb.id],
                )
                relocation_target = relocation_map.get(amb.id)

            if relocation_target is None:
                relocation_target = self.base_location

            amb.abort_dispatch(self.current_time, relocation_target)
            if amb.status == AmbulanceStatus.RELOCATING:
                self._push_event(
                    amb.busy_until,
                    EventType.AMB_RELOCATION_COMPLETE,
                    {"amb_id": amb.id},
                )
            return

        # Arrived on scene
        call = self.active_calls.pop(call_id)
        response_time = self.current_time - call["time"]
        self.call_response_times[call_id] = response_time
        

        # Cancel this call's timeout
        if call_id in self.call_timeouts:
            _, eid = self.call_timeouts.pop(call_id)
            self._cancel_event(eid)

        amb.arrive_at_scene(self.current_time)
        self._push_event(
            amb.busy_until,
            EventType.AMB_SERVICE_COMPLETE,
            {"amb_id": amb.id, "call": call},
        )
        if self.verbose:
            print(f"🚑 Ambulance {amb.id} on‑scene for call {call_id} (RT {response_time/60:.1f} min)")

    def _ev_service_complete(self, data: Dict) -> None:
        amb = self.ambulances[data["amb_id"]]
        call = data["call"]

        if amb.status != AmbulanceStatus.ON_SCENE:
            return

        amb.begin_transport(self.hospital_node, self.current_time)
        self._push_event(
            amb.busy_until,
            EventType.AMB_HOSPITAL_ARRIVAL,
            {"amb_id": amb.id, "call_id": call["call_id"]},
        )

    def _ev_hospital_arrival(self, data: Dict) -> None:
        amb = self.ambulances[data["amb_id"]]
        if amb.status != AmbulanceStatus.TRANSPORT:
            return

        amb.arrive_at_hospital(self.current_time)
        self._push_event(
            amb.busy_until,
            EventType.AMB_TRANSFER_COMPLETE,
            {"amb_id": amb.id, "call_id": data["call_id"]},
        )

    def _ev_transfer_complete(self, data: Dict) -> None:
        amb = self.ambulances[data["amb_id"]]
        call_id = data["call_id"]
        if amb.status != AmbulanceStatus.HOSPITAL:
            return

        # Mark responded
        self.calls_responded += 1
        if self.call_status.get(call_id) == "missed":
            self.calls_rescued += 1
            if self.verbose:
                print(f"✅ Call {call_id} was rescued: previously missed but ultimately responded")
        self.call_status[call_id] = "responded"

        # Response time was tracked at scene arrival; track total time
        total_time = self.current_time - amb.dispatch_time
        self.call_total_times[call_id] = total_time

        # NEW: Log the delivery event
        print(f"🚑 [Delivery] Ambulance {amb.id} delivered Call {call_id} at t={self.current_time:.1f}s.")
        self.completed_deliveries.append({
            "call_id": call_id,
            "ambulance_id": amb.id,
            "delivery_time": self.current_time,
        })

        # Possibly relocation policy
        relocation_target = None
        if self.relocation_policy:
            relocation_map = self.relocation_policy.relocate_ambulances(
                [{"id": amb.id, "location": amb.location}],
                [{"id": a.id, "location": a.location} for a in self.ambulances if a.id != amb.id],
            )
            relocation_target = relocation_map.get(amb.id)

        if relocation_target is None and amb.location != self.base_location:
            relocation_target = self.base_location

        if relocation_target is not None and relocation_target != amb.location:
            amb.relocate(relocation_target, self.current_time)
            self._push_event(
                amb.busy_until,
                EventType.AMB_RELOCATION_COMPLETE,
                {"amb_id": amb.id},
            )
        else:
            amb.status = AmbulanceStatus.IDLE

    def _ev_relocation_complete(self, data: Dict) -> None:
        amb = self.ambulances[data["amb_id"]]
        if amb.status != AmbulanceStatus.RELOCATING:
            return
        amb.location = amb.destination  # type: ignore
        amb.destination = None
        amb.status = AmbulanceStatus.IDLE
        amb.busy_until = self.current_time
        if self.verbose:
            print(f"🧮 Ambulance {amb.id} ready at node {amb.location} ({self._fmt(self.current_time)})")

    def _fmt(self, seconds: float) -> str:
        days = int(seconds // SECONDS_IN_DAY)
        h = int((seconds % SECONDS_IN_DAY) // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        prefix = f"Day {days+1}, " if days else ""
        return f"{prefix}{h:02}:{m:02}:{s:02}"

    def _print_statistics(self) -> None:
        outcome_counts = Counter(self.call_status.values())

        print("\n===== Call Accounting Validation (Final Outcomes Only) =====")
        print(f"Total calls in dataset: {self.total_calls}")
        print(f"Calls responded (delivered): {outcome_counts['responded']}")
        print(f"Timed-out calls (not responded): {outcome_counts['timed_out']}")
        print(f"Ignored calls (dispatch never attempted despite idle ambulances): {outcome_counts['ignored']}")
        print(f"Missed calls (no units at arrival): {outcome_counts['missed']}")
        print(f"Calls rescued (missed --> responded): {self.calls_rescued}")
        print(f"Unclassified calls: {self.total_calls - sum(outcome_counts.values())}")

        # Breakdown of timed-out causes
        if outcome_counts["timed_out"] > 0:
            print(f"Breakdown of timed-out calls:")
            print(f"  • No dispatch before timeout: {self.timeout_causes['no_dispatch']}")
            print(f"  • Ambulance en route but too late: {self.timeout_causes['en_route_but_late']}")

        if self.response_times:
            avg_rt = np.mean(self.response_times) # inspect response time value and see if it's correct
            print(f"Avg response time: {avg_rt/60:.2f} min "
                  f"(min {min(self.response_times)/60:.2f}, max {max(self.response_times)/60:.2f})")

    # NEW: helper to get completed deliveries
    def get_completed_deliveries(self):
        """
        Returns the list of all deliveries that arrived
        at the hospital in this simulation run.
        """
        return self.completed_deliveries
