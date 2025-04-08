"""
Ambulance Simulator
A fully event-driven simulator for ambulance operations with call timeouts.
"""

import os
import sys
import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional, List, Tuple, Set

import heapq
import numpy as np
import random
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import ambulance related classes
from ambulance import Ambulance, AmbulanceStatus

class EventType:
    """Event types for the simulator."""
    CALL_ARRIVAL = "call_arrival" # dispatching an ambulance to a call
    CALL_TIMEOUT = "call_timeout"  # call timed out
    AMB_DISPATCHED = "amb_dispatched"  # ambulance dispatched to a call
    AMB_SCENE_ARRIVAL = "amb_scene_arrival" # ambulance arrived at scene
    AMB_SERVICE_COMPLETE = "amb_service_complete"  # ambulance completed service
    AMB_HOSPITAL_ARRIVAL = "amb_hospital_arrival" # ambulance arrived at hospital
    AMB_TRANSFER_COMPLETE = "amb_transfer_complete" # ambulance completed transfer
    AMB_RELOCATION_COMPLETE = "amb_relocation_complete" # ambulance completed relocation


class AmbulanceSimulator:
    """
    Event-driven simulator for ambulance operations.
    
    This simulator:
      1. Loads synthetic call data
      2. Initializes ambulances at base locations
      3. Uses an event queue to manage simulation events
      4. Handles dispatch, timeouts, and relocation
      5. Tracks statistics
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        call_data: pd.DataFrame,
        num_ambulances: int = 3,
        base_location: int = 241,  # The default base location
        hospital_node: int = 1293,  # The default hospital location
        call_timeout_mean: float = 491.0,  # Mean timeout in seconds (73 + 418)
        call_timeout_std: float = 10.0,   # Standard deviation for timeout
        dispatch_policy = None,
        relocation_policy = None,
        verbose: bool = False,
        day_length: int = 24 * 3600,  # Default day length in seconds
        path_cache: Dict = None,
        node_to_idx: Dict = None,
        idx_to_node: Dict = None,
        manual_mode: bool = False,  # When True, disables automatic dispatch
    ):
        """
        Initialize the simulator.
        
        Args:
            graph: The road network graph
            call_data: DataFrame with call data (columns: day, second_of_day, origin_node, destination_node)
            num_ambulances: Number of ambulances to simulate
            base_location: Base location node ID for ambulances
            hospital_node: Hospital node ID
            call_timeout_mean: Mean time before a call times out (seconds)
            call_timeout_std: Standard deviation for timeout distribution
            dispatch_policy: Policy for dispatching ambulances (optional, defaults to nearest available)
            relocation_policy: Policy for relocating ambulances (optional, defaults to return to base)
            verbose: Whether to print verbose output
            day_length: Default day length in seconds
            path_cache: Path cache for travel times
            node_to_idx: Mapping from node to index
            idx_to_node: Mapping from index to node
            manual_mode: When True, disables automatic dispatch
        """
        # Store parameters
        self.graph = graph
        self.call_data = call_data
        self.num_ambulances = num_ambulances
        self.base_location = base_location
        self.hospital_node = hospital_node
        self.call_timeout_mean = call_timeout_mean
        self.call_timeout_std = call_timeout_std
        self.day_length = day_length
        self.verbose = verbose
        self.path_cache = path_cache
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.manual_mode = manual_mode
        
        # Initialize attributes
        self.current_time = 0
        self.ambulances = []
        self.event_queue = []
        self.event_counter = 0
        self.next_call_id = 1
        self.cancelled_event_ids = set()  # Set of cancelled event IDs for lazy deletion
        
        # Track ambulance events - amb_id -> list of event_ids
        self.ambulance_events = {i: [] for i in range(num_ambulances)}
        
        # Statistics
        self.calls_responded = 0  # Number of patients delivered to hospital
        self.response_times = []
        self.call_response_times = {}  # call_id -> response time
        self.call_total_times = {}  # call_id -> total time from call to drop-off
        self.missed_calls = 0  # Calls missed initially but not timed out
        self.timed_out_calls = 0  # Calls that timed out
        self.total_calls = len(call_data)  # Total number of calls in the dataset
        self.calls_seen = 0  # Total number of calls seen (should equal total_calls)
        
        # Active calls and their timeout events
        self.active_calls = {}  # call_id -> call_data
        self.call_timeouts = {}  # call_id -> (timeout_time, timeout_event_id)
        
        # Ambulance call counts
        self.ambulance_call_counts = {i: 0 for i in range(num_ambulances)}
        
        # Track all call IDs from synthetic data
        self.synthetic_call_ids = set(range(1, self.total_calls + 1))

        # Create ambulances
        for i in range(num_ambulances):
            amb = Ambulance(i, base_location, 
                           path_cache=self.path_cache,
                           node_to_idx=self.node_to_idx,
                           idx_to_node=self.idx_to_node)
            self.ambulances.append(amb)
        
        # Set dispatch and relocation policies
        self.dispatch_policy = dispatch_policy 
        self.relocation_policy = relocation_policy 
        
        # Set up the first day's events
        self._setup_events()
    
    def load_call_data(self, call_data_path: str) -> List[Dict]:
        """Load and preprocess call data."""
        df = pd.read_csv(call_data_path)
        # Sort by second_of_day for accurate chronological ordering
        df = df.sort_values(['day', 'second_of_day'])
        calls = []
        for _, row in df.iterrows():
            t = (row['day'] - 1) * 24 * 3600 + row['second_of_day']
            calls.append({
                "time": t,
                "day": row['day'],
                "second_of_day": row['second_of_day'],
                "origin_node": row['origin_node'],
                "destination_node": row['destination_node'],
                "intensity": row['intensity'] # intensity/priority
            })
        if self.verbose:
            print(f"Loaded {len(calls)} calls over {max(c['day'] for c in calls)} days")
        return calls
    
    def _push_event(self, time_val: float, event_type: str, data: Dict) -> int:
        """Push a new event to the event queue and track it by ambulance ID if provided."""
        event_id = self.event_counter
        heapq.heappush(self.event_queue, (time_val, event_id, event_type, data))
        self.event_counter += 1
        
        # Track event by ambulance ID
        if "amb_id" in data:
            amb_id = data["amb_id"]
            self.ambulance_events[amb_id].append(event_id)
            
        return event_id
        
    def _cancel_event(self, event_id: int) -> bool:
        """
        Mark an event as cancelled using lazy deletion.
        Simply adds the event ID to the cancelled set without modifying the heap.
        
        Returns:
            bool: Always returns True since we're just adding to a set
        """
        self.cancelled_event_ids.add(event_id)
        return True
    
    def initialize(self):
        """Initialize the simulation by setting up initial events."""
        self.current_time = 0
        self.event_queue = []
        self.event_counter = 0
        self.is_running = False
        
        # Reset cancelled event IDs
        self.cancelled_event_ids = set()
        
        # Reset statistics
        self.response_times = []
        self.calls_responded = 0
        self.missed_calls = 0
        self.timed_out_calls = 0
        
        # Reset call tracking
        self.active_calls = {}
        self.call_timeouts = {}
        self.next_call_id = 1
        
        # Reset ambulances
        for amb in self.ambulances:
            amb.status = AmbulanceStatus.IDLE
            amb.location = self.base_location
            amb.current_call = None
            amb.destination = None
            amb.busy_until = 0
            amb.path = []
            amb.calls_responded = 0
            amb.total_response_time = 0
            amb.path_cache = self.path_cache
            amb.node_to_idx = self.node_to_idx
            amb.idx_to_node = self.idx_to_node
        
        # Set up events using _setup_events()
        self._setup_events()
        
        if self.verbose:
            print(f"Simulation initialized with {self.total_calls} calls and {self.num_ambulances} ambulances")
    
    def run(self):
        """Run the simulation until completion."""
        self.current_time = 0
        self.event_queue = []
        self.event_counter = 0
        self.is_running = True
        
        # Reset statistics
        self.response_times = []
        self.calls_responded = 0
        self.missed_calls = 0
        self.timed_out_calls = 0
        
        # Set up events
        self._setup_events()
        
        # Process events until simulation ends
        while self.event_queue and self.is_running:
            self.step()
        
        # Print final statistics
        self.print_statistics()
    
    def step(self):
        """
        Process the next event from the event queue.
        
        Returns:
            tuple: (event_time, event_type, data) or (None, None, None) if no more events.
        """
        if not self.event_queue:
            return None, None, None

        # Pop the next event
        event_time, event_id, event_type, data = heapq.heappop(self.event_queue)
        
        # Skip cancelled events using lazy deletion
        if event_id in self.cancelled_event_ids:
            self.cancelled_event_ids.remove(event_id)  # Clean up the set
            return self.step()  # Recursively process the next event
            
        # Update current time
        self.current_time = event_time

        # Process the event based on its type
        if event_type == EventType.CALL_ARRIVAL:
            if self.verbose:
                print(f"📞 Call arrival at {self.format_time(event_time)}")
                
            # Check if a relocation target is specified in the event data
            relocation_target = data.get("relocation_target", None)
            self._handle_call_arrival(data, relocation_target)

        elif event_type == EventType.CALL_TIMEOUT:
            if self.verbose:
                print(f"👺 Call timeout at {self.format_time(event_time)}")
            self._handle_call_timeout(data)

        elif event_type == EventType.AMB_SCENE_ARRIVAL:
            if self.verbose:
                print(f"🚑 Ambulance scene arrival at {self.format_time(event_time)}")
            self._handle_ambulance_scene_arrival(data)

        elif event_type == EventType.AMB_SERVICE_COMPLETE:
            if self.verbose:
                print(f"🚑 Service complete at {self.format_time(event_time)}")
            self._handle_ambulance_service_complete(data)

        elif event_type == EventType.AMB_HOSPITAL_ARRIVAL:
            if self.verbose:
                print(f"🏥 Hospital arrival at {self.format_time(event_time)}")
            self._handle_ambulance_hospital_arrival(data)

        elif event_type == EventType.AMB_TRANSFER_COMPLETE:
            if self.verbose:
                print(f"🏥 Transfer complete at {self.format_time(event_time)}")
            self._handle_ambulance_transfer_complete(data)

        elif event_type == EventType.AMB_RELOCATION_COMPLETE:
            if self.verbose:
                print(f"🧮 Relocation complete at {self.format_time(event_time)}")
            self._handle_ambulance_relocation_complete(data)

        return event_time, event_type, data
    
    def _handle_call_arrival(self, call: Dict, relocation_target: int = None):
        """Process a new emergency call arrival."""
        call_id = call["call_id"]
        self.calls_seen += 1
        
        if self.verbose:
            print(f"\n📞 Processing call {call_id} at {self.format_time(self.current_time)}")
            print(f"  Total calls seen: {self.calls_seen}/{self.total_calls}")
        
        # Add call to active calls immediately
        self.active_calls[call_id] = call
        
        # Only dispatch automatically if not in manual mode
        if not self.manual_mode:
            # Try to dispatch an ambulance immediately
            dispatch_success = self._dispatch_ambulance(call, relocation_target)
            
            # If no ambulance was available for immediate dispatch, increment missed_calls
            if not dispatch_success and not any(amb.is_available() for amb in self.ambulances):
                self.missed_calls += 1
                # Log detailed information about why the call was missed
                print(f"\n⚠️ Call {call_id} missed at {self.format_time(self.current_time)}")
                print("  Ambulance statuses:")
                for amb in self.ambulances:
                    print(f"    Ambulance {amb.id}: {amb.status.name}")
                    if amb.status != AmbulanceStatus.IDLE:
                        print(f"      Busy until: {self.format_time(amb.busy_until)}")
                print(f"  Call location: node {call['origin_node']}")
                # Remove from active calls since it's been handled
                self.active_calls.pop(call_id, None)
                if call_id in self.call_timeouts:
                    del self.call_timeouts[call_id]
    
    def _dispatch_ambulance(self, call: Dict, relocation_target: int = None) -> bool:
        """Dispatch an ambulance to a call using the dispatch policy."""
        call_id = call["call_id"]
        call_node = call["origin_node"]
        
        # Get list of available ambulances
        available_ambulances = [
            {'id': amb.id, 'location': amb.location, 'status': amb.status.value, 'busy_until': amb.busy_until}
            for amb in self.ambulances
            if amb.is_available()
        ]
        
        # Get list of all ambulances
        all_ambulances = [
            {'id': amb.id, 'location': amb.location, 'status': amb.status.value, 'busy_until': amb.busy_until}
            for amb in self.ambulances
        ]
        
        # Select ambulance using policy or default to nearest available
        if self.dispatch_policy:
            selected_amb_id = self.dispatch_policy.select_ambulance(
                available_ambulances=available_ambulances,
                all_ambulances=all_ambulances,
                current_time=self.current_time,
                current_call=call
            )
        else:
            selected_amb_id = min(
                available_ambulances,
                key=lambda amb: self.path_cache[amb["location"]][call_node]['travel_time']
            )["id"]
        
        # Get the selected ambulance
        selected_ambulance = next((amb for amb in self.ambulances if amb.id == selected_amb_id), None)
        
        # If no ambulance was selected or the selected ambulance is not available, return False
        if selected_ambulance is None or not selected_ambulance.is_available():
            if self.verbose:
                if selected_ambulance is None:
                    print(f"  No ambulance selected for call {call_id}")
                else:
                    print(f"  Selected ambulance {selected_amb_id} is not available")
            return False
            
        # Dispatch the ambulance
        selected_ambulance.dispatch_to_call(call, self.current_time)
        
        # Schedule scene arrival
        event_id = self._push_event(
            selected_ambulance.busy_until,
            EventType.AMB_SCENE_ARRIVAL,
            {"amb_id": selected_ambulance.id, "call_id": call_id}
        )
        
        # Add call to active calls
        self.active_calls[call_id] = call
        
        # Schedule timeout event
        timeout_time = self.current_time + int(np.random.normal(self.call_timeout_mean, self.call_timeout_std))
        timeout_event_id = self._push_event(
            timeout_time,
            EventType.CALL_TIMEOUT,
            {"call_id": call_id, "relocation_target": relocation_target}
        )
        self.call_timeouts[call_id] = (timeout_time, timeout_event_id)
        
        if self.verbose:
            print(f"\nDispatched ambulance {selected_ambulance.id} to call {call_id}")
            travel_time = selected_ambulance.busy_until - self.current_time
            print(f"  Travel time: {travel_time/60:.1f} minutes")
            print(f"  From node {selected_ambulance.location} to {call_node}")
        
        return True
    
    def _handle_call_timeout(self, data: Dict):
        """Handle a call timeout event."""
        call_id = data["call_id"]
        
        # Check if call is still active (timeout wasn't canceled)
        if call_id not in self.active_calls:
            return
        
        # Check if any ambulance was dispatched to this call
        dispatched_ambulance = None
        for amb in self.ambulances:
            if amb.status == AmbulanceStatus.DISPATCHED and amb.call_id == call_id:
                dispatched_ambulance = amb
                break
        
        # If an ambulance is en route, abort the dispatch and cancel pending events
        if dispatched_ambulance:
            amb_id = dispatched_ambulance.id
            
            # Determine relocation target 
            relocation_target = data.get("relocation_target", self.base_location)
            
            # Abort the dispatch and start relocating to chosen target
            dispatched_ambulance.abort_dispatch(self.current_time, relocation_target)
            
            # Cancel all pending events for this ambulance
            for event_id in self.ambulance_events[amb_id]:
                self._cancel_event(event_id)
            
            # Clear the ambulance's event list
            self.ambulance_events[amb_id] = []
            
            # Schedule relocation completion if the ambulance is now relocating
            if dispatched_ambulance.status == AmbulanceStatus.RELOCATING:
                event_id = self._push_event(
                    dispatched_ambulance.busy_until,
                    EventType.AMB_RELOCATION_COMPLETE,
                    {"amb_id": amb_id}
                )
                self.ambulance_events[amb_id].append(event_id)
            
            if self.verbose:
                print(f"\nCall {call_id} timed out at {self.format_time(self.current_time)}")
                print(f"  Aborted dispatch of ambulance {dispatched_ambulance.id}")
                print(f"  Ambulance is now relocating to node {relocation_target}")
            
        # Count as timed out call (only if not already counted as missed)
        # Note: We check active_calls BEFORE popping it
        if call_id in self.active_calls:  # Only count if still active (not already missed)
            self.timed_out_calls += 1
        
        # Get the call data and remove from active calls
        call = self.active_calls.pop(call_id)
        
        # Remove timeout tracking
        if call_id in self.call_timeouts:
            del self.call_timeouts[call_id]
        
        if self.verbose and not dispatched_ambulance:
            print(f"\nCall {call_id} timed out at {self.format_time(self.current_time)}")
    
    def _handle_ambulance_scene_arrival(self, data: Dict):
        """Handle ambulance arrival at call scene."""
        amb_id = data["amb_id"]
        call_id = data["call_id"]
        ambulance = self.ambulances[amb_id]
        
        # Check if ambulance is still dispatched (not aborted)
        if ambulance.status != AmbulanceStatus.DISPATCHED:
            return
            
        # Check if call is still active
        if call_id not in self.active_calls:
            # Call timed out or was otherwise canceled
            # Just return ambulance to idle state
            ambulance.status = AmbulanceStatus.IDLE
            ambulance.current_call = None
            ambulance.call_id = None
            return
        
        # Get call data
        call = self.active_calls.pop(call_id)
        
        # Calculate and store response time (time from call to scene arrival)
        response_time = self.current_time - call["time"]
        self.call_response_times[call_id] = response_time
        
        # Cancel the timeout event for this call
        if call_id in self.call_timeouts:
            _, timeout_event_id = self.call_timeouts[call_id]
            self._cancel_event(timeout_event_id)  # Time doesn't matter with lazy deletion
            del self.call_timeouts[call_id]
        
        # Start on-scene service
        ambulance.arrive_at_scene(self.current_time)
        
        # Schedule service completion
        event_id = self._push_event(
            ambulance.busy_until,
            EventType.AMB_SERVICE_COMPLETE,
            {
                "amb_id": amb_id,
                "call": call
            }
        )
        
        if self.verbose:
            print(f"\nAmbulance {amb_id} arrived at scene for call {call_id} at {self.format_time(self.current_time)}")
            print(f"  Response time: {response_time/60:.1f} minutes")
            service_time = ambulance.busy_until - self.current_time
            print(f"  On-scene service time: {service_time/60:.1f} minutes")
            print(f"  Will complete service at: {self.format_time(ambulance.busy_until)}")
    
    def _handle_ambulance_service_complete(self, data: Dict):
        """Handle completion of on-scene service."""
        amb_id = data["amb_id"]
        call = data["call"]
        ambulance = self.ambulances[amb_id]
        
        # Verify ambulance is actually in ON_SCENE status (not aborted)
        if ambulance.status != AmbulanceStatus.ON_SCENE:
            if self.verbose:
                print(f"⚠️ Warning: Ambulance {amb_id} is not ON_SCENE for service completion. " +
                      f"Current status: {ambulance.status.name}. Call {call.get('call_id', 'unknown')} may be dropped.")
            return
            
        # Verify destination is valid
        if self.hospital_node is None:
            print("ALERT ALERT: No hospital node set")
            # Can't transport without a hospital
            # Set ambulance to idle
            ambulance.status = AmbulanceStatus.IDLE
            ambulance.current_call = None
            return
        
        # Begin transport to hospital
        ambulance.begin_transport(self.hospital_node, self.current_time)
        
        # Schedule hospital arrival
        event_id = self._push_event(
            ambulance.busy_until,
            EventType.AMB_HOSPITAL_ARRIVAL,
            {"amb_id": amb_id, "call_id": call["call_id"]}
        )
        
        if self.verbose:
            print(f"\nAmbulance {amb_id} completed on-scene service at {self.format_time(self.current_time)}")
            print(f"  Transporting to hospital")
            transport_time = ambulance.busy_until - self.current_time
            print(f"  Transport time: {transport_time/60:.1f} minutes")
    
    def _handle_ambulance_hospital_arrival(self, data: Dict):
        """Handle ambulance arrival at hospital."""
        amb_id = data["amb_id"]
        call_id = data.get("call_id")  # Get the call_id from the event data
        ambulance = self.ambulances[amb_id]
        
        # Verify ambulance is actually in TRANSPORT status (not aborted)
        if ambulance.status != AmbulanceStatus.TRANSPORT:
            return
        
        # Start hospital transfer
        ambulance.arrive_at_hospital(self.current_time)
        
        # Schedule transfer completion
        event_id = self._push_event(
            ambulance.busy_until,
            EventType.AMB_TRANSFER_COMPLETE,
            {"amb_id": amb_id, "call_id": call_id}  # Include call_id in the transfer completion event
        )
        
        if self.verbose:
            print(f"\nAmbulance {amb_id} arrived at hospital at {self.format_time(self.current_time)}")
            transfer_time = ambulance.busy_until - self.current_time
            print(f"  Hospital transfer time: {transfer_time/60:.1f} minutes")
            if call_id is not None:
                print(f"  For call: {call_id}")
    
    def _handle_ambulance_transfer_complete(self, data: Dict):
        """Handle ambulance transfer completion at hospital."""
        amb_id = data["amb_id"]
        call_id = data.get("call_id")  # Get the call_id
        ambulance = self.ambulances[amb_id]
        
        # Verify ambulance is actually in HOSPITAL status (not aborted)
        if ambulance.status != AmbulanceStatus.HOSPITAL:
            return
        
        # Only count as responded if patient was delivered to hospital
        if call_id is not None:
            self.calls_responded += 1
            self.ambulance_call_counts[amb_id] += 1
            
            # Get the stored response time
            response_time = self.call_response_times.get(call_id, 0)
            self.response_times.append(response_time)
            
            # Calculate total time from call to drop-off
            # Get the call data from the ambulance's current_call
            call = ambulance.current_call
            if call:
                total_time = self.current_time - call["time"]
                self.call_total_times[call_id] = total_time
                
                if self.verbose:
                    print(f"\nAmbulance {amb_id} delivered patient to hospital for call {call_id}")
                    print(f"  Response time: {response_time/60:.1f} minutes")
                    print(f"  Total time from call to drop-off: {total_time/60:.1f} minutes")
        
        # Set ambulance to idle
        ambulance.status = AmbulanceStatus.IDLE
        ambulance.current_call = None
        
        # Only relocate this specific ambulance if needed
        if self.relocation_policy:
            # Use provided policy to check if this ambulance should relocate
            relocations = self.relocation_policy.relocate_ambulances(
                [{'id': ambulance.id, 'location': ambulance.location}],  # Just this ambulance
                [{'id': amb.id, 'location': amb.location} for amb in self.ambulances if amb.id != ambulance.id]  # All others
            )
            
            # If the policy wants to relocate this ambulance
            if ambulance.id in relocations:
                new_loc = relocations[ambulance.id]
                if new_loc is not None and ambulance.location != new_loc:
                    ambulance.relocate(new_loc, self.current_time)
                    
                    # Schedule relocation completion
                    event_id = self._push_event(
                        ambulance.busy_until,
                        EventType.AMB_RELOCATION_COMPLETE,
                        {"amb_id": ambulance.id}
                    )
                    
                    if self.verbose:
                        print(f"\nAmbulance {ambulance.id} started relocating to {new_loc} at {self.format_time(self.current_time)}")
                        relocation_time = ambulance.busy_until - self.current_time
                        print(f"  Relocation time: {relocation_time/60:.1f} minutes")
        else:
            # Default policy: return to base if not already there
            if ambulance.location != self.base_location:
                ambulance.relocate(self.base_location, self.current_time)
                
                # Schedule relocation completion
                event_id = self._push_event(
                    ambulance.busy_until,
                    EventType.AMB_RELOCATION_COMPLETE,
                    {"amb_id": ambulance.id}
                )
                
                if self.verbose:
                    print(f"\nAmbulance {ambulance.id} started relocating to {self.base_location} at {self.format_time(self.current_time)}")
                    relocation_time = ambulance.busy_until - self.current_time
                    print(f"  Relocation time: {relocation_time/60:.1f} minutes")
        
        if self.verbose:
            print(f"\nAmbulance {amb_id} completed hospital transfer at {self.format_time(self.current_time)}")
            if ambulance.status == AmbulanceStatus.RELOCATING:
                print(f"  Will begin relocation to {ambulance.destination}")
            else:
                print(f"  Status: {ambulance.status.name}")
    
    def _handle_ambulance_relocation_complete(self, data: Dict):
        """Handle completion of ambulance relocation."""
        amb_id = data["amb_id"]
        ambulance = self.ambulances[amb_id]
        
        # Verify ambulance is actually in RELOCATING status (not aborted)
        if ambulance.status != AmbulanceStatus.RELOCATING:
            return
            
        # Verify destination is valid
        if ambulance.destination is None:
            # No destination - set to idle
            ambulance.status = AmbulanceStatus.IDLE
            return
        
        # Complete relocation
        ambulance.location = ambulance.destination
        ambulance.destination = None
        ambulance.status = AmbulanceStatus.IDLE
        
        if self.verbose:
            print(f"\nAmbulance {amb_id} completed relocation to {ambulance.location} at {self.format_time(self.current_time)}")
    
    def _relocate_ambulances(self):
        """
        Use the relocation policy to relocate idle ambulances.
        """
        # Get idle ambulances
        idle_ambulances = [
            amb for amb in self.ambulances if amb.status == AmbulanceStatus.IDLE
        ]
        
        # If no idle ambulances, nothing to do
        if not idle_ambulances:
            return
            
        # Get busy ambulances
        busy_ambulances = [
            {'id': amb.id, 'location': amb.location}
            for amb in self.ambulances if amb.status != AmbulanceStatus.IDLE
        ]
        
        # Get relocations from policy or use default
        if self.relocation_policy:
            # Use provided policy
            relocations = self.relocation_policy.relocate_ambulances(
                [{'id': amb.id, 'location': amb.location} for amb in idle_ambulances],
                busy_ambulances
            )
        else:
            # Default policy: always return to base
            relocations = {}
            for amb in idle_ambulances:
                # Only relocate if not already at base
                if amb.location != self.base_location:
                    relocations[amb.id] = self.base_location
        
        # Apply relocations
        for amb in idle_ambulances:
            if amb.id in relocations:
                new_loc = relocations[amb.id]
                
                # Only relocate if not already at the location and location is valid
                if new_loc is not None and amb.location != new_loc:
                    amb.relocate(new_loc, self.current_time)
                    
                    # Schedule relocation completion
                    event_id = self._push_event(
                        amb.busy_until,
                        EventType.AMB_RELOCATION_COMPLETE,
                        {"amb_id": amb.id}
                    )
                    
                    if self.verbose:
                        print(f"\nAmbulance {amb.id} started relocating to {new_loc} at {self.format_time(self.current_time)}")
                        relocation_time = amb.busy_until - self.current_time
                        print(f"  Relocation time: {relocation_time/60:.1f} minutes")
    
    def print_statistics(self):
        """Print simulation statistics."""
        print("\n===== Simulation Statistics =====")
        print(f"Total calls in dataset: {self.total_calls}")
        print(f"Calls seen: {self.calls_seen}")
        print(f"Calls responded: {self.calls_responded} (delivered to hospital)")
        print(f"Timed-out calls: {self.timed_out_calls}")
        print(f"Missed calls: {self.missed_calls} (missed initially but not timed out)")
        
        # Verify totals
        total_accounted = self.calls_responded + self.missed_calls + self.timed_out_calls
        if total_accounted != self.calls_seen:
            print(f"\nWARNING: Call counting mismatch!")
            print(f"Total accounted for: {total_accounted}")
            print(f"Total calls seen: {self.calls_seen}")
            print(f"Difference: {abs(total_accounted - self.calls_seen)}")
        
        # Calculate response rate
        response_rate = (self.calls_responded / self.total_calls * 100) if self.total_calls > 0 else 0
        print(f"Response rate: {response_rate:.1f}%")
        
        if self.response_times:
            avg_rt = sum(self.response_times) / len(self.response_times)
            print(f"Average response time: {avg_rt/60:.1f} minutes")
            print(f"Min response time: {min(self.response_times)/60:.1f} minutes")
            print(f"Max response time: {max(self.response_times)/60:.1f} minutes")
        
        if self.call_total_times:
            total_times = list(self.call_total_times.values())
            avg_total = sum(total_times) / len(total_times)
            print(f"\nTotal time statistics (from call to drop-off):")
            print(f"Average total time: {avg_total/60:.1f} minutes")
            print(f"Min total time: {min(total_times)/60:.1f} minutes")
            print(f"Max total time: {max(total_times)/60:.1f} minutes")
        
        print("\nAmbulance Statistics:")
        for amb_id, count in self.ambulance_call_counts.items():
            print(f"  Ambulance {amb_id}: {count} calls")
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Convert seconds to a human-readable format."""
        days = int(seconds / (24 * 3600))
        hours = int((seconds % (24 * 3600)) / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"Day {days+1}, {hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _setup_events(self):
        """Schedule all call arrival events for all days."""
        # Clear the event queue (just in case)
        self.event_queue = []
        self.event_counter = 0
        self.cancelled_event_ids = set()  # Clear cancelled event IDs
        self.active_calls = {}
        self.call_timeouts = {}

        # Track all call IDs from the data
        self.synthetic_call_ids = set()
        self.calls_seen = 0  # Reset calls_seen counter

        # Populate the event queue with call arrival events from the calls_data
        for idx, row in self.call_data.iterrows():
            # Calculate absolute time from day and second_of_day
            day = row.get("day", 1)
            t = ((day - 1) * self.day_length) + row["second_of_day"]  # Time of the call
            
            # Use the actual row number (1-based) as the call ID
            call_id = idx + 1
            
            call_data = {
                "time": t,
                "day": day,
                "origin_node": row["origin_node"],
                "destination_node": row["destination_node"],
                "call_id": call_id,
                "intensity": row.get("intensity", 1.0)  # Include call priority if available
            }
            self.synthetic_call_ids.add(call_id)  # Track this call ID
            self._push_event(t, EventType.CALL_ARRIVAL, call_data)
            
        if self.verbose:
            num_days = self.call_data["day"].max() if "day" in self.call_data.columns else 1
            print(f"Scheduled {len(self.call_data)} calls over {num_days} days")
            print(f"Call IDs: {sorted(self.synthetic_call_ids)}")
            print(f"Total calls in data: {self.total_calls}")
            print(f"Unique call IDs: {len(self.synthetic_call_ids)}")
            
            # Verify call counting
            if len(self.synthetic_call_ids) != self.total_calls:
                print(f"\nWARNING: Call counting mismatch in setup!")
                print(f"Total calls in data: {self.total_calls}")
                print(f"Unique call IDs: {len(self.synthetic_call_ids)}")
                print(f"Difference: {abs(len(self.synthetic_call_ids) - self.total_calls)}")

    def run_day(self):
        """Run the simulation for a full day."""
        while True:
            event_time, event_type, _ = self.step()
            if event_type is None:
                break
        return {
            "calls_responded": self.calls_responded,
            "missed_calls": self.missed_calls,
            "timed_out_calls": self.timed_out_calls,
            "response_times": self.response_times,
        }

    def _print_day_statistics(self):
        """Print statistics for the day's simulation."""
        calls_processed = self.calls_responded
        
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        
        print("\n----- Day Statistics -----")
        print(f"Total calls: {self.total_calls}")
        print(f"Calls processed: {calls_processed}/{self.total_calls} "
             f"({100 * calls_processed / self.total_calls:.1f}%)")
        print(f"Missed calls: {self.missed_calls}")
        print(f"Timed out calls: {self.timed_out_calls}")
        print(f"Average response time: {avg_response_time:.1f} seconds "
             f"({avg_response_time / 60:.1f} minutes)")
        print("--------------------------\n")
