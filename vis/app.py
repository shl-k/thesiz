from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import sys
from pathlib import Path
import time
import threading
import pickle
import json
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator

app = Flask(__name__, 
    static_folder='static',
    template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

# Load data
print("Loading simulation data...")
graph = pickle.load(open(os.path.join(project_root, "data/processed/princeton_graph.gpickle"), "rb"))
calls = pd.read_csv(os.path.join(project_root, "data/processed/synthetic_calls.csv"))
path_cache = pickle.load(open(os.path.join(project_root, "data/matrices/path_cache.pkl"), "rb"))
node_to_idx = {int(k): v for k, v in json.load(open(os.path.join(project_root, "data/matrices/node_id_to_idx.json"))).items()}
idx_to_node = {int(k): int(v) for k, v in json.load(open(os.path.join(project_root, "data/matrices/idx_to_node_id.json"))).items()}
node_to_latlon = json.load(open(os.path.join(project_root, "data/matrices/node_to_lat_lon.json")))

# Initialize simulator
simulator = AmbulanceSimulator(
    graph=graph,
    call_data=calls,
    num_ambulances=3,
    base_location=241,
    hospital_node=1293,
    path_cache=path_cache,
    node_to_idx=node_to_idx,
    idx_to_node=idx_to_node,
    manual_mode=False,
    verbose=True
)

# Simulation control
simulation_running = False
simulation_thread = None

@app.route('/')
def index():
    print("Serving index.html")  # Debug print
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    global simulation_running
    simulation_running = False

@socketio.on('start_simulation')
def handle_start_simulation():
    global simulation_running, simulation_thread
    
    print("Starting simulation...")  # Debug print
    if not simulation_running:
        simulation_running = True
        simulator.initialize()
        print("Simulator initialized")  # Debug print
        
        # Don't start the simulation thread automatically
        # Just send the initial state
        state = get_simulation_state()
        print("Initial simulation state:", state)  # Debug print
        socketio.emit('simulation_state', state)
        
        # Send a message that simulation is ready for manual stepping
        socketio.emit('simulation_ready', {"message": "Simulation initialized. Use the Step button to advance."})

@socketio.on('pause_simulation')
def handle_pause_simulation():
    global simulation_running
    simulation_running = False
    socketio.emit('simulation_paused')

@socketio.on('step_simulation')
def handle_step_simulation():
    time, event_type, payload = simulator.step()
    state = get_simulation_state()
    socketio.emit('simulation_state', state)

def run_simulation():
    global simulation_running
    print("Simulation thread started")  # Debug print
    while simulation_running:
        step_time, event_type, payload = simulator.step()
        state = get_simulation_state()
        print("Simulation step - State:", state)  # Debug print
        socketio.emit('simulation_state', state)
        time.sleep(0.1)

def get_simulation_state():
    """Return a JSON-serialisable snapshot of the simulator state."""
    print("\n--- SIMULATION STATE UPDATE ---")
    print(f"Current time: {simulator.current_time:.2f} seconds")
    print(f"Active calls: {len(simulator.active_calls)}")
    print(f"Available ambulances: {sum(amb.status.value == 0 for amb in simulator.ambulances)}")
    
    ambulances = []
    for amb in simulator.ambulances:
        # Simplified logging - just show location and status
        print(f"Ambulance {amb.id}: Location {amb.location}, Status: {amb.status.name}")
        
        # Get current location
        if amb.path and len(amb.path) > 1:
            # If ambulance is moving, use the first node in its path
            current_node = amb.path[0]
        else:
            current_node = amb.location
            
        lat, lng = map(float, node_to_latlon[str(current_node)])

        # Get the full path if it exists, but don't show paths back to HQ
        path_coords = []
        if amb.path and amb.status.value != 4:  # Don't show path if ambulance is at hospital
            # Only show path if it's not going back to base
            if amb.destination != simulator.base_location:
                path_coords = [
                    list(map(float, node_to_latlon[str(node)]))
                    for node in amb.path
                ]
        elif amb.destination and amb.destination != simulator.base_location:
            # If there's a destination but no path, create a direct path (but not to base)
            path_coords = [
                list(map(float, node_to_latlon[str(amb.location)])),
                list(map(float, node_to_latlon[str(amb.destination)]))
            ]

        ambulances.append({
            "id": amb.id,
            "location": {"lat": lat, "lng": lng},
            "status": int(amb.status.value),
            "path": path_coords
        })

    active_calls = [
        {
            "id": int(cid),
            "location": {
                "lat": float(node_to_latlon[str(call["origin_node"])][0]),
                "lng": float(node_to_latlon[str(call["origin_node"])][1]),
            },
            "priority": int(call.get("priority", 1)),
        }
        for cid, call in simulator.active_calls.items()
    ]

    # Calculate patients served (calls responded)
    patients_served = simulator.calls_responded
    print(f"\nPatients served: {patients_served}")
    print(f"Missed calls: {simulator.missed_calls}")
    print("--- END STATE UPDATE ---\n")

    stats = {
        "active_calls": len(active_calls),
        "available_ambulances": sum(amb.status.value == 0 for amb in simulator.ambulances),
        "patients_served": patients_served,
        "missed_calls": simulator.missed_calls,
    }

    return {"ambulances": ambulances, "active_calls": active_calls, "statistics": stats}

if __name__ == '__main__':
    socketio.run(app, debug=False, port=5001) 