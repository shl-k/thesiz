"""
Test script to verify response times from the path cache.
"""

import pickle
import numpy as np
import pandas as pd
import statistics

# Load the path cache
print("Loading path cache...")
with open("data/matrices/path_cache.pkl", "rb") as f:
    PATH_CACHE = pickle.load(f)

# Load synthetic calls to get realistic origin nodes
print("Loading call data...")
calls_df = pd.read_csv('data/processed/synthetic_calls.csv')
call_origins = calls_df['origin_node'].unique()

# Load critical nodes
pfars_node = 241  # PFARS HQ (ambulance base)

# 1. Calculate response times from base to all call locations
base_to_call_times = []
for origin in call_origins:
    if pfars_node in PATH_CACHE and origin in PATH_CACHE[pfars_node]:
        time = PATH_CACHE[pfars_node][origin]['travel_time']
        distance = PATH_CACHE[pfars_node][origin]['length']
        speed = distance / time if time > 0 else 0
        base_to_call_times.append({
            'origin': origin,
            'time_seconds': time,
            'time_minutes': time / 60,
            'distance_meters': distance,
            'speed_mps': speed,
            'speed_kmph': speed * 3.6
        })

# 2. Calculate random response times (ambulance could be anywhere)
random_response_times = []
for origin in call_origins:
    # Sample 5 random starting positions for each call
    for _ in range(5):
        random_start = np.random.choice(list(PATH_CACHE.keys()))
        if origin in PATH_CACHE[random_start]:
            time = PATH_CACHE[random_start][origin]['travel_time']
            distance = PATH_CACHE[random_start][origin]['length']
            speed = distance / time if time > 0 else 0
            random_response_times.append({
                'start': random_start,
                'destination': origin,
                'time_seconds': time,
                'time_minutes': time / 60,
                'distance_meters': distance,
                'speed_mps': speed,
                'speed_kmph': speed * 3.6
            })

# Process and print statistics
if base_to_call_times:
    base_times = [entry['time_seconds'] for entry in base_to_call_times]
    base_times_min = [t / 60 for t in base_times]
    base_distances = [entry['distance_meters'] for entry in base_to_call_times]
    base_speeds = [entry['speed_kmph'] for entry in base_to_call_times]
    
    print("\n===== RESPONSE TIMES FROM BASE =====")
    print(f"Count: {len(base_times)}")
    print(f"Min time: {min(base_times_min):.2f} minutes")
    print(f"Max time: {max(base_times_min):.2f} minutes")
    print(f"Mean time: {statistics.mean(base_times_min):.2f} minutes")
    print(f"Median time: {statistics.median(base_times_min):.2f} minutes")
    print(f"90th percentile: {np.percentile(base_times_min, 90):.2f} minutes")
    print(f"Average distance: {statistics.mean(base_distances):.2f} meters")
    print(f"Average speed: {statistics.mean(base_speeds):.2f} km/h")

if random_response_times:
    random_times = [entry['time_seconds'] for entry in random_response_times]
    random_times_min = [t / 60 for t in random_times]
    random_distances = [entry['distance_meters'] for entry in random_response_times]
    random_speeds = [entry['speed_kmph'] for entry in random_response_times]
    
    print("\n===== RANDOM RESPONSE TIMES =====")
    print(f"Count: {len(random_times)}")
    print(f"Min time: {min(random_times_min):.2f} minutes")
    print(f"Max time: {max(random_times_min):.2f} minutes")
    print(f"Mean time: {statistics.mean(random_times_min):.2f} minutes")
    print(f"Median time: {statistics.median(random_times_min):.2f} minutes")
    print(f"90th percentile: {np.percentile(random_times_min, 90):.2f} minutes")
    print(f"Average distance: {statistics.mean(random_distances):.2f} meters")
    print(f"Average speed: {statistics.mean(random_speeds):.2f} km/h")

# Print extreme examples
print("\n===== FASTEST RESPONSES =====")
sorted_responses = sorted(base_to_call_times, key=lambda x: x['time_seconds'])
for resp in sorted_responses[:5]:
    print(f"Origin: {resp['origin']}, Time: {resp['time_minutes']:.2f} min, Distance: {resp['distance_meters']:.1f}m, Speed: {resp['speed_kmph']:.1f} km/h")

print("\n===== SLOWEST RESPONSES =====")
for resp in sorted_responses[-5:]:
    print(f"Origin: {resp['origin']}, Time: {resp['time_minutes']:.2f} min, Distance: {resp['distance_meters']:.1f}m, Speed: {resp['speed_kmph']:.1f} km/h") 