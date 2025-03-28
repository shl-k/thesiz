"""
Test script to validate synthetic demand generation against historical data
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.geo_utils import osm_graph
from src.data.princeton_data_prep import sparsify_graph, generate_demand_with_temporal_pattern

# Constants
MEDICAL_TRIPS_FILE = 'princeton_data/medical_trips.csv'
OUTPUT_DIR = 'princeton_data/test_results'

def test_synthetic_demand(G, medical_trips_file=MEDICAL_TRIPS_FILE):
    """
    Run tests to validate synthetic demand generation against historical data
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading historical data...")
    historical_trips = pd.read_csv(MEDICAL_TRIPS_FILE)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    synthetic_calls = generate_demand_with_temporal_pattern(G)
    
    # Test 1: Check total daily calls
    print("\nTest 1: Daily Call Volume")
    # Get unique days in historical data
    historical_days = pd.to_datetime(historical_trips['datetime']).dt.date.nunique()
    if historical_days == 0:  # Handle case where datetime might not be parsed correctly
        historical_days = len(historical_trips) / 8  # Assume 8 calls per day if can't determine
    
    historical_daily_avg = len(historical_trips) / historical_days
    synthetic_daily_avg = len(synthetic_calls) / synthetic_calls['day'].nunique()
    print(f"Historical daily average: {historical_daily_avg:.2f} calls")
    print(f"Synthetic daily average: {synthetic_daily_avg:.2f} calls")
    print(f"Target daily average: 8.00 calls")
    
    # Test 2: Compare hourly distributions
    print("\nTest 2: Hourly Distribution")
    
    # Extract hour from historical data
    if 'hour' in historical_trips.columns:
        historical_hourly = historical_trips.groupby('hour').size()
    else:
        # If hour not directly available, derive it from datetime
        historical_trips['hour'] = pd.to_datetime(historical_trips['datetime']).dt.hour
        historical_hourly = historical_trips.groupby('hour').size()
    
    historical_hourly_frac = historical_hourly / historical_hourly.sum()
    
    # Extract hour from synthetic minute-level data
    synthetic_calls['hour'] = synthetic_calls['minute_of_day'] // 60
    synthetic_hourly = synthetic_calls.groupby('hour').size()
    synthetic_hourly_frac = synthetic_hourly / synthetic_hourly.sum()
    
    print("\nHourly Distribution Comparison:")
    # Ensure all hours are represented in both distributions
    full_hours = range(24)
    historical_hourly_full = np.zeros(24)
    synthetic_hourly_full = np.zeros(24)
    
    for hour in full_hours:
        historical_hourly_full[hour] = historical_hourly_frac.get(hour, 0)
        synthetic_hourly_full[hour] = synthetic_hourly_frac.get(hour, 0)
    
    comparison_df = pd.DataFrame({
        'Historical': historical_hourly_full,
        'Synthetic': synthetic_hourly_full
    }, index=range(24))
    print(comparison_df)
    
    # Calculate KL divergence for hourly distributions (with small epsilon to avoid log(0))
    epsilon = 1e-10
    kl_div = np.sum(historical_hourly_full * np.log((historical_hourly_full + epsilon) / (synthetic_hourly_full + epsilon)))
    print(f"\nKL Divergence (hourly distributions): {kl_div:.4f}")
    
    # Test 3: Spatial distribution tests
    print("\nTest 3: Spatial Distribution")
    
    # Map historical locations to nodes
    node_coords = np.array([[G.nodes[node]['y'], G.nodes[node]['x']] 
                           for node in G.nodes()])
    node_ids = list(G.nodes())
    historical_node_counts = np.zeros(len(G.nodes()))
    
    origins = historical_trips[['origin_lat', 'origin_lon']].values
    for origin in origins:
        distances = np.sqrt(np.sum((node_coords - origin)**2, axis=1))
        nearest_node_idx = np.argmin(distances)
        historical_node_counts[nearest_node_idx] += 1
    
    # Get synthetic node counts - using origin_node_idx
    synthetic_node_counts = np.zeros(len(G.nodes()))
    for node_idx in synthetic_calls['origin_node_idx']:
        synthetic_node_counts[int(node_idx)] += 1
    
    # Normalize counts
    historical_spatial = historical_node_counts / np.sum(historical_node_counts)
    synthetic_spatial = synthetic_node_counts / np.sum(synthetic_node_counts)
    
    # Calculate spatial statistics
    print("\nSpatial Statistics:")
    print(f"Historical active nodes: {np.sum(historical_spatial > 0)}")
    print(f"Synthetic active origin nodes: {np.sum(synthetic_spatial > 0)}")
    print(f"Historical unique nodes: {len(historical_trips['origin_lat'].unique())}")
    print(f"Synthetic unique origin nodes: {len(synthetic_calls['origin_node'].unique())}")
    print(f"Destination node: {synthetic_calls['destination_node'].iloc[0]}")
    
    # Calculate spatial correlation (handle case where one might be constant)
    if np.std(historical_spatial) > 0 and np.std(synthetic_spatial) > 0:
        correlation = np.corrcoef(historical_spatial, synthetic_spatial)[0,1]
    else:
        correlation = 0
    print(f"Spatial correlation: {correlation:.4f}")
    
    # Test 4: Plot comparisons
    print("\nTest 4: Generating Comparison Plots...")
    
    # Hourly distribution plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(24), historical_hourly_full, 
             alpha=0.5, label='Historical', width=0.4)
    plt.bar([h+0.4 for h in range(24)], synthetic_hourly_full, 
             alpha=0.5, label='Synthetic', width=0.4)
    plt.xlabel('Hour of Day')
    plt.ylabel('Fraction of Daily Calls')
    plt.title('Comparison of Hourly Call Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(24))
    plt.savefig(f'{OUTPUT_DIR}/demand_hourly_comparison.png')
    plt.close()
    
    # Spatial distribution comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Historical heatmap
    sc1 = ax1.scatter(node_coords[:,1], node_coords[:,0], 
                     c=historical_spatial, cmap='plasma',
                     s=50, alpha=0.6)
    ax1.set_title('Historical Spatial Distribution')
    plt.colorbar(sc1, ax=ax1)
    
    # Synthetic heatmap
    sc2 = ax2.scatter(node_coords[:,1], node_coords[:,0], 
                     c=synthetic_spatial, cmap='plasma',
                     s=50, alpha=0.6)
    ax2.set_title('Synthetic Spatial Distribution')
    plt.colorbar(sc2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/demand_spatial_comparison.png')
    plt.close()
    
    # Test 5: Additional visualizations
    print("\nTest 5: Additional Visualizations...")
    
    # Plot minute-of-day distribution
    minute_bins = np.linspace(0, 24*60, 25)  # 24 bins, one per hour
    synthetic_minute_counts = synthetic_calls['minute_of_day'].values
    
    plt.figure(figsize=(15, 6))
    plt.hist(synthetic_minute_counts, bins=minute_bins, alpha=0.7)
    plt.xlabel('Minute of Day')
    plt.ylabel('Number of Calls')
    plt.title('Distribution of Calls by Minute of Day')
    plt.xticks(minute_bins[::2], [f"{int(m//60)}:00" for m in minute_bins[::2]])
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{OUTPUT_DIR}/minute_distribution.png')
    plt.close()
    
    # Plot call intensity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(synthetic_calls['intensity'], bins=range(1, max(synthetic_calls['intensity'])+2))
    plt.title('Distribution of Call Intensity Values')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.xticks(range(1, max(synthetic_calls['intensity'])+1))
    plt.savefig(f'{OUTPUT_DIR}/call_intensity_distribution.png')
    plt.close()
    
    # Plot calls by day
    plt.figure(figsize=(10, 6))
    day_counts = synthetic_calls.groupby('day').size()
    plt.bar(day_counts.index, day_counts.values)
    plt.axhline(y=8, color='r', linestyle='--', label='Target (8 calls/day)')
    plt.title('Calls Per Day in Synthetic Data')
    plt.xlabel('Day')
    plt.ylabel('Number of Calls')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}/calls_by_day.png')
    plt.close()
    
    # Plot map with hospital location highlighted
    plt.figure(figsize=(12, 10))
    # Plot all nodes
    plt.scatter(node_coords[:,1], node_coords[:,0], c='lightgray', s=20, alpha=0.5)
    
    # Get hospital coordinates
    hospital_node = synthetic_calls['destination_node'].iloc[0]
    hospital_idx = node_ids.index(hospital_node)
    hospital_coords = node_coords[hospital_idx]
    
    # Plot hospital
    plt.scatter(hospital_coords[1], hospital_coords[0], c='red', s=200, marker='*', 
                label='Hospital', edgecolor='black', zorder=10)
    
    # Plot common origin locations
    top_origins = synthetic_calls['origin_node'].value_counts().head(10).index
    for origin_node in top_origins:
        origin_idx = node_ids.index(origin_node)
        origin_coords = node_coords[origin_idx]
        plt.scatter(origin_coords[1], origin_coords[0], c='blue', s=100, alpha=0.7,
                   edgecolor='black', zorder=5)
    
    plt.title('Princeton Map with Hospital and Top 10 Origin Locations')
    plt.legend(['Network Nodes', 'Hospital', 'Common Origins'])
    plt.savefig(f'{OUTPUT_DIR}/hospital_map.png')
    plt.close()
    
    # Results summary
    results = {
        'daily_calls': float(synthetic_daily_avg),
        'hourly_correlation': float(np.corrcoef(historical_hourly_full, synthetic_hourly_full)[0,1]),
        'spatial_correlation': float(correlation),
        'kl_divergence': float(kl_div),
        'active_nodes': {
            'historical': int(np.sum(historical_spatial > 0)),
            'synthetic_origins': int(np.sum(synthetic_spatial > 0))
        },
        'unique_nodes': {
            'synthetic_origins': int(len(synthetic_calls['origin_node'].unique()))
        },
        'intensity_stats': {
            'min': float(synthetic_calls['intensity'].min()),
            'max': float(synthetic_calls['intensity'].max()),
            'mean': float(synthetic_calls['intensity'].mean())
        },
        'days_generated': int(synthetic_calls['day'].nunique()),
        'hospital_node': int(hospital_node)
    }
    
    # Save results to JSON
    with open(f'{OUTPUT_DIR}/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == "__main__":
    # Get Princeton graph
    print('Getting Princeton graph...')
    G_pton_original = osm_graph(location='Princeton, NJ', network_type='drive')
    
    # Sparsify the graph
    print('\nSparsifying graph...')
    G_pton = sparsify_graph(G_pton_original, min_edge_length=30, simplify=True)
    
    # Run the tests
    print("\nRunning validation tests...")
    test_results = test_synthetic_demand(G_pton)
    
    print("\nTest Results Summary:")
    print(json.dumps(test_results, indent=2))
    print(f"\nDetailed results and visualizations saved to {OUTPUT_DIR}") 