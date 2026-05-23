#!/usr/bin/env python3
"""
Quick test script for Step 5 - Extract trips with limited data
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from od_matrix_generator import ODMatrixGenerator

def test_trip_extraction():
    """Test trip extraction with a small sample"""
    
    print("Testing trip extraction with limited data...")
    
    # Load the time-mapped trajectory data
    step4_file = 'output/step4_time_mapped_sample.csv'
    
    if not os.path.exists(step4_file):
        print(f"Error: {step4_file} not found. Please run steps 1-4 first.")
        return
    
    print(f"\nLoading data from {step4_file}...")
    df = pd.read_csv(step4_file)
    
    print(f"Loaded {len(df)} records")
    print(f"Unique vehicles: {df['taxi_id'].nunique()}")
    
    # Limit to first 100 vehicles for testing
    vehicle_ids = df['taxi_id'].unique()[:100]
    df_test = df[df['taxi_id'].isin(vehicle_ids)].copy()
    
    print(f"\nTesting with {len(vehicle_ids)} vehicles")
    print(f"Test data: {len(df_test)} records")
    
    # Initialize generator
    generator = ODMatrixGenerator(num_regions=29, interval_minutes=15)
    
    # Extract trips
    print("\nExtracting trips...")
    trips = generator.extract_trips(
        df_test,
        time_col='date_time',
        region_col='region_id',
        vehicle_col='taxi_id',
        time_threshold_minutes=30
    )
    
    print(f"\n✓ Extraction completed!")
    print(f"Total trips found: {len(trips)}")
    
    if len(trips) > 0:
        print(f"\nTrip statistics:")
        print(f"  Average duration: {trips['duration_minutes'].mean():.2f} minutes")
        print(f"  Median duration: {trips['duration_minutes'].median():.2f} minutes")
        print(f"  Unique OD pairs: {len(trips.groupby(['origin_region', 'dest_region']))}")
        
        # Save test results
        test_output = 'output/step5_test_trips.csv'
        trips.to_csv(test_output, index=False, encoding='utf-8-sig')
        print(f"\n✓ Test results saved to: {test_output}")
        
        # Show sample trips
        print(f"\nSample trips:")
        print(trips[['vehicle_id', 'origin_region', 'dest_region', 'duration_minutes']].head(10))
    
    return trips


if __name__ == "__main__":
    test_trip_extraction()
