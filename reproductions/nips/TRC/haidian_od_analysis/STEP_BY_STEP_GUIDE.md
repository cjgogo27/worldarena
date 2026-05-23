# Step-by-Step Execution Guide

## Data Cleaning Rules

The trajectory data cleaning process includes the following steps:

### Step 2: Load and Filter Trajectory Data

**Cleaning Rules Applied:**

1. **Remove Null Values**
   - Remove records with missing `taxi_id`, `date_time`, `longitude`, or `latitude`
   - Ensures data completeness

2. **Coordinate Range Filter** (Haidian District + Buffer)
   - Longitude: 116.0 - 116.5
   - Latitude: 39.9 - 40.2
   - Removes points outside Haidian district and surrounding areas

3. **Remove Duplicate Records**
   - Based on: `taxi_id`, `date_time`, `longitude`, `latitude`
   - Eliminates exact duplicates

4. **Optional: Speed Filter** (Currently disabled)
   - Can be enabled to remove abnormal high-speed points
   - Requires calculating point-to-point speed

**Data Flow:**
```
Raw Data → Remove Nulls → Filter Coordinates → Remove Duplicates → Clean Data
```

## How to Run Step-by-Step

### Option 1: Interactive Mode (Recommended)

```bash
cd /data/alice/cjtest/TRC/haidian_od_analysis
python run_step_by_step.py
```

This will show you a menu:
```
1. Load region data
2. Load trajectory data
3. Spatial mapping (trajectory → regions)
4. Temporal mapping (time → 15-min slots)
5. Extract trips and generate OD matrices
6. Save final results
7. Run all remaining steps
0. Exit
```

**Recommendation:** Execute steps 1-4 first, then test step 5 with limited data.

### Option 2: Test Step 5 Separately

If step 5 is slow, test with limited data first:

```bash
# Test with 100 vehicles
python test_step5.py
```

This will:
- Load only first 100 vehicles from step4 results
- Extract trips quickly
- Show statistics and sample results
- Help identify any issues

### Option 3: Direct Python

Run steps individually in Python:

```python
from main import HaidianODAnalysisPipeline

config = {
    'shapefile_path': '/data/alice/cjtest/TRC/海淀区边界_110108_Shapefile_(poi86.com)/110108.shp',
    'region_mapping_path': '/data/alice/cjtest/TRC/haidian_od_analysis/config/region_mapping.csv',
    'trajectory_path': '/data/alice/cjtest/TRC/all_taxi_data.csv',
    'output_dir': '/data/alice/cjtest/TRC/haidian_od_analysis/output',
    'num_regions': 29,
    'interval_minutes': 15,
    'trip_time_threshold': 30,
}

pipeline = HaidianODAnalysisPipeline(config)

# Run one step at a time
pipeline.step1_load_and_prepare_regions()
# Check results in output/step1_*.csv and visualizations/

pipeline.step2_load_and_filter_trajectory()
# Check results in output/step2_*.csv

pipeline.step3_spatial_mapping()
# Check results in output/step3_*.csv

pipeline.step4_temporal_mapping()
# Check results in output/step4_*.csv

# For step 5, monitor progress carefully
pipeline.step5_extract_trips_and_generate_od()

pipeline.step6_save_results()
```

## Performance Optimization for Step 5

Step 5 (Extract trips) is the most computationally intensive step.

**Expected Performance:**
- ~100 vehicles: 1-2 minutes
- ~1,000 vehicles: 10-20 minutes
- ~10,000 vehicles: 2-3 hours

**Optimization Tips:**

1. **Limit data during testing:**
   ```python
   # In step2, only read first N days
   df_traj = df_traj[df_traj['date'] == '2008-02-02']
   ```

2. **Increase time threshold:**
   ```python
   config['trip_time_threshold'] = 60  # 60 minutes instead of 30
   ```
   This reduces the number of trips extracted but runs faster.

3. **Filter by vehicle count:**
   ```python
   # Keep only vehicles with reasonable number of records
   vehicle_counts = df_traj.groupby('taxi_id').size()
   valid_vehicles = vehicle_counts[
       (vehicle_counts >= 10) & (vehicle_counts <= 1000)
   ].index
   df_traj = df_traj[df_traj['taxi_id'].isin(valid_vehicles)]
   ```

## Checking Results After Each Step

After each step, check:

### Step 1 - Region Data
- `output/step1_regions.csv` - Should have 29 regions
- `output/visualizations/step1_regions_map.png` - Map of Haidian

### Step 2 - Trajectory Data
- `output/step2_cleaned_trajectory_sample.csv` - Sample of cleaned data
- Check: longitude in [116.0, 116.5], latitude in [39.9, 40.2]
- `output/visualizations/step2_trajectory_distribution.png` - Should show points clustered in Haidian

### Step 3 - Spatial Mapping
- `output/step3_mapped_trajectory_sample.csv` - Should have `region_id` column
- `output/step3_region_point_statistics.csv` - Point count per region
- Check: Most regions should have some points
- `output/visualizations/step3_region_distribution.png` - Bar chart showing distribution

### Step 4 - Temporal Mapping
- `output/step4_time_mapped_sample.csv` - Should have `time_slot` column (0-95)
- `output/step4_time_slot_statistics.csv` - Records per time slot
- Check: All time slots should be 0-95
- `output/visualizations/step4_time_distribution.png` - 24-hour pattern

### Step 5 - OD Matrix
- `output/step5_trips.csv` - Trip records with origin, destination, duration
- Check: duration_minutes should be reasonable (typically 5-60 minutes)
- `output/visualizations/step5_trip_analysis.png` - Trip statistics
- `output/visualizations/step5_od_heatmaps.png` - Flow patterns by time

## Troubleshooting

### Issue: Step 5 is stuck

**Symptoms:** Progress bar at 0% and not moving

**Solutions:**
1. **Check CPU usage** - If high (>90%), it's working, just slow
2. **Test with small data first** - Use `test_step5.py`
3. **Check memory** - If system is swapping, reduce data size
4. **Reduce vehicles** - Filter to fewer vehicles in step 2

### Issue: Low region mapping rate (Step 3)

**Symptoms:** Most points show `region_id = NaN`

**Solutions:**
1. Check shapefile coordinate system
2. Verify trajectory points are in Haidian district
3. Increase buffer zone in coordinate filter

### Issue: Memory error

**Solutions:**
1. Process data in chunks
2. Reduce sample size in step 2
3. Process only specific dates

## Expected Outputs

**All outputs use English labels** for:
- Chart titles
- Axis labels  
- Legend text
- File descriptions

This ensures consistency and readability for international collaboration.

---

**Quick Start:**
```bash
# 1. Run interactively
python run_step_by_step.py

# 2. Test step 5 with limited data first
python test_step5.py

# 3. Once confirmed working, run full pipeline
python main.py
```
