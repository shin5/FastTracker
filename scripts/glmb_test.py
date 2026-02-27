#!/usr/bin/env python3
"""GLMB algorithm test script - replicates GUI workflow via API."""
import json
import math
import sys
import requests

BASE_URL = "http://localhost:5000"

def compute_bearing(lat1, lon1, lat2, lon2):
    """Bearing from (lat1,lon1) to (lat2,lon2). Returns degrees, 0=North CW."""
    dLon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    y = math.sin(dLon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dLon)
    return math.degrees(math.atan2(y, x))

def auto_compute_search_sector(sensor_lat, sensor_lon, launch_lat, launch_lon, target_lat, target_lon):
    """Replicate GUI autoComputeSearchSector() logic."""
    bearing_to_launch = compute_bearing(sensor_lat, sensor_lon, launch_lat, launch_lon)
    bearing_to_target = compute_bearing(sensor_lat, sensor_lon, target_lat, target_lon)

    span = bearing_to_target - bearing_to_launch
    while span > 180: span -= 360
    while span < -180: span += 360

    center = bearing_to_launch + span / 2
    while center > 180: center -= 360
    while center < -180: center += 360

    margin = 20
    sector = abs(span) + margin * 2
    sector = max(sector, 30)
    sector = min(sector, 360)

    return round(center), round(sector)


def run_glmb_test():
    # Scenario parameters
    launch_lat, launch_lon = 39.0, 125.7
    target_lat, target_lon = 35.0, 132.0
    sensor_lat, sensor_lon = 34.79, 131.13

    # Step 1: Generate trajectory
    gen_payload = {
        "launch": {"lat": launch_lat, "lon": launch_lon},
        "target": {"lat": target_lat, "lon": target_lon},
        "params": {
            "missile_type": "ballistic",
            "boost_duration": 65.0,
            "boost_acceleration": 30.0,
            "launch_angle": 0.7,
            "dt": 0.5,
            "auto_adjust": True,
            "initial_mass": 20000.0,
            "fuel_fraction": 0.65,
            "specific_impulse": 250.0,
            "drag_coefficient": 0.3,
            "cross_section_area": 1.0,
            "enable_separation": True,
            "warhead_mass_fraction": 0.3,
        }
    }

    print("Step 1: Generating trajectory...")
    resp = requests.post(f"{BASE_URL}/api/generate", json=gen_payload, timeout=60)
    gen_data = resp.json()
    if not gen_data.get("success"):
        print(f"ERROR: Trajectory generation failed: {gen_data.get('error')}")
        sys.exit(1)

    trajectory = gen_data.get("trajectory", [])
    print(f"  Trajectory points: {len(trajectory)}, Duration: {trajectory[-1]['time']:.1f}s")

    # Step 2: Auto-compute search sector (same as GUI Auto Sector)
    search_center_deg, search_sector_deg = auto_compute_search_sector(
        sensor_lat, sensor_lon, launch_lat, launch_lon, target_lat, target_lon)
    search_sector_rad = search_sector_deg * math.pi / 180
    antenna_boresight_deg = search_center_deg  # boresight = search center

    print(f"  Auto Sector: center={search_center_deg}°, sector={search_sector_deg}°")
    print(f"  Antenna boresight: {antenna_boresight_deg}° (GUI bearing)")

    # Step 3: Run tracker with GLMB
    tracker_payload = {
        "trajectory": trajectory,
        "params": {
            "missile_type": "ballistic",
            "enable_separation": True,
            "warhead_mass_fraction": 0.3,

            # Sensor position
            "sensor_lat": sensor_lat,
            "sensor_lon": sensor_lon,

            # Radar parameters
            "radar_max_range": 0,  # auto
            "radar_min_range": 0,

            # Beam steering (auto-computed)
            "beam_width": 0.052,
            "num_beams": 50,
            "min_search_beams": 1,
            "search_sector": search_sector_rad,
            "search_center": search_center_deg,
            "antenna_boresight": antenna_boresight_deg,
            "range_resolution": 150,

            # Elevation scan
            "elev_scan_min": 0,
            "elev_scan_max": 20,
            "elev_bars_per_frame": 3,
            "elev_cycle_steps": 9,

            # Tracker
            "gate_threshold": 10,
            "confirm_hits": 4,
            "delete_misses": 10,
            "min_snr": 10,
            "max_jump_velocity": 10000,
            "min_init_distance": 30000,

            # GLMB
            "association_method": "glmb",
            "glmb_pd": 0.85,
            "glmb_k_best": 5,
            "glmb_max_hypotheses": 50,
            "glmb_score_decay": 0.9,
            "glmb_survival": 0.99,
            "glmb_birth_weight": 0.01,
            "glmb_clutter_density": 1e-6,
            "glmb_init_existence": 0.2,

            # Multi-run
            "num_runs": 10,
            "seed": 1,
        }
    }

    print(f"\nStep 3: Running tracker with GLMB (10 runs, seed=1)...")
    resp = requests.post(f"{BASE_URL}/api/run-tracker", json=tracker_payload, timeout=600)
    if resp.status_code != 200 or not resp.text:
        print(f"  HTTP {resp.status_code}: {resp.text[:2000]}")
        sys.exit(1)
    result = resp.json()

    if not result.get("success"):
        print(f"ERROR: {result.get('error')}")
        stdout = result.get("stdout", "")
        if stdout:
            print(f"STDOUT (last 2000):\n{stdout[-2000:]}")
        sys.exit(1)

    # Print evaluation summary
    summary = result.get("eval_summary", {})
    print("\n=== GLMB Evaluation Summary ===")
    key_metrics = [
        "Total Frames", "Avg GT Targets", "Avg Tracks",
        "Precision", "Recall", "F1 Score", "Mean OSPA",
        "True Positives", "False Positives", "False Negatives",
        "False Track Rate", "Track Purity",
        "Confirmation Rate", "Avg Track Duration",
        "Mostly Tracked (MT)", "Partially Tracked (PT)", "Mostly Lost (ML)",
    ]
    for k in key_metrics:
        v = summary.get(k)
        if v is not None:
            if isinstance(v, dict):
                print(f"  {k}: {v.get('mean', 'N/A')} +/- {v.get('std', 'N/A')}")
            else:
                print(f"  {k}: {v}")

    # Print stdout
    stdout = result.get("stdout", "")
    if stdout:
        lines = stdout.strip().split("\n")
        print(f"\n=== C++ Output (first 80 lines) ===")
        for line in lines[:80]:
            print(f"  {line}")

    return summary

if __name__ == "__main__":
    run_glmb_test()
