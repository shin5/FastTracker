"""
FastTracker Web Application

Flask-based web GUI for ballistic trajectory generation and visualization.
"""

import os
import sys
import json
import subprocess
import tempfile
import csv
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.trajectory_gen import latlon_to_meters

app = Flask(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
FASTTRACKER_EXE = PROJECT_ROOT / 'build' / 'Release' / 'fasttracker.exe'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate_trajectory():
    """Generate ballistic missile trajectory via C++ executable."""
    data = request.get_json()

    launch = data.get('launch', {})
    target = data.get('target', {})
    params = data.get('params', {})

    launch_lat = float(launch.get('lat', 39.0))
    launch_lon = float(launch.get('lon', 125.7))
    target_lat = float(target.get('lat', 35.7))
    target_lon = float(target.get('lon', 139.7))

    boost_duration = float(params.get('boost_duration', 65.0))
    launch_angle = float(params.get('launch_angle', 0.7))
    dt = float(params.get('dt', 0.5))
    auto_adjust = bool(params.get('auto_adjust', True))

    # Physics model parameters
    initial_mass = float(params.get('initial_mass', 20000.0))
    fuel_fraction = float(params.get('fuel_fraction', 0.65))
    specific_impulse = float(params.get('specific_impulse', 250.0))
    drag_coefficient = float(params.get('drag_coefficient', 0.3))
    cross_section_area = float(params.get('cross_section_area', 1.0))

    if not FASTTRACKER_EXE.exists():
        return jsonify({'success': False, 'error': f'FastTracker executable not found at {FASTTRACKER_EXE}'})

    # Convert lat/lon to meters (target = origin)
    dx, dy = latlon_to_meters(target_lat, target_lon, launch_lat, launch_lon)
    launch_x, launch_y = dx, dy
    target_x, target_y = 0.0, 0.0

    mode = 'auto-adjust' if auto_adjust else 'trajectory'

    cmd = [
        str(FASTTRACKER_EXE),
        '--mode', mode,
        '--launch-x', str(round(launch_x, 1)),
        '--launch-y', str(round(launch_y, 1)),
        '--target-x', str(round(target_x, 1)),
        '--target-y', str(round(target_y, 1)),
        '--target-lat', str(target_lat),
        '--target-lon', str(target_lon),
        '--launch-angle', str(launch_angle),
        '--boost-duration', str(boost_duration),
        '--framerate', str(round(1.0 / dt, 2)),
        '--initial-mass', str(initial_mass),
        '--fuel-fraction', str(fuel_fraction),
        '--specific-impulse', str(specific_impulse),
        '--drag-coefficient', str(drag_coefficient),
        '--cross-section', str(cross_section_area),
        '--distance-threshold', '10000',
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT)
        )

        if result.returncode != 0:
            return jsonify({
                'success': False,
                'error': f'Trajectory generation failed: {result.stderr or result.stdout}'
            })

        # Parse JSON summary from stdout
        summary = json.loads(result.stdout)

        # Read trajectory.csv
        traj_csv = read_csv_safe(PROJECT_ROOT / 'trajectory.csv')
        if not traj_csv:
            return jsonify({'success': False, 'error': 'trajectory.csv not generated'})

        # Build Plotly 3D data from CSV
        plotly_data = build_plotly_data_from_csv(traj_csv)

        # Build trajectory data (ground_truth-compatible format for run-tracker)
        trajectory_data = []
        for row in traj_csv:
            trajectory_data.append({
                'time': row['time'],
                'x': row['x'],
                'y': row['y'],
                'altitude': row['altitude'],
                'vx': row['vx'],
                'vy': row['vy'],
                'vz': row['vz'],
                'speed': row['speed'],
                'phase': row['phase'],
                'lat': row['lat'],
                'lon': row['lon'],
            })

        response = {
            'success': True,
            'trajectory': trajectory_data,
            'plotly': plotly_data,
            'summary': {
                'range_km': round(summary['range_km'], 1),
                'flight_duration': round(summary['flight_duration'], 1),
                'max_speed': round(summary['max_speed'], 1),
                'max_altitude': round(summary['max_altitude'], 1),
                'num_steps': summary['num_steps'],
            }
        }

        # Include auto-adjust results if available
        if 'adjusted' in summary:
            response['auto_adjust'] = {
                'adjusted': summary['adjusted'],
                'launch_angle': summary.get('adj_launch_angle'),
                'specific_impulse': summary.get('adj_specific_impulse'),
                'fuel_fraction': summary.get('adj_fuel_fraction'),
                'impact_distance_km': summary.get('impact_distance_km'),
                'search_iterations': summary.get('search_iterations'),
            }

        return jsonify(response)

    except json.JSONDecodeError as e:
        return jsonify({'success': False, 'error': f'Failed to parse C++ output: {e}'})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Trajectory generation timed out (60s)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/run-tracker', methods=['POST'])
def run_tracker():
    """Run FastTracker C++ simulation and return results."""
    data = request.get_json()
    trajectory = data.get('trajectory', [])
    params = data.get('params', {})

    if not trajectory:
        return jsonify({'success': False, 'error': 'No trajectory data'})

    if not FASTTRACKER_EXE.exists():
        return jsonify({'success': False, 'error': f'FastTracker executable not found at {FASTTRACKER_EXE}'})

    # Determine simulation parameters from trajectory
    duration = trajectory[-1]['time']
    dt = trajectory[1]['time'] - trajectory[0]['time'] if len(trajectory) > 1 else 0.1
    framerate = 1.0 / dt

    # Extract launch/target positions from trajectory data
    launch_x = trajectory[0]['x']
    launch_y = trajectory[0]['y']
    target_x = trajectory[-1]['x']
    target_y = trajectory[-1]['y']

    # Missile parameters from frontend
    launch_angle = float(params.get('launch_angle', 0.7))
    boost_duration_val = float(params.get('boost_duration', 65.0))
    boost_accel = float(params.get('boost_acceleration', 30.0))

    # Physics parameters from frontend
    initial_mass = float(params.get('initial_mass', 20000.0))
    fuel_fraction = float(params.get('fuel_fraction', 0.65))
    specific_impulse = float(params.get('specific_impulse', 250.0))
    drag_coefficient = float(params.get('drag_coefficient', 0.3))
    cross_section_area = float(params.get('cross_section_area', 1.0))

    # Sensor parameters from frontend
    sensor_lat = params.get('sensor_lat')
    sensor_lon = params.get('sensor_lon')
    radar_max_range_m = float(params.get('radar_max_range', 0))  # 0=auto
    radar_fov_rad = float(params.get('radar_fov', 2 * 3.14159265))

    # Convert sensor lat/lon to meters (target = origin)
    sensor_x_m = 0.0
    sensor_y_m = 0.0
    if sensor_lat is not None and sensor_lon is not None:
        # Get target lat/lon from trajectory (last point has lat/lon)
        target_lat_traj = trajectory[-1].get('lat')
        target_lon_traj = trajectory[-1].get('lon')
        if target_lat_traj is not None and target_lon_traj is not None:
            sensor_x_m, sensor_y_m = latlon_to_meters(
                float(target_lat_traj), float(target_lon_traj),
                float(sensor_lat), float(sensor_lon)
            )

    # Run FastTracker with missile + sensor + physics parameters
    cmd = [
        str(FASTTRACKER_EXE),
        '--scenario', 'single-ballistic',
        '--num-targets', '1',
        '--duration', str(round(duration, 1)),
        '--framerate', str(round(framerate, 1)),
        '--launch-x', str(round(launch_x, 1)),
        '--launch-y', str(round(launch_y, 1)),
        '--target-x', str(round(target_x, 1)),
        '--target-y', str(round(target_y, 1)),
        '--launch-angle', str(launch_angle),
        '--boost-duration', str(boost_duration_val),
        '--boost-accel', str(boost_accel),
        '--initial-mass', str(initial_mass),
        '--fuel-fraction', str(fuel_fraction),
        '--specific-impulse', str(specific_impulse),
        '--drag-coefficient', str(drag_coefficient),
        '--cross-section', str(cross_section_area),
        '--sensor-x', str(round(sensor_x_m, 1)),
        '--sensor-y', str(round(sensor_y_m, 1)),
        '--radar-max-range', str(round(radar_max_range_m, 1)),
        '--radar-fov', str(round(radar_fov_rad, 4)),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT)
        )

        # Read results
        results_data = read_csv_safe(PROJECT_ROOT / 'results.csv')
        track_data = read_csv_safe(PROJECT_ROOT / 'track_details.csv')
        gt_data = read_csv_safe(PROJECT_ROOT / 'ground_truth.csv')
        eval_data = read_csv_safe(PROJECT_ROOT / 'evaluation_results.csv')
        meas_data = read_csv_safe(PROJECT_ROOT / 'measurements.csv')

        # Parse evaluation summary from stdout
        eval_summary = parse_eval_summary(result.stdout)

        return jsonify({
            'success': True,
            'tracks': track_data,
            'ground_truth': gt_data,
            'measurements': meas_data,
            'evaluation': eval_data,
            'eval_summary': eval_summary,
            'sensor_x': sensor_x_m,
            'sensor_y': sensor_y_m,
            'stdout': result.stdout[-2000:] if result.stdout else '',
        })

    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Simulation timed out (120s)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def read_csv_safe(filepath):
    """Read CSV file and return as list of dicts."""
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric values
            converted = {}
            for k, v in row.items():
                try:
                    if '.' in v:
                        converted[k] = float(v)
                    else:
                        converted[k] = int(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
    return rows


def parse_eval_summary(stdout):
    """Parse evaluation metrics from FastTracker stdout."""
    summary = {}
    if not stdout:
        return summary

    lines = stdout.split('\n')
    for line in lines:
        line = line.strip()
        if 'RMSE' in line and ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    val = float(parts[1].strip().split()[0])
                    summary[key] = val
                except (ValueError, IndexError):
                    pass
        elif 'OSPA' in line and ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    val = float(parts[1].strip().split()[0])
                    summary['OSPA'] = val
                except (ValueError, IndexError):
                    pass

    return summary


def build_plotly_data_from_csv(traj_csv):
    """Build Plotly-compatible JSON for 3D trajectory visualization from CSV data."""
    n = len(traj_csv)

    times = [row['time'] for row in traj_csv]
    x_km = [row['x'] / 1000.0 for row in traj_csv]
    y_km = [row['y'] / 1000.0 for row in traj_csv]
    z_km = [row['altitude'] / 1000.0 for row in traj_csv]
    speeds = [row['speed'] for row in traj_csv]
    phases = [row['phase'] for row in traj_csv]
    latlons = [[row['lat'], row['lon']] for row in traj_csv]

    # Phase colors
    phase_colors = {
        'BOOST': 'rgb(255, 140, 0)',
        'MIDCOURSE': 'rgb(50, 200, 255)',
        'TERMINAL': 'rgb(255, 32, 80)',
    }
    colors = [phase_colors.get(p, 'rgb(200, 200, 200)') for p in phases]

    # Hover text
    hover = [
        f"Time: {times[i]:.1f}s<br>"
        f"Phase: {phases[i]}<br>"
        f"Pos: ({x_km[i]:.1f}km, {y_km[i]:.1f}km)<br>"
        f"Alt: {z_km[i]:.1f}km<br>"
        f"Speed: {speeds[i]:.0f} m/s ({speeds[i]*3.6:.0f} km/h)"
        for i in range(n)
    ]

    return {
        'trajectory': {
            'x': x_km,
            'y': y_km,
            'z': z_km,
            'text': hover,
            'speeds': speeds,
            'phases': phases,
            'colors': colors,
            'times': times,
        },
        'latlons': latlons,
    }


if __name__ == '__main__':
    print("=" * 50)
    print("FastTracker Web GUI")
    print("=" * 50)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"FastTracker exe: {FASTTRACKER_EXE} ({'found' if FASTTRACKER_EXE.exists() else 'NOT FOUND'})")
    print()
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)
