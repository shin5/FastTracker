"""
FastTracker Web Application

Flask-based web GUI for ballistic trajectory generation and visualization.
"""

import os
import sys
import json
import math
import re
import subprocess
import tempfile
import csv
import threading
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.trajectory_gen import latlon_to_meters

app = Flask(__name__)


@app.after_request
def add_no_cache_headers(response):
    """Prevent browser caching of HTML/JS/JSON to ensure latest data is always served."""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

import platform
if platform.system() == 'Windows':
    FASTTRACKER_EXE = PROJECT_ROOT / 'build' / 'Release' / 'fasttracker.exe'
else:
    FASTTRACKER_EXE = PROJECT_ROOT / 'build' / 'fasttracker'

# Async job tracking: job_id -> job state dict
_jobs = {}


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

    # Missile type (ballistic or hgv)
    missile_type = params.get('missile_type', 'ballistic')

    # Physics model parameters
    initial_mass = float(params.get('initial_mass', 20000.0))
    fuel_fraction = float(params.get('fuel_fraction', 0.65))
    specific_impulse = float(params.get('specific_impulse', 250.0))
    drag_coefficient = float(params.get('drag_coefficient', 0.3))
    cross_section_area = float(params.get('cross_section_area', 1.0))

    # Auto-adjust options
    target_max_altitude = float(params.get('target_max_altitude', 0.0))
    lock_angle = bool(params.get('lock_angle', False))
    lock_isp = bool(params.get('lock_isp', False))
    lock_fuel = bool(params.get('lock_fuel', False))

    # Separation options
    enable_separation = bool(params.get('enable_separation', False))
    warhead_mass_fraction = float(params.get('warhead_mass_fraction', 0.3))

    # HGV-specific parameters
    cruise_altitude = float(params.get('cruise_altitude', 40000.0))
    glide_ratio = float(params.get('glide_ratio', 4.0))
    terminal_dive_range = float(params.get('terminal_dive_range', 20000.0))
    pullup_duration = float(params.get('pullup_duration', 30.0))
    bank_angle_max = float(params.get('bank_angle_max', 1.0))
    num_skips = int(params.get('num_skips', 0))

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
        '--missile-type', missile_type,
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
        '--target-max-altitude', str(target_max_altitude),
    ]

    # HGV-specific parameters
    if missile_type == 'hgv':
        cmd.extend([
            '--cruise-altitude', str(cruise_altitude),
            '--glide-ratio', str(glide_ratio),
            '--terminal-dive-range', str(terminal_dive_range),
            '--pullup-duration', str(pullup_duration),
            '--bank-angle-max', str(bank_angle_max),
        ])
        if num_skips > 0:
            cmd.extend(['--num-skips', str(num_skips)])

    if lock_angle:
        cmd.append('--lock-angle')
    if lock_isp:
        cmd.append('--lock-isp')
    if lock_fuel:
        cmd.append('--lock-fuel')
    if enable_separation:
        cmd.append('--enable-separation')
        cmd.extend(['--warhead-mass-fraction', str(warhead_mass_fraction)])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT), encoding='utf-8', errors='replace'
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
            entry = {
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
            }
            if 'object_id' in row:
                entry['object_id'] = row['object_id']
            trajectory_data.append(entry)

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
                **({'separation_time': round(summary['separation_time'], 1)} if 'separation_time' in summary else {}),
                **({'booster_impact_time': round(summary['booster_impact_time'], 1)} if 'booster_impact_time' in summary else {}),
            }
        }

        # Include auto-adjust results if available
        if 'adjusted' in summary:
            response['auto_adjust'] = {
                'adjusted': summary['adjusted'],
                'launch_angle': summary.get('adj_launch_angle'),
                'specific_impulse': summary.get('adj_specific_impulse'),
                'fuel_fraction': summary.get('adj_fuel_fraction'),
                'fuel_mass': summary.get('adj_fuel_mass'),
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


def _build_tracker_cmd(data):
    """Build FastTracker subprocess command from request data.

    Returns (cmd, sensor_x_m, sensor_y_m) or raises on error.
    """
    trajectory = data.get('trajectory', [])
    params = data.get('params', {})

    if not trajectory:
        raise ValueError('No trajectory data')
    if not FASTTRACKER_EXE.exists():
        raise FileNotFoundError(f'FastTracker executable not found at {FASTTRACKER_EXE}')

    warhead_traj = [t for t in trajectory if t.get('object_id', 0) == 0]
    if not warhead_traj:
        warhead_traj = trajectory

    duration = warhead_traj[-1]['time']
    dt = trajectory[1]['time'] - trajectory[0]['time'] if len(trajectory) > 1 else 0.1
    framerate = 1.0 / dt

    launch_x = warhead_traj[0]['x']
    launch_y = warhead_traj[0]['y']
    target_x = warhead_traj[-1]['x']
    target_y = warhead_traj[-1]['y']

    missile_type = params.get('missile_type', 'ballistic')
    launch_angle = float(params.get('launch_angle', 0.7))
    boost_duration_val = float(params.get('boost_duration', 65.0))
    boost_accel = float(params.get('boost_acceleration', 30.0))
    initial_mass = float(params.get('initial_mass', 20000.0))
    fuel_fraction = float(params.get('fuel_fraction', 0.65))
    specific_impulse = float(params.get('specific_impulse', 250.0))
    drag_coefficient = float(params.get('drag_coefficient', 0.3))
    cross_section_area = float(params.get('cross_section_area', 1.0))

    sensor_lat = params.get('sensor_lat')
    sensor_lon = params.get('sensor_lon')
    radar_max_range_m = float(params.get('radar_max_range', 0))
    radar_fov_rad = float(params.get('radar_fov', 2 * 3.14159265))
    enable_sep = bool(params.get('enable_separation', False))
    warhead_mf = float(params.get('warhead_mass_fraction', 0.3))

    pfa = params.get('pfa')
    pd_ref = params.get('pd_ref')
    pd_ref_range_km = params.get('pd_ref_range_km')
    range_noise = params.get('range_noise')
    azimuth_noise = params.get('azimuth_noise')
    elevation_noise = params.get('elevation_noise')
    doppler_noise = params.get('doppler_noise')

    gate_threshold = params.get('gate_threshold')
    confirm_hits = params.get('confirm_hits')
    delete_misses = params.get('delete_misses')
    min_snr = params.get('min_snr')
    process_pos_noise = params.get('process_pos_noise')
    process_vel_noise = params.get('process_vel_noise')
    process_acc_noise = params.get('process_acc_noise')

    num_runs = int(params.get('num_runs', 1))
    random_seed = int(params.get('seed', 0))
    cluster_count = int(params.get('cluster_count', 0))
    cluster_spread = float(params.get('cluster_spread', 5000.0))
    launch_time_spread = float(params.get('launch_time_spread', 5.0))

    beam_width_rad = float(params.get('beam_width', 0.052))
    num_beams = int(params.get('num_beams', 10))
    min_search_beams = int(params.get('min_search_beams', 1))
    track_confirmed_only = bool(params.get('track_confirmed_only', False))
    search_sector_rad = float(params.get('search_sector', -1))
    search_center_deg = float(params.get('search_center', 0))
    search_center_rad = math.radians(90 - search_center_deg)
    antenna_boresight_deg = float(params.get('antenna_boresight', 0))
    antenna_boresight_rad = math.radians(90 - antenna_boresight_deg)
    search_elevation_deg = float(params.get('search_elevation', 0))
    search_elevation_rad = math.radians(search_elevation_deg)

    sensor_x_m = 0.0
    sensor_y_m = 0.0
    if sensor_lat is not None and sensor_lon is not None:
        target_lat_traj = warhead_traj[-1].get('lat')
        target_lon_traj = warhead_traj[-1].get('lon')
        if target_lat_traj is not None and target_lon_traj is not None:
            sensor_x_m, sensor_y_m = latlon_to_meters(
                float(target_lat_traj), float(target_lon_traj),
                float(sensor_lat), float(sensor_lon)
            )

    hgv_cruise_alt = float(params.get('cruise_altitude', 40000.0))
    hgv_glide_ratio = float(params.get('glide_ratio', 4.0))
    hgv_terminal_dive = float(params.get('terminal_dive_range', 20000.0))
    hgv_pullup_dur = float(params.get('pullup_duration', 30.0))
    hgv_bank_max = float(params.get('bank_angle_max', 1.0))
    hgv_num_skips = int(params.get('num_skips', 0))

    cmd = [
        str(FASTTRACKER_EXE),
        '--scenario', 'single-ballistic',
        '--missile-type', missile_type,
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

    if enable_sep:
        cmd.append('--enable-separation')
        cmd.extend(['--warhead-mass-fraction', str(warhead_mf)])

    if missile_type == 'hgv':
        cmd.extend([
            '--cruise-altitude', str(hgv_cruise_alt),
            '--glide-ratio', str(hgv_glide_ratio),
            '--terminal-dive-range', str(hgv_terminal_dive),
            '--pullup-duration', str(hgv_pullup_dur),
            '--bank-angle-max', str(hgv_bank_max),
        ])
        if hgv_num_skips > 0:
            cmd.extend(['--num-skips', str(hgv_num_skips)])

    if pfa is not None and float(pfa) > 0:
        cmd.extend(['--pfa', str(float(pfa))])
    if pd_ref is not None and 0 < float(pd_ref) < 1:
        cmd.extend(['--pd-ref', str(float(pd_ref))])
    if pd_ref_range_km is not None and float(pd_ref_range_km) > 0:
        cmd.extend(['--pd-ref-range', str(float(pd_ref_range_km) * 1000.0)])
    if range_noise is not None:
        cmd.extend(['--range-noise', str(float(range_noise))])
    if azimuth_noise is not None:
        cmd.extend(['--azimuth-noise', str(float(azimuth_noise))])
    if elevation_noise is not None:
        cmd.extend(['--elevation-noise', str(float(elevation_noise))])
    if doppler_noise is not None:
        cmd.extend(['--doppler-noise', str(float(doppler_noise))])

    if gate_threshold is not None:
        cmd.extend(['--gate-threshold', str(float(gate_threshold))])
    if confirm_hits is not None:
        cmd.extend(['--confirm-hits', str(int(confirm_hits))])
    if delete_misses is not None:
        cmd.extend(['--delete-misses', str(int(delete_misses))])
    if min_snr is not None:
        cmd.extend(['--min-snr', str(float(min_snr))])
    if process_pos_noise is not None:
        cmd.extend(['--process-pos-noise', str(float(process_pos_noise))])
    if process_vel_noise is not None:
        cmd.extend(['--process-vel-noise', str(float(process_vel_noise))])
    if process_acc_noise is not None:
        cmd.extend(['--process-acc-noise', str(float(process_acc_noise))])

    if num_runs > 1:
        cmd.extend(['--num-runs', str(num_runs)])
    if random_seed > 0:
        cmd.extend(['--seed', str(random_seed)])

    if cluster_count > 0:
        cmd.extend(['--cluster-count', str(cluster_count)])
        cmd.extend(['--cluster-spread', str(cluster_spread)])
        cmd.extend(['--launch-time-spread', str(launch_time_spread)])

    cmd.extend(['--beam-width', str(beam_width_rad)])
    cmd.extend(['--num-beams', str(num_beams)])
    cmd.extend(['--min-search-beams', str(min_search_beams)])
    if track_confirmed_only:
        cmd.append('--track-confirmed-only')
    if search_sector_rad > 0:
        cmd.extend(['--search-sector', str(search_sector_rad)])
    cmd.extend(['--search-center', str(search_center_rad)])
    cmd.extend(['--antenna-boresight', str(antenna_boresight_rad)])
    cmd.extend(['--search-elevation', str(search_elevation_rad)])

    return cmd, sensor_x_m, sensor_y_m


def _run_job_thread(job_id, cmd):
    """Background thread: run tracker subprocess and update job state."""
    job = _jobs.get(job_id)
    if not job:
        return

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(PROJECT_ROOT), encoding='utf-8', errors='replace',
            bufsize=1
        )
        job['process'] = process
        job['status'] = 'running'

        stdout_lines = []
        num_runs = job.get('num_runs', 1)
        current_run = 0

        for raw_line in process.stdout:
            line = raw_line.rstrip()
            stdout_lines.append(line)
            job['stdout_lines'] = stdout_lines

            if job.get('cancelled'):
                process.terminate()
                break

            # Parse "Frame X/Y | ..."
            if line.startswith('Frame '):
                try:
                    rest = line[6:]
                    slash = rest.index('/')
                    space = rest.index(' ', slash)
                    cur = int(rest[:slash])
                    tot = int(rest[slash + 1:space])
                    run_base = (current_run / max(num_runs, 1)) * 100.0
                    run_range = 100.0 / max(num_runs, 1)
                    pct = run_base + (cur / tot) * run_range
                    job['progress_pct'] = min(pct, 99.0)
                    job['progress_msg'] = line
                except Exception:
                    pass
            # Parse "--- Run N/M ---"
            elif '--- Run ' in line:
                try:
                    m = re.search(r'Run (\d+)/(\d+)', line)
                    if m:
                        current_run = int(m.group(1)) - 1
                        tot_runs = int(m.group(2))
                        pct = (current_run / max(tot_runs, 1)) * 100.0
                        job['progress_pct'] = pct
                        job['progress_msg'] = f'Run {current_run + 1}/{tot_runs}'
                except Exception:
                    pass

        stderr_text = process.stderr.read()
        process.wait()

        if job.get('cancelled'):
            job['status'] = 'cancelled'
            return

        if process.returncode != 0:
            job['status'] = 'error'
            job['error'] = (
                f'Tracker failed (rc={process.returncode}): '
                f'{stderr_text[-500:] if stderr_text else "no stderr"}'
            )
            return

        stdout_text = '\n'.join(stdout_lines)
        results_data = read_csv_safe(PROJECT_ROOT / 'results.csv')
        track_data = read_csv_safe(PROJECT_ROOT / 'track_details.csv')
        gt_data = read_csv_safe(PROJECT_ROOT / 'ground_truth.csv')
        eval_data = read_csv_safe(PROJECT_ROOT / 'evaluation_results.csv')
        meas_data = read_csv_safe(PROJECT_ROOT / 'measurements.csv')
        eval_summary = parse_eval_summary(stdout_text)

        job['status'] = 'complete'
        job['progress_pct'] = 100.0
        job['result'] = {
            'success': True,
            'tracks': track_data,
            'ground_truth': gt_data,
            'measurements': meas_data,
            'results': results_data,
            'evaluation': eval_data,
            'eval_summary': eval_summary,
            'sensor_x': job.get('sensor_x', 0.0),
            'sensor_y': job.get('sensor_y', 0.0),
            'stdout': stdout_text[-2000:],
        }

    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)


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
    # Use warhead (object_id=0) entries only — trajectory may contain booster data
    warhead_traj = [t for t in trajectory if t.get('object_id', 0) == 0]
    if not warhead_traj:
        warhead_traj = trajectory  # fallback if no object_id field

    duration = warhead_traj[-1]['time']
    dt = trajectory[1]['time'] - trajectory[0]['time'] if len(trajectory) > 1 else 0.1
    framerate = 1.0 / dt

    # Extract launch/target positions from warhead trajectory data
    launch_x = warhead_traj[0]['x']
    launch_y = warhead_traj[0]['y']
    target_x = warhead_traj[-1]['x']
    target_y = warhead_traj[-1]['y']

    # Missile parameters from frontend
    missile_type = params.get('missile_type', 'ballistic')
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

    # Separation parameters
    enable_sep = bool(params.get('enable_separation', False))
    warhead_mf = float(params.get('warhead_mass_fraction', 0.3))

    # Sensor performance parameters
    pfa = params.get('pfa')
    pd_ref = params.get('pd_ref')
    pd_ref_range_km = params.get('pd_ref_range_km')
    range_noise = params.get('range_noise')
    azimuth_noise = params.get('azimuth_noise')
    elevation_noise = params.get('elevation_noise')
    doppler_noise = params.get('doppler_noise')

    # Tracker parameters
    gate_threshold = params.get('gate_threshold')
    confirm_hits = params.get('confirm_hits')
    delete_misses = params.get('delete_misses')
    min_snr = params.get('min_snr')
    process_pos_noise = params.get('process_pos_noise')
    process_vel_noise = params.get('process_vel_noise')
    process_acc_noise = params.get('process_acc_noise')

    # Multi-run parameters
    num_runs = int(params.get('num_runs', 1))
    random_seed = int(params.get('seed', 0))

    # Cluster parameters
    cluster_count = int(params.get('cluster_count', 0))
    cluster_spread = float(params.get('cluster_spread', 5000.0))
    launch_time_spread = float(params.get('launch_time_spread', 5.0))

    # Beam steering parameters
    beam_width_rad = float(params.get('beam_width', 0.052))
    num_beams = int(params.get('num_beams', 10))
    min_search_beams = int(params.get('min_search_beams', 1))
    track_confirmed_only = bool(params.get('track_confirmed_only', False))
    search_sector_rad = float(params.get('search_sector', -1))
    # search_center, antenna_boresight: GUI bearing (deg, 0=North CW) → C++ atan2 (rad, 0=East CCW)
    search_center_deg = float(params.get('search_center', 0))
    search_center_rad = math.radians(90 - search_center_deg)
    antenna_boresight_deg = float(params.get('antenna_boresight', 0))
    antenna_boresight_rad = math.radians(90 - antenna_boresight_deg)
    search_elevation_deg = float(params.get('search_elevation', 0))
    search_elevation_rad = math.radians(search_elevation_deg)

    # Convert sensor lat/lon to meters (warhead impact = origin)
    sensor_x_m = 0.0
    sensor_y_m = 0.0
    if sensor_lat is not None and sensor_lon is not None:
        # Use warhead's last point (not booster) for coordinate reference
        target_lat_traj = warhead_traj[-1].get('lat')
        target_lon_traj = warhead_traj[-1].get('lon')
        if target_lat_traj is not None and target_lon_traj is not None:
            sensor_x_m, sensor_y_m = latlon_to_meters(
                float(target_lat_traj), float(target_lon_traj),
                float(sensor_lat), float(sensor_lon)
            )

    # Debug logging for coordinate tracing
    print(f"[run_tracker DEBUG] warhead_traj[0]: x={warhead_traj[0].get('x')}, y={warhead_traj[0].get('y')}, object_id={warhead_traj[0].get('object_id')}")
    print(f"[run_tracker DEBUG] warhead_traj[-1]: x={warhead_traj[-1].get('x')}, y={warhead_traj[-1].get('y')}, lat={warhead_traj[-1].get('lat')}, lon={warhead_traj[-1].get('lon')}, object_id={warhead_traj[-1].get('object_id')}")
    if len(warhead_traj) < len(trajectory):
        print(f"[run_tracker DEBUG] NOTE: {len(trajectory)-len(warhead_traj)} booster entries filtered out")
    print(f"[run_tracker DEBUG] sensor_lat={sensor_lat}, sensor_lon={sensor_lon}")
    print(f"[run_tracker DEBUG] sensor_x_m={sensor_x_m}, sensor_y_m={sensor_y_m}")
    print(f"[run_tracker DEBUG] launch_x={launch_x}, launch_y={launch_y}, target_x={target_x}, target_y={target_y}")

    # HGV-specific parameters
    hgv_cruise_alt = float(params.get('cruise_altitude', 40000.0))
    hgv_glide_ratio = float(params.get('glide_ratio', 4.0))
    hgv_terminal_dive = float(params.get('terminal_dive_range', 20000.0))
    hgv_pullup_dur = float(params.get('pullup_duration', 30.0))
    hgv_bank_max = float(params.get('bank_angle_max', 1.0))
    hgv_num_skips = int(params.get('num_skips', 0))

    # Run FastTracker with missile + sensor + physics parameters
    cmd = [
        str(FASTTRACKER_EXE),
        '--scenario', 'single-ballistic',
        '--missile-type', missile_type,
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

    if enable_sep:
        cmd.append('--enable-separation')
        cmd.extend(['--warhead-mass-fraction', str(warhead_mf)])

    # HGV-specific parameters
    if missile_type == 'hgv':
        cmd.extend([
            '--cruise-altitude', str(hgv_cruise_alt),
            '--glide-ratio', str(hgv_glide_ratio),
            '--terminal-dive-range', str(hgv_terminal_dive),
            '--pullup-duration', str(hgv_pullup_dur),
            '--bank-angle-max', str(hgv_bank_max),
        ])
        if hgv_num_skips > 0:
            cmd.extend(['--num-skips', str(hgv_num_skips)])

    # Sensor performance parameters
    if pfa is not None and float(pfa) > 0:
        cmd.extend(['--pfa', str(float(pfa))])
    if pd_ref is not None and 0 < float(pd_ref) < 1:
        cmd.extend(['--pd-ref', str(float(pd_ref))])
    if pd_ref_range_km is not None and float(pd_ref_range_km) > 0:
        cmd.extend(['--pd-ref-range', str(float(pd_ref_range_km) * 1000.0)])
    if range_noise is not None:
        cmd.extend(['--range-noise', str(float(range_noise))])
    if azimuth_noise is not None:
        cmd.extend(['--azimuth-noise', str(float(azimuth_noise))])
    if elevation_noise is not None:
        cmd.extend(['--elevation-noise', str(float(elevation_noise))])
    if doppler_noise is not None:
        cmd.extend(['--doppler-noise', str(float(doppler_noise))])

    # Tracker parameters
    if gate_threshold is not None:
        cmd.extend(['--gate-threshold', str(float(gate_threshold))])
    if confirm_hits is not None:
        cmd.extend(['--confirm-hits', str(int(confirm_hits))])
    if delete_misses is not None:
        cmd.extend(['--delete-misses', str(int(delete_misses))])
    if min_snr is not None:
        cmd.extend(['--min-snr', str(float(min_snr))])
    if process_pos_noise is not None:
        cmd.extend(['--process-pos-noise', str(float(process_pos_noise))])
    if process_vel_noise is not None:
        cmd.extend(['--process-vel-noise', str(float(process_vel_noise))])
    if process_acc_noise is not None:
        cmd.extend(['--process-acc-noise', str(float(process_acc_noise))])

    # Multi-run parameters
    if num_runs > 1:
        cmd.extend(['--num-runs', str(num_runs)])
    if random_seed > 0:
        cmd.extend(['--seed', str(random_seed)])

    # Cluster parameters
    if cluster_count > 0:
        cmd.extend(['--cluster-count', str(cluster_count)])
        cmd.extend(['--cluster-spread', str(cluster_spread)])
        cmd.extend(['--launch-time-spread', str(launch_time_spread)])

    # Beam steering parameters
    cmd.extend(['--beam-width', str(beam_width_rad)])
    cmd.extend(['--num-beams', str(num_beams)])
    cmd.extend(['--min-search-beams', str(min_search_beams)])
    if track_confirmed_only:
        cmd.append('--track-confirmed-only')
    if search_sector_rad > 0:
        cmd.extend(['--search-sector', str(search_sector_rad)])
    cmd.extend(['--search-center', str(search_center_rad)])
    cmd.extend(['--antenna-boresight', str(antenna_boresight_rad)])
    cmd.extend(['--search-elevation', str(search_elevation_rad)])

    # Log the C++ command for debugging — write to file to avoid stdout buffering
    debug_log = PROJECT_ROOT / 'debug_run_tracker.log'
    with open(debug_log, 'w') as dbg:
        dbg.write(f"=== run_tracker debug ===\n")
        dbg.write(f"Full command ({len(cmd)} args):\n")
        dbg.write(' '.join(cmd) + '\n\n')
        dbg.write(f"sensor_x_m = {sensor_x_m}\n")
        dbg.write(f"sensor_y_m = {sensor_y_m}\n")
        dbg.write(f"launch_x = {launch_x}\n")
        dbg.write(f"launch_y = {launch_y}\n")
        dbg.write(f"target_x = {target_x}\n")
        dbg.write(f"target_y = {target_y}\n")
        dbg.write(f"sensor_lat = {sensor_lat}\n")
        dbg.write(f"sensor_lon = {sensor_lon}\n")

    try:
        timeout = 120 * max(num_runs, 1) * max(1, 1 + cluster_count // 3)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT), encoding='utf-8', errors='replace'
        )

        # Read results
        results_data = read_csv_safe(PROJECT_ROOT / 'results.csv')
        track_data = read_csv_safe(PROJECT_ROOT / 'track_details.csv')
        gt_data = read_csv_safe(PROJECT_ROOT / 'ground_truth.csv')
        eval_data = read_csv_safe(PROJECT_ROOT / 'evaluation_results.csv')
        meas_data = read_csv_safe(PROJECT_ROOT / 'measurements.csv')

        # Parse evaluation summary from stdout
        eval_summary = parse_eval_summary(result.stdout)

        # Debug: append post-run data to log file
        with open(debug_log, 'a') as dbg:
            dbg.write(f"\n=== Post-run ===\n")
            dbg.write(f"returncode = {result.returncode}\n")
            dbg.write(f"stdout (last 500): {result.stdout[-500:] if result.stdout else 'EMPTY'}\n")
            dbg.write(f"stderr (last 200): {result.stderr[-200:] if result.stderr else 'EMPTY'}\n")
            if gt_data:
                g0 = gt_data[0]
                dbg.write(f"gt_data[0]: x={g0.get('x')}, y={g0.get('y')}, z={g0.get('z')}\n")
            if meas_data:
                m0 = next((m for m in meas_data if m.get('is_clutter', 0) == 0), meas_data[0])
                dbg.write(f"first_meas: range={m0.get('range')}, az={m0.get('azimuth')}, el={m0.get('elevation')}\n")
            dbg.write(f"JSON sensor_x={sensor_x_m}, sensor_y={sensor_y_m}\n")
            dbg.write(f"gt_data count={len(gt_data)}, meas_data count={len(meas_data)}\n")

        return jsonify({
            'success': True,
            'tracks': track_data,
            'ground_truth': gt_data,
            'measurements': meas_data,
            'results': results_data,
            'evaluation': eval_data,
            'eval_summary': eval_summary,
            'sensor_x': sensor_x_m,
            'sensor_y': sensor_y_m,
            'stdout': result.stdout[-2000:] if result.stdout else '',
        })

    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': f'Simulation timed out ({timeout}s)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/run-tracker-start', methods=['POST'])
def run_tracker_start():
    """Start an async tracker job. Returns {job_id}."""
    data = request.get_json()
    try:
        cmd, sensor_x_m, sensor_y_m = _build_tracker_cmd(data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

    params = data.get('params', {})
    num_runs = int(params.get('num_runs', 1))

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        'status': 'starting',
        'progress_pct': 0.0,
        'progress_msg': 'Starting...',
        'stdout_lines': [],
        'process': None,
        'result': None,
        'error': None,
        'cancelled': False,
        'sensor_x': sensor_x_m,
        'sensor_y': sensor_y_m,
        'num_runs': num_runs,
    }

    t = threading.Thread(target=_run_job_thread, args=(job_id, cmd), daemon=True)
    t.start()
    return jsonify({'success': True, 'job_id': job_id})


@app.route('/api/run-tracker-status/<job_id>', methods=['GET'])
def run_tracker_status(job_id):
    """Poll job status. Returns progress and result when complete."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'})

    response = {
        'success': True,
        'status': job['status'],
        'progress_pct': job.get('progress_pct', 0.0),
        'progress_msg': job.get('progress_msg', ''),
    }

    if job['status'] == 'complete':
        response['result'] = job['result']
        _jobs.pop(job_id, None)
    elif job['status'] in ('error', 'cancelled'):
        response['error'] = job.get('error', '')
        _jobs.pop(job_id, None)

    return jsonify(response)


@app.route('/api/run-tracker-cancel/<job_id>', methods=['POST'])
def run_tracker_cancel(job_id):
    """Cancel a running tracker job."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'})

    job['cancelled'] = True
    process = job.get('process')
    if process:
        process.terminate()
        try:
            process.wait(timeout=3)
        except Exception:
            process.kill()

    job['status'] = 'cancelled'
    _jobs.pop(job_id, None)
    return jsonify({'success': True})


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
    """Parse evaluation metrics from FastTracker stdout.

    All C++ output keys are English. Parses all 'key: value' lines from
    the evaluation summary sections (Accuracy, Detection, Track Quality,
    Data Statistics, Performance, Aggregated Results).
    """
    import re
    summary = {}
    if not stdout:
        return summary

    # Known metric keys to parse (all English)
    metric_keys = [
        'Total Frames', 'Avg GT Targets', 'Avg Tracks',
        'Position RMSE', 'Position MAE', 'Velocity RMSE', 'Velocity MAE', 'Mean OSPA',
        'True Positives', 'False Positives', 'False Negatives',
        'Precision', 'Recall', 'F1 Score',
        'Avg Track Duration', 'Confirmation Rate', 'False Track Rate', 'Track Purity',
        'Mostly Tracked (MT)', 'Partially Tracked (PT)', 'Mostly Lost (ML)',
        'Wall-clock time', 'GPU total', 'GPU avg/frame',
        'GPU min/frame', 'GPU max/frame',
        'Predict total', 'Association total', 'Update total',
        'Realtime factor',
    ]

    lines = stdout.split('\n')
    in_aggregated = False
    in_resolved = False
    resolved = {}
    for line in lines:
        line = line.strip()

        # Check for resolved parameters section
        if line == '[Resolved Parameters]':
            in_resolved = True
            continue
        if in_resolved:
            if not line or line.startswith('[') or line.startswith('==='):
                in_resolved = False
            elif ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                try:
                    resolved[key] = float(parts[1].strip())
                except (ValueError, TypeError):
                    pass
                continue

        # Check for aggregated results section
        if 'Aggregated Results' in line:
            in_aggregated = True
            m = re.search(r'\((\d+) runs\)', line)
            if m:
                summary['num_runs'] = int(m.group(1))
            continue

        # Parse aggregated results (lines with +/-)
        if in_aggregated and '+/-' in line and ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val_str = parts[1].strip()
                try:
                    mean_std = val_str.split('+/-')
                    mean_val = float(mean_std[0].strip())
                    std_val = float(mean_std[1].strip().split()[0])
                    summary[f'agg_{key}'] = {'mean': mean_val, 'std': std_val}
                except (ValueError, IndexError):
                    pass
            continue

        # Parse known metric keys
        if ':' not in line:
            continue
        for key in metric_keys:
            if line.startswith(key + ':'):
                val_str = line[len(key) + 1:].strip()
                try:
                    val = float(val_str.split()[0])
                    summary[key] = val
                except (ValueError, IndexError):
                    pass
                break

    if resolved:
        summary['resolved'] = resolved

    return summary


def build_plotly_data_from_csv(traj_csv):
    """Build Plotly-compatible JSON for 3D trajectory visualization from CSV data."""

    # Check if object_id column exists (separation mode)
    has_object_id = len(traj_csv) > 0 and 'object_id' in traj_csv[0]

    # Split by object_id
    warhead_rows = [r for r in traj_csv if not has_object_id or r.get('object_id', 0) == 0]
    booster_rows = [r for r in traj_csv if has_object_id and r.get('object_id', 0) == 1]

    def build_trace(rows, label='warhead'):
        n = len(rows)
        times = [row['time'] for row in rows]
        x_km = [row['x'] / 1000.0 for row in rows]
        y_km = [row['y'] / 1000.0 for row in rows]
        z_km = [row['altitude'] / 1000.0 for row in rows]
        speeds = [row['speed'] for row in rows]
        phases = [row['phase'] for row in rows]
        latlons = [[row['lat'], row['lon']] for row in rows]

        phase_colors = {
            'BOOST': 'rgb(255, 140, 0)',
            'MIDCOURSE': 'rgb(50, 200, 255)',
            'TERMINAL': 'rgb(255, 32, 80)',
            'PULLUP': 'rgb(255, 220, 50)',
            'GLIDE': 'rgb(0, 255, 128)',
        }
        colors = [phase_colors.get(p, 'rgb(200, 200, 200)') for p in phases]

        hover = [
            f"[{label}] Time: {times[i]:.1f}s<br>"
            f"Phase: {phases[i]}<br>"
            f"Pos: ({x_km[i]:.1f}km, {y_km[i]:.1f}km)<br>"
            f"Alt: {z_km[i]:.1f}km<br>"
            f"Speed: {speeds[i]:.0f} m/s ({speeds[i]*3.6:.0f} km/h)"
            for i in range(n)
        ]

        return {
            'x': x_km, 'y': y_km, 'z': z_km,
            'text': hover, 'speeds': speeds, 'phases': phases,
            'colors': colors, 'times': times,
        }, latlons

    warhead_trace, warhead_latlons = build_trace(warhead_rows, 'Warhead' if booster_rows else 'Missile')

    result = {
        'trajectory': warhead_trace,
        'latlons': warhead_latlons,
    }

    if booster_rows:
        booster_trace, booster_latlons = build_trace(booster_rows, 'Booster')
        result['booster'] = booster_trace
        result['booster_latlons'] = booster_latlons

    return result


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
