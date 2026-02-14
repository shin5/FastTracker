"""
3D Trajectory Viewer for FastTracker

Creates interactive 3D visualizations of track trajectories using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
from pathlib import Path
from datetime import datetime
from .data_loader import CSVDataLoader


def create_3d_trajectory_plot(data, color_by='state', time_range=None, output_path=None, show=True, show_ground_truth=True):
    """
    Create an interactive 3D trajectory visualization.

    Args:
        data: TrackingData object from data_loader
        color_by: Color scheme - 'state', 'track_id', or 'model_prob'
        time_range: Tuple of (start_time, end_time) to filter tracks
        output_path: Optional path to save HTML file
        show: Whether to display the figure in browser
        show_ground_truth: Whether to display ground truth trajectories
    """
    if not data.track_trajectories:
        print("ERROR: No track trajectories available for 3D visualization.")
        return None

    # Create figure
    fig = go.Figure()

    # Color schemes
    state_colors = {
        0: 'rgb(255, 255, 0)',  # Tentative - Yellow
        1: 'rgb(0, 255, 0)',    # Confirmed - Green
        2: 'rgb(255, 0, 0)'     # Lost - Red
    }

    # Process each track
    for track_id, trajectory in data.track_trajectories.items():
        # Apply time filter if specified
        if time_range:
            start_time, end_time = time_range
            trajectory = trajectory[
                (trajectory['time'] >= start_time) &
                (trajectory['time'] <= end_time)
            ]

        if len(trajectory) == 0:
            continue

        # Extract data
        x = trajectory['x'].values
        y = trajectory['y'].values
        # For 2D data, use velocity magnitude as Z
        if 'z' in trajectory.columns:
            z = trajectory['z'].values
        else:
            # Use velocity magnitude for Z axis
            vx = trajectory['vx'].values
            vy = trajectory['vy'].values
            z = np.sqrt(vx**2 + vy**2)  # Speed as height

        time = trajectory['time'].values
        state = trajectory['state'].values if 'state' in trajectory.columns else np.zeros(len(trajectory))

        # Determine color
        if color_by == 'state':
            # Use most common state for the track
            most_common_state = int(np.bincount(state.astype(int)).argmax())
            color = state_colors.get(most_common_state, 'rgb(128, 128, 128)')
            color_label = f"State: {most_common_state}"
        elif color_by == 'track_id':
            # Use track ID for color (cycle through color palette)
            colors_palette = px.colors.qualitative.Plotly
            color = colors_palette[track_id % len(colors_palette)]
            color_label = f"Track {track_id}"
        elif color_by == 'model_prob':
            # Color by dominant model (if available)
            if 'model_prob_cv' in trajectory.columns:
                cv = trajectory['model_prob_cv'].values
                high = trajectory['model_prob_high'].values
                med = trajectory['model_prob_med'].values

                # Find dominant model
                probs = np.stack([cv, high, med], axis=1)
                dominant_model = np.argmax(probs, axis=1)

                # Average dominant model
                avg_dominant = int(np.bincount(dominant_model).argmax())
                model_colors = ['rgb(0, 0, 255)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']
                color = model_colors[avg_dominant]
                model_names = ['CV', 'High-Accel', 'Med-Accel']
                color_label = f"Model: {model_names[avg_dominant]}"
            else:
                color = 'rgb(128, 128, 128)'
                color_label = "No model data"
        else:
            color = 'rgb(128, 128, 128)'
            color_label = f"Track {track_id}"

        # Create hover text
        hover_text = [
            f"Track {track_id}<br>" +
            f"Time: {t:.2f}s<br>" +
            f"Pos: ({x[i]:.1f}, {y[i]:.1f}, {z[i]:.1f})<br>" +
            f"State: {int(state[i])}"
            for i, t in enumerate(time)
        ]

        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'Track {track_id}',
            line=dict(color=color, width=3),
            marker=dict(size=3, color=color),
            text=hover_text,
            hoverinfo='text',
            showlegend=True if len(data.track_trajectories) <= 20 else False
        ))

        # Add start and end markers
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers',
            marker=dict(size=8, color='white', symbol='circle',
                       line=dict(color=color, width=2)),
            name=f'Start {track_id}',
            hovertext=f'Track {track_id} Start<br>Time: {time[0]:.2f}s',
            hoverinfo='text',
            showlegend=False
        ))

        fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode='markers',
            marker=dict(size=8, color=color, symbol='diamond'),
            name=f'End {track_id}',
            hovertext=f'Track {track_id} End<br>Time: {time[-1]:.2f}s',
            hoverinfo='text',
            showlegend=False
        ))

    # Add ground truth trajectories if available and requested
    if show_ground_truth and data.ground_truth_trajectories:
        for target_id, trajectory in data.ground_truth_trajectories.items():
            # Apply time filter if specified
            if time_range:
                start_time, end_time = time_range
                trajectory = trajectory[
                    (trajectory['time'] >= start_time) &
                    (trajectory['time'] <= end_time)
                ]

            if len(trajectory) == 0:
                continue

            # Extract data
            x = trajectory['x'].values
            y = trajectory['y'].values
            # For 2D data, use velocity magnitude as Z
            if 'z' in trajectory.columns:
                z = trajectory['z'].values
            else:
                # Use velocity magnitude for Z axis
                vx = trajectory['vx'].values
                vy = trajectory['vy'].values
                z = np.sqrt(vx**2 + vy**2)  # Speed as height

            time = trajectory['time'].values

            # Create hover text
            hover_text = [
                f"Ground Truth {target_id}<br>" +
                f"Time: {t:.2f}s<br>" +
                f"Pos: ({x[i]:.1f}, {y[i]:.1f}, {z[i]:.1f})"
                for i, t in enumerate(time)
            ]

            # Add ground truth trajectory line (solid, bright color for visibility)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                name=f'Ground Truth {target_id}',
                line=dict(color='rgba(255, 215, 0, 0.9)', width=4),  # Gold color
                marker=dict(size=2, color='rgba(255, 215, 0, 0.9)'),
                text=hover_text,
                hoverinfo='text',
                showlegend=True,
                legendgroup=f'gt_{target_id}'
            ))

            # Add start marker for ground truth
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers',
                marker=dict(size=10, color='gold', symbol='diamond',
                           line=dict(color='white', width=2)),
                name=f'GT {target_id} Start',
                hovertext=f'Ground Truth {target_id} Start<br>Time: {time[0]:.2f}s',
                hoverinfo='text',
                showlegend=False,
                legendgroup=f'gt_{target_id}'
            ))

            # Add end marker for ground truth
            fig.add_trace(go.Scatter3d(
                x=[x[-1]], y=[y[-1]], z=[z[-1]],
                mode='markers',
                marker=dict(size=10, color='gold', symbol='cross',
                           line=dict(color='white', width=2)),
                name=f'GT {target_id} End',
                hovertext=f'Ground Truth {target_id} End<br>Time: {time[-1]:.2f}s',
                hoverinfo='text',
                showlegend=False,
                legendgroup=f'gt_{target_id}'
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text='FastTracker - 3D Track Trajectories',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title='X Position (m)', backgroundcolor="rgb(230, 230,230)",
                      gridcolor="white", showbackground=True),
            yaxis=dict(title='Y Position (m)', backgroundcolor="rgb(230, 230,230)",
                      gridcolor="white", showbackground=True),
            zaxis=dict(title='Z / Speed (m or m/s)', backgroundcolor="rgb(230, 230,230)",
                      gridcolor="white", showbackground=True),
            aspectmode='auto'
        ),
        width=1200,
        height=800,
        hovermode='closest',
        showlegend=True if len(data.track_trajectories) <= 20 else False,
        template='plotly_dark'
    )

    # Add annotation with statistics
    num_tracks = len(data.track_trajectories)
    time_span = 0
    if data.frame_data is not None:
        time_span = data.frame_data['time'].max() - data.frame_data['time'].min()

    annotation_text = (
        f"Tracks: {num_tracks}<br>"
        f"Time span: {time_span:.2f}s<br>"
        f"Color by: {color_by}"
    )

    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(0, 0, 0, 0.7)",
        bordercolor="white",
        borderwidth=1,
        font=dict(color="white", size=12),
        align="left"
    )

    # Save to HTML if path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"3D trajectory saved to: {output_path}")

    # Show if requested
    if show:
        fig.show()

    return fig


def create_trajectory_comparison(data, track_ids, output_path=None, show=True):
    """
    Create a focused comparison of specific tracks.

    Args:
        data: TrackingData object
        track_ids: List of track IDs to compare
        output_path: Optional output HTML path
        show: Whether to display
    """
    if not data.track_trajectories:
        print("ERROR: No track trajectories available.")
        return None

    # Create subplots - one 3D plot and one 2D projection
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
        subplot_titles=('3D Trajectories', 'XY Projection')
    )

    colors = px.colors.qualitative.Plotly

    for idx, track_id in enumerate(track_ids):
        if track_id not in data.track_trajectories:
            print(f"Warning: Track {track_id} not found")
            continue

        trajectory = data.track_trajectories[track_id]
        color = colors[idx % len(colors)]

        x = trajectory['x'].values
        y = trajectory['y'].values
        vx = trajectory['vx'].values
        vy = trajectory['vy'].values
        z = np.sqrt(vx**2 + vy**2)  # Speed as Z

        # 3D plot
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                name=f'Track {track_id}',
                line=dict(color=color, width=4),
                marker=dict(size=4, color=color)
            ),
            row=1, col=1
        )

        # 2D projection
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='lines+markers',
                name=f'Track {track_id}',
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                showlegend=False
            ),
            row=1, col=2
        )

    fig.update_layout(
        title='Track Comparison',
        height=600,
        width=1400,
        template='plotly_dark'
    )

    fig.update_xaxes(title_text="X Position (m)", row=1, col=2)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=2)

    if output_path:
        fig.write_html(str(output_path))
        print(f"Comparison saved to: {output_path}")

    if show:
        fig.show()

    return fig


def create_animated_3d_plot(data, frame_step=1, speed_ms=100, output_path=None, show=True):
    """
    Create an animated 3D trajectory visualization with time progression.

    Args:
        data: TrackingData object from data_loader
        frame_step: Time step between animation frames (1 = every frame)
        speed_ms: Animation speed in milliseconds per frame
        output_path: Optional path to save HTML file
        show: Whether to display the figure in browser

    Returns:
        Plotly figure object with animation
    """
    if data.frame_data is None or len(data.frame_data) == 0:
        print("ERROR: No frame data available for animation.")
        return None

    # Get time range
    times = np.sort(data.frame_data['time'].unique())

    # Sample frames based on frame_step
    times = times[::frame_step]

    if len(times) == 0:
        print("ERROR: No time steps available.")
        return None

    print(f"Creating animation with {len(times)} frames...")

    # Prepare ground truth data if available
    gt_data = {}
    gt_lines = {}  # Store full trajectory lines
    if data.ground_truth_trajectories:
        for gt_id, gt_traj in data.ground_truth_trajectories.items():
            gt_data[gt_id] = gt_traj
            # Store full trajectory for the line
            x = gt_traj['x'].values
            y = gt_traj['y'].values
            if 'z' in gt_traj.columns:
                z = gt_traj['z'].values
            else:
                vx = gt_traj['vx'].values
                vy = gt_traj['vy'].values
                z = np.sqrt(vx**2 + vy**2)
            gt_lines[gt_id] = (x, y, z)

    # Prepare track data - group by time
    track_data_by_time = {}
    for track_id, traj in data.track_trajectories.items():
        for _, row in traj.iterrows():
            t = row['time']
            if t not in track_data_by_time:
                track_data_by_time[t] = []
            track_data_by_time[t].append((track_id, row))

    # Create frames
    frames = []
    for frame_idx, current_time in enumerate(times):
        frame_data = []

        # Add ground truth full trajectory lines (always visible)
        for gt_id, (x, y, z) in gt_lines.items():
            frame_data.append(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    name=f'Ground Truth {gt_id}',
                    line=dict(color='rgba(255, 215, 0, 0.7)', width=3),
                    showlegend=(frame_idx == 0),
                    legendgroup=f'gt_{gt_id}'
                )
            )

        # Add ground truth current position markers
        for gt_id, gt_traj in gt_data.items():
            # Find position at current time
            gt_at_time = gt_traj[gt_traj['time'] <= current_time]
            if len(gt_at_time) > 0:
                row = gt_at_time.iloc[-1]
                x_pos = row['x']
                y_pos = row['y']
                if 'z' in gt_traj.columns:
                    z_pos = row['z']
                else:
                    z_pos = np.sqrt(row['vx']**2 + row['vy']**2)

                frame_data.append(
                    go.Scatter3d(
                        x=[x_pos], y=[y_pos], z=[z_pos],
                        mode='markers',
                        marker=dict(size=12, color='gold', symbol='diamond',
                                   line=dict(color='white', width=2)),
                        name=f'GT {gt_id} Position',
                        showlegend=False,
                        legendgroup=f'gt_{gt_id}',
                        hovertext=f'Ground Truth {gt_id}<br>Time: {row["time"]:.2f}s<br>Pos: ({x_pos:.1f}, {y_pos:.1f}, {z_pos:.1f})',
                        hoverinfo='text'
                    )
                )

        # Add track trajectories up to current time
        track_history = {}  # Store trajectory history for each track
        for t in times:
            if t > current_time:
                break
            if t in track_data_by_time:
                for track_id, row in track_data_by_time[t]:
                    if track_id not in track_history:
                        track_history[track_id] = {'x': [], 'y': [], 'z': [], 'state': []}
                    track_history[track_id]['x'].append(row['x'])
                    track_history[track_id]['y'].append(row['y'])
                    if 'z' in row:
                        track_history[track_id]['z'].append(row['z'])
                    else:
                        track_history[track_id]['z'].append(np.sqrt(row['vx']**2 + row['vy']**2))
                    track_history[track_id]['state'].append(row.get('state', 1))

        # Plot track trajectories
        colors_palette = px.colors.qualitative.Plotly
        for track_id, hist in track_history.items():
            if len(hist['x']) == 0:
                continue

            # Determine color by most recent state
            state = hist['state'][-1]
            state_colors = {0: 'yellow', 1: 'lime', 2: 'red'}
            color = state_colors.get(state, colors_palette[track_id % len(colors_palette)])

            # Add trajectory line
            frame_data.append(
                go.Scatter3d(
                    x=hist['x'], y=hist['y'], z=hist['z'],
                    mode='lines',
                    name=f'Track {track_id}',
                    line=dict(color=color, width=2),
                    showlegend=(frame_idx == 0),
                    legendgroup=f'track_{track_id}'
                )
            )

            # Add current position marker
            frame_data.append(
                go.Scatter3d(
                    x=[hist['x'][-1]], y=[hist['y'][-1]], z=[hist['z'][-1]],
                    mode='markers',
                    marker=dict(size=8, color=color, symbol='circle'),
                    name=f'Track {track_id} Current',
                    showlegend=False,
                    legendgroup=f'track_{track_id}',
                    hovertext=f'Track {track_id}<br>Time: {current_time:.2f}s<br>State: {state}',
                    hoverinfo='text'
                )
            )

        # Create frame
        frames.append(
            go.Frame(
                data=frame_data,
                name=f't_{current_time:.2f}',
                layout=go.Layout(
                    title=dict(text=f'FastTracker Animation - Time: {current_time:.2f}s')
                )
            )
        )

    # Create initial figure with first frame data
    fig = go.Figure(data=frames[0].data if frames else [], frames=frames)

    # Update layout with animation controls
    fig.update_layout(
        title=dict(
            text=f'FastTracker - Animated 3D Trajectories (Time: {times[0]:.2f}s)',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title='X Position (m)', backgroundcolor="rgb(230, 230, 230)",
                      gridcolor="white", showbackground=True),
            yaxis=dict(title='Y Position (m)', backgroundcolor="rgb(230, 230, 230)",
                      gridcolor="white", showbackground=True),
            zaxis=dict(title='Z / Speed (m or m/s)', backgroundcolor="rgb(230, 230, 230)",
                      gridcolor="white", showbackground=True),
            aspectmode='auto'
        ),
        width=1400,
        height=900,
        hovermode='closest',
        template='plotly_dark',
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=speed_ms, redraw=True),
                                         fromcurrent=True,
                                         mode='immediate',
                                         transition=dict(duration=0))]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                          mode='immediate',
                                          transition=dict(duration=0))])
                ],
                direction='left',
                pad=dict(r=10, t=87),
                x=0.1,
                xanchor='left',
                y=0,
                yanchor='top'
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor='top',
                y=0.02,
                xanchor='left',
                x=0.1,
                currentvalue=dict(
                    prefix='Time: ',
                    visible=True,
                    xanchor='right'
                ),
                pad=dict(b=10, t=50),
                len=0.8,
                steps=[
                    dict(
                        args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                            mode='immediate',
                                            transition=dict(duration=0))],
                        method='animate',
                        label=f'{times[i]:.2f}s'
                    ) for i, f in enumerate(frames)
                ]
            )
        ]
    )

    # Save to HTML if path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Animated 3D trajectory saved to: {output_path}")

    # Show if requested
    if show:
        fig.show()

    return fig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='FastTracker 3D Trajectory Viewer')
    parser.add_argument('--path', type=str, default='.',
                       help='Path to directory containing CSV files')
    parser.add_argument('--tracks', type=str, default='track_details.csv',
                       help='Track details CSV filename')
    parser.add_argument('--results', type=str, default='results.csv',
                       help='Results CSV filename (for metadata)')
    parser.add_argument('--color-by', type=str, default='state',
                       choices=['state', 'track_id', 'model_prob'],
                       help='Color scheme for tracks')
    parser.add_argument('--time-start', type=float, default=None,
                       help='Start time for filtering (seconds)')
    parser.add_argument('--time-end', type=float, default=None,
                       help='End time for filtering (seconds)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output HTML file path')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display in browser')
    parser.add_argument('--compare', type=str, default=None,
                       help='Comma-separated track IDs to compare (e.g., "1,2,3")')
    parser.add_argument('--no-ground-truth', action='store_true',
                       help='Do not display ground truth trajectories')
    parser.add_argument('--animate', action='store_true',
                       help='Create animated visualization with time progression')
    parser.add_argument('--frame-step', type=int, default=3,
                       help='Frame step for animation (1=every frame, higher=faster)')
    parser.add_argument('--speed', type=int, default=100,
                       help='Animation speed in milliseconds per frame')

    args = parser.parse_args()

    # Default output filename
    if args.output is None and args.no_show:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"trajectory_3d_{timestamp}.html"

    # Load data
    print("Loading data...")
    loader = CSVDataLoader(args.path)
    data = loader.load_all(results_file=args.results, track_file=args.tracks)

    if not data.track_trajectories:
        print("ERROR: No track trajectories loaded. Make sure track_details.csv exists.")
        return

    # Time range filter
    time_range = None
    if args.time_start is not None or args.time_end is not None:
        time_range = (
            args.time_start if args.time_start is not None else 0.0,
            args.time_end if args.time_end is not None else float('inf')
        )

    # Create visualization
    if args.compare:
        # Comparison mode
        track_ids = [int(tid.strip()) for tid in args.compare.split(',')]
        print(f"Creating comparison for tracks: {track_ids}")
        create_trajectory_comparison(data, track_ids,
                                     output_path=args.output,
                                     show=not args.no_show)
    elif args.animate:
        # Animated mode
        print("Creating animated 3D trajectory visualization...")
        create_animated_3d_plot(data,
                               frame_step=args.frame_step,
                               speed_ms=args.speed,
                               output_path=args.output,
                               show=not args.no_show)
    else:
        # Standard 3D view
        print("Creating 3D trajectory visualization...")
        create_3d_trajectory_plot(data,
                                  color_by=args.color_by,
                                  time_range=time_range,
                                  output_path=args.output,
                                  show=not args.no_show,
                                  show_ground_truth=not args.no_ground_truth)

    print("Done!")


if __name__ == '__main__':
    main()
