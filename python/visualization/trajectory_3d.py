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

            # Add ground truth trajectory line (dashed, white/gray)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                name=f'GT {target_id}',
                line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dash'),
                text=hover_text,
                hoverinfo='text',
                showlegend=False
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
