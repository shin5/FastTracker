"""
Performance Dashboard for FastTracker

Generates a multi-panel dashboard visualizing tracking performance metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from .data_loader import CSVDataLoader


def create_dashboard(data, output_path=None, show=True):
    """
    Create a comprehensive performance dashboard.

    Args:
        data: TrackingData object from data_loader
        output_path: Optional path to save figure (PNG/PDF)
        show: Whether to display the figure interactively
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('FastTracker Performance Dashboard', fontsize=16, fontweight='bold')

    # Create grid layout (3 rows, 2 columns)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: FPS Time Series (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_fps_timeseries(ax1, data)

    # Panel 2: Processing Time Breakdown (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_processing_time(ax2, data)

    # Panel 3: Track Count Progression (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_track_counts(ax3, data)

    # Panel 4: Measurements vs Tracks (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_measurements_vs_tracks(ax4, data)

    # Panel 5: Position Error (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    plot_position_error(ax5, data)

    # Panel 6: Statistics Summary (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    plot_statistics_summary(ax6, data)

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to: {output_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_fps_timeseries(ax, data):
    """Plot FPS over time."""
    if data.performance_stats is None:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return

    time = data.performance_stats['time']
    fps = data.performance_stats['fps']

    ax.plot(time, fps, color='#2E86AB', linewidth=1.5, alpha=0.8)
    ax.fill_between(time, 0, fps, color='#2E86AB', alpha=0.2)

    # Add mean line
    mean_fps = fps.mean()
    ax.axhline(mean_fps, color='#E63946', linestyle='--', linewidth=2,
               label=f'Mean: {mean_fps:.1f} FPS')

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('FPS', fontsize=10)
    ax.set_title('Processing Speed Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)


def plot_processing_time(ax, data):
    """Plot processing time breakdown."""
    if data.frame_data is None:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return

    time = data.frame_data['time']
    proc_time = data.frame_data['processing_time_ms']

    ax.plot(time, proc_time, color='#F77F00', linewidth=1.5, alpha=0.8)
    ax.fill_between(time, 0, proc_time, color='#F77F00', alpha=0.2)

    # Add statistics
    mean_time = proc_time.mean()
    max_time = proc_time.max()
    ax.axhline(mean_time, color='#06D6A0', linestyle='--', linewidth=2,
               label=f'Mean: {mean_time:.2f} ms')

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Processing Time (ms)', fontsize=10)
    ax.set_title('Processing Time Per Frame', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)

    # Add text annotation for max
    ax.text(0.98, 0.98, f'Max: {max_time:.2f} ms',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_track_counts(ax, data):
    """Plot track count progression over time."""
    if data.frame_data is None:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return

    time = data.frame_data['time']
    num_tracks = data.frame_data['num_tracks']
    num_confirmed = data.frame_data['num_confirmed']

    # Calculate tentative tracks
    num_tentative = num_tracks - num_confirmed

    # Stacked area plot
    ax.fill_between(time, 0, num_confirmed,
                    color='#06D6A0', alpha=0.7, label='Confirmed')
    ax.fill_between(time, num_confirmed, num_tracks,
                    color='#FDCA40', alpha=0.7, label='Tentative')

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Number of Tracks', fontsize=10)
    ax.set_title('Track Count Progression', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)


def plot_measurements_vs_tracks(ax, data):
    """Plot measurements vs track counts."""
    if data.frame_data is None:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return

    time = data.frame_data['time']
    measurements = data.frame_data['num_measurements']
    tracks = data.frame_data['num_tracks']

    ax.plot(time, measurements, color='#E63946', linewidth=2,
            label='Measurements', alpha=0.8)
    ax.plot(time, tracks, color='#2E86AB', linewidth=2,
            label='Tracks', alpha=0.8)

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Measurements vs Tracks', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)


def plot_position_error(ax, data):
    """Plot position error over time."""
    if data.evaluation_data is None:
        ax.text(0.5, 0.5, 'Evaluation data not available', ha='center', va='center')
        ax.set_title('Position Error (Not Available)', fontsize=12, fontweight='bold')
        return

    time = data.evaluation_data['time']
    pos_error = data.evaluation_data['avg_position_error']
    ospa = data.evaluation_data['ospa_distance']

    ax.plot(time, pos_error, color='#E63946', linewidth=2,
            label='Avg Position Error (m)', alpha=0.8)
    ax.plot(time, ospa, color='#9D4EDD', linewidth=2,
            label='OSPA Distance (m)', alpha=0.8)

    # Add mean lines
    mean_pos_error = pos_error.mean()
    ax.axhline(mean_pos_error, color='#E63946', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Error / Distance (m)', fontsize=10)
    ax.set_title('Tracking Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)


def plot_statistics_summary(ax, data):
    """Display statistical summary as text."""
    ax.axis('off')

    # Compute statistics
    stats_text = "=== Performance Summary ===\n\n"

    if data.frame_data is not None:
        num_frames = len(data.frame_data)
        duration = data.frame_data['time'].max() - data.frame_data['time'].min()
        max_tracks = data.frame_data['num_tracks'].max()
        max_confirmed = data.frame_data['num_confirmed'].max()
        avg_measurements = data.frame_data['num_measurements'].mean()

        stats_text += f"Simulation:\n"
        stats_text += f"  Frames: {num_frames}\n"
        stats_text += f"  Duration: {duration:.2f} s\n"
        stats_text += f"  Avg Measurements: {avg_measurements:.1f}\n\n"

        stats_text += f"Tracks:\n"
        stats_text += f"  Max Total: {max_tracks}\n"
        stats_text += f"  Max Confirmed: {max_confirmed}\n"
        if max_tracks > 0:
            confirm_rate = (max_confirmed / max_tracks) * 100
            stats_text += f"  Peak Confirm Rate: {confirm_rate:.1f}%\n\n"

    if data.performance_stats is not None:
        mean_fps = data.performance_stats['fps'].mean()
        max_fps = data.performance_stats['fps'].max()
        min_fps = data.performance_stats['fps'].min()
        std_fps = data.performance_stats['fps'].std()

        stats_text += f"Performance:\n"
        stats_text += f"  Mean FPS: {mean_fps:.1f}\n"
        stats_text += f"  Max FPS: {max_fps:.1f}\n"
        stats_text += f"  Min FPS: {min_fps:.1f}\n"
        stats_text += f"  Std Dev: {std_fps:.1f}\n\n"

    if data.evaluation_data is not None:
        num_gt = data.evaluation_data['num_ground_truth'].iloc[0] if 'num_ground_truth' in data.evaluation_data else 0
        mean_pos_error = data.evaluation_data['avg_position_error'].mean()
        mean_ospa = data.evaluation_data['ospa_distance'].mean()
        mean_tp = data.evaluation_data['true_positives'].mean() if 'true_positives' in data.evaluation_data else 0
        mean_fp = data.evaluation_data['false_positives'].mean() if 'false_positives' in data.evaluation_data else 0

        stats_text += f"Accuracy:\n"
        stats_text += f"  Ground Truth Targets: {int(num_gt)}\n"
        stats_text += f"  Avg Position Error: {mean_pos_error:.2f} m\n"
        stats_text += f"  Avg OSPA Distance: {mean_ospa:.2f} m\n"
        stats_text += f"  Avg True Positives: {mean_tp:.1f}\n"
        stats_text += f"  Avg False Positives: {mean_fp:.1f}\n"

    # Display text
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    ax.set_title('Statistics Summary', fontsize=12, fontweight='bold')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate FastTracker performance dashboard')
    parser.add_argument('--path', type=str, default='.',
                       help='Path to directory containing CSV files')
    parser.add_argument('--results', type=str, default='results.csv',
                       help='Results CSV filename')
    parser.add_argument('--eval', type=str, default='evaluation_results.csv',
                       help='Evaluation CSV filename')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (PNG/PDF). If not specified, displays only.')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the dashboard interactively')

    args = parser.parse_args()

    # Default output filename if not specified but saving is implied
    if args.output is None and args.no_show:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"dashboard_{timestamp}.png"

    # Load data
    print("Loading data...")
    loader = CSVDataLoader(args.path)
    data = loader.load_all(args.results, args.eval)

    # Create dashboard
    print("Generating dashboard...")
    create_dashboard(data, output_path=args.output, show=not args.no_show)

    print("Done!")


if __name__ == '__main__':
    main()
