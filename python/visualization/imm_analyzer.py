"""
IMM Analyzer for FastTracker

Analyzes and visualizes Interacting Multiple Model (IMM) filter behavior.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from .data_loader import CSVDataLoader


def analyze_imm_behavior(data, track_id=None, output_path=None, show=True):
    """
    Analyze IMM model probabilities for tracks.

    Args:
        data: TrackingData object
        track_id: Specific track ID to analyze (None = all tracks)
        output_path: Optional output file path
        show: Whether to display the plot
    """
    if not data.track_trajectories:
        print("ERROR: No track data available for IMM analysis.")
        return None

    # Filter tracks with model probability data
    tracks_with_imm = {
        tid: traj for tid, traj in data.track_trajectories.items()
        if 'model_prob_cv' in traj.columns
    }

    if not tracks_with_imm:
        print("ERROR: No IMM model probability data found in tracks.")
        return None

    # Analyze specific track or all tracks
    if track_id is not None:
        if track_id not in tracks_with_imm:
            print(f"ERROR: Track {track_id} not found or has no IMM data.")
            return None
        tracks_to_analyze = {track_id: tracks_with_imm[track_id]}
    else:
        tracks_to_analyze = tracks_with_imm

    # Create figure
    num_tracks = len(tracks_to_analyze)
    fig = plt.figure(figsize=(14, 4 * num_tracks))
    fig.suptitle('IMM Model Probability Analysis', fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(num_tracks, 2, figure=fig, hspace=0.4, wspace=0.3)

    for idx, (tid, trajectory) in enumerate(tracks_to_analyze.items()):
        # Extract data
        time = trajectory['time'].values
        cv_prob = trajectory['model_prob_cv'].values
        high_prob = trajectory['model_prob_high'].values
        med_prob = trajectory['model_prob_med'].values

        # Panel 1: Stacked area plot of model probabilities
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.fill_between(time, 0, cv_prob, alpha=0.7, color='#2E86AB', label='CV Model')
        ax1.fill_between(time, cv_prob, cv_prob + high_prob,
                        alpha=0.7, color='#E63946', label='High-Accel Model')
        ax1.fill_between(time, cv_prob + high_prob, cv_prob + high_prob + med_prob,
                        alpha=0.7, color='#06D6A0', label='Med-Accel Model')

        ax1.set_xlabel('Time (s)', fontsize=10)
        ax1.set_ylabel('Model Probability', fontsize=10)
        ax1.set_title(f'Track {tid}: Model Probabilities Over Time',
                     fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Panel 2: Acceleration profile and dominant model
        ax2 = fig.add_subplot(gs[idx, 1])

        # Calculate acceleration magnitude
        if 'ax' in trajectory.columns and 'ay' in trajectory.columns:
            ax_vals = trajectory['ax'].values
            ay_vals = trajectory['ay'].values
            accel_mag = np.sqrt(ax_vals**2 + ay_vals**2)

            # Plot acceleration
            ax2.plot(time, accel_mag, color='#F77F00', linewidth=2, label='Acceleration Magnitude')
            ax2.set_ylabel('Acceleration (m/s²)', fontsize=10, color='#F77F00')
            ax2.tick_params(axis='y', labelcolor='#F77F00')

        # Determine dominant model at each time step
        probs = np.stack([cv_prob, high_prob, med_prob], axis=1)
        dominant_model = np.argmax(probs, axis=1)
        model_names = ['CV', 'High-Accel', 'Med-Accel']
        model_colors = ['#2E86AB', '#E63946', '#06D6A0']

        # Create secondary y-axis for dominant model
        ax2_twin = ax2.twinx()

        # Plot dominant model as scatter points
        for model_idx in range(3):
            mask = dominant_model == model_idx
            if np.any(mask):
                ax2_twin.scatter(time[mask], np.ones(np.sum(mask)) * model_idx,
                               color=model_colors[model_idx],
                               label=model_names[model_idx],
                               alpha=0.6, s=50)

        ax2_twin.set_ylabel('Dominant Model', fontsize=10)
        ax2_twin.set_yticks([0, 1, 2])
        ax2_twin.set_yticklabels(model_names)
        ax2_twin.set_ylim([-0.5, 2.5])

        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_title(f'Track {tid}: Acceleration & Model Switching',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"IMM analysis saved to: {output_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def generate_imm_statistics(data, output_path=None):
    """
    Generate statistical summary of IMM behavior.

    Args:
        data: TrackingData object
        output_path: Optional output text file path
    """
    if not data.track_trajectories:
        print("ERROR: No track data available.")
        return None

    # Filter tracks with IMM data
    tracks_with_imm = {
        tid: traj for tid, traj in data.track_trajectories.items()
        if 'model_prob_cv' in traj.columns
    }

    if not tracks_with_imm:
        print("ERROR: No IMM data found.")
        return None

    # Compute statistics
    stats = []
    stats.append("=" * 60)
    stats.append("IMM FILTER ANALYSIS - STATISTICAL SUMMARY")
    stats.append("=" * 60)
    stats.append("")

    for tid, trajectory in tracks_with_imm.items():
        stats.append(f"Track {tid}:")
        stats.append("-" * 40)

        # Extract probabilities
        cv = trajectory['model_prob_cv'].values
        high = trajectory['model_prob_high'].values
        med = trajectory['model_prob_med'].values

        # Time in each model (based on dominant model)
        probs = np.stack([cv, high, med], axis=1)
        dominant = np.argmax(probs, axis=1)

        time_in_cv = np.sum(dominant == 0) / len(dominant) * 100
        time_in_high = np.sum(dominant == 1) / len(dominant) * 100
        time_in_med = np.sum(dominant == 2) / len(dominant) * 100

        stats.append(f"  Time in CV Model:         {time_in_cv:.1f}%")
        stats.append(f"  Time in High-Accel Model: {time_in_high:.1f}%")
        stats.append(f"  Time in Med-Accel Model:  {time_in_med:.1f}%")

        # Average probabilities
        stats.append(f"  Avg CV Probability:       {cv.mean():.3f}")
        stats.append(f"  Avg High Probability:     {high.mean():.3f}")
        stats.append(f"  Avg Med Probability:      {med.mean():.3f}")

        # Model switches
        switches = np.sum(np.diff(dominant) != 0)
        stats.append(f"  Model Switches:           {switches}")

        # Acceleration statistics (if available)
        if 'ax' in trajectory.columns and 'ay' in trajectory.columns:
            ax_vals = trajectory['ax'].values
            ay_vals = trajectory['ay'].values
            accel_mag = np.sqrt(ax_vals**2 + ay_vals**2)
            stats.append(f"  Max Acceleration:         {accel_mag.max():.2f} m/s^2")
            stats.append(f"  Avg Acceleration:         {accel_mag.mean():.2f} m/s^2")

        stats.append("")

    # Overall summary
    stats.append("=" * 60)
    stats.append("OVERALL SUMMARY")
    stats.append("=" * 60)
    stats.append(f"Total tracks analyzed: {len(tracks_with_imm)}")

    # Aggregate statistics
    all_cv = []
    all_high = []
    all_med = []
    for trajectory in tracks_with_imm.values():
        all_cv.extend(trajectory['model_prob_cv'].values)
        all_high.extend(trajectory['model_prob_high'].values)
        all_med.extend(trajectory['model_prob_med'].values)

    stats.append(f"Overall avg CV prob:    {np.mean(all_cv):.3f}")
    stats.append(f"Overall avg High prob:  {np.mean(all_high):.3f}")
    stats.append(f"Overall avg Med prob:   {np.mean(all_med):.3f}")
    stats.append("")

    # Convert to string
    report = "\n".join(stats)

    # Print to console (handle encoding issues)
    try:
        print(report)
    except UnicodeEncodeError:
        # Fallback for console encoding issues
        print(report.encode('ascii', 'replace').decode('ascii'))

    # Save to file if specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nStatistics saved to: {output_path}")

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='FastTracker IMM Analysis Tool')
    parser.add_argument('--path', type=str, default='.',
                       help='Path to directory containing CSV files')
    parser.add_argument('--tracks', type=str, default='track_details.csv',
                       help='Track details CSV filename')
    parser.add_argument('--track-id', type=int, default=None,
                       help='Specific track ID to analyze (default: all)')
    parser.add_argument('--output-plot', type=str, default=None,
                       help='Output plot file path (PNG/PDF)')
    parser.add_argument('--output-stats', type=str, default=None,
                       help='Output statistics file path (TXT)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')

    args = parser.parse_args()

    # Default output filenames
    if args.output_plot is None and args.no_show:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_plot = f"imm_analysis_{timestamp}.png"

    # Load data
    print("Loading data...")
    loader = CSVDataLoader(args.path)
    data = loader.load_all(track_file=args.tracks)

    if not data.track_trajectories:
        print("ERROR: No track trajectories loaded.")
        return

    # Generate plot
    print("Generating IMM probability plots...")
    analyze_imm_behavior(data,
                        track_id=args.track_id,
                        output_path=args.output_plot,
                        show=not args.no_show)

    # Generate statistics
    print("\nGenerating IMM statistics...")
    generate_imm_statistics(data, output_path=args.output_stats)

    print("\nDone!")


if __name__ == '__main__':
    main()
