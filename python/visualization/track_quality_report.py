"""
Track Quality Report for FastTracker

Generates comprehensive statistics and visualizations for track quality analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from .data_loader import CSVDataLoader


def generate_track_quality_report(data, output_plot=None, output_text=None, show=True):
    """
    Generate comprehensive track quality report.

    Args:
        data: TrackingData object
        output_plot: Optional output plot file path
        output_text: Optional output text report path
        show: Whether to display plots
    """
    if data.frame_data is None:
        print("ERROR: No frame data available.")
        return None

    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Track Quality Analysis Report', fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Track continuity (tracks over time)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_track_continuity(ax1, data)

    # Panel 2: Confirmation rate over time
    ax2 = fig.add_subplot(gs[0, 1])
    plot_confirmation_rate(ax2, data)

    # Panel 3: Track lifetime histogram
    ax3 = fig.add_subplot(gs[1, 0])
    plot_track_lifetime_histogram(ax3, data)

    # Panel 4: Track state distribution
    ax4 = fig.add_subplot(gs[1, 1])
    plot_track_state_distribution(ax4, data)

    # Panel 5: False positive/negative analysis
    ax5 = fig.add_subplot(gs[2, 0])
    plot_fp_fn_analysis(ax5, data)

    # Panel 6: Performance summary
    ax6 = fig.add_subplot(gs[2, 1])
    plot_performance_summary(ax6, data)

    # Save plot if specified
    if output_plot:
        output_plot = Path(output_plot)
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Quality report plot saved to: {output_plot}")

    # Show if requested
    if show:
        plt.show()

    # Generate text report
    if output_text or not show:
        report_text = generate_text_report(data)
        if output_text:
            output_text = Path(output_text)
            output_text.parent.mkdir(parents=True, exist_ok=True)
            with open(output_text, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Text report saved to: {output_text}")
        try:
            print("\n" + report_text)
        except UnicodeEncodeError:
            print("\n" + report_text.encode('ascii', 'replace').decode('ascii'))

    return fig


def plot_track_continuity(ax, data):
    """Plot track count over time."""
    time = data.frame_data['time']
    tracks = data.frame_data['num_tracks']
    confirmed = data.frame_data['num_confirmed']

    ax.plot(time, tracks, color='#2E86AB', linewidth=2, label='Total Tracks', alpha=0.8)
    ax.plot(time, confirmed, color='#06D6A0', linewidth=2, label='Confirmed Tracks', alpha=0.8)
    ax.fill_between(time, 0, confirmed, color='#06D6A0', alpha=0.2)

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Number of Tracks', fontsize=10)
    ax.set_title('Track Continuity Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)


def plot_confirmation_rate(ax, data):
    """Plot confirmation rate over time."""
    time = data.frame_data['time']

    # Calculate confirmation rate
    tracks = data.frame_data['num_tracks'].values
    confirmed = data.frame_data['num_confirmed'].values
    conf_rate = np.where(tracks > 0, (confirmed / tracks) * 100, 0)

    ax.plot(time, conf_rate, color='#E63946', linewidth=2, alpha=0.8)
    ax.fill_between(time, 0, conf_rate, color='#E63946', alpha=0.2)

    # Add mean line
    mean_rate = np.mean(conf_rate[tracks > 0]) if np.any(tracks > 0) else 0
    ax.axhline(mean_rate, color='#F77F00', linestyle='--', linewidth=2,
               label=f'Mean: {mean_rate:.1f}%')

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Confirmation Rate (%)', fontsize=10)
    ax.set_title('Track Confirmation Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim([0, 100])


def plot_track_lifetime_histogram(ax, data):
    """Plot histogram of track lifetimes."""
    if not data.track_trajectories:
        ax.text(0.5, 0.5, 'No track trajectory data available',
               ha='center', va='center')
        ax.set_title('Track Lifetime Distribution', fontsize=12, fontweight='bold')
        return

    # Calculate track lifetimes
    lifetimes = []
    for trajectory in data.track_trajectories.values():
        lifetime = trajectory['time'].max() - trajectory['time'].min()
        lifetimes.append(lifetime)

    lifetimes = np.array(lifetimes)

    # Create histogram
    ax.hist(lifetimes, bins=20, color='#9D4EDD', alpha=0.7, edgecolor='black')

    # Add statistics
    mean_lifetime = lifetimes.mean()
    median_lifetime = np.median(lifetimes)
    ax.axvline(mean_lifetime, color='#E63946', linestyle='--', linewidth=2,
              label=f'Mean: {mean_lifetime:.2f}s')
    ax.axvline(median_lifetime, color='#06D6A0', linestyle='--', linewidth=2,
              label=f'Median: {median_lifetime:.2f}s')

    ax.set_xlabel('Track Lifetime (s)', fontsize=10)
    ax.set_ylabel('Number of Tracks', fontsize=10)
    ax.set_title('Track Lifetime Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')


def plot_track_state_distribution(ax, data):
    """Plot distribution of track states."""
    if not data.track_trajectories:
        ax.text(0.5, 0.5, 'No track trajectory data available',
               ha='center', va='center')
        ax.set_title('Track State Distribution', fontsize=12, fontweight='bold')
        return

    # Count track states across all frames
    state_counts = {0: 0, 1: 0, 2: 0}  # tentative, confirmed, lost

    for trajectory in data.track_trajectories.values():
        if 'state' in trajectory.columns:
            states = trajectory['state'].values
            unique, counts = np.unique(states.astype(int), return_counts=True)
            for state, count in zip(unique, counts):
                if state in state_counts:
                    state_counts[state] += count

    # Create bar chart
    labels = ['Tentative', 'Confirmed', 'Lost']
    values = [state_counts[0], state_counts[1], state_counts[2]]
    colors = ['#FDCA40', '#06D6A0', '#E63946']

    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Occurrences', fontsize=10)
    ax.set_title('Track State Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def plot_fp_fn_analysis(ax, data):
    """Plot false positive and false negative analysis."""
    if data.evaluation_data is None:
        ax.text(0.5, 0.5, 'No evaluation data available',
               ha='center', va='center')
        ax.set_title('False Positive/Negative Analysis', fontsize=12, fontweight='bold')
        return

    time = data.evaluation_data['time']

    # Check if FP/FN columns exist
    if 'false_positives' in data.evaluation_data and 'false_negatives' in data.evaluation_data:
        fp = data.evaluation_data['false_positives']
        fn = data.evaluation_data['false_negatives']

        ax.plot(time, fp, color='#E63946', linewidth=2, label='False Positives', alpha=0.8)
        ax.plot(time, fn, color='#F77F00', linewidth=2, label='False Negatives', alpha=0.8)

        # Add mean lines
        ax.axhline(fp.mean(), color='#E63946', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axhline(fn.mean(), color='#F77F00', linestyle='--', linewidth=1.5, alpha=0.5)

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('False Positive/Negative Analysis', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'FP/FN data not available in evaluation results',
               ha='center', va='center')


def plot_performance_summary(ax, data):
    """Display performance summary as text."""
    ax.axis('off')

    # Compute summary statistics
    summary = []
    summary.append("=== Quality Metrics ===\n")

    if data.frame_data is not None:
        # Track statistics
        max_tracks = data.frame_data['num_tracks'].max()
        max_confirmed = data.frame_data['num_confirmed'].max()
        avg_tracks = data.frame_data['num_tracks'].mean()
        avg_confirmed = data.frame_data['num_confirmed'].mean()

        summary.append(f"Max Total Tracks: {max_tracks}")
        summary.append(f"Max Confirmed: {max_confirmed}")
        summary.append(f"Avg Total Tracks: {avg_tracks:.1f}")
        summary.append(f"Avg Confirmed: {avg_confirmed:.1f}\n")

        # Confirmation rate
        tracks = data.frame_data['num_tracks'].values
        confirmed = data.frame_data['num_confirmed'].values
        conf_rate = np.where(tracks > 0, (confirmed / tracks) * 100, 0)
        avg_conf_rate = np.mean(conf_rate[tracks > 0]) if np.any(tracks > 0) else 0

        summary.append(f"Avg Confirmation Rate: {avg_conf_rate:.1f}%\n")

    if data.track_trajectories:
        # Track lifetime statistics
        lifetimes = []
        for trajectory in data.track_trajectories.values():
            lifetime = trajectory['time'].max() - trajectory['time'].min()
            lifetimes.append(lifetime)

        lifetimes = np.array(lifetimes)
        summary.append(f"Unique Tracks: {len(data.track_trajectories)}")
        summary.append(f"Avg Track Lifetime: {lifetimes.mean():.2f}s")
        summary.append(f"Max Track Lifetime: {lifetimes.max():.2f}s\n")

    if data.evaluation_data is not None:
        # Evaluation metrics
        if 'false_positives' in data.evaluation_data:
            avg_fp = data.evaluation_data['false_positives'].mean()
            summary.append(f"Avg False Positives: {avg_fp:.1f}")

        if 'false_negatives' in data.evaluation_data:
            avg_fn = data.evaluation_data['false_negatives'].mean()
            summary.append(f"Avg False Negatives: {avg_fn:.1f}")

    summary_text = "\n".join(summary)

    ax.text(0.05, 0.95, summary_text,
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    ax.set_title('Performance Summary', fontsize=12, fontweight='bold')


def generate_text_report(data):
    """Generate detailed text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("FASTTRACKER TRACK QUALITY REPORT")
    lines.append("=" * 70)
    lines.append("")

    # General statistics
    if data.frame_data is not None:
        lines.append("[GENERAL STATISTICS]")
        lines.append(f"  Total Frames: {len(data.frame_data)}")
        duration = data.frame_data['time'].max() - data.frame_data['time'].min()
        lines.append(f"  Duration: {duration:.2f} seconds")
        lines.append(f"  Max Total Tracks: {data.frame_data['num_tracks'].max()}")
        lines.append(f"  Max Confirmed Tracks: {data.frame_data['num_confirmed'].max()}")
        lines.append(f"  Avg Total Tracks: {data.frame_data['num_tracks'].mean():.2f}")
        lines.append(f"  Avg Confirmed Tracks: {data.frame_data['num_confirmed'].mean():.2f}")
        lines.append("")

    # Confirmation rate
    if data.frame_data is not None:
        lines.append("[CONFIRMATION RATE]")
        tracks = data.frame_data['num_tracks'].values
        confirmed = data.frame_data['num_confirmed'].values
        conf_rate = np.where(tracks > 0, (confirmed / tracks) * 100, 0)
        valid_rates = conf_rate[tracks > 0]

        if len(valid_rates) > 0:
            lines.append(f"  Mean: {valid_rates.mean():.2f}%")
            lines.append(f"  Median: {np.median(valid_rates):.2f}%")
            lines.append(f"  Min: {valid_rates.min():.2f}%")
            lines.append(f"  Max: {valid_rates.max():.2f}%")
        lines.append("")

    # Track lifetime
    if data.track_trajectories:
        lines.append("[TRACK LIFETIME ANALYSIS]")
        lifetimes = []
        for trajectory in data.track_trajectories.values():
            lifetime = trajectory['time'].max() - trajectory['time'].min()
            lifetimes.append(lifetime)

        lifetimes = np.array(lifetimes)
        lines.append(f"  Number of Unique Tracks: {len(data.track_trajectories)}")
        lines.append(f"  Mean Lifetime: {lifetimes.mean():.3f} seconds")
        lines.append(f"  Median Lifetime: {np.median(lifetimes):.3f} seconds")
        lines.append(f"  Min Lifetime: {lifetimes.min():.3f} seconds")
        lines.append(f"  Max Lifetime: {lifetimes.max():.3f} seconds")
        lines.append("")

    # Evaluation metrics
    if data.evaluation_data is not None:
        lines.append("[EVALUATION METRICS]")

        if 'false_positives' in data.evaluation_data:
            fp = data.evaluation_data['false_positives']
            lines.append(f"  Avg False Positives: {fp.mean():.2f}")
            lines.append(f"  Max False Positives: {fp.max()}")

        if 'false_negatives' in data.evaluation_data:
            fn = data.evaluation_data['false_negatives']
            lines.append(f"  Avg False Negatives: {fn.mean():.2f}")
            lines.append(f"  Max False Negatives: {fn.max()}")

        if 'avg_position_error' in data.evaluation_data:
            pos_err = data.evaluation_data['avg_position_error']
            lines.append(f"  Avg Position Error: {pos_err.mean():.2f} m")

        if 'ospa_distance' in data.evaluation_data:
            ospa = data.evaluation_data['ospa_distance']
            lines.append(f"  Avg OSPA Distance: {ospa.mean():.2f} m")

        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='FastTracker Track Quality Report Generator')
    parser.add_argument('--path', type=str, default='.',
                       help='Path to directory containing CSV files')
    parser.add_argument('--results', type=str, default='results.csv',
                       help='Results CSV filename')
    parser.add_argument('--eval', type=str, default='evaluation_results.csv',
                       help='Evaluation CSV filename')
    parser.add_argument('--tracks', type=str, default='track_details.csv',
                       help='Track details CSV filename')
    parser.add_argument('--output-plot', type=str, default=None,
                       help='Output plot file path (PNG/PDF)')
    parser.add_argument('--output-text', type=str, default=None,
                       help='Output text report path')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')

    args = parser.parse_args()

    # Default output filenames
    if args.output_plot is None and args.no_show:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_plot = f"quality_report_{timestamp}.png"

    # Load data
    print("Loading data...")
    loader = CSVDataLoader(args.path)
    data = loader.load_all(args.results, args.eval, args.tracks)

    # Generate report
    print("Generating track quality report...")
    generate_track_quality_report(data,
                                  output_plot=args.output_plot,
                                  output_text=args.output_text,
                                  show=not args.no_show)

    print("\nDone!")


if __name__ == '__main__':
    main()
