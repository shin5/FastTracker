"""
Trajectory Player for FastTracker

Interactive GUI for replaying track trajectories from CSV logs.
"""

import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from collections import deque
import argparse
from pathlib import Path
from .data_loader import CSVDataLoader


class TrajectoryPlayer(QtWidgets.QMainWindow):
    """Main window for trajectory playback."""

    def __init__(self, data, playback_speed=1.0, trail_length=30):
        """
        Initialize the trajectory player.

        Args:
            data: TrackingData object from data_loader
            playback_speed: Playback speed multiplier (1.0 = real-time)
            trail_length: Number of past positions to show as trail
        """
        super().__init__()
        self.data = data
        self.playback_speed = playback_speed
        self.trail_length = trail_length

        # Playback state
        self.current_frame = 0
        self.is_playing = False
        self.total_frames = len(data.frame_data) if data.frame_data is not None else 0

        # Track visualization data
        self.track_plots = {}  # track_id -> plot items
        self.track_trails = {}  # track_id -> deque of (x, y) positions
        self.track_arrows = {}  # track_id -> arrow items
        self.ground_truth_plots = {}  # target_id -> plot items
        self.ground_truth_trails = {}  # target_id -> deque of (x, y) positions

        # Color schemes
        self.colors = {
            'confirmed': (0, 255, 0, 200),    # Green
            'tentative': (255, 255, 0, 200),  # Yellow
            'lost': (255, 0, 0, 200),         # Red
            'ground_truth': (255, 255, 255, 150)  # White (semi-transparent)
        }

        # Setup UI
        self.setup_ui()

        # Animation timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Load first frame
        if self.total_frames > 0:
            self.update_display()

    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle('FastTracker Trajectory Player')
        self.resize(1280, 800)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Top: Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.setLabel('left', 'Y Position (m)')
        self.plot_widget.setLabel('bottom', 'X Position (m)')
        self.plot_widget.setTitle('Track Trajectories')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        main_layout.addWidget(self.plot_widget)

        # Middle: Frame slider
        slider_layout = QtWidgets.QHBoxLayout()
        slider_label = QtWidgets.QLabel('Frame:')
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        self.frame_label = QtWidgets.QLabel(f'0 / {self.total_frames}')
        self.frame_label.setMinimumWidth(100)

        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_label)
        main_layout.addLayout(slider_layout)

        # Bottom: Control buttons
        control_layout = QtWidgets.QHBoxLayout()

        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_play)

        self.stop_button = QtWidgets.QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop)

        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(['0.25x', '0.5x', '1x', '2x', '5x'])
        self.speed_combo.setCurrentText('1x')
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)

        self.info_label = QtWidgets.QLabel('Ready')
        self.info_label.setStyleSheet('QLabel { color: white; padding: 5px; }')

        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(QtWidgets.QLabel('Speed:'))
        control_layout.addWidget(self.speed_combo)
        control_layout.addStretch()
        control_layout.addWidget(self.info_label)

        main_layout.addLayout(control_layout)

        # Legend
        legend_layout = QtWidgets.QHBoxLayout()
        legend_layout.addWidget(QtWidgets.QLabel('<span style="color: #00FF00;">■</span> Confirmed'))
        legend_layout.addWidget(QtWidgets.QLabel('<span style="color: #FFFF00;">■</span> Tentative'))
        legend_layout.addWidget(QtWidgets.QLabel('<span style="color: #FF0000;">■</span> Lost'))
        legend_layout.addWidget(QtWidgets.QLabel('<span style="color: #FFFFFF;">○</span> Ground Truth'))
        legend_layout.addStretch()
        main_layout.addLayout(legend_layout)

    def toggle_play(self):
        """Toggle between play and pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        """Start playback."""
        self.is_playing = True
        self.play_button.setText('Pause')

        # Calculate timer interval based on frame rate and playback speed
        if self.data.frame_data is not None and len(self.data.frame_data) > 1:
            time_diff = self.data.frame_data['time'].iloc[1] - self.data.frame_data['time'].iloc[0]
            interval_ms = int((time_diff * 1000) / self.playback_speed)
            self.timer.start(max(interval_ms, 16))  # Minimum 16ms (60 FPS)
        else:
            self.timer.start(33)  # Default 30 FPS

    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self.play_button.setText('Play')
        self.timer.stop()

    def stop(self):
        """Stop playback and reset to frame 0."""
        self.pause()
        self.current_frame = 0
        self.frame_slider.setValue(0)
        self.update_display()

    def update_frame(self):
        """Update to next frame (called by timer)."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self.update_display()
        else:
            # Reached end, stop playback
            self.pause()

    def on_slider_changed(self, value):
        """Handle slider value change."""
        self.current_frame = value
        self.update_display()

    def on_speed_changed(self, text):
        """Handle playback speed change."""
        speed_map = {
            '0.25x': 0.25,
            '0.5x': 0.5,
            '1x': 1.0,
            '2x': 2.0,
            '5x': 5.0
        }
        self.playback_speed = speed_map.get(text, 1.0)

        # Restart timer if playing
        if self.is_playing:
            self.pause()
            self.play()

    def update_display(self):
        """Update the display for the current frame."""
        if self.data.frame_data is None or self.current_frame >= self.total_frames:
            return

        # Get current time
        current_time = self.data.frame_data.iloc[self.current_frame]['time']

        # Update frame label
        self.frame_label.setText(f'{self.current_frame} / {self.total_frames} ({current_time:.2f}s)')

        # Update info label
        num_tracks = self.data.frame_data.iloc[self.current_frame]['num_tracks']
        num_confirmed = self.data.frame_data.iloc[self.current_frame]['num_confirmed']
        self.info_label.setText(f'Tracks: {num_tracks} | Confirmed: {num_confirmed} | Time: {current_time:.2f}s')

        # Get tracks at current time from track_trajectories
        current_tracks = self.get_tracks_at_time(current_time)

        # Get ground truth at current time
        current_ground_truth = self.get_ground_truth_at_time(current_time)

        # Update track visualizations
        self.update_tracks(current_tracks)

        # Update ground truth visualizations
        self.update_ground_truth(current_ground_truth)

    def get_tracks_at_time(self, time):
        """Get all tracks that exist at the specified time."""
        tracks = []

        for track_id, trajectory in self.data.track_trajectories.items():
            # Find the closest time entry
            time_diffs = np.abs(trajectory['time'].values - time)
            closest_idx = np.argmin(time_diffs)

            if time_diffs[closest_idx] < 0.05:  # Within 50ms
                track_data = trajectory.iloc[closest_idx]
                tracks.append({
                    'id': track_id,
                    'x': track_data['x'],
                    'y': track_data['y'],
                    'vx': track_data['vx'],
                    'vy': track_data['vy'],
                    'state': int(track_data['state'])
                })

        return tracks

    def get_ground_truth_at_time(self, time):
        """Get all ground truth targets that exist at the specified time."""
        targets = []

        for target_id, trajectory in self.data.ground_truth_trajectories.items():
            # Find the closest time entry
            time_diffs = np.abs(trajectory['time'].values - time)
            closest_idx = np.argmin(time_diffs)

            if time_diffs[closest_idx] < 0.05:  # Within 50ms
                target_data = trajectory.iloc[closest_idx]
                targets.append({
                    'id': target_id,
                    'x': target_data['x'],
                    'y': target_data['y']
                })

        return targets

    def update_tracks(self, tracks):
        """Update track visualizations."""
        # Clear old tracks that don't exist anymore
        current_track_ids = {track['id'] for track in tracks}
        old_track_ids = set(self.track_plots.keys())
        removed_ids = old_track_ids - current_track_ids

        for track_id in removed_ids:
            if track_id in self.track_plots:
                self.plot_widget.removeItem(self.track_plots[track_id])
                del self.track_plots[track_id]
            if track_id in self.track_arrows:
                self.plot_widget.removeItem(self.track_arrows[track_id])
                del self.track_arrows[track_id]
            if track_id in self.track_trails:
                del self.track_trails[track_id]

        # Update or create tracks
        for track in tracks:
            track_id = track['id']
            x, y = track['x'], track['y']
            vx, vy = track['vx'], track['vy']
            state = track['state']

            # Get color based on state
            if state == 1:  # CONFIRMED
                color = self.colors['confirmed']
            elif state == 2:  # LOST
                color = self.colors['lost']
            else:  # TENTATIVE
                color = self.colors['tentative']

            # Update or create position marker
            if track_id not in self.track_plots:
                self.track_plots[track_id] = pg.ScatterPlotItem(
                    size=10, brush=pg.mkBrush(color))
                self.plot_widget.addItem(self.track_plots[track_id])
                self.track_trails[track_id] = deque(maxlen=self.trail_length)

            self.track_plots[track_id].setData([x], [y])
            self.track_plots[track_id].setBrush(pg.mkBrush(color))

            # Update trail
            self.track_trails[track_id].append((x, y))
            if len(self.track_trails[track_id]) > 1:
                trail_x = [pos[0] for pos in self.track_trails[track_id]]
                trail_y = [pos[1] for pos in self.track_trails[track_id]]

                # Create or update trail plot
                trail_key = f'trail_{track_id}'
                if hasattr(self, 'trail_plots') and trail_key in self.trail_plots:
                    self.plot_widget.removeItem(self.trail_plots[trail_key])

                if not hasattr(self, 'trail_plots'):
                    self.trail_plots = {}

                trail_color = (*color[:3], 100)  # More transparent
                self.trail_plots[trail_key] = pg.PlotDataItem(
                    trail_x, trail_y, pen=pg.mkPen(trail_color, width=2))
                self.plot_widget.addItem(self.trail_plots[trail_key])

            # Update velocity arrow
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 1.0:  # Only show arrow if moving
                arrow_length = min(speed * 10, 5000)  # Scale arrow
                angle = np.arctan2(vy, vx)
                dx = arrow_length * np.cos(angle)
                dy = arrow_length * np.sin(angle)

                arrow_key = f'arrow_{track_id}'
                if hasattr(self, 'arrow_plots') and arrow_key in self.arrow_plots:
                    self.plot_widget.removeItem(self.arrow_plots[arrow_key])

                if not hasattr(self, 'arrow_plots'):
                    self.arrow_plots = {}

                arrow = pg.ArrowItem(
                    angle=np.degrees(angle),
                    tipAngle=30,
                    tailLen=arrow_length / 1000,
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen(color)
                )
                arrow.setPos(x, y)
                self.arrow_plots[arrow_key] = arrow
                self.plot_widget.addItem(arrow)

    def update_ground_truth(self, targets):
        """Update ground truth visualizations."""
        # Clear old ground truth that don't exist anymore
        current_target_ids = {target['id'] for target in targets}
        old_target_ids = set(self.ground_truth_plots.keys())
        removed_ids = old_target_ids - current_target_ids

        for target_id in removed_ids:
            if target_id in self.ground_truth_plots:
                self.plot_widget.removeItem(self.ground_truth_plots[target_id])
                del self.ground_truth_plots[target_id]
            if target_id in self.ground_truth_trails:
                del self.ground_truth_trails[target_id]

        # Update or create ground truth targets
        color = self.colors['ground_truth']

        for target in targets:
            target_id = target['id']
            x, y = target['x'], target['y']

            # Update or create position marker (smaller and semi-transparent)
            if target_id not in self.ground_truth_plots:
                self.ground_truth_plots[target_id] = pg.ScatterPlotItem(
                    size=6, brush=pg.mkBrush(color), symbol='o')
                self.plot_widget.addItem(self.ground_truth_plots[target_id])
                self.ground_truth_trails[target_id] = deque(maxlen=self.trail_length)

            self.ground_truth_plots[target_id].setData([x], [y])

            # Update trail
            self.ground_truth_trails[target_id].append((x, y))
            if len(self.ground_truth_trails[target_id]) > 1:
                trail_x = [pos[0] for pos in self.ground_truth_trails[target_id]]
                trail_y = [pos[1] for pos in self.ground_truth_trails[target_id]]

                # Create or update trail plot (dashed line)
                trail_key = f'gt_trail_{target_id}'
                if hasattr(self, 'gt_trail_plots') and trail_key in self.gt_trail_plots:
                    self.plot_widget.removeItem(self.gt_trail_plots[trail_key])

                if not hasattr(self, 'gt_trail_plots'):
                    self.gt_trail_plots = {}

                # Dashed line for ground truth
                gt_pen = pg.mkPen(color, width=1, style=pg.QtCore.Qt.DashLine)
                self.gt_trail_plots[trail_key] = pg.PlotDataItem(
                    trail_x, trail_y, pen=gt_pen)
                self.plot_widget.addItem(self.gt_trail_plots[trail_key])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='FastTracker Trajectory Player')
    parser.add_argument('--path', type=str, default='.',
                       help='Path to directory containing CSV files')
    parser.add_argument('--results', type=str, default='results.csv',
                       help='Results CSV filename')
    parser.add_argument('--eval', type=str, default='evaluation_results.csv',
                       help='Evaluation CSV filename')
    parser.add_argument('--tracks', type=str, default='track_details.csv',
                       help='Track details CSV filename')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Initial playback speed (default: 1.0)')
    parser.add_argument('--trail', type=int, default=30,
                       help='Trail length in frames (default: 30)')

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    loader = CSVDataLoader(args.path)
    data = loader.load_all(args.results, args.eval, args.tracks)

    if not data.track_trajectories:
        print("ERROR: No track trajectories loaded. Cannot display trajectory player.")
        print("Make sure track_details.csv exists and contains track data.")
        sys.exit(1)

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)

    # Create and show player
    player = TrajectoryPlayer(data, playback_speed=args.speed, trail_length=args.trail)
    player.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
