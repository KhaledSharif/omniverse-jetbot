"""
Jetbot Mobile Robot Control with PyInput Keyboard Control and Rich TUI

This example demonstrates:
- Loading and controlling a Jetbot differential drive robot
- PyInput keyboard integration for external keyboard input
- Rich TUI with live-updating interface and visual button feedback
- Navigation control with linear and angular velocity
- Demonstration recording for imitation learning

Controls:
    Movement:
        W: Move forward
        S: Move backward
        A: Turn left
        D: Turn right
        Space: Stop (emergency brake)

    Recording:
        ` (backtick): Toggle recording on/off
        [: Mark episode as success
        ]: Mark episode as failure

    System:
        R: Reset robot to start position
        G: Spawn new random goal
        Esc: Exit application
"""

from isaacsim import SimulationApp

# SimulationApp will be created in JetbotKeyboardController.__init__()
# This allows tests to mock it before creation
simulation_app = None

import argparse
import numpy as np
import threading
import sys
import termios
from pynput import keyboard

# Isaac Sim imports must happen AFTER SimulationApp is created
# They will be imported inside __init__ after app initialization
World = None
ArticulationAction = None
WheeledRobot = None
DifferentialController = None
get_assets_root_path = None

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TUIRenderer:  # pragma: no cover
    """Renders the terminal user interface for robot control using Rich library."""

    def __init__(self):  # pragma: no cover
        """Initialize the TUI renderer."""
        self.pressed_keys = set()

        # Telemetry data
        self.position = np.array([0.0, 0.0])  # x, y
        self.heading = 0.0  # theta in radians
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.goal_position = np.array([0.0, 0.0])
        self.distance_to_goal = 0.0
        self.angle_to_goal = 0.0
        self.last_command = "Ready"

        # Recording status
        self.recording_active = False
        self.recording_stats = {}
        self.checkpoint_flash_active = False
        self.recording_enabled = False

    def set_pressed_key(self, key):  # pragma: no cover
        """Mark a key as pressed for highlighting."""
        self.pressed_keys.add(key)

    def clear_pressed_key(self, key):  # pragma: no cover
        """Mark a key as released."""
        self.pressed_keys.discard(key)

    def update_telemetry(self, position, heading, linear_vel, angular_vel,
                         goal_position=None, distance_to_goal=0.0, angle_to_goal=0.0):  # pragma: no cover
        """Update telemetry values."""
        self.position = position[:2] if len(position) > 2 else position
        self.heading = heading
        self.linear_velocity = linear_vel
        self.angular_velocity = angular_vel
        if goal_position is not None:
            self.goal_position = goal_position[:2] if len(goal_position) > 2 else goal_position
        self.distance_to_goal = distance_to_goal
        self.angle_to_goal = angle_to_goal

    def set_last_command(self, command):  # pragma: no cover
        """Set the last executed command."""
        self.last_command = command

    def set_recording_status(self, is_recording: bool, stats: dict):  # pragma: no cover
        """Update recording status for display."""
        self.recording_active = is_recording
        self.recording_stats = stats.copy() if stats else {}

    def set_checkpoint_flash(self, active: bool):  # pragma: no cover
        """Set checkpoint flash indicator state."""
        self.checkpoint_flash_active = active

    def set_recording_enabled(self, enabled: bool):  # pragma: no cover
        """Set whether recording mode is enabled."""
        self.recording_enabled = enabled

    def _render_recording_panel(self) -> Panel:  # pragma: no cover
        """Render recording status panel with controls."""
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold")
        table.add_column(justify="left")

        # Recording status indicator
        if self.recording_active:
            status_text = Text("REC", style="bold red blink")
        else:
            status_text = Text("IDLE", style="dim")

        # Add checkpoint flash indicator
        if self.checkpoint_flash_active:
            status_row = Text.assemble(status_text, " ", Text("SAVED", style="bold green reverse"))
        else:
            status_row = status_text

        table.add_row(Text("STATUS:", style="bold underline"), status_row)
        table.add_row("", "")

        # Episode stats
        total_frames = self.recording_stats.get('total_frames', 0)
        current_frames = self.recording_stats.get('current_episode_frames', 0)
        num_episodes = self.recording_stats.get('num_episodes', 0)
        num_success = self.recording_stats.get('num_success', 0)
        num_failed = self.recording_stats.get('num_failed', 0)
        current_return = self.recording_stats.get('current_episode_return', 0.0)

        table.add_row(Text("EPISODES:", style="bold underline"), "")
        table.add_row("Total:", f"{num_episodes}")
        table.add_row("Success:", Text(f"{num_success}", style="green"))
        table.add_row("Failed:", Text(f"{num_failed}", style="red"))
        table.add_row("", "")

        table.add_row(Text("FRAMES:", style="bold underline"), "")
        table.add_row("Total:", f"{total_frames}")
        table.add_row("Current Ep:", f"{current_frames}")
        table.add_row("Return:", f"{current_return:.2f}")
        table.add_row("", "")

        # Recording controls
        table.add_row(Text("CONTROLS:", style="bold underline"), "")

        backtick_btn = self._create_button("`", "`", width=4)
        table.add_row(backtick_btn, Text("Start/Stop", style="dim"))

        left_bracket_btn = self._create_button("[", "[", width=4)
        table.add_row(left_bracket_btn, Text("Mark Success", style="green dim"))

        right_bracket_btn = self._create_button("]", "]", width=4)
        table.add_row(right_bracket_btn, Text("Mark Failed", style="red dim"))

        table.add_row("", "")
        table.add_row(Text("AUTO-SAVE:", style="dim italic"), Text("Every 5s", style="dim italic"))

        border_style = "red" if self.recording_active else "dim"
        return Panel(table, title="Recording", border_style=border_style)

    def _create_button(self, label, key, width=6):  # pragma: no cover
        """Create a button with optional highlighting."""
        is_pressed = key in self.pressed_keys
        text = f" {label:^{width-2}} "

        if is_pressed:
            return Text(text, style="reverse bold")
        else:
            return Text(text, style="dim")

    def _render_controls(self):  # pragma: no cover
        """Render navigation control panel."""
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="center")
        table.add_column(justify="center")
        table.add_column(justify="center")

        # Title
        title = Text("NAVIGATION CONTROL", style="bold cyan")
        table.add_row(Text("", style=""))
        table.add_row(title, "", "")
        table.add_row(Text("", style=""))

        # W - Forward
        table.add_row("", self._create_button("W", "w"), "")
        table.add_row("", Text("Forward", style="dim italic"), "")
        table.add_row(Text("", style=""))

        # A, S, D row
        table.add_row(
            self._create_button("A", "a"),
            self._create_button("S", "s"),
            self._create_button("D", "d")
        )
        table.add_row(
            Text("Left", style="dim italic"),
            Text("Back", style="dim italic"),
            Text("Right", style="dim italic")
        )
        table.add_row(Text("", style=""))

        # Space - Stop
        table.add_row("", self._create_button("SPACE", "space", width=10), "")
        table.add_row("", Text("Stop", style="dim italic red"), "")

        # Separator
        table.add_row(Text("", style=""))
        table.add_row(Text("-" * 30, style="dim"))
        table.add_row(Text("", style=""))

        # Secondary controls
        table.add_row(
            self._create_button("R", "r"),
            Text("Reset Position", style="dim"),
            ""
        )
        table.add_row(
            self._create_button("G", "g"),
            Text("New Goal", style="dim"),
            ""
        )
        table.add_row(
            Text("[Esc]", style="yellow"),
            Text("Exit", style="dim"),
            ""
        )

        return Panel(table, title="Controls", border_style="blue")

    def _render_state_panel(self):  # pragma: no cover
        """Render robot state telemetry panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan")
        table.add_column(justify="left")

        # Position data
        table.add_row(Text("POSITION (m)", style="bold underline"), "")
        table.add_row("X:", f"{self.position[0]:.3f}")
        table.add_row("Y:", f"{self.position[1]:.3f}")
        table.add_row("", "")

        # Orientation data (convert to degrees)
        heading_deg = np.degrees(self.heading)
        table.add_row(Text("HEADING", style="bold underline"), "")
        table.add_row("Theta:", f"{heading_deg:>7.1f} deg")
        table.add_row("", "")

        # Velocity data
        table.add_row(Text("VELOCITY", style="bold underline"), "")
        table.add_row("Linear:", f"{self.linear_velocity:>6.3f} m/s")
        table.add_row("Angular:", f"{self.angular_velocity:>6.3f} rad/s")
        table.add_row("", "")

        # Goal data
        table.add_row(Text("GOAL", style="bold underline"), "")
        table.add_row("Goal X:", f"{self.goal_position[0]:.3f}")
        table.add_row("Goal Y:", f"{self.goal_position[1]:.3f}")
        table.add_row("Distance:", f"{self.distance_to_goal:.3f} m")

        angle_deg = np.degrees(self.angle_to_goal)
        table.add_row("Angle:", f"{angle_deg:>6.1f} deg")

        # Goal reached indicator
        if self.distance_to_goal < 0.15:
            table.add_row("", Text("GOAL REACHED!", style="bold green blink"))
        table.add_row("", "")

        # Last command
        table.add_row(Text("LAST COMMAND", style="bold underline"), "")
        table.add_row("", Text(self.last_command, style="italic"))

        return Panel(table, title="Robot State", border_style="green")

    def render(self):  # pragma: no cover
        """Render the complete TUI layout."""
        # Create main layout
        layout = Layout()

        # Mode header
        mode_text = Text("JETBOT NAVIGATION", style="bold white on blue", justify="center")

        # Split into rows: header + content
        layout.split_column(
            Layout(Panel(mode_text, border_style="bright_blue"), size=3, name="header"),
            Layout(name="content")
        )

        # Split content based on whether recording is enabled
        if self.recording_enabled:
            layout["content"].split_row(
                Layout(name="controls", ratio=35),
                Layout(name="recording", ratio=25),
                Layout(name="state", ratio=40)
            )
            layout["recording"].update(self._render_recording_panel())
        else:
            layout["content"].split_row(
                Layout(name="controls", ratio=40),
                Layout(name="state", ratio=60)
            )

        # Render controls
        layout["controls"].update(self._render_controls())

        # Render state panel
        layout["state"].update(self._render_state_panel())

        return layout


class SceneManager:
    """Manages scene objects for navigation tasks.

    This class handles spawning and managing objects in the Isaac Sim scene,
    including goal markers for target locations.

    Attributes:
        world: The Isaac Sim World object
        goal_marker: The visual goal marker (or None)
        goal_position: Target position as numpy array (or None)
        workspace_bounds: Dict defining the workspace boundaries
    """

    DEFAULT_WORKSPACE_BOUNDS = {
        'x': [-2.0, 2.0],
        'y': [-2.0, 2.0],
    }

    def __init__(self, world, workspace_bounds: dict = None):
        """Initialize the SceneManager.

        Args:
            world: The Isaac Sim World object
            workspace_bounds: Optional dict with 'x', 'y' keys, each containing
                             [min, max] bounds for random position generation
        """
        self.world = world
        self.goal_marker = None
        self.goal_position = None
        self.workspace_bounds = workspace_bounds or self.DEFAULT_WORKSPACE_BOUNDS.copy()
        self.goal_counter = 0

    def spawn_goal_marker(self, position: list = None, color: tuple = (0, 1, 0, 0.5)) -> np.ndarray:
        """Spawn a visual goal marker in the scene.

        Args:
            position: [x, y, z] position. If None, uses random position.
            color: RGBA tuple (default semi-transparent green)

        Returns:
            The goal position as numpy array
        """
        if position is None:
            position = self._random_position()

        self.goal_position = np.array(position)
        self.goal_counter += 1

        prim_path = f"/World/GoalMarker_{self.goal_counter:03d}"

        # Try to use real Isaac Sim primitives
        try:
            from isaacsim.core.api.objects import VisualCuboid
            self.goal_marker = self.world.scene.add(
                VisualCuboid(
                    prim_path=prim_path,
                    name=f"goal_marker_{self.goal_counter:03d}",
                    position=self.goal_position,
                    scale=np.array([0.15, 0.15, 0.02]),  # Flat marker on ground
                    color=np.array(color[:3])
                )
            )
        except ImportError:
            # Fall back to mock for testing
            marker_mock = type('GoalMarker', (), {
                'name': 'goal_marker',
                'position': self.goal_position,
                'color': color
            })()
            self.goal_marker = self.world.scene.add(marker_mock)

        return self.goal_position

    def get_goal_position(self) -> np.ndarray:
        """Get the goal position.

        Returns:
            Goal position as numpy array, or None if not set
        """
        return self.goal_position

    def reset_goal(self) -> np.ndarray:
        """Reset the goal to a new random position.

        Returns:
            New goal position as numpy array
        """
        return self.spawn_goal_marker()

    def check_goal_reached(self, robot_position: np.ndarray, threshold: float = 0.15) -> bool:
        """Check if the robot has reached the goal.

        Args:
            robot_position: Current robot position [x, y] or [x, y, z]
            threshold: Maximum distance to consider goal reached

        Returns:
            True if robot is at goal position, False otherwise
        """
        if self.goal_position is None:
            return False

        robot_pos_2d = robot_position[:2]
        goal_pos_2d = self.goal_position[:2]
        distance = np.linalg.norm(robot_pos_2d - goal_pos_2d)

        return distance < threshold

    def _random_position(self) -> list:
        """Generate a random position within workspace bounds.

        Returns:
            [x, y, z] position list (z=0.01 for ground level)
        """
        x = np.random.uniform(self.workspace_bounds['x'][0], self.workspace_bounds['x'][1])
        y = np.random.uniform(self.workspace_bounds['y'][0], self.workspace_bounds['y'][1])
        z = 0.01  # Slightly above ground

        return [x, y, z]


class DemoRecorder:
    """Records demonstrations for imitation learning and RL training.

    This class captures state-action trajectories during teleoperation,
    with support for episode segmentation and success/failure labeling.

    Attributes:
        obs_dim: Dimension of observation vectors
        action_dim: Dimension of action vectors
        observations: List of recorded observations
        actions: List of recorded actions
        rewards: List of recorded rewards
        dones: List of done flags
        is_recording: Whether recording is active
    """

    def __init__(self, obs_dim: int, action_dim: int):
        """Initialize the DemoRecorder.

        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Data buffers
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

        # Episode tracking
        self.episode_starts = []
        self.episode_lengths = []
        self.episode_returns = []
        self.episode_success = []

        # Recording state
        self.is_recording = False
        self.current_episode_start = 0
        self.current_episode_return = 0.0
        self._pending_success = None

    def start_recording(self) -> None:
        """Start recording a new episode."""
        self.is_recording = True
        self.current_episode_start = len(self.observations)
        self.current_episode_return = 0.0
        self._pending_success = None

    def stop_recording(self) -> None:
        """Stop recording."""
        self.is_recording = False

    def record_step(self, obs: np.ndarray, action: np.ndarray,
                    reward: float, done: bool) -> None:
        """Record a single timestep.

        Args:
            obs: Observation vector
            action: Action vector
            reward: Reward value
            done: Whether episode is done
        """
        if not self.is_recording:
            return

        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.current_episode_return += reward

    def mark_episode_success(self, success: bool) -> None:
        """Mark the current episode as successful or failed.

        Args:
            success: True if episode was successful
        """
        self._pending_success = success

    def finalize_episode(self) -> None:
        """Finalize the current episode and record its metadata."""
        episode_length = len(self.observations) - self.current_episode_start

        if episode_length > 0:
            self.episode_starts.append(self.current_episode_start)
            self.episode_lengths.append(episode_length)
            self.episode_returns.append(self.current_episode_return)

            # Default to False if not explicitly marked
            success = self._pending_success if self._pending_success is not None else False
            self.episode_success.append(success)

        # Reset for next episode
        self.current_episode_start = len(self.observations)
        self.current_episode_return = 0.0
        self._pending_success = None

    def get_stats(self) -> dict:
        """Get current recording statistics.

        Returns:
            Dictionary with recording statistics
        """
        num_success = sum(1 for s in self.episode_success if s)
        num_failed = len(self.episode_success) - num_success

        return {
            'is_recording': self.is_recording,
            'total_frames': len(self.observations),
            'current_episode_frames': len(self.observations) - self.current_episode_start,
            'current_episode_return': self.current_episode_return,
            'num_episodes': len(self.episode_starts),
            'num_success': num_success,
            'num_failed': num_failed,
        }

    def clear(self) -> None:
        """Clear all recorded data and reset state."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.episode_starts = []
        self.episode_lengths = []
        self.episode_returns = []
        self.episode_success = []
        self.is_recording = False
        self.current_episode_start = 0
        self.current_episode_return = 0.0
        self._pending_success = None

    def save(self, filepath: str, metadata: dict = None) -> None:
        """Save demonstrations to NPZ file.

        Args:
            filepath: Path to save the .npz file
            metadata: Optional dictionary of additional metadata
        """
        # Auto-finalize any pending episode data
        pending_frames = len(self.observations) - self.current_episode_start
        if pending_frames > 0:
            if self._pending_success is None:
                self._pending_success = False
            self.finalize_episode()

        # Build metadata
        save_metadata = {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'num_episodes': len(self.episode_starts),
            'total_frames': len(self.observations),
        }
        if metadata:
            save_metadata.update(metadata)

        # Convert lists to arrays
        if self.observations:
            obs_array = np.array(self.observations)
            action_array = np.array(self.actions)
        else:
            obs_array = np.array([]).reshape(0, self.obs_dim)
            action_array = np.array([]).reshape(0, self.action_dim)

        np.savez_compressed(
            filepath,
            observations=obs_array,
            actions=action_array,
            rewards=np.array(self.rewards, dtype=np.float32),
            dones=np.array(self.dones, dtype=bool),
            episode_starts=np.array(self.episode_starts, dtype=np.int64),
            episode_lengths=np.array(self.episode_lengths, dtype=np.int64),
            episode_returns=np.array(self.episode_returns, dtype=np.float32),
            episode_success=np.array(self.episode_success, dtype=bool),
            metadata=np.array(save_metadata, dtype=object),
        )

    @classmethod
    def load(cls, filepath: str) -> 'DemoRecorder':
        """Load demonstrations from NPZ file.

        Args:
            filepath: Path to the .npz file

        Returns:
            DemoRecorder instance with loaded data
        """
        data = np.load(filepath, allow_pickle=True)
        metadata = data['metadata'].item()

        recorder = cls(
            obs_dim=metadata['obs_dim'],
            action_dim=metadata['action_dim']
        )

        recorder.observations = list(data['observations'])
        recorder.actions = list(data['actions'])
        recorder.rewards = list(data['rewards'])
        recorder.dones = list(data['dones'])
        recorder.episode_starts = list(data['episode_starts'])
        recorder.episode_lengths = list(data['episode_lengths'])
        recorder.episode_returns = list(data['episode_returns'])
        recorder.episode_success = list(data['episode_success'])

        recorder.current_episode_start = len(recorder.observations)

        return recorder


class ActionMapper:
    """Maps keyboard commands to normalized action space.

    Action space layout (2D):
        [0]: linear_velocity  - Forward/backward speed (m/s)
        [1]: angular_velocity - Left/right turn rate (rad/s)
    """

    # Key to action mapping
    # Each key maps to a 2D action vector [linear_vel, angular_vel]
    KEY_TO_ACTION = {
        'w': [1.0, 0.0],     # Forward
        's': [-1.0, 0.0],    # Backward
        'a': [0.0, 1.0],     # Turn left
        'd': [0.0, -1.0],    # Turn right
        'space': [0.0, 0.0], # Stop
    }

    def __init__(self):
        """Initialize the ActionMapper."""
        self.action_dim = 2

    def map_key(self, key: str) -> np.ndarray:
        """Map a keyboard key to an action vector.

        Args:
            key: The key character (e.g., 'w', 'a', 's', 'd')

        Returns:
            2D action vector as float32 numpy array
        """
        if key is None or key not in self.KEY_TO_ACTION:
            return np.zeros(self.action_dim, dtype=np.float32)

        return np.array(self.KEY_TO_ACTION[key], dtype=np.float32)


class ObservationBuilder:
    """Builds observation vectors from robot state.

    Observation layout (10D):
        [0:2]  - robot position (x, y)
        [2]    - robot heading (theta)
        [3]    - linear velocity
        [4]    - angular velocity
        [5:7]  - goal position (x, y)
        [7]    - distance to goal
        [8]    - angle to goal (relative heading)
        [9]    - goal reached flag
    """

    def __init__(self):
        """Initialize the ObservationBuilder."""
        self.obs_dim = 10

    def build(self, robot_position: np.ndarray, robot_heading: float,
              linear_velocity: float, angular_velocity: float,
              goal_position: np.ndarray, goal_reached: bool) -> np.ndarray:
        """Build an observation vector from robot state.

        Args:
            robot_position: Robot [x, y] or [x, y, z] position
            robot_heading: Robot heading angle in radians
            linear_velocity: Current linear velocity (m/s)
            angular_velocity: Current angular velocity (rad/s)
            goal_position: Goal [x, y] or [x, y, z] position
            goal_reached: Whether goal has been reached

        Returns:
            10D observation vector as float32
        """
        # Extract 2D positions
        robot_pos_2d = np.array(robot_position[:2], dtype=np.float32)
        goal_pos_2d = np.array(goal_position[:2], dtype=np.float32)

        # Compute distance to goal
        dist_to_goal = np.linalg.norm(robot_pos_2d - goal_pos_2d)

        # Compute angle to goal (relative to robot heading)
        delta = goal_pos_2d - robot_pos_2d
        goal_angle_world = np.arctan2(delta[1], delta[0])
        angle_to_goal = goal_angle_world - robot_heading
        # Normalize to [-pi, pi]
        angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))

        obs = np.array([
            robot_pos_2d[0],           # [0] x
            robot_pos_2d[1],           # [1] y
            robot_heading,             # [2] theta
            linear_velocity,           # [3] linear vel
            angular_velocity,          # [4] angular vel
            goal_pos_2d[0],            # [5] goal x
            goal_pos_2d[1],            # [6] goal y
            dist_to_goal,              # [7] distance
            angle_to_goal,             # [8] angle to goal
            1.0 if goal_reached else 0.0,  # [9] goal reached flag
        ], dtype=np.float32)

        return obs


class RewardComputer:
    """Computes rewards for navigation task.

    Supports both sparse and dense reward modes.
    """

    # Reward constants
    GOAL_REACHED_REWARD = 10.0
    DISTANCE_SCALE = 1.0
    HEADING_BONUS_SCALE = 0.1

    def __init__(self, mode: str = 'dense'):
        """Initialize the RewardComputer.

        Args:
            mode: 'dense' for shaped rewards, 'sparse' for only goal completion
        """
        self.mode = mode

    def compute(self, obs: np.ndarray, action: np.ndarray,
                next_obs: np.ndarray, info: dict) -> float:
        """Compute reward for a transition.

        Args:
            obs: Previous observation
            action: Action taken
            next_obs: Resulting observation
            info: Additional info dict with flags like 'goal_reached'

        Returns:
            Scalar reward value
        """
        if self.mode == 'sparse':
            return self._sparse_reward(info)
        else:
            return self._dense_reward(obs, next_obs, info)

    def _sparse_reward(self, info: dict) -> float:
        """Compute sparse reward (only on goal reached)."""
        if info.get('goal_reached', False):
            return self.GOAL_REACHED_REWARD
        return 0.0

    def _dense_reward(self, obs: np.ndarray, next_obs: np.ndarray, info: dict) -> float:
        """Compute dense shaped reward."""
        reward = 0.0

        # Goal reached bonus
        if info.get('goal_reached', False):
            reward += self.GOAL_REACHED_REWARD
            return reward

        # Distance-based shaping (reward getting closer to goal)
        prev_dist = obs[7]  # distance to goal
        curr_dist = next_obs[7]
        reward += (prev_dist - curr_dist) * self.DISTANCE_SCALE

        # Heading bonus (reward facing the goal)
        angle_to_goal = abs(next_obs[8])  # angle to goal
        heading_bonus = (np.pi - angle_to_goal) / np.pi  # 1.0 when facing goal, 0.0 when facing away
        reward += heading_bonus * self.HEADING_BONUS_SCALE

        return reward


class DemoPlayer:
    """Plays back recorded demonstrations.

    This class loads recorded demonstration data and provides methods
    to access individual episodes and filter by success status.
    """

    def __init__(self, filepath: str):
        """Load demonstrations from NPZ file.

        Args:
            filepath: Path to the .npz file containing demonstrations
        """
        data = np.load(filepath, allow_pickle=True)

        self.observations = data['observations']
        self.actions = data['actions']
        self.episode_starts = data['episode_starts']
        self.episode_lengths = data['episode_lengths']
        self.episode_success = data['episode_success']

        self.num_episodes = len(self.episode_starts)
        self.total_frames = len(self.observations)

    def get_episode(self, episode_idx: int) -> tuple:
        """Get observations and actions for a specific episode.

        Args:
            episode_idx: Index of the episode to retrieve

        Returns:
            Tuple of (observations, actions) arrays for the episode

        Raises:
            IndexError: If episode_idx is out of range
        """
        if episode_idx < 0 or episode_idx >= self.num_episodes:
            raise IndexError(f"Episode index {episode_idx} out of range [0, {self.num_episodes})")

        start = self.episode_starts[episode_idx]
        length = self.episode_lengths[episode_idx]
        end = start + length

        return self.observations[start:end], self.actions[start:end]

    def get_successful_episodes(self) -> list:
        """Get indices of all successful episodes.

        Returns:
            List of episode indices where success flag is True
        """
        return [i for i in range(self.num_episodes) if self.episode_success[i]]


class JetbotKeyboardController:
    """Controller for Jetbot robot with keyboard input via pynput and Rich TUI."""

    # Control parameters
    MAX_LINEAR_VELOCITY = 0.3   # m/s
    MAX_ANGULAR_VELOCITY = 1.0  # rad/s

    # Jetbot physical parameters
    WHEEL_RADIUS = 0.03    # meters
    WHEEL_BASE = 0.1125    # meters (distance between wheels)

    # Start position
    START_POSITION = np.array([0.0, 0.0, 0.05])
    START_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion (w, x, y, z)

    def __init__(self, enable_recording: bool = False, demo_path: str = None,
                 reward_mode: str = "dense"):
        """Initialize the Jetbot robot and keyboard controller.

        Args:
            enable_recording: Enable demonstration recording mode
            demo_path: Path to save demonstrations (auto-generated with timestamp if None)
            reward_mode: Reward computation mode ('dense' or 'sparse')
        """
        # Create SimulationApp if not already created (e.g., by tests)
        global simulation_app, World, ArticulationAction, WheeledRobot
        global DifferentialController, get_assets_root_path

        if simulation_app is None:
            simulation_app = SimulationApp({"headless": False})
        self.simulation_app = simulation_app

        # Import Isaac Sim modules after SimulationApp is created
        if World is None:
            from isaacsim.core.api import World as _World
            from isaacsim.core.utils.types import ArticulationAction as _ArticulationAction
            from isaacsim.robot.wheeled_robots.robots import WheeledRobot as _WheeledRobot
            from isaacsim.robot.wheeled_robots.controllers import DifferentialController as _DifferentialController
            from isaacsim.core.utils.nucleus import get_assets_root_path as _get_assets_root_path

            World = _World
            ArticulationAction = _ArticulationAction
            WheeledRobot = _WheeledRobot
            DifferentialController = _DifferentialController
            get_assets_root_path = _get_assets_root_path

        # Create TUI renderer
        self.tui = TUIRenderer()
        self.tui.set_last_command("Initializing...")
        self.tui.set_recording_enabled(enable_recording)

        # Create world
        self.world = World(stage_units_in_meters=1.0)

        # Get assets root path
        assets_root = get_assets_root_path()

        # Create Jetbot
        self.jetbot = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/Jetbot",
                name="my_jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
                position=self.START_POSITION,
                orientation=self.START_ORIENTATION
            )
        )

        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Create differential controller
        self.controller = DifferentialController(
            name="jetbot_controller",
            wheel_radius=self.WHEEL_RADIUS,
            wheel_base=self.WHEEL_BASE
        )

        # Control state
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.should_exit = False

        # Thread-safe command queue
        self.command_lock = threading.Lock()
        self.pending_commands = []

        # Recording configuration
        self.enable_recording = enable_recording
        self.reward_mode = reward_mode

        # Generate timestamped filename if no path provided
        if demo_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.demo_save_path = f"demos/recording_{timestamp}.npz"
        else:
            self.demo_save_path = demo_path

        # Recording components (initialized if enable_recording is True)
        self.recorder = None
        self.action_mapper = None
        self.obs_builder = None
        self.reward_computer = None
        self.current_obs = None

        # Checkpoint/auto-save state
        self.checkpoint_frame_counter = 0
        self.checkpoint_interval_frames = 50  # 5 seconds at ~10 Hz
        self.checkpoint_flash_frames = 0
        self.checkpoint_flash_duration = 10

        # Initialize scene manager for goal spawning
        self.scene_manager = SceneManager(self.world)

        # Initialize recording components if enabled
        if self.enable_recording:
            self._init_recording_components()

        # Terminal settings (for disabling echo)
        self.old_terminal_settings = None

        # Setup keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.listener.start()

        init_msg = "Initialization complete"
        if self.enable_recording:
            init_msg += " [RECORDING MODE]"
        self.tui.set_last_command(init_msg)

    def _init_recording_components(self):
        """Initialize recording components for demonstration collection."""
        # Initialize demo recorder (10D obs, 2D action)
        self.recorder = DemoRecorder(obs_dim=10, action_dim=2)

        # Initialize action mapper and observation builder
        self.action_mapper = ActionMapper()
        self.obs_builder = ObservationBuilder()

        # Initialize reward computer
        self.reward_computer = RewardComputer(mode=self.reward_mode)

        self.tui.set_last_command("Recording components initialized")

    def _setup_recording_scene(self):
        """Set up the scene for recording (spawn goal marker)."""
        if self.scene_manager is not None:
            self.scene_manager.spawn_goal_marker()
            self.tui.set_last_command("Recording scene ready - Press ` to start")

    def _get_robot_pose(self) -> tuple:
        """Get current robot position and heading.

        Returns:
            Tuple of (position, heading) where position is [x, y, z] and heading is radians
        """
        position, orientation = self.jetbot.get_world_pose()

        # Convert quaternion to heading angle (yaw)
        # Quaternion is [w, x, y, z]
        w, x, y, z = orientation
        # Yaw from quaternion
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        heading = np.arctan2(siny_cosp, cosy_cosp)

        return position, heading

    def _build_current_observation(self) -> np.ndarray:
        """Build observation from current robot and scene state."""
        if self.obs_builder is None:
            return None

        # Get robot state
        position, heading = self._get_robot_pose()

        # Get goal state
        goal_position = self.scene_manager.get_goal_position()
        if goal_position is None:
            goal_position = np.zeros(3)

        # Check goal reached
        goal_reached = self.scene_manager.check_goal_reached(position)

        return self.obs_builder.build(
            robot_position=position,
            robot_heading=heading,
            linear_velocity=self.current_linear_vel,
            angular_velocity=self.current_angular_vel,
            goal_position=goal_position,
            goal_reached=goal_reached
        )

    def _record_step(self, action: np.ndarray):
        """Record a single step during demonstration collection."""
        if self.recorder is None or not self.recorder.is_recording:
            return

        # Build next observation
        next_obs = self._build_current_observation()

        # Get robot position for goal check
        position, _ = self._get_robot_pose()
        goal_reached = self.scene_manager.check_goal_reached(position)

        info = {
            'goal_reached': goal_reached
        }

        # Compute reward
        reward = 0.0
        if self.reward_computer is not None and self.current_obs is not None:
            reward = self.reward_computer.compute(
                self.current_obs, action, next_obs, info
            )

        # Record step
        done = goal_reached
        self.recorder.record_step(self.current_obs, action, reward, done)

        # Update state
        self.current_obs = next_obs

        # Update TUI with recording stats
        stats = self.recorder.get_stats()
        self.tui.set_recording_status(self.recorder.is_recording, stats)

        # Auto-finalize episode on goal reached
        if goal_reached:
            self.recorder.mark_episode_success(True)
            self.recorder.finalize_episode()
            self._reset_recording_episode()
            self.tui.set_last_command("Goal reached! Episode finalized.")

    def _reset_recording_episode(self):
        """Reset scene for a new recording episode."""
        if self.scene_manager is not None:
            self.scene_manager.reset_goal()
        self._reset_robot()
        self.current_obs = self._build_current_observation()

    def _on_key_press(self, key):
        """Handle key press events from pynput."""
        try:
            # Handle character keys
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()
                self.tui.set_pressed_key(key_char)
                self._queue_command(('char', key_char))
            # Handle special keys
            elif key == keyboard.Key.space:
                self.tui.set_pressed_key('space')
                self._queue_command(('char', 'space'))
            elif key == keyboard.Key.esc:
                self.tui.set_pressed_key('esc')
                self._queue_command(('special', 'esc'))
        except Exception as e:
            self.tui.set_last_command(f"Error: {e}")

    def _on_key_release(self, key):
        """Handle key release events from pynput."""
        try:
            if hasattr(key, 'char') and key.char:
                self.tui.clear_pressed_key(key.char.lower())
            elif key == keyboard.Key.space:
                self.tui.clear_pressed_key('space')
            elif key == keyboard.Key.esc:
                self.tui.clear_pressed_key('esc')
        except Exception:
            pass

    def _queue_command(self, command):
        """Add a command to the thread-safe queue."""
        with self.command_lock:
            self.pending_commands.append(command)

    def _process_commands(self):
        """Process all pending commands from the queue.

        Returns:
            str or None: The last 'char' command processed, or None if no char commands.
        """
        with self.command_lock:
            commands = self.pending_commands.copy()
            self.pending_commands.clear()

        last_char_cmd = None
        for cmd_type, cmd_value in commands:
            if cmd_type == 'special':
                if cmd_value == 'esc':
                    self.should_exit = True
                    self.tui.set_last_command("Exiting...")
            elif cmd_type == 'char':
                # Recording commands
                if cmd_value in ('`', '[', ']'):
                    self._handle_recording_command(cmd_value)
                # System commands
                elif cmd_value == 'r':
                    self._reset_robot()
                    self.tui.set_last_command("Reset to start position")
                elif cmd_value == 'g':
                    self._spawn_new_goal()
                # Movement commands
                elif cmd_value in ('w', 's', 'a', 'd', 'space'):
                    self._process_movement_command(cmd_value)
                    last_char_cmd = cmd_value

        return last_char_cmd

    def _process_movement_command(self, key: str):
        """Process movement commands."""
        if key == 'w':
            self.current_linear_vel = self.MAX_LINEAR_VELOCITY
            self.current_angular_vel = 0.0
            self.tui.set_last_command("Forward")
        elif key == 's':
            self.current_linear_vel = -self.MAX_LINEAR_VELOCITY
            self.current_angular_vel = 0.0
            self.tui.set_last_command("Backward")
        elif key == 'a':
            self.current_linear_vel = 0.0
            self.current_angular_vel = self.MAX_ANGULAR_VELOCITY
            self.tui.set_last_command("Turn Left")
        elif key == 'd':
            self.current_linear_vel = 0.0
            self.current_angular_vel = -self.MAX_ANGULAR_VELOCITY
            self.tui.set_last_command("Turn Right")
        elif key == 'space':
            self.current_linear_vel = 0.0
            self.current_angular_vel = 0.0
            self.tui.set_last_command("Stop")

    def _handle_recording_command(self, key: str):
        """Handle recording control keys."""
        if key == '`':
            if self.recorder is None:
                return
            if self.recorder.is_recording:
                self.recorder.stop_recording()
                self.tui.set_last_command("Recording stopped")
            else:
                self.recorder.start_recording()
                self.tui.set_last_command("Recording started")

        elif key == '[':
            if self.recorder is not None and len(self.recorder.observations) > self.recorder.current_episode_start:
                self.recorder.mark_episode_success(True)
                self.recorder.finalize_episode()
                self._reset_recording_episode()
                stats = self.recorder.get_stats()
                self.tui.set_recording_status(True, stats)
                self.tui.set_last_command(f"Episode {stats['num_episodes']} SUCCESS - Scene reset")

        elif key == ']':
            if self.recorder is not None and len(self.recorder.observations) > self.recorder.current_episode_start:
                self.recorder.mark_episode_success(False)
                self.recorder.finalize_episode()
                self._reset_recording_episode()
                stats = self.recorder.get_stats()
                self.tui.set_recording_status(True, stats)
                self.tui.set_last_command(f"Episode {stats['num_episodes']} FAILED - Scene reset")

    def _spawn_new_goal(self):
        """Spawn a new random goal marker."""
        if self.scene_manager is not None:
            self.scene_manager.spawn_goal_marker()
            self.tui.set_last_command("New goal spawned")

    def _reset_robot(self):
        """Reset robot to start position."""
        self.jetbot.set_world_pose(
            position=self.START_POSITION,
            orientation=self.START_ORIENTATION
        )
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

    def _apply_control(self):
        """Apply current velocity commands to the robot."""
        # Get wheel velocities from differential controller
        actions = self.controller.forward(
            command=[self.current_linear_vel, self.current_angular_vel]
        )

        # Apply to robot
        self.jetbot.apply_wheel_actions(actions)

    def _perform_checkpoint_save(self) -> bool:
        """Perform checkpoint save of recording data."""
        if self.recorder is None or len(self.recorder.observations) == 0:
            return False

        self.recorder.save(self.demo_save_path)
        self.tui.set_last_command(f"Checkpoint: {len(self.recorder.observations)} frames")
        self.checkpoint_flash_frames = self.checkpoint_flash_duration
        return True

    def _update_tui_state(self):
        """Update TUI with current robot state."""
        position, heading = self._get_robot_pose()

        # Get goal info
        goal_position = self.scene_manager.get_goal_position()
        if goal_position is None:
            goal_position = np.zeros(3)

        # Calculate distance and angle to goal
        robot_pos_2d = position[:2]
        goal_pos_2d = goal_position[:2]
        distance_to_goal = np.linalg.norm(robot_pos_2d - goal_pos_2d)

        delta = goal_pos_2d - robot_pos_2d
        goal_angle_world = np.arctan2(delta[1], delta[0])
        angle_to_goal = goal_angle_world - heading
        angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))

        self.tui.update_telemetry(
            position=position,
            heading=heading,
            linear_vel=self.current_linear_vel,
            angular_vel=self.current_angular_vel,
            goal_position=goal_position,
            distance_to_goal=distance_to_goal,
            angle_to_goal=angle_to_goal
        )

    def _disable_terminal_echo(self):
        """Disable terminal echo to prevent key presses from appearing in TUI."""
        try:
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~termios.ECHO
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
        except Exception as e:
            self.tui.set_last_command(f"Warning: Could not disable echo: {e}")

    def _restore_terminal_settings(self):
        """Restore original terminal settings."""
        try:
            if self.old_terminal_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
        except Exception:
            pass

    def run(self):
        """Main simulation loop with Rich TUI."""
        self.tui.set_last_command("Starting simulation...")

        # Set up recording scene BEFORE first reset if enabled
        if self.enable_recording:
            self._setup_recording_scene()

        # Always spawn a goal marker for navigation
        if self.scene_manager.goal_position is None:
            self.scene_manager.spawn_goal_marker()

        # Reset world after scene setup
        self.world.reset()

        # Build initial observation if recording enabled
        if self.enable_recording:
            self.current_obs = self._build_current_observation()

        reset_needed = False
        last_key_processed = None

        ready_msg = "Ready - WASD to move, Space to stop"
        if self.enable_recording:
            ready_msg += " | `=Record, [=Success, ]=Fail"
        self.tui.set_last_command(ready_msg)

        # Disable terminal echo
        self._disable_terminal_echo()

        try:
            with Live(self.tui.render(), refresh_per_second=10, screen=True) as live:
                while self.simulation_app.is_running() and not self.should_exit:
                    self.world.step(render=True)

                    if self.world.is_stopped() and not reset_needed:
                        reset_needed = True

                    if self.world.is_playing():
                        if reset_needed:
                            self.world.reset()
                            self._reset_robot()
                            reset_needed = False

                        # Process pending commands
                        last_char_cmd = self._process_commands()
                        if last_char_cmd is not None:
                            last_key_processed = last_char_cmd

                        # Apply current velocity control
                        self._apply_control()

                        # Record step if recording is active
                        if self.enable_recording and self.action_mapper is not None:
                            action = np.array([self.current_linear_vel, self.current_angular_vel], dtype=np.float32)
                            # Normalize action for recording
                            normalized_action = np.array([
                                self.current_linear_vel / self.MAX_LINEAR_VELOCITY,
                                self.current_angular_vel / self.MAX_ANGULAR_VELOCITY
                            ], dtype=np.float32)
                            self._record_step(normalized_action)

                        # Checkpoint auto-save
                        if self.enable_recording and self.recorder is not None:
                            if self.recorder.is_recording:
                                self.checkpoint_frame_counter += 1
                                if self.checkpoint_frame_counter >= self.checkpoint_interval_frames:
                                    self._perform_checkpoint_save()
                                    self.checkpoint_frame_counter = 0

                            if self.checkpoint_flash_frames > 0:
                                self.checkpoint_flash_frames -= 1

                        # Update TUI flash state
                        if self.enable_recording:
                            self.tui.set_checkpoint_flash(self.checkpoint_flash_frames > 0)

                        # Update TUI state
                        self._update_tui_state()
                        live.update(self.tui.render())

        finally:
            # Auto-save recording on exit if there's data
            if self.enable_recording and self.recorder is not None:
                if len(self.recorder.observations) > 0:
                    self.tui.set_last_command("Saving recording data...")

                    pending_frames = len(self.recorder.observations) - self.recorder.current_episode_start
                    if pending_frames > 0 and self.recorder.is_recording:
                        self.recorder.mark_episode_success(False)
                        self.recorder.finalize_episode()

                    self.recorder.save(self.demo_save_path)
                    print(f"\nAuto-saved {len(self.recorder.observations)} frames to {self.demo_save_path}")

            self._restore_terminal_settings()
            self.listener.stop()
            self.simulation_app.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Jetbot Keyboard Control with optional recording mode'
    )
    parser.add_argument(
        '--enable-recording', action='store_true',
        help='Enable demonstration recording mode with navigation task'
    )
    parser.add_argument(
        '--demo-path', type=str, default=None,
        help='Path to save demonstrations (default: demos/recording_TIMESTAMP.npz)'
    )
    parser.add_argument(
        '--reward-mode', type=str, default='dense',
        choices=['dense', 'sparse'],
        help='Reward computation mode (default: dense)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    controller = JetbotKeyboardController(
        enable_recording=args.enable_recording,
        demo_path=args.demo_path,
        reward_mode=args.reward_mode
    )
    controller.run()
