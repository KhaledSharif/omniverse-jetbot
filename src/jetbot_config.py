"""Shared Jetbot physical constants and utility functions.

Single source of truth for robot parameters used across
jetbot_keyboard_control.py, jetbot_rl_env.py, and replay.py.
"""

import numpy as np

# Jetbot physical parameters
WHEEL_RADIUS = 0.03      # meters
WHEEL_BASE = 0.1125      # meters (distance between wheels)

# Velocity limits
MAX_LINEAR_VELOCITY = 0.3    # m/s
MAX_ANGULAR_VELOCITY = 1.0   # rad/s

# Start pose
START_POSITION = np.array([0.0, 0.0, 0.05])
START_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion (w, x, y, z)

# Workspace bounds (square arena)
DEFAULT_WORKSPACE_BOUNDS = {
    'x': [-2.0, 2.0],
    'y': [-2.0, 2.0],
}


def quaternion_to_yaw(orientation) -> float:
    """Convert Isaac Sim quaternion (w, x, y, z) to yaw angle in radians.

    Args:
        orientation: Array-like [w, x, y, z] quaternion

    Returns:
        Yaw angle in radians
    """
    w, x, y, z = orientation
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))
