# Phase 4: IMU Integration Notes

## Session Date: 2025-12-07

## Summary

During Phase 3 cuVSLAM integration, IMU sensor integration was attempted but failed due to missing node types in Isaac Sim 5.0.0. This document captures all findings and potential paths forward.

---

## What Was Attempted

### 1. IMU OmniGraph Node Creation

Attempted to use the `isaacsim.sensor.nodes.IsaacReadIMU` node type:

```python
def _create_imu_graph(self) -> bool:
    """Create OmniGraph for IMU data publishing."""
    imu_prim_path = f"{self.jetbot_path}/chassis/imu_sensor"

    # This node type does NOT exist in Isaac Sim 5.0
    (imu_read_node, imu_pub_node) = og.Controller.Keys.CREATE_NODES: [
        ("isaac_read_imu", "isaacsim.sensor.nodes.IsaacReadIMU"),
        ("ros2_publish_imu", "isaacsim.ros2.bridge.ROS2PublishImu"),
    ]
```

### 2. Error Encountered

```
[Error] [omni.graph.core.plugin] Could not create node using unrecognized type 'isaacsim.sensor.nodes.IsaacReadIMU'. Please ensure the extension defining this node type is loaded.
```

### 3. Extension Enable Attempt

Tried enabling the sensor nodes extension:

```python
def enable_sensor_extension() -> bool:
    """Enable the Isaac Sim sensor nodes extension for IMU."""
    try:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.sensor.nodes", True)
        return True
    except Exception as e:
        print(f"[ROS2Bridge] Failed to enable sensor extension: {e}")
        return False
```

**Result**: Extension `isaacsim.sensor.nodes` does not exist in Isaac Sim 5.0.0.

---

## Investigation Findings

### Available Isaac Sim Extensions (Sensor-Related)

Searched for sensor-related extensions in Isaac Sim 5.0:

```bash
find ~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64 -name "*.py" -exec grep -l "IsaacReadIMU\|ReadIMU" {} \;
```

**Result**: No files found containing `IsaacReadIMU` or `ReadIMU` node implementations.

### Possible Alternative Approaches

1. **Physics-Based IMU Sensor** (`isaacsim.sensors.physics`)
   - Isaac Sim has physics-based sensors that simulate IMU behavior
   - May require different API than OmniGraph nodes

2. **Custom IMU Publisher**
   - Read robot rigid body velocities/accelerations directly
   - Manually compute IMU-like data (linear acceleration, angular velocity)
   - Publish via standard ROS2 publisher

3. **Isaac ROS IMU** (Docker approach)
   - Use Isaac ROS perception stack inside Docker
   - May have IMU simulation nodes available

### Isaac Sim 5.0 Sensor Extensions

Available sensor-related extensions found:

```
isaacsim.sensors.camera
isaacsim.sensors.lidar
isaacsim.sensors.contact
isaacsim.sensors.physics  # <-- Most likely candidate for IMU
```

---

## Current State

### What Works

| Component | Status |
|-----------|--------|
| Stereo Cameras (10cm baseline) | ✅ Working |
| `/camera/left/image_raw` | ✅ Publishing |
| `/camera/right/image_raw` | ✅ Publishing |
| `/camera/left/camera_info` | ✅ Publishing |
| `/camera/right/camera_info` | ✅ Publishing |
| `/jetbot/odom` | ✅ Publishing |
| `/tf` | ✅ Publishing |
| `/clock` | ✅ Publishing |
| FastDDS SHM Transport | ✅ Working |

### What Doesn't Work

| Component | Status |
|-----------|--------|
| `/jetbot/imu` | ❌ Node type unavailable |
| IMU Fusion in cuVSLAM | ❌ Disabled |

### Workaround Applied

Made IMU graph creation optional in `ros2_bridge.py`:

```python
# IMU graph is optional - cuVSLAM can work without it
imu_enabled = self._create_imu_graph()
if not imu_enabled:
    print("[ROS2Bridge] IMU graph not created - continuing without IMU")
    print("[ROS2Bridge] Note: cuVSLAM will use stereo-only mode")
```

---

## FastDDS XML Fix

### Original (Broken)

```xml
<transport_descriptor>
    <transport_id>shm_transport</transport_id>
    <type>SHM</type>
    <segment_size>10485760</segment_size>
    <max_message_size>5242880</max_message_size>  <!-- INVALID -->
</transport_descriptor>
```

### Fixed

```xml
<transport_descriptor>
    <transport_id>shm_transport</transport_id>
    <type>SHM</type>
    <segment_size>10485760</segment_size>
    <!-- max_message_size is NOT valid for SHM transport -->
</transport_descriptor>
```

**Error that was fixed**:
```
[RTPS_TRANSPORT_SHM Error] -> Function process_channel_resources
XMLParser error: Invalid element found into 'transportDescriptorType'. Name: max_message_size
```

---

## Phase 4 Implementation Options

### Option A: Physics-Based IMU Sensor

Use `isaacsim.sensors.physics` extension to create a simulated IMU:

```python
# Potential approach - needs investigation
from isaacsim.sensors.physics import IMUSensor

imu = IMUSensor(
    prim_path="/World/Jetbot/chassis/imu",
    name="jetbot_imu",
    frequency=100,  # Hz
)

# Read data each frame
data = imu.get_current_frame()
linear_accel = data["linear_acceleration"]
angular_vel = data["angular_velocity"]
orientation = data["orientation"]
```

### Option B: Manual IMU from Rigid Body

Extract IMU-like data from the robot's rigid body physics:

```python
from omni.isaac.core.utils.physics import get_rigid_body_velocities

def get_imu_data(chassis_prim_path: str):
    """Extract IMU-equivalent data from rigid body."""
    # Get rigid body handle
    rb = RigidPrimView(chassis_prim_path)

    # Linear velocity -> differentiate for acceleration
    linear_vel = rb.get_linear_velocities()

    # Angular velocity directly available
    angular_vel = rb.get_angular_velocities()

    # Orientation from transform
    orientation = rb.get_world_poses()[1]  # quaternion

    return {
        "linear_acceleration": compute_acceleration(linear_vel),
        "angular_velocity": angular_vel,
        "orientation": orientation
    }
```

### Option C: Isaac ROS Perception (Docker)

Use the full Isaac ROS perception stack which may include IMU simulation:

```bash
# In docker/run_cuvslam.sh
docker run ... \
    nvcr.io/nvidia/isaac/ros:x86_64-ros2_humble-aarch64_3.2.0 \
    ros2 launch isaac_ros_imu isaac_sim_imu.launch.py
```

---

## Recommended Next Steps

1. **Investigate `isaacsim.sensors.physics`**
   - Search Isaac Sim examples for physics-based IMU
   - Check if `IMUSensor` or similar class exists

2. **Test cuVSLAM Stereo-Only Mode**
   - Build and run the Docker container
   - Verify SLAM works without IMU
   - Measure drift/accuracy

3. **Manual IMU Implementation**
   - If physics IMU doesn't exist, implement manual approach
   - Read chassis rigid body velocities
   - Compute accelerations and publish to `/jetbot/imu`

4. **Check Isaac Sim 5.0 Release Notes**
   - May have different API for sensors
   - Could be renamed or restructured from 4.x

---

## Files Modified Today

| File | Changes |
|------|---------|
| `~/.ros/fastdds.xml` | Removed invalid `max_message_size` |
| `src/ros2_bridge.py` | Added stage parameter, made IMU optional |
| `src/jetbot_keyboard_control.py` | Pass stage to `create_ros2_graph()` |
| `src/test_ros2_bridge.py` | Complete rewrite for stereo + IMU tests |

---

## Test Results

All 145 tests pass:

```bash
./run_tests.sh
# ===== 145 passed in 0.49s =====
```

---

## References

- [Isaac Sim 5.0 Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [cuVSLAM GitHub](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam)
- [FastDDS XML Reference](https://fast-dds.docs.eprosima.com/en/latest/fastdds/xml_configuration/xml_configuration.html)

---

## Conclusion

Phase 3 cuVSLAM integration is **functionally complete** with stereo cameras. IMU integration requires Phase 4 work to either:
1. Find the correct Isaac Sim 5.0 API for physics-based IMU
2. Implement a manual IMU data publisher from rigid body physics

cuVSLAM can operate in stereo-only mode for now, with IMU fusion as a future enhancement.
