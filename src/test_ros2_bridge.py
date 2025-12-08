"""Tests for ROS2 Bridge and RViz configuration.

This module tests the ROS2 bridge functionality (Phase 1) and
RViz configuration files (Phase 2 & Phase 3 cuVSLAM) for the Isaac Sim Jetbot.
"""

import os
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

# =============================================================================
# MOCK SETUP - Must be done BEFORE importing ros2_bridge
# =============================================================================

# Mock omni.graph.core module - use simple MagicMock, don't override Controller
mock_og = MagicMock()
# Just set the Keys constants we need (accessed via auto-created attributes)
mock_og.Controller.Keys.CREATE_NODES = "CREATE_NODES"
mock_og.Controller.Keys.CONNECT = "CONNECT"
mock_og.Controller.Keys.SET_VALUES = "SET_VALUES"
sys.modules['omni'] = MagicMock()
sys.modules['omni.graph'] = MagicMock()
sys.modules['omni.graph.core'] = mock_og

# Mock isaacsim modules
mock_isaacsim = MagicMock()
mock_extensions = MagicMock()
sys.modules['isaacsim'] = mock_isaacsim
sys.modules['isaacsim.core'] = MagicMock()
sys.modules['isaacsim.core.utils'] = MagicMock()
sys.modules['isaacsim.core.utils.extensions'] = mock_extensions

# Mock usdrt for TF/odom graph
mock_usdrt = MagicMock()
mock_usdrt_sdf = MagicMock()
mock_usdrt_sdf.Path = MagicMock(side_effect=lambda x: f"SdfPath({x})")
sys.modules['usdrt'] = mock_usdrt
sys.modules['usdrt.Sdf'] = mock_usdrt_sdf

# Mock pxr for stereo camera/IMU creation
mock_pxr = MagicMock()
mock_gf = MagicMock()
mock_gf.Vec3d = MagicMock(side_effect=lambda x, y, z: (x, y, z))
mock_gf.Vec2f = MagicMock(side_effect=lambda x, y: (x, y))
mock_pxr.Gf = mock_gf
mock_pxr.UsdGeom = MagicMock()
mock_pxr.UsdPhysics = MagicMock()
sys.modules['pxr'] = mock_pxr

# =============================================================================
# IMPORT MODULE UNDER TEST
# =============================================================================

from ros2_bridge import ROS2Bridge, enable_ros2_bridge_extension, enable_sensor_extension

# Get a reference to the actual og module imported by ros2_bridge
# This ensures we're testing the same object that ros2_bridge uses
import ros2_bridge
og_controller_edit = ros2_bridge.og.Controller.edit

# =============================================================================
# FIXTURES
# =============================================================================

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def mock_enable_extension():
    """Fixture to mock enable_extension function."""
    with patch('ros2_bridge.enable_extension') as mock:
        yield mock


@pytest.fixture
def bridge():
    """Fixture to create a fresh ROS2Bridge instance."""
    return ROS2Bridge()


@pytest.fixture
def bridge_custom_path():
    """Fixture to create ROS2Bridge with custom robot path."""
    return ROS2Bridge(robot_prim_path="/CustomWorld/MyRobot")


@pytest.fixture
def mock_stage():
    """Fixture to create a mock USD stage."""
    stage = MagicMock()
    return stage


# =============================================================================
# TEST: enable_ros2_bridge_extension function
# =============================================================================

class TestEnableRos2BridgeExtension:
    """Tests for the enable_ros2_bridge_extension function."""

    def test_enable_ros2_bridge_extension_success(self, mock_enable_extension):
        """Test that function returns True when extension enables successfully."""
        mock_enable_extension.return_value = None  # No exception

        result = enable_ros2_bridge_extension()

        assert result is True
        mock_enable_extension.assert_called_once_with("isaacsim.ros2.bridge")

    def test_enable_ros2_bridge_extension_failure(self, mock_enable_extension):
        """Test that function returns False when extension fails to enable."""
        mock_enable_extension.side_effect = RuntimeError("Extension not found")

        result = enable_ros2_bridge_extension()

        assert result is False


class TestEnableSensorExtension:
    """Tests for the enable_sensor_extension function."""

    def test_enable_sensor_extension_success(self, mock_enable_extension):
        """Test that function returns True when extension enables successfully."""
        mock_enable_extension.return_value = None  # No exception

        result = enable_sensor_extension()

        assert result is True
        mock_enable_extension.assert_called_once_with("isaacsim.sensor.nodes")

    def test_enable_sensor_extension_failure(self, mock_enable_extension):
        """Test that function returns False when extension fails to enable."""
        mock_enable_extension.side_effect = RuntimeError("Extension not found")

        result = enable_sensor_extension()

        assert result is False


# =============================================================================
# TEST: ROS2Bridge Initialization
# =============================================================================

class TestROS2BridgeInit:
    """Tests for ROS2Bridge initialization."""

    def test_ros2_bridge_init_default_path(self, bridge):
        """Test default robot prim path is /World/Jetbot."""
        assert bridge.robot_prim_path == "/World/Jetbot"

    def test_ros2_bridge_init_custom_path(self, bridge_custom_path):
        """Test custom robot prim path is stored correctly."""
        assert bridge_custom_path.robot_prim_path == "/CustomWorld/MyRobot"

    def test_ros2_bridge_chassis_path(self, bridge):
        """Test chassis prim path is derived correctly."""
        assert bridge.chassis_prim_path == "/World/Jetbot/chassis"

    def test_ros2_bridge_custom_chassis_path(self, bridge_custom_path):
        """Test chassis path with custom robot path."""
        assert bridge_custom_path.chassis_prim_path == "/CustomWorld/MyRobot/chassis"

    def test_ros2_bridge_stereo_paths(self, bridge):
        """Test stereo camera prim paths are set correctly."""
        assert bridge.stereo_mount_path == "/World/Jetbot/chassis/stereo_camera"
        assert bridge.left_camera_path == "/World/Jetbot/chassis/stereo_camera/left_camera"
        assert bridge.right_camera_path == "/World/Jetbot/chassis/stereo_camera/right_camera"

    def test_ros2_bridge_imu_path(self, bridge):
        """Test IMU sensor prim path is set correctly."""
        assert bridge.imu_path == "/World/Jetbot/chassis/imu_sensor"

    def test_ros2_bridge_graph_paths(self, bridge):
        """Test graph paths are set correctly."""
        assert bridge.clock_graph_path == "/ROS2ClockGraph"
        assert bridge.stereo_graph_path == "/ROS2StereoGraph"
        assert bridge.imu_graph_path == "/ROS2IMUGraph"
        assert bridge.tf_odom_graph_path == "/ROS2TFOdomGraph"

    def test_ros2_bridge_initial_state(self, bridge):
        """Test initial state flags are False."""
        assert bridge._graph_created is False
        assert bridge._cameras_created is False
        assert bridge._imu_created is False


# =============================================================================
# TEST: ROS2Bridge Stereo Camera Constants
# =============================================================================

class TestROS2BridgeStereoConstants:
    """Tests for ROS2Bridge stereo camera constants."""

    def test_baseline(self):
        """Test BASELINE constant is 10cm."""
        assert ROS2Bridge.BASELINE == 0.10

    def test_camera_width(self):
        """Test CAMERA_WIDTH constant."""
        assert ROS2Bridge.CAMERA_WIDTH == 640

    def test_camera_height(self):
        """Test CAMERA_HEIGHT constant."""
        assert ROS2Bridge.CAMERA_HEIGHT == 480

    def test_left_image_topic(self):
        """Test LEFT_IMAGE_TOPIC constant."""
        assert ROS2Bridge.LEFT_IMAGE_TOPIC == "/camera/left/image_raw"

    def test_left_info_topic(self):
        """Test LEFT_INFO_TOPIC constant."""
        assert ROS2Bridge.LEFT_INFO_TOPIC == "/camera/left/camera_info"

    def test_right_image_topic(self):
        """Test RIGHT_IMAGE_TOPIC constant."""
        assert ROS2Bridge.RIGHT_IMAGE_TOPIC == "/camera/right/image_raw"

    def test_right_info_topic(self):
        """Test RIGHT_INFO_TOPIC constant."""
        assert ROS2Bridge.RIGHT_INFO_TOPIC == "/camera/right/camera_info"


# =============================================================================
# TEST: ROS2Bridge IMU and Navigation Constants
# =============================================================================

class TestROS2BridgeNavigationConstants:
    """Tests for ROS2Bridge IMU and navigation constants."""

    def test_namespace(self):
        """Test NAMESPACE constant."""
        assert ROS2Bridge.NAMESPACE == "jetbot"

    def test_imu_topic(self):
        """Test IMU_TOPIC constant."""
        assert ROS2Bridge.IMU_TOPIC == "/jetbot/imu"

    def test_odom_topic(self):
        """Test ODOM_TOPIC constant."""
        assert ROS2Bridge.ODOM_TOPIC == "/jetbot/odom"

    def test_tf_topic(self):
        """Test TF_TOPIC constant."""
        assert ROS2Bridge.TF_TOPIC == "/tf"

    def test_clock_topic(self):
        """Test CLOCK_TOPIC constant."""
        assert ROS2Bridge.CLOCK_TOPIC == "/clock"


# =============================================================================
# TEST: ROS2Bridge Frame ID Constants
# =============================================================================

class TestROS2BridgeFrameConstants:
    """Tests for ROS2Bridge frame ID constants."""

    def test_odom_frame(self):
        """Test ODOM_FRAME matches Isaac Sim's world frame."""
        assert ROS2Bridge.ODOM_FRAME == "world"

    def test_base_frame(self):
        """Test BASE_FRAME matches Isaac Sim's chassis frame."""
        assert ROS2Bridge.BASE_FRAME == "chassis"

    def test_left_camera_frame(self):
        """Test LEFT_CAMERA_FRAME constant for cuVSLAM."""
        assert ROS2Bridge.LEFT_CAMERA_FRAME == "left_camera_optical"

    def test_right_camera_frame(self):
        """Test RIGHT_CAMERA_FRAME constant for cuVSLAM."""
        assert ROS2Bridge.RIGHT_CAMERA_FRAME == "right_camera_optical"

    def test_imu_frame(self):
        """Test IMU_FRAME constant for cuVSLAM."""
        assert ROS2Bridge.IMU_FRAME == "imu_link"


# =============================================================================
# TEST: ROS2Bridge Stereo Camera Creation
# =============================================================================

class TestROS2BridgeStereoCameraCreation:
    """Tests for stereo camera creation."""

    def test_create_stereo_cameras_success(self, bridge, mock_stage):
        """Test successful stereo camera creation."""
        result = bridge.create_stereo_cameras(mock_stage)

        assert result is True
        assert bridge._cameras_created is True

    def test_create_stereo_cameras_already_created(self, bridge, mock_stage):
        """Test stereo cameras not recreated if already created."""
        bridge._cameras_created = True
        # Reset the mock to track only calls in this test
        mock_pxr.UsdGeom.Xform.Define.reset_mock()

        result = bridge.create_stereo_cameras(mock_stage)

        assert result is True
        # UsdGeom.Xform.Define should not be called since cameras already exist
        mock_pxr.UsdGeom.Xform.Define.assert_not_called()

    def test_create_stereo_cameras_sets_flag(self, bridge, mock_stage):
        """Test that _cameras_created flag is set after successful creation."""
        assert bridge._cameras_created is False
        bridge.create_stereo_cameras(mock_stage)
        assert bridge._cameras_created is True


# =============================================================================
# TEST: ROS2Bridge IMU Sensor Creation
# =============================================================================

class TestROS2BridgeIMUCreation:
    """Tests for IMU sensor creation."""

    def test_create_imu_sensor_success(self, bridge, mock_stage):
        """Test successful IMU sensor creation."""
        result = bridge.create_imu_sensor(mock_stage)

        assert result is True
        assert bridge._imu_created is True

    def test_create_imu_sensor_already_created(self, bridge, mock_stage):
        """Test IMU sensor not recreated if already created."""
        bridge._imu_created = True

        result = bridge.create_imu_sensor(mock_stage)

        assert result is True

    def test_create_imu_sensor_sets_flag(self, bridge, mock_stage):
        """Test that _imu_created flag is set after successful creation."""
        assert bridge._imu_created is False
        bridge.create_imu_sensor(mock_stage)
        assert bridge._imu_created is True


# =============================================================================
# TEST: ROS2Bridge Graph Creation
# =============================================================================

class TestROS2BridgeGraphCreation:
    """Tests for ROS2Bridge graph creation methods."""

    def test_create_ros2_graph_success(self, bridge, mock_enable_extension):
        """Test successful graph creation returns True."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        result = bridge.create_ros2_graph()

        assert result is True
        assert bridge._graph_created is True
        # Should call og.Controller.edit 4 times (clock, stereo, imu, tf_odom)
        assert og_controller_edit.call_count == 4

    def test_create_ros2_graph_already_created(self, bridge, mock_enable_extension):
        """Test that graph creation is skipped if already created."""
        bridge._graph_created = True

        result = bridge.create_ros2_graph()

        assert result is True
        mock_enable_extension.assert_not_called()

    def test_create_ros2_graph_extension_fails(self, bridge, mock_enable_extension):
        """Test graph creation fails if extension fails."""
        mock_enable_extension.side_effect = RuntimeError("Extension not found")

        result = bridge.create_ros2_graph()

        assert result is False
        assert bridge._graph_created is False

    def test_create_ros2_graph_clock_fails(self, bridge, mock_enable_extension):
        """Test graph creation fails if clock graph creation fails."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.side_effect = RuntimeError("OmniGraph error")

        result = bridge.create_ros2_graph()

        assert result is False
        assert bridge._graph_created is False
        og_controller_edit.side_effect = None

    def test_create_ros2_graph_stereo_fails(self, bridge, mock_enable_extension):
        """Test graph creation fails if stereo graph creation fails."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        # First call (clock) succeeds, second call (stereo) fails
        og_controller_edit.side_effect = [None, RuntimeError("Stereo graph error")]

        result = bridge.create_ros2_graph()

        assert result is False
        assert bridge._graph_created is False
        og_controller_edit.side_effect = None

    def test_create_ros2_graph_imu_fails(self, bridge, mock_enable_extension):
        """Test graph creation fails if IMU graph creation fails."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        # First two calls succeed, third call (imu) fails
        og_controller_edit.side_effect = [None, None, RuntimeError("IMU graph error")]

        result = bridge.create_ros2_graph()

        assert result is False
        assert bridge._graph_created is False
        og_controller_edit.side_effect = None

    def test_create_ros2_graph_tf_odom_fails(self, bridge, mock_enable_extension):
        """Test graph creation fails if TF/odom graph creation fails."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        # First three calls succeed, fourth call fails
        og_controller_edit.side_effect = [None, None, None, RuntimeError("TF/Odom error")]

        result = bridge.create_ros2_graph()

        assert result is False
        assert bridge._graph_created is False
        og_controller_edit.side_effect = None

    def test_create_clock_graph_parameters(self, bridge, mock_enable_extension):
        """Test clock graph creation uses correct parameters."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        bridge.create_ros2_graph()

        # Get the first call (clock graph)
        assert len(og_controller_edit.call_args_list) >= 1
        clock_call = og_controller_edit.call_args_list[0]
        graph_config = clock_call[0][0]

        assert graph_config["graph_path"] == "/ROS2ClockGraph"
        assert graph_config["evaluator_name"] == "execution"

    def test_create_stereo_graph_parameters(self, bridge, mock_enable_extension):
        """Test stereo camera graph creation uses correct parameters."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        bridge.create_ros2_graph()

        # Get the second call (stereo graph)
        assert len(og_controller_edit.call_args_list) >= 2
        stereo_call = og_controller_edit.call_args_list[1]
        graph_config = stereo_call[0][0]

        assert graph_config["graph_path"] == "/ROS2StereoGraph"
        assert graph_config["evaluator_name"] == "execution"

    def test_create_imu_graph_parameters(self, bridge, mock_enable_extension):
        """Test IMU graph creation uses correct parameters."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        bridge.create_ros2_graph()

        # Get the third call (IMU graph)
        assert len(og_controller_edit.call_args_list) >= 3
        imu_call = og_controller_edit.call_args_list[2]
        graph_config = imu_call[0][0]

        assert graph_config["graph_path"] == "/ROS2IMUGraph"
        assert graph_config["evaluator_name"] == "execution"

    def test_create_tf_odom_graph_parameters(self, bridge, mock_enable_extension):
        """Test TF/odom graph creation uses correct parameters."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        bridge.create_ros2_graph()

        # Get the fourth call (TF/odom graph)
        assert len(og_controller_edit.call_args_list) >= 4
        tf_odom_call = og_controller_edit.call_args_list[3]
        graph_config = tf_odom_call[0][0]

        assert graph_config["graph_path"] == "/ROS2TFOdomGraph"
        assert graph_config["evaluator_name"] == "execution"


# =============================================================================
# TEST: ROS2Bridge Helper Methods
# =============================================================================

class TestROS2BridgeHelpers:
    """Tests for ROS2Bridge helper methods."""

    def test_get_published_topics(self, bridge):
        """Test get_published_topics returns correct dictionary."""
        topics = bridge.get_published_topics()

        assert topics["left_image"] == "/camera/left/image_raw"
        assert topics["left_info"] == "/camera/left/camera_info"
        assert topics["right_image"] == "/camera/right/image_raw"
        assert topics["right_info"] == "/camera/right/camera_info"
        assert topics["imu"] == "/jetbot/imu"
        assert topics["odom"] == "/jetbot/odom"
        assert topics["tf"] == "/tf"
        assert topics["clock"] == "/clock"

    def test_get_published_topics_has_all_keys(self, bridge):
        """Test get_published_topics contains all expected keys."""
        topics = bridge.get_published_topics()
        expected_keys = {"left_image", "left_info", "right_image", "right_info",
                        "imu", "odom", "tf", "clock"}

        assert set(topics.keys()) == expected_keys

    def test_is_enabled_false_initially(self, bridge):
        """Test is_enabled returns False before graph creation."""
        assert bridge.is_enabled() is False

    def test_is_enabled_true_after_creation(self, bridge, mock_enable_extension):
        """Test is_enabled returns True after successful graph creation."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        bridge.create_ros2_graph()

        assert bridge.is_enabled() is True

    def test_get_stereo_config(self, bridge):
        """Test get_stereo_config returns correct dictionary."""
        config = bridge.get_stereo_config()

        assert config["baseline"] == 0.10
        assert config["width"] == 640
        assert config["height"] == 480
        assert config["left_frame"] == "left_camera_optical"
        assert config["right_frame"] == "right_camera_optical"

    def test_get_stereo_config_has_all_keys(self, bridge):
        """Test get_stereo_config contains all expected keys."""
        config = bridge.get_stereo_config()
        expected_keys = {"baseline", "width", "height", "left_frame", "right_frame"}

        assert set(config.keys()) == expected_keys


# =============================================================================
# TEST: RViz Configuration File (Original jetbot.rviz)
# =============================================================================

class TestRVizConfigFile:
    """Tests for rviz/jetbot.rviz configuration file."""

    @pytest.fixture
    def rviz_config_path(self):
        """Get path to RViz config file."""
        return PROJECT_ROOT / "rviz" / "jetbot.rviz"

    @pytest.fixture
    def rviz_config(self, rviz_config_path):
        """Load and parse RViz config file."""
        import yaml
        with open(rviz_config_path, 'r') as f:
            return yaml.safe_load(f)

    def test_rviz_config_exists(self, rviz_config_path):
        """Test that rviz/jetbot.rviz file exists."""
        assert rviz_config_path.exists(), f"RViz config not found at {rviz_config_path}"

    def test_rviz_config_valid_yaml(self, rviz_config_path):
        """Test that config file is valid YAML."""
        import yaml
        try:
            with open(rviz_config_path, 'r') as f:
                config = yaml.safe_load(f)
            assert config is not None
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML: {e}")

    def test_rviz_config_fixed_frame(self, rviz_config):
        """Test Fixed Frame is 'world' (matching Isaac Sim TF)."""
        fixed_frame = rviz_config["Visualization Manager"]["Global Options"]["Fixed Frame"]
        assert fixed_frame == "world"

    def test_rviz_config_has_displays(self, rviz_config):
        """Test config has displays defined."""
        displays = rviz_config["Visualization Manager"]["Displays"]
        assert isinstance(displays, list)
        assert len(displays) > 0

    def _find_display(self, rviz_config, name):
        """Helper to find display by name."""
        displays = rviz_config["Visualization Manager"]["Displays"]
        for display in displays:
            if display.get("Name") == name:
                return display
        return None

    def test_rviz_config_has_tf_display(self, rviz_config):
        """Test config includes TF display."""
        tf_display = self._find_display(rviz_config, "TF")
        assert tf_display is not None, "TF display not found in config"
        assert tf_display["Enabled"] is True

    def test_rviz_config_has_grid_display(self, rviz_config):
        """Test config includes Grid display."""
        grid_display = self._find_display(rviz_config, "Grid")
        assert grid_display is not None, "Grid display not found in config"
        assert grid_display["Enabled"] is True


# =============================================================================
# TEST: cuVSLAM RViz Configuration File
# =============================================================================

class TestCuvslamRVizConfig:
    """Tests for rviz/cuvslam.rviz configuration file."""

    @pytest.fixture
    def cuvslam_config_path(self):
        """Get path to cuVSLAM RViz config file."""
        return PROJECT_ROOT / "rviz" / "cuvslam.rviz"

    @pytest.fixture
    def cuvslam_config(self, cuvslam_config_path):
        """Load and parse cuVSLAM RViz config file."""
        import yaml
        with open(cuvslam_config_path, 'r') as f:
            return yaml.safe_load(f)

    def test_cuvslam_config_exists(self, cuvslam_config_path):
        """Test that rviz/cuvslam.rviz file exists."""
        assert cuvslam_config_path.exists(), f"cuVSLAM RViz config not found at {cuvslam_config_path}"

    def test_cuvslam_config_valid_yaml(self, cuvslam_config_path):
        """Test that config file is valid YAML."""
        import yaml
        try:
            with open(cuvslam_config_path, 'r') as f:
                config = yaml.safe_load(f)
            assert config is not None
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML: {e}")

    def test_cuvslam_config_fixed_frame(self, cuvslam_config):
        """Test Fixed Frame is 'map' for SLAM."""
        fixed_frame = cuvslam_config["Visualization Manager"]["Global Options"]["Fixed Frame"]
        assert fixed_frame == "map"

    def _find_display(self, config, name):
        """Helper to find display by name."""
        displays = config["Visualization Manager"]["Displays"]
        for display in displays:
            if display.get("Name") == name:
                return display
        return None

    def test_cuvslam_left_camera_topic(self, cuvslam_config):
        """Test Left Camera topic matches ROS2Bridge constant."""
        left_display = self._find_display(cuvslam_config, "Left Camera")
        assert left_display is not None, "Left Camera display not found"
        assert left_display["Topic"]["Value"] == ROS2Bridge.LEFT_IMAGE_TOPIC

    def test_cuvslam_right_camera_topic(self, cuvslam_config):
        """Test Right Camera topic matches ROS2Bridge constant."""
        right_display = self._find_display(cuvslam_config, "Right Camera")
        assert right_display is not None, "Right Camera display not found"
        assert right_display["Topic"]["Value"] == ROS2Bridge.RIGHT_IMAGE_TOPIC

    def test_cuvslam_left_camera_qos(self, cuvslam_config):
        """Test Left Camera uses Best Effort QoS."""
        left_display = self._find_display(cuvslam_config, "Left Camera")
        assert left_display is not None
        assert left_display["Topic"]["Reliability Policy"] == "Best Effort"

    def test_cuvslam_right_camera_qos(self, cuvslam_config):
        """Test Right Camera uses Best Effort QoS."""
        right_display = self._find_display(cuvslam_config, "Right Camera")
        assert right_display is not None
        assert right_display["Topic"]["Reliability Policy"] == "Best Effort"

    def test_cuvslam_has_slam_landmarks(self, cuvslam_config):
        """Test config includes SLAM Landmarks PointCloud2 display."""
        landmarks = self._find_display(cuvslam_config, "SLAM Landmarks")
        assert landmarks is not None, "SLAM Landmarks display not found"
        assert landmarks["Topic"]["Value"] == "/visual_slam/vis/landmarks_cloud"

    def test_cuvslam_has_slam_trajectory(self, cuvslam_config):
        """Test config includes SLAM Trajectory Path display."""
        trajectory = self._find_display(cuvslam_config, "SLAM Trajectory")
        assert trajectory is not None, "SLAM Trajectory display not found"
        assert trajectory["Topic"]["Value"] == "/visual_slam/tracking/slam_path"

    def test_cuvslam_has_slam_pose(self, cuvslam_config):
        """Test config includes SLAM Pose Odometry display."""
        pose = self._find_display(cuvslam_config, "SLAM Pose")
        assert pose is not None, "SLAM Pose display not found"
        assert pose["Topic"]["Value"] == "/visual_slam/tracking/odometry"

    def test_cuvslam_robot_odometry_topic(self, cuvslam_config):
        """Test Robot Odometry topic matches ROS2Bridge constant."""
        odom_display = self._find_display(cuvslam_config, "Robot Odometry")
        assert odom_display is not None, "Robot Odometry display not found"
        assert odom_display["Topic"]["Value"] == ROS2Bridge.ODOM_TOPIC

    def test_cuvslam_has_tf_display(self, cuvslam_config):
        """Test config includes TF display."""
        tf_display = self._find_display(cuvslam_config, "TF")
        assert tf_display is not None, "TF display not found"
        assert tf_display["Enabled"] is True

    def test_cuvslam_has_grid_display(self, cuvslam_config):
        """Test config includes Grid display."""
        grid_display = self._find_display(cuvslam_config, "Grid")
        assert grid_display is not None, "Grid display not found"
        assert grid_display["Enabled"] is True


# =============================================================================
# TEST: View Jetbot Launch Script
# =============================================================================

class TestViewJetbotScript:
    """Tests for rviz/view_jetbot.sh launch script."""

    @pytest.fixture
    def script_path(self):
        """Get path to launch script."""
        return PROJECT_ROOT / "rviz" / "view_jetbot.sh"

    @pytest.fixture
    def script_content(self, script_path):
        """Read script content."""
        with open(script_path, 'r') as f:
            return f.read()

    def test_view_jetbot_script_exists(self, script_path):
        """Test that rviz/view_jetbot.sh file exists."""
        assert script_path.exists(), f"Launch script not found at {script_path}"

    def test_view_jetbot_script_executable(self, script_path):
        """Test script has executable permission."""
        mode = script_path.stat().st_mode
        assert mode & stat.S_IXUSR, "Script is not executable by owner"

    def test_view_jetbot_sources_ros2(self, script_content):
        """Test script sources ROS2 Jazzy setup."""
        assert "source /opt/ros/jazzy/setup.bash" in script_content

    def test_view_jetbot_launches_rviz(self, script_content):
        """Test script launches rviz2."""
        assert "rviz2" in script_content

    def test_view_jetbot_uses_rviz_config(self, script_content):
        """Test script uses the jetbot.rviz config file."""
        assert "jetbot.rviz" in script_content


# =============================================================================
# TEST: Docker Infrastructure
# =============================================================================

class TestDockerInfrastructure:
    """Tests for Docker cuVSLAM infrastructure."""

    def test_dockerfile_exists(self):
        """Test that docker/Dockerfile.cuvslam exists."""
        dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.cuvslam"
        assert dockerfile.exists(), f"Dockerfile not found at {dockerfile}"

    def test_run_cuvslam_script_exists(self):
        """Test that docker/run_cuvslam.sh exists."""
        script = PROJECT_ROOT / "docker" / "run_cuvslam.sh"
        assert script.exists(), f"Run script not found at {script}"

    def test_run_cuvslam_script_executable(self):
        """Test docker/run_cuvslam.sh has executable permission."""
        script = PROJECT_ROOT / "docker" / "run_cuvslam.sh"
        mode = script.stat().st_mode
        assert mode & stat.S_IXUSR, "Script is not executable by owner"

    def test_entrypoint_exists(self):
        """Test that docker/entrypoint.sh exists."""
        entrypoint = PROJECT_ROOT / "docker" / "entrypoint.sh"
        assert entrypoint.exists(), f"Entrypoint not found at {entrypoint}"

    def test_entrypoint_executable(self):
        """Test docker/entrypoint.sh has executable permission."""
        entrypoint = PROJECT_ROOT / "docker" / "entrypoint.sh"
        mode = entrypoint.stat().st_mode
        assert mode & stat.S_IXUSR, "Entrypoint is not executable by owner"

    def test_launch_file_exists(self):
        """Test that docker/cuvslam.launch.py exists."""
        launch_file = PROJECT_ROOT / "docker" / "cuvslam.launch.py"
        assert launch_file.exists(), f"Launch file not found at {launch_file}"

    @pytest.fixture
    def run_cuvslam_content(self):
        """Read run_cuvslam.sh content."""
        script = PROJECT_ROOT / "docker" / "run_cuvslam.sh"
        with open(script, 'r') as f:
            return f.read()

    def test_run_cuvslam_uses_gpu(self, run_cuvslam_content):
        """Test run_cuvslam.sh passes --gpus all to docker."""
        assert "--gpus all" in run_cuvslam_content

    def test_run_cuvslam_network_host(self, run_cuvslam_content):
        """Test run_cuvslam.sh uses host network for ROS2 communication."""
        assert "--network host" in run_cuvslam_content

    def test_run_cuvslam_mounts_fastdds(self, run_cuvslam_content):
        """Test run_cuvslam.sh mounts FastDDS config."""
        assert "fastdds.xml" in run_cuvslam_content


# =============================================================================
# TEST: run_slam.sh Host Launcher
# =============================================================================

class TestRunSlamScript:
    """Tests for run_slam.sh host launcher script."""

    @pytest.fixture
    def script_path(self):
        """Get path to run_slam.sh script."""
        return PROJECT_ROOT / "run_slam.sh"

    @pytest.fixture
    def script_content(self, script_path):
        """Read script content."""
        with open(script_path, 'r') as f:
            return f.read()

    def test_run_slam_exists(self, script_path):
        """Test that run_slam.sh exists."""
        assert script_path.exists(), f"run_slam.sh not found at {script_path}"

    def test_run_slam_executable(self, script_path):
        """Test run_slam.sh has executable permission."""
        mode = script_path.stat().st_mode
        assert mode & stat.S_IXUSR, "Script is not executable by owner"

    def test_run_slam_uses_isaac_sim(self, script_content):
        """Test script uses Isaac Sim Python."""
        assert "isaac-sim" in script_content
        assert "python.sh" in script_content

    def test_run_slam_enables_ros2(self, script_content):
        """Test script passes --enable-ros2 flag."""
        assert "--enable-ros2" in script_content

    def test_run_slam_mentions_cuvslam(self, script_content):
        """Test script mentions cuVSLAM Docker container."""
        assert "cuvslam" in script_content.lower() or "cuVSLAM" in script_content

    def test_run_slam_mentions_rviz(self, script_content):
        """Test script mentions RViz visualization."""
        assert "rviz" in script_content.lower()


# =============================================================================
# TEST: FastDDS Configuration
# =============================================================================

class TestFastDDSConfig:
    """Tests for FastDDS configuration file."""

    @pytest.fixture
    def fastdds_path(self):
        """Get path to FastDDS config file."""
        return Path.home() / ".ros" / "fastdds.xml"

    def test_fastdds_config_exists(self, fastdds_path):
        """Test that ~/.ros/fastdds.xml exists."""
        assert fastdds_path.exists(), f"FastDDS config not found at {fastdds_path}"

    def test_fastdds_config_valid_xml(self, fastdds_path):
        """Test that fastdds.xml is valid XML."""
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(fastdds_path)
            assert tree.getroot() is not None
        except ET.ParseError as e:
            pytest.fail(f"Invalid XML: {e}")

    def test_fastdds_has_shm_transport(self, fastdds_path):
        """Test FastDDS config includes SHM transport."""
        with open(fastdds_path, 'r') as f:
            content = f.read()
        assert "SHM" in content

    def test_fastdds_has_udp_transport(self, fastdds_path):
        """Test FastDDS config includes UDP transport as fallback."""
        with open(fastdds_path, 'r') as f:
            content = f.read()
        assert "UDPv4" in content


# =============================================================================
# TEST: Topic Consistency Between ROS2Bridge and RViz Configs
# =============================================================================

class TestTopicConsistency:
    """Tests to verify topic names match between ROS2Bridge and RViz configs."""

    @pytest.fixture
    def cuvslam_config(self):
        """Load cuVSLAM RViz config."""
        import yaml
        config_path = PROJECT_ROOT / "rviz" / "cuvslam.rviz"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _find_display(self, rviz_config, name):
        """Helper to find display by name."""
        displays = rviz_config["Visualization Manager"]["Displays"]
        for display in displays:
            if display.get("Name") == name:
                return display
        return None

    def test_left_image_topic_consistency(self, cuvslam_config):
        """Test Left Camera topic in RViz matches ROS2Bridge constant."""
        left_display = self._find_display(cuvslam_config, "Left Camera")
        assert left_display["Topic"]["Value"] == ROS2Bridge.LEFT_IMAGE_TOPIC

    def test_right_image_topic_consistency(self, cuvslam_config):
        """Test Right Camera topic in RViz matches ROS2Bridge constant."""
        right_display = self._find_display(cuvslam_config, "Right Camera")
        assert right_display["Topic"]["Value"] == ROS2Bridge.RIGHT_IMAGE_TOPIC

    def test_odom_topic_consistency(self, cuvslam_config):
        """Test Odometry topic in RViz matches ROS2Bridge constant."""
        odom_display = self._find_display(cuvslam_config, "Robot Odometry")
        assert odom_display["Topic"]["Value"] == ROS2Bridge.ODOM_TOPIC
