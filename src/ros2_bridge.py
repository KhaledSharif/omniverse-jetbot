"""ROS2 Bridge for Isaac Sim Jetbot with Stereo Cameras and IMU.

This module sets up OmniGraph nodes to publish sensor data to ROS2 topics
for NVIDIA cuVSLAM integration:

Stereo Cameras:
- /camera/left/image_raw (sensor_msgs/Image)
- /camera/left/camera_info (sensor_msgs/CameraInfo)
- /camera/right/image_raw (sensor_msgs/Image)
- /camera/right/camera_info (sensor_msgs/CameraInfo)

IMU:
- /jetbot/imu (sensor_msgs/Imu)

Odometry & TF:
- /jetbot/odom (nav_msgs/Odometry)
- /tf (tf2_msgs/TFMessage)
- /clock (rosgraph_msgs/Clock)

The OmniGraph-based approach uses Isaac Sim's internal ROS2 libraries,
avoiding Python version conflicts with system ROS2 installations.
"""

import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension


def enable_ros2_bridge_extension() -> bool:
    """Enable the Isaac Sim ROS2 bridge extension.

    Returns:
        True if extension was enabled successfully
    """
    try:
        enable_extension("isaacsim.ros2.bridge")
        print("[ROS2Bridge] Enabled isaacsim.ros2.bridge extension")
        return True
    except Exception as e:
        print(f"[ROS2Bridge] Failed to enable extension: {e}")
        return False


def enable_sensor_extension() -> bool:
    """Enable the Isaac Sim sensor physics extension for IMU.

    Returns:
        True if extension was enabled successfully
    """
    try:
        enable_extension("isaacsim.sensors.physics")
        print("[ROS2Bridge] Enabled isaacsim.sensors.physics extension")
        return True
    except Exception as e:
        print(f"[ROS2Bridge] Failed to enable sensor extension: {e}")
        return False


class ROS2Bridge:
    """Manages ROS2 OmniGraph nodes for Isaac Sim Jetbot with stereo cameras and IMU."""

    # Stereo camera configuration
    BASELINE = 0.10  # 10cm baseline between left and right cameras
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # Topic names for cuVSLAM compatibility
    LEFT_IMAGE_TOPIC = "/camera/left/image_raw"
    LEFT_INFO_TOPIC = "/camera/left/camera_info"
    RIGHT_IMAGE_TOPIC = "/camera/right/image_raw"
    RIGHT_INFO_TOPIC = "/camera/right/camera_info"

    # IMU topic
    NAMESPACE = "jetbot"
    IMU_TOPIC = f"/{NAMESPACE}/imu"

    # Odometry topics
    ODOM_TOPIC = f"/{NAMESPACE}/odom"
    TF_TOPIC = "/tf"
    CLOCK_TOPIC = "/clock"

    # Frame IDs (matching USD stage hierarchy)
    ODOM_FRAME = "world"  # USD root frame
    BASE_FRAME = "chassis"
    LEFT_CAMERA_FRAME = "left_camera_optical"
    RIGHT_CAMERA_FRAME = "right_camera_optical"
    IMU_FRAME = "imu_link"

    def __init__(self, robot_prim_path: str = "/World/Jetbot"):
        """Initialize ROS2 Bridge with stereo cameras and IMU.

        Args:
            robot_prim_path: USD prim path of the Jetbot robot
        """
        self.robot_prim_path = robot_prim_path
        self.chassis_prim_path = f"{robot_prim_path}/chassis"

        # Stereo camera paths (will be created on chassis)
        self.stereo_mount_path = f"{self.chassis_prim_path}/stereo_camera"
        self.left_camera_path = f"{self.stereo_mount_path}/left_camera"
        self.right_camera_path = f"{self.stereo_mount_path}/right_camera"

        # IMU path
        self.imu_path = f"{self.chassis_prim_path}/imu_sensor"

        # Graph paths
        self.clock_graph_path = "/ROS2ClockGraph"
        self.stereo_graph_path = "/ROS2StereoGraph"
        self.imu_graph_path = "/ROS2IMUGraph"
        self.tf_odom_graph_path = "/ROS2TFOdomGraph"

        self._graph_created = False
        self._cameras_created = False
        self._imu_created = False

    def _inspect_cameras(self, stage) -> None:
        """Inspect and print all cameras in the USD stage.

        Args:
            stage: USD stage object
        """
        try:
            from pxr import Usd, UsdGeom
            import math

            cameras = []
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Camera):
                    cameras.append(prim)

            print(f"\n[ROS2Bridge] Found {len(cameras)} camera(s) in stage:")
            print("-" * 60)

            for cam_prim in cameras:
                camera = UsdGeom.Camera(cam_prim)
                xformable = UsdGeom.Xformable(cam_prim)

                path = str(cam_prim.GetPath())
                name = cam_prim.GetName()

                # World transform
                world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                pos = world_tf.ExtractTranslation()

                # Camera properties
                fl = camera.GetFocalLengthAttr().Get()
                ha = camera.GetHorizontalApertureAttr().Get()

                # FOV calculation
                fov_str = "N/A"
                if fl and ha:
                    fov = 2 * math.atan(ha / (2 * fl)) * 180 / math.pi
                    fov_str = f"{fov:.1f}°"

                # Get local transform ops
                xform_ops = xformable.GetOrderedXformOps()
                local_pos = None
                local_rot = None
                for op in xform_ops:
                    op_name = op.GetOpName()
                    if "translate" in op_name.lower():
                        local_pos = op.Get()
                    elif "rotate" in op_name.lower():
                        local_rot = op.Get()

                print(f"  {name}:")
                print(f"    Path: {path}")
                print(f"    World Pos: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                if local_pos:
                    print(f"    Local Pos: ({local_pos[0]:.4f}, {local_pos[1]:.4f}, {local_pos[2]:.4f})")
                if local_rot:
                    print(f"    Local Rot: ({local_rot[0]:.1f}, {local_rot[1]:.1f}, {local_rot[2]:.1f})")
                print(f"    Focal: {fl}, Aperture: {ha}, FOV: {fov_str}")

            print("-" * 60 + "\n")

        except Exception as e:
            print(f"[ROS2Bridge] Camera inspection failed: {e}")

    def create_stereo_cameras(self, stage) -> bool:
        """Create stereo camera prims on the Jetbot chassis.

        Args:
            stage: USD stage object

        Returns:
            True if cameras created successfully
        """
        if self._cameras_created:
            print("[ROS2Bridge] Stereo cameras already created")
            return True

        try:
            from pxr import Gf, UsdGeom

            # Create stereo camera mount
            mount_xform = UsdGeom.Xform.Define(stage, self.stereo_mount_path)
            # Position mount at front of robot, matching jetbot_camera height (~0.106)
            mount_xform.AddTranslateOp().Set(Gf.Vec3d(0.07, 0, 0.06))

            # Create left camera (+Y offset for 10cm baseline)
            left_camera = UsdGeom.Camera.Define(stage, self.left_camera_path)
            left_xform = UsdGeom.Xformable(left_camera.GetPrim())
            left_xform.AddTranslateOp().Set(Gf.Vec3d(0, self.BASELINE / 2, 0))
            left_xform.AddRotateXYZOp().Set(Gf.Vec3d(90, 0, -90))

            # Set left camera properties
            left_camera.GetFocalLengthAttr().Set(1.93)  # ~90 degree FOV
            left_camera.GetHorizontalApertureAttr().Set(3.6)
            left_camera.GetVerticalApertureAttr().Set(2.7)
            left_camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

            # Create right camera (-Y offset for 10cm baseline)
            right_camera = UsdGeom.Camera.Define(stage, self.right_camera_path)
            right_xform = UsdGeom.Xformable(right_camera.GetPrim())
            right_xform.AddTranslateOp().Set(Gf.Vec3d(0, -self.BASELINE / 2, 0))
            right_xform.AddRotateXYZOp().Set(Gf.Vec3d(90, 0, -90))

            # Set right camera properties (same as left)
            right_camera.GetFocalLengthAttr().Set(1.93)
            right_camera.GetHorizontalApertureAttr().Set(3.6)
            right_camera.GetVerticalApertureAttr().Set(2.7)
            right_camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

            self._cameras_created = True
            print(f"[ROS2Bridge] Created stereo cameras:")
            print(f"  Left:  {self.left_camera_path}")
            print(f"  Right: {self.right_camera_path}")
            print(f"  Baseline: {self.BASELINE}m")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create stereo cameras: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_imu_sensor(self, stage) -> bool:
        """Create IMU sensor prim on the Jetbot chassis.

        Args:
            stage: USD stage object

        Returns:
            True if IMU created successfully
        """
        if self._imu_created:
            print("[ROS2Bridge] IMU sensor already created")
            return True

        try:
            from pxr import Gf, UsdGeom, UsdPhysics

            # Create IMU xform at chassis center
            imu_xform = UsdGeom.Xform.Define(stage, self.imu_path)
            imu_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.02))

            self._imu_created = True
            print(f"[ROS2Bridge] Created IMU sensor at {self.imu_path}")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create IMU sensor: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_ros2_graph(self, stage=None) -> bool:
        """Create the OmniGraph for ROS2 publishing.

        Args:
            stage: USD stage object (optional). If provided, stereo cameras
                   and IMU sensor will be created on the stage first.

        Returns:
            True if graph created successfully
        """
        if self._graph_created:
            print("[ROS2Bridge] Graph already created")
            return True

        # Enable ROS2 bridge extension first
        if not enable_ros2_bridge_extension():
            print("[ROS2Bridge] Cannot create graph - extension not available")
            return False

        # Enable sensor extension for IMU
        if not enable_sensor_extension():
            print("[ROS2Bridge] Warning: Sensor extension not available, IMU may not work")

        # If stage provided, create stereo cameras and IMU sensor prims
        if stage is not None:
            if not self.create_stereo_cameras(stage):
                print("[ROS2Bridge] Warning: Failed to create stereo cameras")
            if not self.create_imu_sensor(stage):
                print("[ROS2Bridge] Warning: Failed to create IMU sensor")
            # Inspect all cameras in the stage
            self._inspect_cameras(stage)

        try:
            # Create clock publisher graph
            if not self._create_clock_graph():
                return False

            # Create stereo camera publisher graph
            if not self._create_stereo_graph():
                return False

            # Create IMU publisher graph (optional - may not be available)
            imu_enabled = self._create_imu_graph()
            if not imu_enabled:
                print("[ROS2Bridge] IMU graph not created - continuing without IMU")

            # Create TF and odometry publisher graph
            if not self._create_tf_odom_graph():
                return False

            self._graph_created = True
            print(f"[ROS2Bridge] Created OmniGraph nodes for cuVSLAM")
            print(f"[ROS2Bridge] Publishing topics:")
            print(f"  Stereo:")
            print(f"    - {self.LEFT_IMAGE_TOPIC}")
            print(f"    - {self.LEFT_INFO_TOPIC}")
            print(f"    - {self.RIGHT_IMAGE_TOPIC}")
            print(f"    - {self.RIGHT_INFO_TOPIC}")
            if imu_enabled:
                print(f"  IMU:")
                print(f"    - {self.IMU_TOPIC}")
            print(f"  Navigation:")
            print(f"    - {self.ODOM_TOPIC}")
            print(f"    - {self.TF_TOPIC}")
            print(f"    - {self.CLOCK_TOPIC}")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create graph: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_clock_graph(self) -> bool:
        """Create the clock publisher graph.

        Returns:
            True if successful
        """
        try:
            keys = og.Controller.Keys

            og.Controller.edit(
                {"graph_path": self.clock_graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ],
                    keys.SET_VALUES: [
                        ("PublishClock.inputs:topicName", self.CLOCK_TOPIC),
                    ],
                }
            )
            print(f"[ROS2Bridge] Created clock graph at {self.clock_graph_path}")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create clock graph: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_stereo_graph(self) -> bool:
        """Create the stereo camera publisher graph.

        Returns:
            True if successful
        """
        try:
            keys = og.Controller.Keys

            og.Controller.edit(
                {"graph_path": self.stereo_graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        # Left camera
                        ("LeftRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("LeftCameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("LeftCameraInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                        # Right camera
                        ("RightRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("RightCameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("RightCameraInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    ],
                    keys.CONNECT: [
                        # Left camera execution flow
                        ("OnPlaybackTick.outputs:tick", "LeftRenderProduct.inputs:execIn"),
                        ("LeftRenderProduct.outputs:execOut", "LeftCameraHelper.inputs:execIn"),
                        ("LeftRenderProduct.outputs:execOut", "LeftCameraInfo.inputs:execIn"),
                        ("LeftRenderProduct.outputs:renderProductPath", "LeftCameraHelper.inputs:renderProductPath"),
                        ("LeftRenderProduct.outputs:renderProductPath", "LeftCameraInfo.inputs:renderProductPath"),
                        # Right camera execution flow
                        ("OnPlaybackTick.outputs:tick", "RightRenderProduct.inputs:execIn"),
                        ("RightRenderProduct.outputs:execOut", "RightCameraHelper.inputs:execIn"),
                        ("RightRenderProduct.outputs:execOut", "RightCameraInfo.inputs:execIn"),
                        ("RightRenderProduct.outputs:renderProductPath", "RightCameraHelper.inputs:renderProductPath"),
                        ("RightRenderProduct.outputs:renderProductPath", "RightCameraInfo.inputs:renderProductPath"),
                    ],
                    keys.SET_VALUES: [
                        # Left camera render product
                        ("LeftRenderProduct.inputs:cameraPrim", self.left_camera_path),
                        ("LeftRenderProduct.inputs:width", self.CAMERA_WIDTH),
                        ("LeftRenderProduct.inputs:height", self.CAMERA_HEIGHT),
                        # Left camera image publisher
                        ("LeftCameraHelper.inputs:frameId", self.LEFT_CAMERA_FRAME),
                        ("LeftCameraHelper.inputs:topicName", self.LEFT_IMAGE_TOPIC),
                        ("LeftCameraHelper.inputs:type", "rgb"),
                        # Left camera info publisher
                        ("LeftCameraInfo.inputs:frameId", self.LEFT_CAMERA_FRAME),
                        ("LeftCameraInfo.inputs:topicName", self.LEFT_INFO_TOPIC),
                        # Right camera render product
                        ("RightRenderProduct.inputs:cameraPrim", self.right_camera_path),
                        ("RightRenderProduct.inputs:width", self.CAMERA_WIDTH),
                        ("RightRenderProduct.inputs:height", self.CAMERA_HEIGHT),
                        # Right camera image publisher
                        ("RightCameraHelper.inputs:frameId", self.RIGHT_CAMERA_FRAME),
                        ("RightCameraHelper.inputs:topicName", self.RIGHT_IMAGE_TOPIC),
                        ("RightCameraHelper.inputs:type", "rgb"),
                        # Right camera info publisher
                        ("RightCameraInfo.inputs:frameId", self.RIGHT_CAMERA_FRAME),
                        ("RightCameraInfo.inputs:topicName", self.RIGHT_INFO_TOPIC),
                    ],
                }
            )

            print(f"[ROS2Bridge] Created stereo camera graph at {self.stereo_graph_path}")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create stereo camera graph: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_imu_graph(self) -> bool:
        """Create the IMU publisher graph.

        Returns:
            True if successful
        """
        try:
            keys = og.Controller.Keys

            og.Controller.edit(
                {"graph_path": self.imu_graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("ReadIMU", "isaacsim.sensors.physics.IsaacReadIMU"),
                        ("PublishIMU", "isaacsim.ros2.bridge.ROS2PublishImu"),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "ReadIMU.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "PublishIMU.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishIMU.inputs:timeStamp"),
                        ("ReadIMU.outputs:angVel", "PublishIMU.inputs:angularVelocity"),
                        ("ReadIMU.outputs:linAcc", "PublishIMU.inputs:linearAcceleration"),
                        ("ReadIMU.outputs:orientation", "PublishIMU.inputs:orientation"),
                    ],
                    keys.SET_VALUES: [
                        ("ReadIMU.inputs:imuPrim", self.imu_path),
                        ("PublishIMU.inputs:topicName", self.IMU_TOPIC),
                        ("PublishIMU.inputs:frameId", self.IMU_FRAME),
                        ("PublishIMU.inputs:publishAngularVelocity", True),
                        ("PublishIMU.inputs:publishLinearAcceleration", True),
                        ("PublishIMU.inputs:publishOrientation", True),
                    ],
                }
            )

            print(f"[ROS2Bridge] Created IMU graph at {self.imu_graph_path}")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create IMU graph: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_tf_odom_graph(self) -> bool:
        """Create TF and odometry publisher graph.

        Returns:
            True if successful
        """
        try:
            import usdrt.Sdf
            keys = og.Controller.Keys

            og.Controller.edit(
                {"graph_path": self.tf_odom_graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                        ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                        ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                    ],
                    keys.CONNECT: [
                        # TF publishing
                        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                        # Odometry computation and publishing
                        ("OnPlaybackTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "PublishOdometry.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
                        ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                        ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                        ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                        ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                    ],
                    keys.SET_VALUES: [
                        # TF settings - publish full robot articulation tree
                        ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(self.robot_prim_path)]),
                        ("PublishTF.inputs:topicName", self.TF_TOPIC),
                        # Odometry settings
                        ("ComputeOdometry.inputs:chassisPrim", [usdrt.Sdf.Path(self.chassis_prim_path)]),
                        ("PublishOdometry.inputs:topicName", self.ODOM_TOPIC),
                        ("PublishOdometry.inputs:odomFrameId", self.ODOM_FRAME),
                        ("PublishOdometry.inputs:chassisFrameId", self.BASE_FRAME),
                    ],
                }
            )

            print(f"[ROS2Bridge] Created TF/Odom graph at {self.tf_odom_graph_path}")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create TF/Odom graph: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_published_topics(self) -> dict:
        """Get dictionary of published topic names.

        Returns:
            Dict mapping topic type to topic name
        """
        return {
            "left_image": self.LEFT_IMAGE_TOPIC,
            "left_info": self.LEFT_INFO_TOPIC,
            "right_image": self.RIGHT_IMAGE_TOPIC,
            "right_info": self.RIGHT_INFO_TOPIC,
            "imu": self.IMU_TOPIC,
            "odom": self.ODOM_TOPIC,
            "tf": self.TF_TOPIC,
            "clock": self.CLOCK_TOPIC,
        }

    def is_enabled(self) -> bool:
        """Check if ROS2 bridge graph has been created.

        Returns:
            True if graph is active
        """
        return self._graph_created

    def get_stereo_config(self) -> dict:
        """Get stereo camera configuration.

        Returns:
            Dict with stereo camera settings
        """
        return {
            "baseline": self.BASELINE,
            "width": self.CAMERA_WIDTH,
            "height": self.CAMERA_HEIGHT,
            "left_frame": self.LEFT_CAMERA_FRAME,
            "right_frame": self.RIGHT_CAMERA_FRAME,
        }
