"""Launch file for NVIDIA cuVSLAM with Isaac Sim Jetbot.

This launch file configures cuVSLAM for stereo visual SLAM,
receiving data from Isaac Sim via ROS2 topics.

Note: IMU fusion is available but disabled by default. Enable with:
    ros2 launch ... enable_imu_fusion:=true

Frame Configuration:
- Isaac Sim publishes TF: world -> chassis -> cameras
- cuVSLAM configured with odom_frame='world' to match Isaac Sim's USD hierarchy
- Static TFs added: map -> world, camera -> camera_optical

Topics subscribed (from Isaac Sim):
- /camera/left/image_raw
- /camera/left/camera_info
- /camera/right/image_raw
- /camera/right/camera_info

Topics published:
- /visual_slam/tracking/odometry
- /visual_slam/tracking/slam_path
- /visual_slam/vis/landmarks_cloud
- /tf (map -> world transform)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from /clock topic'
    )

    enable_slam_arg = DeclareLaunchArgument(
        'enable_slam',
        default_value='true',
        description='Enable SLAM (mapping). Set to false for localization only.'
    )

    enable_imu_arg = DeclareLaunchArgument(
        'enable_imu_fusion',
        default_value='false',  # Disabled - Isaac Sim 5.0 IMU node unavailable
        description='Enable IMU fusion for better tracking'
    )

    # cuVSLAM node
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam',
        name='visual_slam',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),

            # Frame configuration (matching Isaac Sim)
            'map_frame': 'map',
            'odom_frame': 'world',  # Isaac Sim publishes world->chassis in TF
            'base_frame': 'chassis',
            'input_left_camera_frame': 'left_camera_optical',
            'input_right_camera_frame': 'right_camera_optical',
            'input_imu_frame': 'imu_link',

            # Camera configuration
            'num_cameras': 2,
            'enable_rectified_pose': True,

            # IMU fusion
            'enable_imu_fusion': LaunchConfiguration('enable_imu_fusion'),
            'gyro_noise_density': 0.000244,
            'gyro_random_walk': 0.000019393,
            'accel_noise_density': 0.001862,
            'accel_random_walk': 0.003,
            'calibration_frequency': 200.0,
            'imu_jitter_threshold_ms': 10.0,

            # SLAM configuration
            'enable_slam_visualization': True,
            'enable_landmarks_view': True,
            'enable_observations_view': True,

            # Image processing
            'rectified_images': False,
            'image_jitter_threshold_ms': 34.0,
            'img_height': 480,
            'img_width': 640,

            # Performance tuning
            'enable_debug_mode': False,
            'debug_dump_path': '/tmp/cuvslam_debug',
        }],
        remappings=[
            # Stereo camera topics from Isaac Sim
            ('visual_slam/image_0', '/camera/left/image_raw'),
            ('visual_slam/camera_info_0', '/camera/left/camera_info'),
            ('visual_slam/image_1', '/camera/right/image_raw'),
            ('visual_slam/camera_info_1', '/camera/right/camera_info'),
            # IMU topic from Isaac Sim
            ('visual_slam/imu', '/jetbot/imu'),
        ]
    )

    # Static transform: map -> world (initial identity, cuVSLAM will update)
    map_to_world_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_world_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'world'],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    # Static transform: chassis -> left_camera
    # Position: stereo_mount (0.07, 0, 0.06) + left offset (0, 0.05, 0) = (0.07, 0.05, 0.06)
    # Rotation: XYZ Euler (90, 0, -90) = quaternion (0.5, -0.5, -0.5, 0.5)
    chassis_to_left_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='chassis_to_left_camera_tf',
        arguments=['0.07', '0.05', '0.06', '0.5', '-0.5', '-0.5', '0.5', 'chassis', 'left_camera'],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    # Static transform: chassis -> right_camera
    # Position: stereo_mount (0.07, 0, 0.06) + right offset (0, -0.05, 0) = (0.07, -0.05, 0.06)
    # Rotation: XYZ Euler (90, 0, -90) = quaternion (0.5, -0.5, -0.5, 0.5)
    chassis_to_right_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='chassis_to_right_camera_tf',
        arguments=['0.07', '-0.05', '0.06', '0.5', '-0.5', '-0.5', '0.5', 'chassis', 'right_camera'],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    # Static transform: left_camera -> left_camera_optical
    # Quaternion [-0.5, 0.5, -0.5, 0.5] rotates camera link frame to optical frame
    # (Z forward, X right, Y down in optical frame)
    left_camera_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='left_camera_optical_tf',
        arguments=['0', '0', '0', '-0.5', '0.5', '-0.5', '0.5', 'left_camera', 'left_camera_optical'],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    # Static transform: right_camera -> right_camera_optical
    right_camera_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='right_camera_optical_tf',
        arguments=['0', '0', '0', '-0.5', '0.5', '-0.5', '0.5', 'right_camera', 'right_camera_optical'],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        enable_slam_arg,
        enable_imu_arg,
        visual_slam_node,
        map_to_world_tf,
        chassis_to_left_camera_tf,
        chassis_to_right_camera_tf,
        left_camera_optical_tf,
        right_camera_optical_tf,
    ])
