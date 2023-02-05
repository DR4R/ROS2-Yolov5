from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
import os

def generate_launch_description():
    
    param_dir = LaunchConfiguration(
        'param_dir',
        default=os.path.join(
            get_package_share_directory('yolov5'),
            'param',
            'yolo_parameters.yaml'
        )
    )
    
    params = DeclareLaunchArgument(
        'param_dir',
        default_value=param_dir,
        description='Full path of parameter file'
    )
    
    camera_node = Node(
        package='yolov5',
        executable='img_publisher',
        name='camera',
        output='screen'
    )
    
    # image_show_node = Node(
        
    # )
    
    yolo_node = Node(
        package='yolov5',
        executable='yolov5',
        name='detection',
        output='screen'
    )
    
    return LaunchDescription([
        params,
        camera_node,
        # image_show_node,
        yolo_node
    ])