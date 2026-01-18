import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # Define paths to your world and model directories
    world_file = 'gazebo_models/worlds/test_city.world'
    pioneer_model_path = '/models/Pioneer3AT/model.sdf'

    # Set GAZEBO_MODEL_PATH to include both directories
    env = os.environ.copy()
    gazebo_model_path = (
        '/gazebo_models/models:'  'plz update model paths acording to your'
        '/pioneer3at_gz/models'
    )
    env['GAZEBO_MODEL_PATH'] = gazebo_model_path

    return LaunchDescription([
        # Start Gazebo with the world file
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_factory.so'],
            output='screen',
            env=env
        ),

        # Spawn Pioneer3AT model in the world
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-file', pioneer_model_path,
                '-entity', 'pioneer3at',
                '-x', '0.0', '-y', '0.0', '-z', '0.1'
            ],
            output='screen',
            env=env
        ),

        # 🔥 Fire Detection Node (Generative Replay – Fire Task Only)
        Node(
            package='pioneer3at_gz',
            executable='fire_detection',
            name='fire_detection_node',
            output='screen'
        ),
    ])
