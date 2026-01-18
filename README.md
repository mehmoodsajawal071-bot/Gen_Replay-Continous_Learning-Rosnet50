This repository implements a continual learning framework for robotic visual perception using Generative Replay to mitigate catastrophic forgetting across sequential tasks. The system is evaluated on three real-world inspired perception tasks and deployed in a ROS 2 + Gazebo simulation using the Pioneer3AT mobile robot.

For retraining the model on custom datasets 
cd continual-learning-gen-replay

Install dependencies:
pip install -r requirements.txt

Run training:
python src/main.py



Running the ROS2-humble Gazebo Simulation

Navigate to the ROS 2 workspace:
cd ros2_ws


Source the workspace:
source install/setup.bash


Launch the Pioneer3AT simulation:
ros2 launch pioneer3at_gz pioneer3at.launch.py

