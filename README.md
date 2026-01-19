This repository implements a continual learning framework for robotic visual perception using Generative Replay to mitigate catastrophic forgetting across sequential tasks. The system is evaluated on three real-world inspired perception tasks and deployed in a ROS 2 + Gazebo simulation using the Pioneer3AT mobile robot.

## Retraining the Model on a Custom Dataset

```bash
cd continual-learning-gen-replay
pip install -r requirements.txt
python src/main.py
```

## Running the ROS 2 Humble Gazebo Simulation

```bash
cd ros2_ws
source install/setup.bash
ros2 launch pioneer3at_gz pioneer3at.launch.py
```

