This repository implements a continual learning framework for robotic visual perception using Generative Replay to mitigate catastrophic forgetting across sequential tasks. The system is evaluated on three real-world inspired perception tasks and deployed in a ROS 2 + Gazebo simulation using the Pioneer3AT mobile robot.

## Retraining the Model on a Custom Dataset

```bash
cd continual-learning-gen-replay
pip install -r requirements.txt<img width="611" height="329" alt="Screenshot from 2026-01-19 17-47-26" src="https://github.com/user-attachments/assets/52f39f52-99c5-434d-a751-55226be13bf6" />

python src/main.py
```

## Running the ROS 2 Humble Gazebo Simulation

```bash
cd ros2_ws
source install/setup.bash
ros2 launch pioneer3at_gz pioneer3at.launch.py
```

