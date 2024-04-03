# PointPillar-ROS-Node

This repository contains a ROS node that utilizes the PointPillar neural network model ([MMDetection3d](https://mmdetection3d.readthedocs.io/en/latest/)) to process point cloud data from ROSbags. The repository is specifically designed to work in a ROS 1 environment.

Please note that the instructions provided assume a ROS Noetic environment. The repo is tested with Ubuntu 20.

## Prerequisites

Before using this repository, please ensure that you have the following prerequisites installed:

- CUDA 11.6 and MMDetection3D library [[installation instruction](https://mengwoods.github.io/post/tech/001-install-openmm3d-lib/en/)]
- ROS Noetic
- jsk-recognition-msgs, jsk-rviz-plugins

You can install the required ROS packages using the following commands:
```bash
sudo apt-get install ros-noetic-jsk-recognition-msgs ros-noetic-jsk-rviz-plugins
```

Additionally, you will need to download the required models weight and config files from [model zoo](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars) and place them in directory: [`pointpillar_ros/model/pointpillar/`](./pointpillar_ros/model/pointpillar/). Update model weight and confi files' names in [launchfile](./pointpillar_ros/launch/pointpillar_ros.launch).

## Usage

To use this code, follow the steps below:

1. Clone this repository to your `catkin_ws/src` directory.
2. Update the point cloud topic and model names in [launchfile](./pointpillar_ros/launch/pointpillar_ros.launch).
3. Build the package:
    ```bash
    catkin config --install
    catkin build pointpillar_ros
    ```
4. Launch the ROS node by executing the following commands:
    ```bash
    source install/setup.bash # or source install/pointpillar_ros/setup.bash
    roslaunch pointpillar_ros pointpillar_ros.launch
    ```

## Troubleshooting

If you encounter a `ModuleNotFoundError` related to the `netifaces` module, you can resolve it by installing the module using pip `pip install netifaces`.

Hope this helps! If you have any further questions, feel free to ask.
