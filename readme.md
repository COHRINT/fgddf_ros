# fgddf-ROS

## Setup
- Install fglib from ofer modified repository: https://github.com/oferdagan/fglib/tree/Gaussian_mul_expand_branch
    - Alternatively, install from source included in this repository
    - Run ```python setup.py install``` in the fglib directory

## Running an Experiment
Follow the following steps to run a fgddf_ros experiment. Each number should be executed on a separate, new terminal. In all of the following steps, XXX denotes the IP address of the master computer and YYY denotes the IP address of the robot
1. On master computer:
    ```
    source <ros_path>/setup.bash
    source ./network.sh XXX XXX
    roscore
    ```
2. On master computer:
    ```
    source <ros_path>/setup.bash
    source ./network.sh XXX XXX
    roslaunch vrpn_client_ros sample.launch server:=192.168.20.100
    ```
3. On master computer:
    ```
    source <ros_path>/setup.bash
    source ./network.sh XXX XXX
    rosrun fgddf_ros boss.py
    ```
4. On master computer:
    ```
    source <ros_path>/setup.bash
    source ./network.sh XXX XXX
    rosbag record /truth_data /results
    ```
5. On EACH robot:
    ```
    source ./network.sh XXX YYY
    ./startup.sh aspen
    ```
6. On EACH robot:
    ```
    source ./network.sh XXX YYY
    rosrun fgddf_ros main.py
    ```