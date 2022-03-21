# fgddf-ROS

## Setup
- Install fglib from ofer modified repository: https://github.com/oferdagan/fglib/tree/Gaussian_mul_expand_branch
    - Alternatively, install from source included in this repository
    - Run ```python setup.py install``` in the fglib directory

## Network Setup
For the following steps, let XX be the IP address of the master computer and YY be the IP address of the robots
- On master computer:
    ```
    export ROS_MASTER_URI=http://192.168.1.XX:11311
    export ROS_IP=192.168.1.XX
    roscore
    ```
    Open a new tab and run the boss script:
    ```
    export ROS_MASTER_URI=http://192.168.1.XX:11311
    export ROS_IP=192.168.1.XX
    rosrun fgddf_ros boss.py
    ```
- On each robot:
    ```
    export ROS_MASTER_URI=http://192.168.1.XX:11311
    export ROS_IP=192.168.1.YY
    rosrun fgddf_ros main.py
    ```

