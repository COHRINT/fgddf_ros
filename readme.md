# fgddf-ROS

## Setup
- Install cohrint_jackal_bringup from repository: https://github.com/COHRINT/cohrint_jackal/tree/master
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
5. OPTIONAL: On EACH robot:
    ```
    source ./network.sh XXX YYY
    ./startup.sh aspen
    ```
6. On EACH robot:
    ```
    source ./network.sh XXX YYY
    rosrun fgddf_ros main.py
    ```
In our testing we have observed that step 5 above can drastically slow down the processing speed of the Jackals and thus also slow down the speed at which the experiment runs. Thus, it is recommended to run the experiement without step 5.

## Code Overview
### Experiement Setup/Configuration
The experiment setup/configuration can be controlled through the addition/modification of input files in the /src/fgDDF. These input files allow the user to change a wide variety of parameters including the number of agents, number of targets, H and R matrices for each agent, etc. Once an input file has been copied into /src/fgDDF and/or modified in /src/fgDDF, line 21 in main.py must be changed to: ```from fgDDF.<import file name> import *```.

Additionally, the algorithm can run either with a channel filter or covariance intersection approach. To configure this, line 298 in main.py can be changed; "HS_CF" will use a channel filter algorithm while "HS_CI" will use a covariance intersection algorithm.
