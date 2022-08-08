#!/usr/bin/env python

"""
Module for all ROS publisher/subscribers
"""

# ROS related libraries
import rospy
import rospkg
import os.path as path
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from fgddf_ros.msg import ChannelFilter
from fgddf_ros.msg import Results

# FGDDF libraries
from fgDDF.inputFile import *

# Misc libraries
import numpy as np

class ROSFxn:
    def __init__(self,agent_name):
        rospack = rospkg.RosPack()
        p = rospack.get_path("fgddf_ros")

        # Create subscriber to save agent position
        self.agent_sub = rospy.Subscriber("vrpn_client_node/"+agent_name+"/pose",PoseStamped,self.agent_callback)

        # Create subscribers to save target positions
        if (target1 is not None):
            self.target1_sub = rospy.Subscriber("vrpn_client_node/"+target1+"/pose",PoseStamped,self.target1_callback)
        if (target2 is not None):
            self.target2_sub = rospy.Subscriber("vrpn_client_node/"+target2+"/pose",PoseStamped,self.target2_callback)
        if (target3 is not None):
            self.target3_sub = rospy.Subscriber("vrpn_client_node/"+target3+"/pose",PoseStamped,self.target3_callback)
        if (target4 is not None):
            self.target4_sub = rospy.Subscriber("vrpn_client_node/"+target4+"/pose",PoseStamped,self.target4_callback)
        if (target5 is not None):
            self.target1_sub = rospy.Subscriber("vrpn_client_node/"+target5+"/pose",PoseStamped,self.target5_callback)
        if (target6 is not None):
            self.target6_sub = rospy.Subscriber("vrpn_client_node/"+target6+"/pose",PoseStamped,self.target6_callback)
        if (target7 is not None):
            self.target7_sub = rospy.Subscriber("vrpn_client_node/"+target7+"/pose",PoseStamped,self.target7_callback)
        if (target8 is not None):
            self.target8_sub = rospy.Subscriber("vrpn_client_node/"+target8+"/pose",PoseStamped,self.target8_callback)
        if (target9 is not None):
            self.target9_sub = rospy.Subscriber("vrpn_client_node/"+target9+"/pose",PoseStamped,self.target9_callback)
        if (target10 is not None):
            self.target10_sub = rospy.Subscriber("vrpn_client_node/"+target10+"/pose",PoseStamped,self.target10_callback)
        if (target11 is not None):
            self.target11_sub = rospy.Subscriber("vrpn_client_node/"+target11+"/pose",PoseStamped,self.target11_callback)
        if (target12 is not None):
            self.target12_sub = rospy.Subscriber("vrpn_client_node/"+target12+"/pose",PoseStamped,self.target12_callback)
        if (target13 is not None):
            self.target13_sub = rospy.Subscriber("vrpn_client_node/"+target13+"/pose",PoseStamped,self.target13_callback)
        if (target14 is not None):
            self.target14_sub = rospy.Subscriber("vrpn_client_node/"+target14+"/pose",PoseStamped,self.target14_callback)
        if (target15 is not None):
            self.target15_sub = rospy.Subscriber("vrpn_client_node/"+target15+"/pose",PoseStamped,self.target15_callback)
        if (target16 is not None):
            self.target16_sub = rospy.Subscriber("vrpn_client_node/"+target16+"/pose",PoseStamped,self.target16_callback)
        if (target17 is not None):
            self.target17_sub = rospy.Subscriber("vrpn_client_node/"+target17+"/pose",PoseStamped,self.target17_callback)
        if (target18 is not None):
            self.target18_sub = rospy.Subscriber("vrpn_client_node/"+target18+"/pose",PoseStamped,self.target18_callback)
        if (target19 is not None):
            self.target19_sub = rospy.Subscriber("vrpn_client_node/"+target19+"/pose",PoseStamped,self.target19_callback)
        if (target20 is not None):
            self.target20_sub = rospy.Subscriber("vrpn_client_node/"+target20+"/pose",PoseStamped,self.target20_callback)

        # Create array to save target positions
        self.target_pos = np.empty([20,2])

        # Run ROS functions
        self.run()
    
    # Run the ros functions
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    # Define ROS callback function to store agent position
    def agent_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.agent_pos = np.array([x,y])

    # Define ROS callback functions to store target position
    def target1_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[0] = np.array([x,y])

    def target2_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[1] = np.array([x,y])

    def target3_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[2] = np.array([x,y])

    def target4_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[3] = np.array([x,y])

    def target5_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[4] = np.array([x,y])

    def target6_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[5] = np.array([x,y])

    def target7_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[6] = np.array([x,y])

    def target8_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[7] = np.array([x,y])

    def target9_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[8] = np.array([x,y])

    def target10_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[9] = np.array([x,y])

    def target11_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[10] = np.array([x,y])

    def target12_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[11] = np.array([x,y])

    def target13_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[12] = np.array([x,y])

    def target14_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[13] = np.array([x,y])

    def target15_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[14] = np.array([x,y])

    def target16_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[15] = np.array([x,y])

    def target17_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[16] = np.array([x,y])

    def target18_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[17] = np.array([x,y])

    def target19_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[18] = np.array([x,y])

    def target20_callback(self,msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_pos[19] = np.array([x,y])

# NEEDS TO BE MOVVED TO MAIN.PY!!
if __name__ == "__main__":
    rospy.init_node("ros_functions")
    rf = ROSFxn(agent_name)