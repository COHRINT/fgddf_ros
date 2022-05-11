#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, TwistStamped
import numpy as np
from fgddf_ros.msg import ChannelFilter, TruthData

TARGET_NAME = "cohrint_tycho_bot_1"


class Boss:
    def __init__(self, nAgents, rate):
        self.nAgents = nAgents
        self.k = 2
        self.rate = rate
        self.has_msg = np.zeros(nAgents)
        self.current_truth = TruthData()
        self.current_truth.bias0 = np.array([[2], [3]])
        self.current_truth.bias1 = np.array([[3], [2]])
        self.current_truth.bias2 = np.array([[5], [5]])
        self.current_truth.target = TARGET_NAME

    def talker(self):
        pub = rospy.Publisher("boss", String, queue_size=10)
        sub = rospy.Subscriber("chatter", ChannelFilter, self.callback)
        truth_pub = rospy.Publisher("/truth_data",TruthData,queue_size=10)
        truth_pos_sub = rospy.Subscriber("/vrpn_client_node/"+TARGET_NAME+"/pose",PoseStamped,self.callback_position)
        truth_vel_sub = rospy.Subscriber("/vrpn_client_node/"+TARGET_NAME+"/twist",TwistStamped,self.callback_velocity)
        rospy.init_node("talker", anonymous=True)
        rate = rospy.Rate(self.rate)
        truth_pub.publish(self.current_truth)
        while not rospy.is_shutdown():
            hello_str = "hello world %s" % rospy.get_time()
            print(self.has_msg)
            if all(self.has_msg):
                pub.publish(hello_str)
                self.has_msg = np.zeros(self.nAgents)
                self.k += 1
                print(f"Time step k = {self.k}")
                truth_pub.publish(self.current_truth)
            rate.sleep()

    def callback(self, data):
        self.has_msg[data.sender] += 1

    def callback_position(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.current_truth.position = np.array([x_pos,y_pos,z_pos])
        # print(self.pos)

    def callback_velocity(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.current_truth.velocity = np.array([x_vel,y_vel,z_vel])
        # print(self.vel)

if __name__ == "__main__":
    try:
        rate = 10  # Hz
        nAgents = 3  # Number of agents
        B = Boss(nAgents, rate)
        B.talker()
    except rospy.ROSInterruptException:
        pass
