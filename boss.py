#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import numpy as np
from fgddf_ros.msg import ChannelFilter


class Boss:
    def __init__(self, nAgents, rate):
        self.nAgents = nAgents
        self.k = 2
        self.rate = rate
        self.has_msg = np.zeros(nAgents)

    def talker(self):
        pub = rospy.Publisher("boss", String, queue_size=10)
        sub = rospy.Subscriber("chatter", ChannelFilter, self.callback)
        rospy.init_node("talker", anonymous=True)
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            hello_str = "hello world %s" % rospy.get_time()
            print(self.has_msg)
            if all(self.has_msg):
                pub.publish(hello_str)
                self.has_msg = np.zeros(self.nAgents)
                self.k += 1
                print(f"Time step k = {self.k}")
            rate.sleep()

    def callback(self, data):
        self.has_msg[data.sender] += 1


if __name__ == "__main__":
    try:
        rate = 10  # Hz
        nAgents = 2  # Number of agents
        B = Boss(nAgents, rate)
        B.talker()
    except rospy.ROSInterruptException:
        pass
