#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import Twist

class Circle:
        def __init__(self):
            self.pub = rospy.Publisher("/case/jackal_velocity_controller/cmd_vel",Twist,queue_size=10)
            self.linear_speed = 0.3
            # self.diameter = 2
            # self.angular_speed = -1 * (np.log(self.diameter / 3.634) / -3.036)
            self.angular_speed = 0.3
            rate = rospy.Rate(10)

            while not rospy.is_shutdown():
                t = Twist()
                t.linear.x = self.linear_speed
                t.angular.z = self.angular_speed
                self.pub.publish(t)
                rate.sleep()

if __name__ == '__main__':
    rospy.init_node("circle_case")
    c = Circle()