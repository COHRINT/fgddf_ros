#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, TwistStamped
import numpy as np
from fgddf_ros.msg import ChannelFilter, TruthData

TARGET1_NAME = "cohrint_tycho_bot_1"
TARGET2_NAME = "cohrint_tycho_bot_2"
TARGET3_NAME = "cohrint_tycho_bot_3"
TARGET4_NAME = "cohrint_tycho_bot_4"
TARGET5_NAME = "cohrint_tycho_bot_5"
TARGET6_NAME = None
TARGET7_NAME = None
TARGET8_NAME = None
TARGET9_NAME = None
TARGET10_NAME = None
TARGET11_NAME = None
TARGET12_NAME = None
TARGET13_NAME = None
TARGET14_NAME = None
TARGET15_NAME = None
TARGET16_NAME = None
TARGET17_NAME = None
TARGET18_NAME = None
TARGET19_NAME = None
TARGET20_NAME = None

class Boss:
    def __init__(self, nAgents, rate, targets):
        self.nAgents = nAgents
        self.k = 2
        self.rate = rate
        self.has_msg = np.zeros(nAgents)

        self.target1_exists = False
        self.target2_exists = False
        self.target3_exists = False
        self.target4_exists = False
        self.target5_exists = False
        self.target6_exists = False
        self.target7_exists = False
        self.target8_exists = False
        self.target9_exists = False
        self.target10_exists = False
        self.target11_exists = False
        self.target12_exists = False
        self.target13_exists = False
        self.target14_exists = False
        self.target15_exists = False
        self.target16_exists = False
        self.target17_exists = False
        self.target18_exists = False
        self.target19_exists = False
        self.target20_exists = False

        for t in targets:
            if t == "T1":
                self.target1_truth = TruthData()
                self.target1_truth.target = TARGET1_NAME
                self.target1_exists = True
            if t == "T2":
                self.target2_truth = TruthData()
                self.target2_truth.target = TARGET2_NAME
                self.target2_exists = True
            if t == "T3":
                self.target3_truth = TruthData()
                self.target3_truth.target = TARGET3_NAME
                self.target3_exists = True
            if t == "T4":
                self.target4_truth = TruthData()
                self.target4_truth.target = TARGET4_NAME
                self.target4_exists = True
            if t == "T5":
                self.target5_truth = TruthData()
                self.target5_truth.target = TARGET5_NAME
                self.target5_exists = True
            if t == "T6":
                self.target6_truth = TruthData()
                self.target6_truth.target = TARGET6_NAME
                self.target6_exists = True
            if t == "T7":
                self.target7_truth = TruthData()
                self.target7_truth.target = TARGET7_NAME
                self.target7_exists = True
            if t == "T8":
                self.target8_truth = TruthData()
                self.target8_truth.target = TARGET8_NAME
                self.target8_exists = True
            if t == "T9":
                self.target9_truth = TruthData()
                self.target9_truth.target = TARGET9_NAME
                self.target9_exists = True
            if t == "T10":
                self.target10_truth = TruthData()
                self.target10_truth.target = TARGET10_NAME
                self.target10_exists = True
            if t == "T11":
                self.target11_truth = TruthData()
                self.target11_truth.target = TARGET11_NAME
                self.target11_exists = True
            if t == "T12":
                self.target12_truth = TruthData()
                self.target12_truth.target = TARGET12_NAME
                self.target12_exists = True
            if t == "T13":
                self.target13_truth = TruthData()
                self.target13_truth.target = TARGET13_NAME
                self.target13_exists = True
            if t == "T14":
                self.target14_truth = TruthData()
                self.target14_truth.target = TARGET14_NAME
                self.target14_exists = True
            if t == "T15":
                self.target15_truth = TruthData()
                self.target15_truth.target = TARGET15_NAME
                self.target15_exists = True
            if t == "T16":
                self.target16_truth = TruthData()
                self.target16_truth.target = TARGET16_NAME
                self.target16_exists = True
            if t == "T17":
                self.target17_truth = TruthData()
                self.target17_truth.target = TARGET17_NAME
                self.target17_exists = True
            if t == "T18":
                self.target18_truth = TruthData()
                self.target18_truth.target = TARGET18_NAME
                self.target18_exists = True
            if t == "T19":
                self.target19_truth = TruthData()
                self.target19_truth.target = TARGET19_NAME
                self.target19_exists = True
            if t == "T20":
                self.target20_truth = TruthData()
                self.target20_truth.target = TARGET20_NAME
                self.target20_exists = True

        # self.current_truth = TruthData()
        # self.current_truth.bias0 = np.array([[2], [3]])
        # self.current_truth.bias1 = np.array([[3], [2]])
        # self.current_truth.bias2 = np.array([[5], [5]])
        # self.current_truth.target = TARGET_NAME

    def talker(self):
        rospy.init_node("talker", anonymous=True)
        rate = rospy.Rate(self.rate)
        
        pub = rospy.Publisher("boss", String, queue_size=10)
        sub = rospy.Subscriber("chatter", ChannelFilter, self.callback)
        truth_pub = rospy.Publisher("/truth_data",TruthData,queue_size=10)

        # truth_pos_sub = rospy.Subscriber("/vrpn_client_node/"+TARGET_NAME+"/pose",PoseStamped,self.callback_position)
        # truth_vel_sub = rospy.Subscriber("/vrpn_client_node/"+TARGET_NAME+"/twist",TwistStamped,self.callback_velocity)
        if self.target1_exists:
            truth_pos_sub_1 = rospy.Subscriber("/vrpn_client_node/"+TARGET1_NAME+"/pose",PoseStamped,self.callback_position_1)
            truth_vel_sub_1 = rospy.Subscriber("/vrpn_client_node/"+TARGET1_NAME+"/twist",TwistStamped,self.callback_velocity_1)
            truth_pub.publish(self.target1_truth)
        if self.target2_exists:
            truth_pos_sub_2 = rospy.Subscriber("/vrpn_client_node/"+TARGET2_NAME+"/pose",PoseStamped,self.callback_position_2)
            truth_vel_sub_2 = rospy.Subscriber("/vrpn_client_node/"+TARGET2_NAME+"/twist",TwistStamped,self.callback_velocity_2)
            truth_pub.publish(self.target2_truth)
        if self.target3_exists:
            truth_pos_sub_3 = rospy.Subscriber("/vrpn_client_node/"+TARGET3_NAME+"/pose",PoseStamped,self.callback_position_3)
            truth_vel_sub_3 = rospy.Subscriber("/vrpn_client_node/"+TARGET3_NAME+"/twist",TwistStamped,self.callback_velocity_3)
            truth_pub.publish(self.target3_truth)
        if self.target4_exists:
            truth_pos_sub_4 = rospy.Subscriber("/vrpn_client_node/"+TARGET4_NAME+"/pose",PoseStamped,self.callback_position_4)
            truth_vel_sub_4 = rospy.Subscriber("/vrpn_client_node/"+TARGET4_NAME+"/twist",TwistStamped,self.callback_velocity_4)
            truth_pub.publish(self.target4_truth)
        if self.target5_exists:
            truth_pos_sub_5 = rospy.Subscriber("/vrpn_client_node/"+TARGET5_NAME+"/pose",PoseStamped,self.callback_position_5)
            truth_vel_sub_5 = rospy.Subscriber("/vrpn_client_node/"+TARGET5_NAME+"/twist",TwistStamped,self.callback_velocity_5)
            truth_pub.publish(self.target5_truth)
        if self.target6_exists:
            truth_pos_sub_6 = rospy.Subscriber("/vrpn_client_node/"+TARGET6_NAME+"/pose",PoseStamped,self.callback_position_6)
            truth_vel_sub_6 = rospy.Subscriber("/vrpn_client_node/"+TARGET6_NAME+"/twist",TwistStamped,self.callback_velocity_6)
            truth_pub.publish(self.target6_truth)
        if self.target7_exists:
            truth_pos_sub_7 = rospy.Subscriber("/vrpn_client_node/"+TARGET7_NAME+"/pose",PoseStamped,self.callback_position_7)
            truth_vel_sub_7 = rospy.Subscriber("/vrpn_client_node/"+TARGET7_NAME+"/twist",TwistStamped,self.callback_velocity_7)
            truth_pub.publish(self.target7_truth)
        if self.target8_exists:
            truth_pos_sub_8 = rospy.Subscriber("/vrpn_client_node/"+TARGET8_NAME+"/pose",PoseStamped,self.callback_position_8)
            truth_vel_sub_8 = rospy.Subscriber("/vrpn_client_node/"+TARGET8_NAME+"/twist",TwistStamped,self.callback_velocity_8)
            truth_pub.publish(self.target8_truth)
        if self.target9_exists:
            truth_pos_sub_9 = rospy.Subscriber("/vrpn_client_node/"+TARGET9_NAME+"/pose",PoseStamped,self.callback_position_9)
            truth_vel_sub_9 = rospy.Subscriber("/vrpn_client_node/"+TARGET9_NAME+"/twist",TwistStamped,self.callback_velocity_9)
            truth_pub.publish(self.target9_truth)
        if self.target10_exists:
            truth_pos_sub_10 = rospy.Subscriber("/vrpn_client_node/"+TARGET10_NAME+"/pose",PoseStamped,self.callback_position_10)
            truth_vel_sub_10 = rospy.Subscriber("/vrpn_client_node/"+TARGET10_NAME+"/twist",TwistStamped,self.callback_velocity_10)
            truth_pub.publish(self.target10_truth)
        if self.target11_exists:
            truth_pos_sub_11 = rospy.Subscriber("/vrpn_client_node/"+TARGET11_NAME+"/pose",PoseStamped,self.callback_position_11)
            truth_vel_sub_11 = rospy.Subscriber("/vrpn_client_node/"+TARGET11_NAME+"/twist",TwistStamped,self.callback_velocity_11)
            truth_pub.publish(self.target11_truth)
        if self.target12_exists:
            truth_pos_sub_12 = rospy.Subscriber("/vrpn_client_node/"+TARGET12_NAME+"/pose",PoseStamped,self.callback_position_12)
            truth_vel_sub_12 = rospy.Subscriber("/vrpn_client_node/"+TARGET12_NAME+"/twist",TwistStamped,self.callback_velocity_12)
            truth_pub.publish(self.target12_truth)
        if self.target13_exists:
            truth_pos_sub_13 = rospy.Subscriber("/vrpn_client_node/"+TARGET13_NAME+"/pose",PoseStamped,self.callback_position_13)
            truth_vel_sub_13 = rospy.Subscriber("/vrpn_client_node/"+TARGET13_NAME+"/twist",TwistStamped,self.callback_velocity_13)
            truth_pub.publish(self.target13_truth)
        if self.target14_exists:
            truth_pos_sub_14 = rospy.Subscriber("/vrpn_client_node/"+TARGET14_NAME+"/pose",PoseStamped,self.callback_position_14)
            truth_vel_sub_14 = rospy.Subscriber("/vrpn_client_node/"+TARGET14_NAME+"/twist",TwistStamped,self.callback_velocity_14)
            truth_pub.publish(self.target14_truth)
        if self.target15_exists:
            truth_pos_sub_15 = rospy.Subscriber("/vrpn_client_node/"+TARGET15_NAME+"/pose",PoseStamped,self.callback_position_15)
            truth_vel_sub_15 = rospy.Subscriber("/vrpn_client_node/"+TARGET15_NAME+"/twist",TwistStamped,self.callback_velocity_15)
            truth_pub.publish(self.target15_truth)
        if self.target16_exists:
            truth_pos_sub_16 = rospy.Subscriber("/vrpn_client_node/"+TARGET16_NAME+"/pose",PoseStamped,self.callback_position_16)
            truth_vel_sub_16 = rospy.Subscriber("/vrpn_client_node/"+TARGET16_NAME+"/twist",TwistStamped,self.callback_velocity_16)
            truth_pub.publish(self.target16_truth)
        if self.target17_exists:
            truth_pos_sub_17 = rospy.Subscriber("/vrpn_client_node/"+TARGET17_NAME+"/pose",PoseStamped,self.callback_position_17)
            truth_vel_sub_17 = rospy.Subscriber("/vrpn_client_node/"+TARGET17_NAME+"/twist",TwistStamped,self.callback_velocity_17)
            truth_pub.publish(self.target17_truth)
        if self.target18_exists:
            truth_pos_sub_18 = rospy.Subscriber("/vrpn_client_node/"+TARGET18_NAME+"/pose",PoseStamped,self.callback_position_18)
            truth_vel_sub_18 = rospy.Subscriber("/vrpn_client_node/"+TARGET18_NAME+"/twist",TwistStamped,self.callback_velocity_18)
            truth_pub.publish(self.target18_truth)
        if self.target19_exists:
            truth_pos_sub_19 = rospy.Subscriber("/vrpn_client_node/"+TARGET19_NAME+"/pose",PoseStamped,self.callback_position_19)
            truth_vel_sub_19 = rospy.Subscriber("/vrpn_client_node/"+TARGET19_NAME+"/twist",TwistStamped,self.callback_velocity_19)
            truth_pub.publish(self.target19_truth)
        if self.target20_exists:
            truth_pos_sub_20 = rospy.Subscriber("/vrpn_client_node/"+TARGET20_NAME+"/pose",PoseStamped,self.callback_position_20)
            truth_vel_sub_20 = rospy.Subscriber("/vrpn_client_node/"+TARGET20_NAME+"/twist",TwistStamped,self.callback_velocity_20)
            truth_pub.publish(self.target20_truth)

        # truth_pub.publish(self.current_truth)
        while not rospy.is_shutdown():
            hello_str = "hello world %s" % rospy.get_time()
            print(self.has_msg)
            if all(self.has_msg):
                pub.publish(hello_str)
                self.has_msg = np.zeros(self.nAgents)
                self.k += 1
                print(f"Time step k = {self.k}")

                if self.target1_exists:
                    truth_pub.publish(self.target1_truth)
                if self.target2_exists:
                    truth_pub.publish(self.target2_truth)
                if self.target3_exists:
                    truth_pub.publish(self.target3_truth)
                if self.target4_exists:
                    truth_pub.publish(self.target4_truth)
                if self.target5_exists:
                    truth_pub.publish(self.target5_truth)
                if self.target6_exists:
                    truth_pub.publish(self.target6_truth)
                if self.target7_exists:
                    truth_pub.publish(self.target7_truth)
                if self.target8_exists:
                    truth_pub.publish(self.target8_truth)
                if self.target9_exists:
                    truth_pub.publish(self.target9_truth)
                if self.target10_exists:
                    truth_pub.publish(self.target10_truth)
                if self.target11_exists:
                    truth_pub.publish(self.target11_truth)
                if self.target12_exists:
                    truth_pub.publish(self.target12_truth)
                if self.target13_exists:
                    truth_pub.publish(self.target13_truth)
                if self.target14_exists:
                    truth_pub.publish(self.target14_truth)
                if self.target15_exists:
                    truth_pub.publish(self.target15_truth)
                if self.target16_exists:
                    truth_pub.publish(self.target16_truth)
                if self.target17_exists:
                    truth_pub.publish(self.target17_truth)
                if self.target18_exists:
                    truth_pub.publish(self.target18_truth)
                if self.target19_exists:
                    truth_pub.publish(self.target19_truth)
                if self.target20_exists:
                    truth_pub.publish(self.target20_truth)
                # truth_pub.publish(self.current_truth)
            rate.sleep()

    def callback(self, data):
        self.has_msg[data.sender] += 1

    def callback_position_1(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target1_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_2(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target2_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_3(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target3_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_4(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target4_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_5(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target5_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_6(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target6_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_7(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target7_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_8(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target8_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_9(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target9_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_10(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target10_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_11(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target11_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_12(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target12_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_13(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target13_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_14(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target14_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_15(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target15_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_16(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target16_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_17(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target17_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_18(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target18_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_19(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target19_truth.position = np.array([x_pos,y_pos,z_pos])
    def callback_position_20(self,msg):
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z
        self.target20_truth.position = np.array([x_pos,y_pos,z_pos])

    def callback_velocity_1(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target1_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_2(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target2_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_3(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target3_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_4(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target4_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_5(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target5_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_6(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target6_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_7(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target7_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_8(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target8_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_9(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target9_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_10(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target10_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_11(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target11_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_12(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target12_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_13(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target13_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_14(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target14_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_15(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target15_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_16(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target16_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_17(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target17_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_18(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target18_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_19(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target19_truth.velocity = np.array([x_vel,y_vel,z_vel])
    def callback_velocity_20(self,msg):
        x_vel = msg.twist.linear.x
        y_vel = msg.twist.linear.y
        z_vel = msg.twist.linear.z
        self.target20_truth.velocity = np.array([x_vel,y_vel,z_vel])

if __name__ == "__main__":
    try:
        rate = 10  # Hz
        nAgents = 2  # Number of agents
        targets = ["T1"]
        B = Boss(nAgents, rate, targets)
        B.talker()
    except rospy.ROSInterruptException:
        pass
