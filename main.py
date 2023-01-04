#!/usr/bin/env python

from typing import Counter
from fglib import graphs, nodes, inference, rv, utils
import networkx as nx
import numpy as np
import random
import scipy.linalg
import scipy.io as sio
from scipy.io import savemat
import itertools
from copy import deepcopy
import time

import fgDDF

from fgDDF.agent import *
from fgDDF.factor_utils import *
from fgDDF.FG_KF import *
from fgDDF.fusionAlgo import *
from fgDDF.inputFile_5T_2A import *

import rospy
import rospkg
import os.path as path
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from fgddf_ros.msg import ChannelFilter
from fgddf_ros.msg import Results
from fgddf_ros.msg import TruthData
from fgddf_ros.msg import CurrentMeasData

# Function definitions
def get_target_pos(current_agent):
    for var in current_agent["measuredVars"]:
        if "T" in var:
            temp = list(var)
            temp = temp[1:]
            target_num = int("".join(temp))
    mu = np.array([0,0])
    noise = np.random.multivariate_normal(mu,current_agent["R"])
    measurement = target_pos[target_num-1] + bias + noise
    measurement = measurement.reshape((len(measurement),1))
    return measurement

def get_agent_bias(current_agent):
    mu = np.array([0,0])
    noise = np.random.multivariate_normal(mu,current_agent["R"])
    measurement = bias + noise
    measurement = measurement.reshape((len(measurement),1))
    return measurement

def convertMsgToDict(msg):
    msgs = {}
    data = {}
    data["dims"] = msg.dims
    n = msg.matrixDim
    data["infMat"] = np.array(msg.infMat).reshape((n, n))
    data["infVec"] = np.array(msg.infVec).reshape((n, 1))
    msgs[1] = data
    return msgs

# ROS callback functions
def callback(data, agent):
    if data.recipient == agent["agent"].id:
        agent["agent"].fusion.fusionLib[data.sender].inMsg = convertMsgToDict(data)

        receive = np.random.choice(2, 1, p=[1-pMsg, pMsg])
        if receive == 0:
            agent["agent"].fusion.fusionLib[data.sender].inMsg = None

def boss_callback(msg):
    pass

def target1_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[0] = np.array([x,y])

def target2_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[1] = np.array([x,y])

def target3_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[2] = np.array([x,y])

def target4_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[3] = np.array([x,y])

def target5_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[4] = np.array([x,y])

def target6_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[5] = np.array([x,y])

def target7_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[6] = np.array([x,y])

def target8_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[7] = np.array([x,y])

def target9_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[8] = np.array([x,y])

def target10_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[9] = np.array([x,y])

def target11_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[10] = np.array([x,y])

def target12_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[11] = np.array([x,y])

def target13_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[12] = np.array([x,y])

def target14_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[13] = np.array([x,y])

def target15_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[14] = np.array([x,y])

def target16_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[15] = np.array([x,y])

def target17_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[16] = np.array([x,y])

def target18_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[17] = np.array([x,y])

def target19_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[18] = np.array([x,y])

def target20_callback(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    target_pos[19] = np.array([x,y])

np.set_printoptions(precision=3)

rospack = rospkg.RosPack()
p = rospack.get_path("fgddf_ros")
matFile = sio.loadmat(path.join(p, "measurements_TRO_1T_2A_Dynamic_250.mat"))

class MeasData:
    def __init__(self) -> None:
        self.data = []

rospy.init_node("talker", anonymous=True)
meas_data = MeasData()
print("Enter agent number: ")
ag_idx = int(input())
ag = agents[ag_idx]

# Create array to save target positions
target_pos = np.empty([20,2])

# Create subscribers to save target positions
if (target1 is not None):
    target1_sub = rospy.Subscriber("vrpn_client_node/"+target1+"/pose",PoseStamped,target1_callback)
if (target2 is not None):
    target2_sub = rospy.Subscriber("vrpn_client_node/"+target2+"/pose",PoseStamped,target2_callback)
if (target3 is not None):
    target3_sub = rospy.Subscriber("vrpn_client_node/"+target3+"/pose",PoseStamped,target3_callback)
if (target4 is not None):
    target4_sub = rospy.Subscriber("vrpn_client_node/"+target4+"/pose",PoseStamped,target4_callback)
if (target5 is not None):
    target1_sub = rospy.Subscriber("vrpn_client_node/"+target5+"/pose",PoseStamped,target5_callback)
if (target6 is not None):
    target6_sub = rospy.Subscriber("vrpn_client_node/"+target6+"/pose",PoseStamped,target6_callback)
if (target7 is not None):
    target7_sub = rospy.Subscriber("vrpn_client_node/"+target7+"/pose",PoseStamped,target7_callback)
if (target8 is not None):
    target8_sub = rospy.Subscriber("vrpn_client_node/"+target8+"/pose",PoseStamped,target8_callback)
if (target9 is not None):
    target9_sub = rospy.Subscriber("vrpn_client_node/"+target9+"/pose",PoseStamped,target9_callback)
if (target10 is not None):
    target10_sub = rospy.Subscriber("vrpn_client_node/"+target10+"/pose",PoseStamped,target10_callback)
if (target11 is not None):
    target11_sub = rospy.Subscriber("vrpn_client_node/"+target11+"/pose",PoseStamped,target11_callback)
if (target12 is not None):
    target12_sub = rospy.Subscriber("vrpn_client_node/"+target12+"/pose",PoseStamped,target12_callback)
if (target13 is not None):
    target13_sub = rospy.Subscriber("vrpn_client_node/"+target13+"/pose",PoseStamped,target13_callback)
if (target14 is not None):
    target14_sub = rospy.Subscriber("vrpn_client_node/"+target14+"/pose",PoseStamped,target14_callback)
if (target15 is not None):
    target15_sub = rospy.Subscriber("vrpn_client_node/"+target15+"/pose",PoseStamped,target15_callback)
if (target16 is not None):
    target16_sub = rospy.Subscriber("vrpn_client_node/"+target16+"/pose",PoseStamped,target16_callback)
if (target17 is not None):
    target17_sub = rospy.Subscriber("vrpn_client_node/"+target17+"/pose",PoseStamped,target17_callback)
if (target18 is not None):
    target18_sub = rospy.Subscriber("vrpn_client_node/"+target18+"/pose",PoseStamped,target18_callback)
if (target19 is not None):
    target19_sub = rospy.Subscriber("vrpn_client_node/"+target19+"/pose",PoseStamped,target19_callback)
if (target20 is not None):
    target20_sub = rospy.Subscriber("vrpn_client_node/"+target20+"/pose",PoseStamped,target20_callback)

sub = rospy.Subscriber("chatter", ChannelFilter, callback, (ag))
boss_sub = rospy.Subscriber("boss", String, boss_callback)
pub = rospy.Publisher("chatter", ChannelFilter, queue_size=10)
data = ChannelFilter()

for i in range(nAgents):
    # YData[i] = matFile["yTruth"][i, 0:1].item()
    uData = matFile["u"]

# Wait for each target's position to be recorded at least onece
if (target1 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target1+"/pose",PoseStamped)
if (target2 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target2+"/pose",PoseStamped)
if (target3 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target3+"/pose",PoseStamped)
if (target4 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target4+"/pose",PoseStamped)
if (target5 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target5+"/pose",PoseStamped)
if (target6 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target6+"/pose",PoseStamped)
if (target7 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target7+"/pose",PoseStamped)
if (target8 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target8+"/pose",PoseStamped)
if (target9 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target9+"/pose",PoseStamped)
if (target10 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target10+"/pose",PoseStamped)
if (target11 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target11+"/pose",PoseStamped)
if (target12 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target12+"/pose",PoseStamped)
if (target13 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target13+"/pose",PoseStamped)
if (target14 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target14+"/pose",PoseStamped)
if (target15 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target15+"/pose",PoseStamped)
if (target16 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target16+"/pose",PoseStamped)
if (target17 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target17+"/pose",PoseStamped)
if (target18 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target18+"/pose",PoseStamped)
if (target19 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target19+"/pose",PoseStamped)
if (target20 is not None):
    rospy.wait_for_message("vrpn_client_node/"+target20+"/pose",PoseStamped)

# instanciate filters and agents:
for i, a in enumerate(agents):
    print("Initializing agent:", i)
    # a = agents[i]
    a["filter"] = FG_KF(variables, varSet[i], a["measData"], uData)
    a["agent"] = agent(
        varSet[i], dynamicList, a["filter"], "HS_CF", i, condVar[i], variables
    )
    a["agent"].set_prior(prior)

    # Add prediction nodes to the agent's graph
    a["agent"] = a["filter"].add_Prediction(a["agent"])

    # Marginalize out time 0:
    for var in a["agent"].varSet:
        if var in a["agent"].dynamicList:
            a["agent"] = a["filter"].filterPastState(
                a["agent"], getattr(a["agent"], var + "_Past")
            )

    # rospy.wait_for_message(meas_topic, PoseWithCovarianceStamped)

    nM = len(a["measData"])  # number of measurements
    for l in range(nM):
        # p = a["measData"][l]["H"].shape[0]  # number of vector elements
        # print(a)
        if (a["measData"][l]["measType"] == "targetPos"):
            a["currentMeas"][l] = get_target_pos(a["measData"][l])
        elif (a["measData"][l]["measType"] == "agentBias"):
            a["currentMeas"][l] = get_agent_bias(a["measData"][l])

        # a["currentMeas"][l] = meas_data.data[l].reshape((len(meas_data.data[l]),1))
        

    a["agent"] = a["filter"].add_Measurement(a["agent"], a["currentMeas"])

# set initial fusion defenitions:
for i, a in enumerate(agents):
    # a = agents[i]
    for n in a["neighbors"]:
        a["agent"].set_fusion(agents[n]["agent"], variables)
        # Add prediction nodes to the agent's CF graph
        tmpCFgraph = a["agent"].fusion.fusionLib[n]
        tmpCFgraph.filter.add_Prediction(tmpCFgraph)

        for var in tmpCFgraph.varSet:
            if var in a["agent"].dynamicList:
                tmpCFgraph.filter.filterPastState(
                    tmpCFgraph, getattr(a["agent"], var + "_Past")
                )
        del tmpCFgraph


# Recive messages, time step 1:
for i, a in enumerate(agents):
    # a = agents[i]
    print("agent", i)
    for n in a["neighbors"]:
        # recieve message (a dictionary of factors):
        print("neighbor", n)
        msg = agents[n]["agent"].sendMsg(agents, n, i)
        a["agent"].fusion.fusionLib[n].inMsg = msg

        outMsg = agents[i]["agent"].sendMsg(agents, i, n)
        agents[i]["agent"].fusion.fusionLib[n].outMsg = outMsg

        receive = np.random.choice(2, 1, p=[1-pMsg, pMsg])
        if receive == 0:
            a["agent"].fusion.fusionLib[n].inMsg = None

# Fuse incoming messages, time step 1:
for a in agents:
    # a = agents[a]
    # if len(agents[a]['neighbors'])>0:
    a["agent"].fuseMsg()
    for key in a["agent"].fusion.commonVars:
        a["agent"] = mergeFactors(a["agent"], a["agent"].fusion.commonVars[key])
        # TODO check if mergeFactors works with dynamic variables

for a in agents:
    # a = agents[a]
    a["agent"].build_semiclique_tree()


tmpGraph = dict()
# Inference
for i, a in enumerate(agents):
    # a = agents[i]
    tmpGraph[i] = a["agent"].add_factors_to_clique_fg()

    a["results"][0] = dict()  # m is the MC run number

    # Calculate full covariance for NEES test
    jointInfMat = buildJointMatrix(a["agent"])
    # jointCovMat=jointInfMat.factor.cov

    print("in main; time step 1")
    print(a["agent"])
    print(a["agent"].clique_fg.graph["cliqueFlag"])
    if a["agent"].clique_fg.graph["cliqueFlag"] == 0:
        # for n in agents[a]['agent'].fg.get_vnodes():
        for n in tmpGraph[i].get_vnodes():
            varStr = str(n)
            for v in list(dynamicList):
                if varStr.find(v) != -1:
                    varStr = v
                    break

            # belief = inference.sum_product(agents[a]['agent'].fg, n);
            belief = inference.sum_product(tmpGraph[i], n)
            a["results"][0][(varStr + "_mu")] = np.array(belief.mean)
            a["results"][0][(varStr + "_cov")] = np.array(belief.cov)

        a["results"][0]["FullCov"] = np.array(jointInfMat.factor.cov)
        a["results"][0]["FullMu"] = np.array(jointInfMat.factor.mean)
        # agents[a]['results'][m]['Lamda']=np.array(int(1))
    else:
        print("in else")
        for n in tmpGraph[i].get_vnodes():
            varCount = 0
            # print(tmpGraph[i].nodes[n]["dims"])
            belief = inference.sum_product(tmpGraph[i], n)
            try:
                vNames = tmpGraph[i].nodes[n]["dims"]
                for d in range(len(vNames)):
                    if d < varCount:
                        continue

                    currentDims = [
                        i for i, d in enumerate(vNames) if d == vNames[varCount]
                    ]
                    varStr = vNames[varCount]
                    for v in list(dynamicList):
                        if varStr.find(v) != -1:
                            varStr = v
                            break

                    a["results"][0][(varStr + "_mu")] = np.array(belief.mean)[
                        currentDims[0] : currentDims[-1] + 1
                    ]
                    a["results"][0][(varStr + "_cov")] = np.array(belief.cov)[
                        currentDims[0] : currentDims[-1] + 1,
                        currentDims[0] : currentDims[-1] + 1,
                    ]

                    varCount = varCount + len(currentDims)

            except:
                varStr = str(n)
                for v in list(dynamicList):
                    if varStr.find(v) != -1:
                        varStr = v
                        break
                a["results"][0][(varStr + "_mu")] = np.array(belief.mean)
                a["results"][0][(varStr + "_cov")] = np.array(belief.cov)

        a["results"][0]["FullCov"] = np.array(jointInfMat.factor.cov)
        a["results"][0]["FullMu"] = np.array(jointInfMat.factor.mean)
del tmpGraph

pub_results = rospy.Publisher("results", Results, queue_size=10)
pub_current_meas_data = rospy.Publisher("current_meas_data", CurrentMeasData, queue_size=10)

k = 2
rospy.sleep(1)
while not rospy.is_shutdown() and (k < 200):
    # Prediction step
    ag["agent"] = ag["filter"].add_Prediction(ag["agent"])

    # Agent's CF graphs prediction step:
    for n in ag["neighbors"]:
        # Add prediction nodes to the agent's CF graph
        tmpCFgraph = ag["agent"].fusion.fusionLib[n]
        tmpCFgraph.filter.add_Prediction(tmpCFgraph)

        for var in tmpCFgraph.varSet:
            if var in ag["agent"].dynamicList:
                tmpCFgraph.filter.filterPastState(
                    tmpCFgraph, getattr(ag["agent"], var + "_Past")
                )

        del tmpCFgraph

    # Conservative filtering
    if conservativeFlag == 1:
        ag["agent"] = ag["filter"].consMarginalizeNodes(ag["agent"])
        # Update CF graph due to conservative filtering
        lamdaMin = ag["agent"].lamdaMin
        for n in ag["neighbors"]:
            ag["agent"].fusion.updateGraph(n, lamdaMin)
    else:
        # Marginalize out time k:
        for var in ag["agent"].varSet:
            if var in ag["agent"].dynamicList:
                ag["agent"] = ag["filter"].filterPastState(
                    ag["agent"],
                    getattr(ag["agent"], var + "_Past"),
                )

    # Measurement update step:
    nM = len(ag["measData"])  # number of measurements
    for l in range(nM):
        if (ag["measData"][l]["measType"] == "targetPos"):
            ag["currentMeas"][l] = get_target_pos(ag["measData"][l])
        elif (ag["measData"][l]["measType"] == "agentBias"):
            ag["currentMeas"][l] = get_agent_bias(ag["measData"][l])

    ag["agent"] = ag["filter"].add_Measurement(ag["agent"], ag["currentMeas"])

    # Recive messages, time step k:
    # TODO: This requires sending data between agents
    for n in ag["neighbors"]:
        # Send message (a dictionary of factors):
        msgs = ag["agent"].sendMsg(agents, ag["agent"].id, n)
        for msg in msgs.values():
            data.sender = ag["agent"].id
            print(f'Sending from {data.sender}')
            data.recipient = n
            data.dims = msg["dims"]
            data.matrixDim = len(data.dims)
            data.infMat = msg["infMat"].flatten()
            data.infVec = msg["infVec"]
            pub.publish(data)

        outMsg = ag["agent"].sendMsg(agents, ag_idx, n)
        ag["agent"].fusion.fusionLib[n].outMsg = outMsg

    rospy.wait_for_message("boss", String)  # Wait for go ahead

    # for n in ag["neighbors"]:
    #     receive = np.random.choice(2, 1, p=[1-pMsg, pMsg])
    #     if receive == 0:
    #         ag["agent"].fusion.fusionLib[n].inMsg = None

    ag["agent"].fuseMsg()

    for key in ag["agent"].fusion.commonVars:
        ag["agent"] = mergeFactors(ag["agent"], ag["agent"].fusion.commonVars[key])

    ag["agent"].build_semiclique_tree()
    tmpGraph = dict()
    # inference
    jointInfMat = buildJointMatrix(ag["agent"])
    tmpGraph[ag["agent"].id] = ag["agent"].add_factors_to_clique_fg()

    print("in main; time step " + str(k))
    print(a["agent"])
    print(a["agent"].clique_fg.graph["cliqueFlag"])
    if ag["agent"].clique_fg.graph["cliqueFlag"] == 0:
        for n in tmpGraph[ag["agent"].id].get_vnodes():
            varStr = str(n)
            for v in list(dynamicList):
                if varStr.find(v) != -1:
                    varStr = v
                    break

            belief = inference.sum_product(tmpGraph[ag["agent"].id], n)
            ag["results"][0][(varStr + "_mu")] = np.array(belief.mean)
            ag["results"][0][(varStr + "_cov")] = np.array(belief.cov)

        ag["results"][0]["FullCov"] = np.array(jointInfMat.factor.cov)
        ag["results"][0]["FullMu"] = np.array(jointInfMat.factor.mean)

    else:
        print("in else")
        for n in tmpGraph[ag["agent"].id].get_vnodes():
            varCount = 0
            belief = inference.sum_product(tmpGraph[ag["agent"].id], n)
            try:
                vNames = tmpGraph[ag["agent"].id].nodes[n]["dims"]
                for d in range(len(vNames)):
                    if d < varCount:
                        continue
                    currentDims = [
                        i for i, d in enumerate(vNames) if d == vNames[varCount]
                    ]
                    varStr = vNames[varCount]
                    for v in list(dynamicList):
                        if varStr.find(v) != -1:
                            varStr = v
                            break
                        print("\nvarStr:")
                        print(varStr)
                    ag["results"][0][(varStr + "_mu")] = np.array(belief.mean)[currentDims[0] : currentDims[-1] + 1]
                    ag["results"][0][(varStr + "_cov")] = np.array(belief.cov)[
                            currentDims[0] : currentDims[-1] + 1,
                            currentDims[0] : currentDims[-1] + 1,
                        ]
                    varCount = varCount + len(currentDims)
            except:
                varStr = str(n)
                for v in list(dynamicList):
                    if varStr.find(v) != -1:
                        varStr = v
                        break
                print("\nvarStr:")
                print(varStr)
                ag["results"][0][(varStr + "_mu")] = np.array(belief.mean)
                ag["results"][0][(varStr + "_cov")] = np.array(belief.cov)

            

        ag["results"][0]["FullCov"] = np.array(jointInfMat.factor.cov)
        ag["results"][0]["FullMu"] = np.array(jointInfMat.factor.mean)
    del tmpGraph
    k += 1

    for md in range(len(ag["measData"])):
        current_meas_data = CurrentMeasData()
        current_meas_data.TimeStep = k-1
        current_meas_data.Agent = "S" + str(ag_idx + 1)
        current_meas_data.MeasuredVars = ag["measData"][md]["measuredVars"]
        current_meas_data.Data = ag["currentMeas"][md]

        pub_current_meas_data.publish(current_meas_data)

    for var in ag["agent"].varSet:
        ag_tag = "S" + str(ag_idx + 1)
        if var != ag_tag:
            agent_results = Results()
            agent_results.TimeStep = k-1
            agent_results.Agent = ag_tag
            agent_results.Bias = bias
            agent_results.Target = var
            agent_results.LambdaMin = ag["agent"].lamdaMin
            agent_results.FullMuDim = np.array(ag["results"][0]["FullMu"].shape)
            agent_results.FullMu = ag["results"][0]["FullMu"].flatten()
            agent_results.FullCovDim = np.array(ag["results"][0]["FullCov"].shape)
            agent_results.FullCov = ag["results"][0]["FullCov"].flatten()
            agent_results.TMuDim = np.array(ag["results"][0][var + "_mu"].shape)
            agent_results.TMu = ag["results"][0][var + "_mu"].flatten()
            agent_results.TCovDim = np.array(ag["results"][0][var + "_cov"].shape)
            agent_results.TCov = ag["results"][0][var +"_cov"].flatten()
            agent_results.SMuDim = np.array(ag["results"][0][ag_tag + "_mu"].shape)
            agent_results.SMu = ag["results"][0][ag_tag + "_mu"].flatten()
            agent_results.SCovDim = np.array(ag["results"][0][ag_tag + "_cov"].shape)
            agent_results.SCov = ag["results"][0][ag_tag + "_cov"].flatten()

            pub_results.publish(agent_results)