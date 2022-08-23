#!/usr/bin/env python

"""
Dynamic target estimation example
The agent has East and north bias (s) and the target has East and North position states (x)
This example uses a linear observation model
"""

from typing import Counter
import networkx as nx
import numpy as np
import scipy.linalg
import itertools
from copy import deepcopy
# import sys
# sys.path.insert(0, "c/Users/dagan/source/pyTorchLib/FG_Lib_example/fglib/fglib")
from fglib import graphs, nodes, inference, rv, utils
import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# import scipy.linalg
import scipy.io as sio
from scipy.io import savemat

import fgDDF
from fgDDF.FG_KF import *
# import time
from fgDDF.inputFile import *
# from fusion import *
from fgDDF.measurementFxn import *
from fgDDF.dynamicsFxn import *
from fgDDF.agent import *
from fgDDF.factor_utils import *
from fgDDF.fusionAlgo import *
from fgDDF.inputFile import *

import rospy
import rospkg
import os.path as path
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from fgddf_ros.msg import ChannelFilter
from fgddf_ros.msg import Results

# ROS callback functions
def callback(data, agent):
    if data.recipient == agent["agent"].id:
        agent["agent"].fusion.fusionLib[data.sender].inMsg = convertMsgToDict(data)

def boss_callback(msg):
    pass

# Converts ROS messages to the dictionary format used by fgDDF code
def convertMsgToDict(msg):
    msgs = {}
    data = {}
    data["dims"] = msg.dims
    n = msg.matrixDim
    data["infMat"] = np.array(msg.infMat).reshape((n, n))
    data["infVec"] = np.array(msg.infVec).reshape((n, 1))
    msgs[1] = data
    return msgs

def vector(*args):
    """Create n-D double numpy array"""
    veclis = []
    for l in args:
        veclis.append(l)
    vec = np.array([veclis], dtype=np.float64).T
    return vec

DEBUG = 0
dt = 0.1
nAgents = 2   # number of agents
nMC = 1
saveFlag = 1
conservativeFlag =  1 # use conservative marginalization
YData = dict()
fusionFlag = 0 # start fusing from time stp 5
#N = 1000 # Time steps
uData = dict()

# Find ROS package path
rospack = rospkg.RosPack()
p = rospack.get_path("fgddf_ros")
# matFile = sio.loadmat(path.join(p, "trackingAndLocalization_2A_1T_MC.mat"))

# matFile = sio.loadmat('testStatic.mat')
#matFile = sio.loadmat('measurements_TRO_1T_2A_Dynamic_250.mat')
# matFile = sio.loadmat('trackingAndLocalization_2A_1T.mat')
matFile = sio.loadmat('/src/fgDDF/trackingAndLocalization_2A_1T_MC.mat')

# Initialize ROS node
rospy.init_node("talker", anonymous=True)

# Poll user for agent number
print("Enter agent number: ")
ag_idx = int(input())
ag = agents[ag_idx]

# Create required ROS publishers and subscibers
sub = rospy.Subscriber("chatter", ChannelFilter, callback, (ag))
boss_sub = rospy.Subscriber("boss", String, boss_callback)
pub = rospy.Publisher("chatter", ChannelFilter, queue_size=10)
pub_results = rospy.Publisher("results", Results, queue_size=10)

# Create data variable
data = ChannelFilter()

# Wait for each target's and landmark's position to be recorded at least onece
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
if (landmark1 is not None):
    rospy.wait_for_message("vrpn_client_node/"+landmark1+"/pose",PoseStamped)
if (landmark2 is not None):
    rospy.wait_for_message("vrpn_client_node/"+landmark2+"/pose",PoseStamped)
if (landmark3 is not None):
    rospy.wait_for_message("vrpn_client_node/"+landmark3+"/pose",PoseStamped)
if (landmark4 is not None):
    rospy.wait_for_message("vrpn_client_node/"+landmark4+"/pose",PoseStamped)
if (landmark5 is not None):
    rospy.wait_for_message("vrpn_client_node/"+landmark5+"/pose",PoseStamped)
if (landmark6 is not None):
    rospy.wait_for_message("vrpn_client_node/"+landmark6+"/pose",PoseStamped)
if (landmark7 is not None):
    rospy.wait_for_message("vrpn_client_node/"+landmark7+"/pose",PoseStamped)
if (landmark8 is not None):
    rospy.wait_for_message("vrpn_client_node/"+landmark8+"/pose",PoseStamped)
if (landmark9 is not None):
    rospy.wait_for_message("vrpn_client_node/"+landmark9+"/pose",PoseStamped)
if (landmark10 is not None):
    rospy.wait_for_message("vrpn_client_node/"+landmark10+"/pose",PoseStamped)

# Start ros functions
rf = ROSFxn(agent_name)

# Read landmark positions
for ll in range(1,nLM+1):
    variables["l"+str(ll)] = rf.landmark_pos(ll-1)

#matFile = sio.loadmat('measurements_TRO_5T_4A_Dynamic_500.mat')
# uData = []
for m in range(nMC):

    print('MC:', m)
    for i in range(nAgents):
        #YData[i] = matFile['ydata']   # [i-1,m:m+1].item() - needed when ydata is a cell array
        YData[i] = matFile['yTruth'][i,m:m+1].item()
        # uData = matFile['u']
        # if uData.shape[1]==1:
        #     uData = uData*np.ones((1,N+1), dtype=np.float64)

    # instantiate filters and agents:
    for a in range(nAgents):
        print('agent:', a)

        agents[a]['filter'] = FG_EKF(variables, varSet[a], agents[a]['measData'], uData, dt)
        agents[a]['agent'] = agent(varSet[a], dynamicList, agents[a]['filter'], 'HS_CF', condVar[a], variables)
        agents[a]['agent'].set_prior(prior)


        for var in agents[a]['agent'].varSet:
            if var in agents[a]['agent'].dynamicList:
                agents[a]['filter'].x_hat[var] = agents[a]['agent'].prior[var+'_0']['x_hat']
        # Add prediction nodes to the agent's graph
        agents[a]['agent']=agents[a]['filter'].add_Prediction(agents[a]['agent'])

        # Marginalize out time 0:
        for var in agents[a]['agent'].varSet:
            if var in agents[a]['agent'].dynamicList:
                agents[a]['agent'] = agents[a]['filter'].filterPastState(agents[a]['agent'], getattr(agents[a]['agent'], var+"_Past"))

        nS = len(agents[a]['measData'])  # number of sensors for agent a
        for l in range(1, nS+1):
            nM = len(agents[a]['measData'][l]['measuredVars'])  # number of measurements the sensor takes
            for n in range(1, nM+1):
                #agents[a]['currentMeas'][l,n] = np.array([YData[a][agents[a]['measData'][l]['measInd'][n],1]]).T
                agents[a]['filter'].currentMeas[l,n] = np.array([YData[a][agents[a]['measData'][l]['measInd'][n],1]]).T  # 1st measurement



        # Inference

        for n in agents[a]['agent'].fg.get_vnodes():
                    #print('agent', a)
            varStr = str(n)
            for v in list(dynamicList):
                if varStr.find(v) != -1:
                    varStr = v
                    break


            belief = inference.sum_product(agents[a]['agent'].fg, n)
            myu = belief.mean
            myu[2] = wrapToPi(myu[2]) # wrapping angle to [-pi pi]

            agents[a]['filter'].x_hat[varStr] = myu

        # add landmark data to x_hat
        for ll in landMarks:
            agents[a]['filter'].x_hat[ll] = variables[ll]


        agents[a]['agent'] = agents[a]['filter'].add_Measurement(agents[a]['agent'])

        # plt.figure(1000+a)
        # nx.draw(agents[a]['agent'].fg, with_labels=True)
        # plt.savefig("predictionGraph_a"+str(a)+".png")

    # set initial fusion definitions:
    for a in range(nAgents):
        for n in agents[a]['neighbors']:
            agents[a]['agent'].set_fusion(agents[n]['agent'], variables)

            # Add prediction nodes to the agent's CF graph
            tmpCFgraph = agents[a]['agent'].fusion.fusionLib[agents[n]['agent'].id]
            tmpCFgraph.filter.add_Prediction(tmpCFgraph)

            for var in tmpCFgraph.varSet:
                if var in agents[a]['agent'].dynamicList:
                    tmpCFgraph.filter.filterPastState(tmpCFgraph, getattr(agents[a]['agent'], var+"_Past"))

            # plt.figure(100+a)
            # nx.draw(tmpCFgraph.fg, with_labels=True)
            # plt.savefig("predictionCF_Graph_a"+str(a)+".png")

            del tmpCFgraph
    if fusionFlag<1:   # only fuse if starting fusion from time step 1
    # Receive messages, time step 1:
        for a in range(nAgents):
            print('\n agent', a)
            for n in agents[a]['neighbors']:
                # receive message (a dictionary of factors):
                print('\n neighbor', n)
                msg = agents[n]['agent'].sendMsg(agents, n, a)
                agents[a]['agent'].fusion.fusionLib[agents[n]['agent'].id].inMsg = msg

        # Fuse incoming messages, time step 1:
        for a in range(nAgents):

            # if len(agents[a]['neighbors'])>0:
            agents[a]['agent'].fuseMsg()

            for key in agents[a]['agent'].fusion.commonVars:

                agents[a]['agent'] = mergeFactors(agents[a]['agent'],agents[a]['agent'].fusion.commonVars[key])

        # plt.figure(10000+a)
        # nx.draw(agents[a]['agent'].fg, with_labels = True)
        # plt.savefig("check_StaticTarget_a"+str(a)+".png")


    for a in range(nAgents):
        agents[a]['agent'].build_semiclique_tree()

    tmpGraph = dict()
    # Inference

    agents = inferState(agents, dynamicList, nAgents, m,  firstRunFlag = 1, saveFlag = 1)
    firstRunFlag = 0

    for a in range(nAgents):
        # tmpGraph[a] = agents[a]['agent'].add_factors_to_clique_fg()

        # agents[a]['results'][m] = dict()    # m is the MC run number

        # Calculate full covariance for NEES test
        jointInfMat=buildJointMatrix(agents[a]['agent'])
        agents[a]['results'][m]['FullCov'] = np.array(jointInfMat.factor.cov)
        agents[a]['results'][m]['FullMu']=np.array(jointInfMat.factor.mean)
        agents[a]['results'][m]['Lambda']=np.array([1])
       # jointCovMat=jointInfMat.factor.cov

    # TODO: HAVE THE ROBOT START DRIVING HERE

    k = 2
    rospy.sleep(1)
    while not rospy.is_shutdown() and (k < 200):
        # T1["u"] = uData[:,k]
        print('\nk=',k)

        # Prediction step
        ag['agent'] = ag['filter'].add_Prediction(ag['agent'])



        # Agent's CF graphs prediction step:
        for n in ag['neighbors']:

            # Add prediction nodes to the agent's CF graph
            tmpCFgraph = ag['agent'].fusion.fusionLib[n]
            tmpCFgraph.filter.add_Prediction(tmpCFgraph)

            for var in tmpCFgraph.varSet:
                if var in ag['agent'].dynamicList:
                    tmpCFgraph.filter.filterPastState(tmpCFgraph, getattr(ag['agent'], var+"_Past"))

            del tmpCFgraph

        # prepare information for measurement
        nS = len(ag['measData'])  # number of sensors for agent a
        for l in range(1, nS+1):
            nM = len(ag['measData'][l]['measuredVars'])  # number of measurements the sensor takes
            for n in range(1, nM+1):
                #agents[a]['currentMeas'][l,n] = np.array([YData[a][agents[a]['measData'][l]['measInd'][n],1]]).T
                ag['filter'].currentMeas[l,n] = np.array([YData[a][ag['measData'][l]['measInd'][n],k]]).T  # k measurement


        # Conservative filtering
        if conservativeFlag == 1 and k>=fusionFlag:
            ag['agent']=ag['filter'].consMarginalizeNodes(ag['agent'])
            # Update CF graph due to conservative filtering
            lamdaMin=ag['agent'].lamdaMin
            for n in ag['neighbors']:
                ag['agent'].fusion.updateGraph(n, lamdaMin)

        else:
            # Marginalize out time k:
            for var in ag['agent'].varSet:
                if var in ag['agent'].dynamicList:
                    ag['agent'] = ag['filter'].filterPastState(ag['agent'], getattr(ag['agent'], var+"_Past"))

            ag['agent'].lamdaMin = 1

        ag['agent'].build_semiclique_tree( )

        agents = inferState(agents, dynamicList, nAgents, m,  firstRunFlag, saveFlag = 0)

        ag['agent'] = ag['filter'].add_Measurement(ag['agent'])

        # Receive messages, time step k:
        if k>=fusionFlag:
            for n in agents[a]['neighbors']:
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

            rospy.wait_for_message("boss", String)
            ag["agent"].fuseMsg()

            for key in ag["agent"].fusion.commonVars:
                ag["agent"] = mergeFactors(ag["agent"], ag["agent"].fusion.commonVars[key])

            ag["agent"].build_semiclique_tree()
            tmpGraph = dict()

            # inference
            jointInfMat=buildJointMatrix(ag['agent'])

            ag['results'][m]['FullCov']=np.append(ag['results'][m]['FullCov'], np.array(jointInfMat.factor.cov), axis = 1)
            ag['results'][m]['FullMu']=np.append(ag['results'][m]['FullMu'], np.array(jointInfMat.factor.mean), axis=1)
            ag['results'][m]['Lambda']=np.append(ag['results'][m]['Lambda'], np.array([ag['agent'].lamdaMin]), axis=0)

        agents = inferState(agents, dynamicList, nAgents, m,  firstRunFlag, saveFlag = 1)

        del tmpGraph
        k += 1

        # Save data from current time step
        for var in ag["agent"].varSet:
            ag_tag = "X" + str(ag_idx + 1)
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

# I think this saves the data --> replace with ROS code
# for a in range(nAgents):
#     # plt.figure(100+a)
#     # nx.draw(agents[a]['agent'].fg, with_labels = True)
#     # plt.savefig("MC_StaticTarget_a"+str(a)+".png")

#     if saveFlag == 1:
#         mdict = agents[a]['results'][0]
#         for m in range(1,nMC):
#             for key in mdict:
#                 mdict[key] = np.append(mdict[key], agents[a]['results'][m][key],axis = 0)
#               # savemat(('NL_Filtering_2agents_1Target_agent_'+str(a)+'.mat'), mdict,appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='column' )
#         #savemat(('NL_Filtering_2agents_1Target_agent_'+str(a)+'_MC.mat'), mdict,appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='column' )
#         savemat(('NL_Filtering_2agents_1Target_agent_'+str(a)+'_MC.mat'), mdict,appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='column' )

#         #savemat(('CoopLocTest_MC.mat'), mdict,appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='column' )