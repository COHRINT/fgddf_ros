"""
Cooperative localization example to test code
"""

import networkx as nx
import numpy as np
import scipy.linalg
from fgDDF.agent import agent
# from fgDDF.FG_KF import FG_KF
from fgDDF.fusionAlgo import *
from fgDDF.measurementFxn import *
from fgDDF.dynamicsFxn import *
from fgDDF.truthFxn import truthRelativeAzimuthMeas, truthRangeMeas, truthAzimuthMeas, truthGpsMeas
import scipy.io as sio


def vector(*args):
    "Create n-D double numpy array"
    veclis = []
    for l in args:
        veclis.append(l)
    vec = np.array([veclis], dtype=np.float64).T
    return vec

DEBUG = 0
dt = 0.1
nAgents = 2   # number of agents
nTargets = 1   # number of targets
nLM = 4      #number of landmarks

matFile = sio.loadmat("catkin_ws/src/fgddf_ros/src/fgDDF/trackingAndLocalization_2A_1T_MC.mat")

N = matFile['tVec'].shape[1]

prior = dict()

variables = dict()
agents = dict()
varSet = dict()
condVar =  dict() # variables for conditional independence (will need to be automated later)
varList = dict()
commonVars = dict()
localVars = dict()

# define local agent variable sets in dictionaries:
varList[0] = {"T1", "X1"}
varList[1] = {"T1", "X2"}
# T1 is the target
# X1, X2 are the agents
# each vector has 2D position and an angle

localVars = {"X1","X2"}

landMarks = {"l1", "l2", "l3", "l4"}

varSet[0] = set(varList[0])
varSet[1] = set(varList[1])

condVar[0], condVar[1] = {'X1'}, {'X2'}

X1 = {'n' : 3}
X2 = {'n' : 3}
T1 = {'n' : 3}

variables["T1"] = T1

variables["X1"], variables["X2"]  = X1, X2



for var in variables:
    if var in localVars:
        variables[var]['Type'] = 'local'
    else:
        variables[var]['Type']= 'common'

# # Replace this with vicon data
# for ll in range(1,nLM+1):
#     variables["l"+str(ll)] = vector(matFile['zeta_l'][ll-1].item(), matFile['eta_l'][ll-1].item())

dynamicList = {"X1", "X2", "T1"}
variables["dynamicList"] = dynamicList


# Define Linear observations:
for a in range(nAgents):
    agents[a] = dict()
    agents[a]['measData'] = dict()
    agents[a]['measData'][1] = dict()
    agents[a]['measData'][2] = dict()
    agents[a]['currentMeas'] = dict()
    agents[a]['neighbors'] = dict()
    agents[a]['results'] = dict()


# agent 1:
agents[0]['measData'][1]['measFxn'] = relativeAzimuthMeas
agents[0]['measData'][1]['measJacobian'] = relativeAzimuthJacobian
agents[0]['measData'][1]['trueMeasFxn'] = truthRelativeAzimuthMeas
agents[0]['measData'][1]['R'] = np.diag(vector(matFile['Rtrue'][0,0:1].item()[0,0]).T[0,:])
agents[0]['measData'][1]['invR'] = np.linalg.inv(agents[0]['measData'][1]['R'])
agents[0]['measData'][1]['measuredVars'] = dict()
agents[0]['measData'][1]['measInd'] = dict()
agents[0]['measData'][1]['measuredVars'] = {1: ['X1', 'l1'], 2: ['X1', 'l2'] , 3: ['X1', 'l3'], 4: ['X1', 'l4'], 5: ['X1', 'T1']  }
agents[0]['measData'][1]['measInd'] = {1: 0, 2: 2, 3: 4, 4: 6, 5: nLM*2}   # measurement indices in the measurement vector
# has to be in the order of the variable vector

agents[0]['measData'][2]['measFxn'] = rangeMeas
agents[0]['measData'][2]['measJacobian'] = relativeRangeJacobian
agents[0]['measData'][2]['trueMeasFxn'] = truthRangeMeas
agents[0]['measData'][2]['R'] = np.diag(vector(matFile['Rtrue'][0,0:1].item()[1,1]).T[0,:])
agents[0]['measData'][2]['invR'] = np.linalg.inv(agents[0]['measData'][2]['R'])
agents[0]['measData'][2]['measuredVars'] = dict()
agents[0]['measData'][2]['measInd'] = dict()
agents[0]['measData'][2]['measuredVars'] = {1: ['X1', 'l1'], 2: ['X1', 'l2'] , 3: ['X1', 'l3'], 4: ['X1', 'l4'], 5: ['X1', 'T1']  }
agents[0]['measData'][2]['measInd'] = {1: 0, 2: 2, 3: 4, 4: 6, 5: nLM*2}


# agent 2:
agents[1]['measData'][1]['measFxn'] = relativeAzimuthMeas
agents[1]['measData'][1]['measJacobian'] = relativeAzimuthJacobian
agents[1]['measData'][1]['trueMeasFxn'] = truthAzimuthMeas
agents[1]['measData'][1]['R'] = np.diag(vector(matFile['Rtrue'][0,1:2].item()[0,0]).T[0,:])
agents[1]['measData'][1]['invR'] = np.linalg.inv(agents[1]['measData'][1]['R'])
agents[1]['measData'][1]['measuredVars'] = dict()
agents[1]['measData'][1]['measInd'] = dict()
agents[1]['measData'][1]['measuredVars'] = {1: ['X2', 'l1'], 2: ['X2', 'l2'] , 3: ['X2', 'l3'], 4: ['X2', 'l4'], 5: ['X2', 'T1'] }
agents[1]['measData'][1]['measInd'] = {1: 0, 2: 2, 3: 4, 4: 6, 5: nLM*2}   # measurement indices in the measurement vector
# has to be in the order of the variable vector
# agents[1]['currentMeas'][1] = np.array([YData[1][0:2,1]]).T


agents[1]['measData'][2]['measFxn'] = rangeMeas
agents[1]['measData'][2]['measJacobian'] = relativeRangeJacobian
agents[1]['measData'][2]['trueMeasFxn'] = truthRangeMeas
agents[1]['measData'][2]['R'] = np.diag(vector(matFile['Rtrue'][0,1:2].item()[1,1]).T[0,:])
agents[1]['measData'][2]['invR'] = np.linalg.inv(agents[1]['measData'][2]['R'])
agents[1]['measData'][2]['measuredVars'] = dict()
agents[1]['measData'][2]['measInd'] = dict()
agents[1]['measData'][2]['measuredVars'] = {1: ['X2', 'l1'], 2: ['X2', 'l2'] , 3: ['X2', 'l3'], 4: ['X2', 'l4'], 5: ['X2', 'T1'] }
agents[1]['measData'][2]['measInd'] = {1: 0, 2: 2, 3: 4, 4: 6, 5: nLM*2}


# Define neighbors:
agents[0]['neighbors'] = [1]
agents[1]['neighbors'] = [0]

# Create factor nodes for prior:
t0 = vector(0, 0, -np.pi/2)
T0 = np.diag(vector(55, 55, np.pi/30).T[0,:])

x10 = vector( 10, 3, -np.pi/3)
x20 = vector( 6, 7, -np.pi/8)
X0 = np.diag(vector(100, 100, np.pi/30).T[0,:])

prior['T1_0']   \
    = {'infMat': np.linalg.inv(T0), 'infVec': np.dot(np.linalg.inv(T0), t0), 'dim': T0.shape[0], 'x_hat': t0  }
prior['X1_0']   \
    = {'infMat': np.linalg.inv(X0), 'infVec': np.dot(np.linalg.inv(X0), x10), 'dim': X0.shape[0], 'x_hat': x10  }

prior['X2_0'] \
    = {'infMat': np.linalg.inv(X0), 'infVec': np.dot(np.linalg.inv(X0), x20), 'dim': X0.shape[0], 'x_hat': x20  }
# Dynamic definitions
variables["X1"]["Q"] = np.diag(vector(0.5, 0.5, 0.09*3).T[0,:]) #/dt**2
variables["X1"]['dynFxn'] = dubinsCarFxn
variables["X1"]['dynJacobian'] = dubinsCarJacobian

variables["X2"]["Q"] = np.diag(vector(0.2, 0.2, 0.09*3).T[0,:]) #/dt**2
variables["X2"]['dynFxn'] = dubinsCarFxn
variables["X2"]['dynJacobian'] = dubinsCarJacobian

variables["T1"]["Q"] = np.diag(vector(0.3, 0.3, 0.4).T[0,:]) #/dt**2
variables["T1"]['dynFxn'] = dubinsUniFxn
variables["T1"]['dynJacobian'] = dubinsUniJacobian
#variables["T1"]["G"] = np.array([[ 0.5*dt**2, 0], [dt,0 ], [ 0, 0.5*dt**2], [ 0, dt]] ,  dtype=np.float64)

for a in range(1, nAgents+1):
    variables["X"+str(a)]["u"] = vector(matFile['vA'][a-1].item(), matFile['phi_a'][a-1].item())*np.ones((1,N), dtype=np.float64)

for t in range(1, nTargets+1):
    variables["T"+str(t)]["u"] = vector(matFile['vT'][t-1].item(), matFile['w_t'][t-1].item())*np.ones((1,N), dtype=np.float64)

# Agent name
agent_name = "cohrint_case"

# Target names
target1 = "cohrint_tycho_bot_1"
target2 = None
target3 = None
target4 = None
target5 = None
target6 = None
target7 = None
target8 = None
target9 = None
target10 = None
target11 = None
target12 = None
target13 = None
target14 = None
target15 = None
target16 = None
target17 = None
target18 = None
target19 = None
target20 = None

# Landmark names
landmark1 = "LandMark_blue"
landmark2 = "LandMark_green"
landmark3 = "LandMark_pink"
landmark4 = "cohrint_tycho_bot_2"
landmark5 = None
landmark6 = None
landmark7 = None
landmark8 = None
landmark9 = None
landmark10 = None