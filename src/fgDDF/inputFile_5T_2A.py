"""
Dynamic target estimation example - 2 agent, 2 target
The agent has East and north bias (s) and the target has East and North position states (x)
This example uses a linear observation model
"""

import networkx as nx
import numpy as np
import scipy.linalg
from fgDDF.agent import agent
from fgDDF.FG_KF import FG_KF
from fgDDF.fusionAlgo import *

DEBUG = 0
dt = 0.1
nAgents = 2   # number of agents
conservativeFlag = 1
pMsg = 1


prior = dict()

variables = dict()
agents = []
varSet = dict()
condVar =  dict() # variables for conditional independence (will need to be automated later)
varList = dict()
commonVars = dict()
localVars = dict()

# define local agent variable sets in dictionaries:
varList[0] = {"T1", "T2", "T3", "S1"}
varList[1] = {"T3", "T4", "T5", "S2"}


localVars = {"S1", "S2", "T1", "T2","T4","T5"} 

varSet[0] = set(varList[0])
varSet[1] = set(varList[1])


condVar[0], condVar[1] = {'S1'}, {'S2'}

S1 = {'n' : 2}
S2 = {'n' : 2}  #{'n' : 2}

T1 = {'n' : 4}
T2 = {'n' : 4}
T3 = {'n' : 4}
T4 = {'n' : 4}
T5 = {'n' : 4}



variables["T1"], variables["T2"], variables["T3"], variables["T4"], variables["T5"] = T1, T2, T3, T4, T5

variables["S1"], variables["S2"]  = S1, S2

for var in variables:
    if var in localVars:
        variables[var]['Type'] = 'local'
    else:
        variables[var]['Type']= 'common'

dynamicList = {"T1", "T2", "T3", "T4", "T5"}
variables["dynamicList"] = dynamicList

# # Define Linear observations:
# for a in range(nAgents):
#     agents[a] = dict()
#     agents[a]['measData'] = dict()
#     agents[a]['measData'][0] = dict()
#     agents[a]['measData'][1] = dict()
#     agents[a]['measData'][2] = dict()
#     agents[a]['measData'][3] = dict()
#     agents[a]['measData'][4] = dict()
#     agents[a]['currentMeas'] = dict()
#     agents[a]['neighbors'] = dict()
#     agents[a]['results'] = dict()

# Define Linear observations:
for _ in range(nAgents):
    ag = dict()
    ag["measData"] = dict()
    for ii in range(4):
        ag["measData"][ii] = dict()
    ag["currentMeas"] = dict()
    ag["neighbors"] = dict()
    ag["results"] = dict()
    agents.append(ag)

# ag["measData"][4] = dict()

# agent 0:
agents[0]['measData'][0]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[0]['measData'][0]['R'] = np.diag([1.0, 10.0])
agents[0]['measData'][0]['invR'] = np.linalg.inv(agents[0]['measData'][0]['R'])
agents[0]['measData'][0]['measuredVars'] = ['T1','S1']   # has to be in the order of the variable vector
agents[0]["measData"][0]["measType"] = "targetPos"
# agents[1]['currentMeas'][1] = np.array([YData[1][0:2,1]]).T

agents[0]['measData'][1]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[0]['measData'][1]['R'] = np.diag([1.0, 10.0])
agents[0]['measData'][1]['invR'] = np.linalg.inv(agents[0]['measData'][1]['R'])
agents[0]['measData'][1]['measuredVars'] = ['T2','S1']   # has to be in the order of the variable vector
agents[0]["measData"][1]["measType"] = "targetPos"


agents[0]['measData'][2]['H'] = np.array([[ 1, 0], [ 0, 1]], dtype=np.float64)
agents[0]['measData'][2]['R'] = np.diag([3.0, 3.0])
agents[0]['measData'][2]['invR'] = np.linalg.inv(agents[0]['measData'][2]['R'])
agents[0]['measData'][2]['measuredVars'] = ['S1']   # has to be in the order of the variable vector
agents[0]["measData"][2]["measType"] = "agentBias"

# agents[1]['currentMeas'][3] = np.array([YData[1][4:6,1]]).T
agents[0]['measData'][3]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[0]['measData'][3]['R'] = np.diag([1.0, 10.0])
agents[0]['measData'][3]['invR'] = np.linalg.inv(agents[0]['measData'][3]['R'])
agents[0]['measData'][3]['measuredVars'] = ['T3','S1']   # has to be in the order of the variable vector
agents[0]["measData"][3]["measType"] = "targetPos"

# agents[0]['measData'][4]['H'] = np.array([[1, 0, 0, 0, 1, 0],
#                                           [0, 0,  1, 0,  0, 1]], dtype=np.float64)
# agents[0]['measData'][4]['R'] = np.diag([1.0, 10.0])
# agents[0]['measData'][4]['invR'] = np.linalg.inv(agents[0]['measData'][4]['R'])
# agents[0]['measData'][4]['measuredVars'] = ['T4','S1']   # has to be in the order of the variable vector
# agents[0]["measData"][4]["measType"] = "targetPos"


# agent 1:
agents[1]['measData'][0]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[1]['measData'][0]['R'] = np.diag([1.0, 10.0])
agents[1]['measData'][0]['invR'] = np.linalg.inv(agents[1]['measData'][0]['R'])
agents[1]['measData'][0]['measuredVars'] = ['T3','S2']   # has to be in the order of the variable vector
agents[1]["measData"][0]["measType"] = "targetPos"
# agents[1]['currentMeas'][1] = np.array([YData[1][0:2,1]]).T

agents[1]['measData'][1]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[1]['measData'][1]['R'] = np.diag([1.0, 10.0])
agents[1]['measData'][1]['invR'] = np.linalg.inv(agents[1]['measData'][1]['R'])
agents[1]['measData'][1]['measuredVars'] = ['T4','S2']   # has to be in the order of the variable vector
agents[1]["measData"][1]["measType"] = "targetPos"


agents[1]['measData'][2]['H'] = np.array([[ 1, 0], [ 0, 1]], dtype=np.float64)
agents[1]['measData'][2]['R'] = np.diag([3.0, 3.0])
agents[1]['measData'][2]['invR'] = np.linalg.inv(agents[1]['measData'][2]['R'])
agents[1]['measData'][2]['measuredVars'] = ['S2']   # has to be in the order of the variable vector
agents[1]["measData"][2]["measType"] = "agentBias"
# agents[1]['currentMeas'][3] = np.array([YData[1][4:6,1]]).T

agents[1]['measData'][3]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[1]['measData'][3]['R'] = np.diag([1.0, 10.0])
agents[1]['measData'][3]['invR'] = np.linalg.inv(agents[1]['measData'][3]['R'])
agents[1]['measData'][3]['measuredVars'] = ['T5','S2']   # has to be in the order of the variable vector
agents[1]["measData"][3]["measType"] = "targetPos"


# agents[1]['measData'][4]['H'] = np.array([[1, 0, 1, 0],
#                                           [0, 1, 0, 1]], dtype=np.float64)
# agents[1]['measData'][4]['R'] = np.diag([1.0, 10.0])
# agents[1]['measData'][4]['invR'] = np.linalg.inv(agents[1]['measData'][4]['R'])
# agents[1]['measData'][4]['measuredVars'] = ['T6','S2']   # has to be in the order of the variable vector
# agents[1]["measData"][4]["measType"] = "targetPos"



# Define neighbors:
agents[0]['neighbors'] = [1]
agents[1]['neighbors'] = [0]


# Create factor nodes for prior:
# Dynamic target initial conditions
x0d = np.array([[0], [0], [0], [0]])
X0d = np.diag([100.0, 100.0, 100.0, 100.0])

# Static target initial conditions
x0s = np.array([[0], [0]])
X0s = np.diag([100.0, 100.0])

s0 = np.array([[5], [5]])
S0 = np.diag([10.0, 10.0])

# prior['T1'] = prior['T6'] \
#     = {'infMat': np.linalg.inv(X0s), 'infVec': np.dot(np.linalg.inv(X0s), x0s), 'dim': X0s.shape[0]  }
prior['T1_0'] = prior['T2_0'] = prior['T3_0'] = prior['T4_0'] = prior['T5_0'] \
    = {'infMat': np.linalg.inv(X0d), 'infVec': np.dot(np.linalg.inv(X0d), x0d), 'dim': X0d.shape[0]  }
prior['S1'] = prior['S2']   \
    = {'infMat': np.linalg.inv(S0), 'infVec': np.dot(np.linalg.inv(S0), s0), 'dim': S0.shape[0]  }


# Dynamic definitions
variables["T1"]["Q"] = np.diag([0.08, 0.08, 0.08, 0.08])
variables["T1"]["F"] = np.array([[ 1, dt, 0, 0], [ 0, 1,0 ,0 ], [ 0, 0, 1, dt], [ 0, 0, 0, 1]] ,  dtype=np.float64)
variables["T1"]["G"] = np.array([[ 0.5*dt**2, 0], [dt,0 ], [ 0, 0.5*dt**2], [ 0, dt]] ,  dtype=np.float64)

variables["T2"]["Q"] = np.diag([0.08, 0.08, 0.08, 0.08])
variables["T2"]["F"] = np.array([[ 1, dt, 0, 0], [ 0, 1,0 ,0 ], [ 0, 0, 1, dt], [ 0, 0, 0, 1]] ,  dtype=np.float64)
variables["T2"]["G"] = np.array([[ 0.5*dt**2, 0], [dt,0 ], [ 0, 0.5*dt**2], [ 0, dt]] ,  dtype=np.float64)

variables["T3"]["Q"] = np.diag([0.08, 0.08, 0.08, 0.08])
variables["T3"]["F"] = np.array([[ 1, dt, 0, 0], [ 0, 1,0 ,0 ], [ 0, 0, 1, dt], [ 0, 0, 0, 1]] ,  dtype=np.float64)
variables["T3"]["G"] = np.array([[ 0.5*dt**2, 0], [dt,0 ], [ 0, 0.5*dt**2], [ 0, dt]] ,  dtype=np.float64)

variables["T4"]["Q"] = np.diag([0.08, 0.08, 0.08, 0.08])
variables["T4"]["F"] = np.array([[ 1, dt, 0, 0], [ 0, 1,0 ,0 ], [ 0, 0, 1, dt], [ 0, 0, 0, 1]] ,  dtype=np.float64)
variables["T4"]["G"] = np.array([[ 0.5*dt**2, 0], [dt,0 ], [ 0, 0.5*dt**2], [ 0, dt]] ,  dtype=np.float64)

variables["T5"]["Q"] = np.diag([0.08, 0.08, 0.08, 0.08])
variables["T5"]["F"] = np.array([[ 1, dt, 0, 0], [ 0, 1,0 ,0 ], [ 0, 0, 1, dt], [ 0, 0, 0, 1]] ,  dtype=np.float64)
variables["T5"]["G"] = np.array([[ 0.5*dt**2, 0], [dt,0 ], [ 0, 0.5*dt**2], [ 0, dt]] ,  dtype=np.float64)

variables["T1"]["uInd"] = [0,1]
variables["T2"]["uInd"] = [0,1]
variables["T3"]["uInd"] = [0,1]
variables["T4"]["uInd"] = [0,1]
variables["T5"]["uInd"] = [0,1]

# Agent bias
bias = np.array([3,4])

# Target names
target1 = "cohrint_tycho_bot_6"
target2 = "cohrint_tycho_bot_7"
target3 = "cohrint_tycho_bot_8"
target4 = "cohrint_tycho_bot_9"
target5 = "cohrint_tycho_bot_10"
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

# Agent names
agent1 = None
agent2 = None
agent3 = None