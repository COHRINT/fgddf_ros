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

prior = dict()

variables = dict()
agents = []
# varList = dict()
# commonVars = dict()

# define local agent variable sets in dictionaries:
localVars = {"S1", "S2", "T1"}
varSet = [set({"T1", "T2", "S1"}), set({"T2","T3","S2"})]
condVar = [{"S1"},{"S2"}]

S1 = {'n' : 2}
S2 = {'n' : 2}  #{'n' : 2}
T1 = {'n' : 4}
T2 = {'n' : 4}
T3 = {'n' : 4}

variables["T1"], variables["T2"], variables["T3"]= T1, T2, T3
variables["S1"], variables["S2"]  = S1, S2

for var in variables:
    if var in localVars:
        variables[var]['Type'] = 'local'
    else:
        variables[var]['Type']= 'common'

dynamicList = {"T1", "T2", "T3"}
variables["dynamicList"] = dynamicList

# Define Linear observations:
for _ in range(nAgents):
    ag = dict()
    ag["measData"] = dict()
    for ii in range(3):
        ag["measData"][ii] = dict()
    ag["currentMeas"] = dict()
    ag["neighbors"] = dict()
    ag["results"] = dict()
    agents.append(ag)

# agents[0]['measData'][3]=dict()

# agent 1:
agents[0]['measData'][0]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[0]['measData'][0]['R'] = np.diag([1.0, 10.0])
agents[0]['measData'][0]['invR'] = np.linalg.inv(agents[0]['measData'][0]['R'])
agents[0]['measData'][0]['measuredVars'] = ['T1','S1']   # has to be in the order of the variable vector
agents[0]["measData"][0]["measType"] = "targetPos"

# agent 1:
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

# agents[0]['measData'][3]['H'] = np.array([[1, 0, 0, 0, 1, 0],
#                                           [0, 0,  1, 0,  0, 1]], dtype=np.float64)
# agents[0]['measData'][3]['R'] = np.diag([1.0, 10.0])
# agents[0]['measData'][3]['invR'] = np.linalg.inv(agents[0]['measData'][3]['R'])
# agents[0]['measData'][3]['measuredVars'] = ['T3','S1']   # has to be in the order of the variable vector
# agents[0]["measData"][3]["measType"] = "targetPos"

# agent 2:
agents[1]['measData'][0]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[1]['measData'][0]['R'] = np.diag([3.0, 3.0])
agents[1]['measData'][0]['invR'] = np.linalg.inv(agents[1]['measData'][0]['R'])
agents[1]['measData'][0]['measuredVars'] = ['T2','S2']   # has to be in the order of the variable vector
agents[1]["measData"][0]["measType"] = "targetPos" 

# agent 2:
agents[1]['measData'][1]['H'] = np.array([[ 1, 0], [ 0, 1]], dtype=np.float64)
agents[1]['measData'][1]['R'] = np.diag([3.0, 3.0])
agents[1]['measData'][1]['invR'] = np.linalg.inv(agents[1]['measData'][1]['R'])
agents[1]['measData'][1]['measuredVars'] = ['S2']   # has to be in the order of the variable vector
agents[1]["measData"][1]["measType"] = "agentBias"

agents[1]['measData'][2]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[1]['measData'][2]['R'] = np.diag([3.0, 3.0])
agents[1]['measData'][2]['invR'] = np.linalg.inv(agents[1]['measData'][2]['R'])
agents[1]['measData'][2]['measuredVars'] = ['T3','S2']   # has to be in the order of the variable vector
agents[1]["measData"][2]["measType"] = "targetPos" 



# Define neighbors:
agents[0]['neighbors'] = [1]
agents[1]['neighbors'] = [0]


# Create factor nodes for prior:
x0 = np.array([[0], [0], [0], [0]])
X0 = np.diag([50.0, 50.0, 50.0, 50.0])


s0 = np.array([[5], [5]])
S0 = np.diag([10.0, 10.0])

prior['T1_0']  = prior['T2_0'] = prior['T3_0'] \
    = {'infMat': np.linalg.inv(X0), 'infVec': np.dot(np.linalg.inv(X0), x0), 'dim': X0.shape[0]  }
  
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

# Set [0,1] for all the targets
variables["T1"]["uInd"] = [0,1]
variables["T2"]["uInd"] = [0,1] 
variables["T3"]["uInd"] = [0,1] 

# Agent bias
bias = np.array([3,4])

# Target names
target1 = "cohrint_tycho_bot_3"
target2 = "cohrint_tycho_bot_4"
target3 = "cohrint_tycho_bot_5"
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

# Agent names
agent1 = None
agent2 = None
agent3 = None