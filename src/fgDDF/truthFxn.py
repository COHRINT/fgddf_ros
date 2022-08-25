"""
Module for utilities.
This module contains auxiliary functions for factor graphs.
Functions:
"""

import numpy as np
import math
from fgDDF.factor_utils import *
from fgDDF.rosFxn import *


def truthRangeMeas(rf, current_agent, agent_idx, target_idx, is_target):
    """
    Computes the nonlinear range measurement functions
    """

    x1 = rf.agent_pos
    if is_target:
        x2 = rf.target_pos[target_idx]
    else:
        x2 = rf.landmark_pos[target_idx]

    noise = np.random.normal(0,current_agent["R"])

    y = math.dist(x1[0:2],x2)
    y = y + noise

    return y

def truthAzimuthMeas(rf, current_agent, agent_idx, target_idx, is_target):
    """
    Computes the nonlinear azimuth measurement function of x2 relative to x1
    x1, x2 are 2D vector of positions
    """

    x1 = rf.agent_pos
    if is_target:
        x2 = rf.target_pos[target_idx]
    else:
        x2 = rf.landmark_pos[target_idx]

    noise = np.random.normal(0,current_agent["R"])

    y = math.atan2(x2[1]-x1[1], x2[0]-x1[0])
    y = y + noise

    return wrapToPi(y)

def truthRelativeAzimuthMeas(rf, current_agent, agent_idx, target_idx, is_target):

    """
    Computes the nonlinear azimuth measurement function of x2 relative to x1
    x1, x2 are 3x1 vectors of 2D position and heading angle
    """

    # print(target_idx)
    # print(is_target)
    # print(dir(current_agent))

    x1 = rf.agent_pos
    if is_target:
        x2 = rf.target_pos[target_idx]
    else:
        # print(rf.landmark_pos)
        x2 = rf.landmark_pos[target_idx]

    noise = np.random.normal(0,current_agent["R"])

    y = math.atan2(x2[1]-x1[1], x2[0]-x1[0])-x1[2]
    y = y + noise

    return wrapToPi(y)

def truthGpsMeas(rf, mData, ind):

    """
    GPS measurement
    """


    y = mData.x_hat[mData.measurementData[ind[0]]['measuredVars'][ind[1]][0]][0:2]

    return y