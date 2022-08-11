"""
Module for utilities.
This module contains auxiliary functions for factor graphs.
Functions:
"""

import numpy as np
import math
from fgDDF.factor_utils import *
from fgDDF.rosFxn import *


def rangeMeas(mData, agent_idx, target_idx):
    """
    Computes the nonlinear range measurement functions
    """

    x1 = rf.agent_pos[agent_idx]
    x2 = rf.target_pos[target_idx]

    hx = math.dist(x1,x2)

    return hx


def rangeJacobian(mData, agent_idx, target_idx):
    """
    Computes the range measurement Jacobian for a state vector of 2D position
    """

    x1 = rf.agent_pos[agent_idx]
    x2 = rf.target_pos[target_idx]

    dist = math.dist(x1, x2)
    dhdx1 = (x1[0]-x2[0])/dist
    dhdx2 = (x1[1]-x2[1])/dist

    H = np.array([dhdx1, dhdx2], dtype=np.float64, ndmin=2)

    return H

def relativeRangeJacobian(mData, agent_idx, target_idx):

    """
    Computes the range measurement Jacobian for a state vector of 2D position and angle of two vehicle
    taking relative measurement from x1 to x2, as in Cooperative localization.
    returns a 6x1 H matrix
    """
    x1 = rf.agent_pos[agent_idx]
    x2 = rf.target_pos[target_idx]

    dist = math.dist(x1, x2)
    dhdx1 = (x1[0]-x2[0])/dist
    dhdx2 = (x1[1]-x2[1])/dist
    dhdx3 = 0
    dhdx4 = -dhdx1
    dhdx5 = -dhdx2
    dhdx6 = 0

    H = np.array([dhdx1, dhdx2, dhdx3, dhdx4, dhdx5, dhdx6], dtype=np.float64, ndmin=2)

    return H


def azimuthMeas(mData, agent_idx, target_idx):
    """
    Computes the nonlinear azimuth measurement function of x2 relative to x1
    x1, x2 are 2D vector of positions
    """

    x1 = rf.agent_pos[agent_idx]
    x2 = rf.target_pos[target_idx]

    hx = math.atan2(x2[1]-x1[1], x2[0]-x1[0])

    return wrapToPi(hx)


def azimuthJacobian(mData, agent_idx, target_idx):

    """
    Computes the azimuth measurement Jacobian for a state vector of 2D position and angle of two vehicle
    taking relative measurement from x1 to x2, as in Cooperative localization.
    returns a 6x2 H matrix - encompassing 2 relative azimuth measurements - from vehicle 1 to 2 and from 2 to 1.
    """
    x1 = rf.agent_pos[agent_idx]
    x2 = rf.target_pos[target_idx]

    dist2 = (x1[0]-x2[0])**2+(x1[1]-x2[1])**2

    dhdx1 = -(x1[1]-x2[1])/dist2
    dhdx2 = -(x1[0]-x2[0])/dist2

    H = np.array([dhdx1, dhdx2], dtype=np.float64, ndmin=2)

    return H

def relativeAzimuthMeas(mData, agent_idx, target_idx):

    """
    Computes the nonlinear azimuth measurement function of x2 relative to x1
    x1, x2 are 3x1 vectors of 2D position and heading angle
    """
    x1 = rf.agent_pos[agent_idx]
    x2 = rf.target_pos[target_idx]

    hx = math.atan2(x2[1]-x1[1], x2[0]-x1[0])-x1[2]

    return wrapToPi(hx)

def relativeAzimuthJacobian(mData, agent_idx, target_idx):

    """
    Computes the relative azimuth measurement Jacobian
    """

    x1 = rf.agent_pos[agent_idx]
    x2 = rf.target_pos[target_idx]

    dist2 = (x1[0]-x2[0])**2+(x1[1]-x2[1])**2

    dhdx1 = (x2[1]-x1[1])/dist2
    dhdx2 = (x1[0]-x2[0])/dist2
    dhdx3 = -1
    dhdx4 = -dhdx1
    dhdx5 = -dhdx2
    dhdx6 = 0

    H = np.array([dhdx1, dhdx2, dhdx3, dhdx4, dhdx5, dhdx6], dtype=np.float64, ndmin=2)

    return H

def gpsMeas(mData, ind):

    """
    GPS measurement
    """


    hx = mData.x_hat[mData.measurementData[ind[0]]['measuredVars'][ind[1]][0]][0:2]

    return hx

def gpsJacobian(mData, ind):

    """
    GPS Jacobian
    """

    #x1 = mData.x_hat[mData.measurementData[ind]['measuredVars'][0]]

   # H = np.eye(x1.shape[0]-1, dtype=np.float64)
    H = np.array([[1, 0, 0],
         [0, 1, 0]])

    return H