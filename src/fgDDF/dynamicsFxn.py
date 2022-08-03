"""Module for utilities.
This module contains auxiliary functions for factor graphs.
Functions:
"""

import numpy as np
from fgDDF.factor_utils import *


def dubinsCarFxn(t, x1, u, L):

    """
    Computes the nonlinear range measurement function
    input:
    position vectors x1 and x2
    """

    theta = x1[2]
    v = u[0]
    phi = u[1]

    dxdt = np.zeros(x1.shape)

    dxdt[0] = v*np.cos(theta)
    dxdt[1] = v*np.sin(theta)
    dxdt[2] = v*np.tan(phi)/L

    return dxdt


def dubinsCarJacobian(x1, u, dt):

    """
    Computes the range measurement Jacobian
    """

    theta = x1[2]
    v = u[0]

    A = np.zeros((x1.shape[0], x1.shape[0]))
    A[0,2] = -v*np.sin(theta)
    A[1,2] = v*np.cos(theta)

    I = np.eye(x1.shape[0])
    F = I+dt*A

    return F

def dubinsCarUJacobian(x1, u, dt, L):
    '''
    df/du Jacobian
    '''
    theta = x1[2]
    v = u[0]
    phi = u[1]

    B = np.zeros((x1.shape[0], u.shape[0]))
    B[0,0] = np.cos(theta)
    B[1,0] = np.sin(theta)
    B[2,0] = np.tan(phi)/L
    B[2,1] = v/L/(np.cos(phi)**2)

    return dt*B

def dubinsUniFxn(t, x1, u,  L):

    """
    Computes the nonlinear range measurement function
    input:
    position vectors x1 and x2
    """

    theta = x1[2]
    v = u[0]
    w = u[1]

    dxdt = np.zeros(x1.shape)

    dxdt[0] = v*np.cos(theta)
    dxdt[1] = v*np.sin(theta)
    dxdt[2] = w

    return dxdt


def dubinsUniJacobian(x1, u, dt):

    """
    Computes the range measurement Jacobian
    """

    theta =  x1[2]
    v = u[0]

    A = np.zeros((x1.shape[0], x1.shape[0]))
    A[0,2] = -v*np.sin(theta)
    A[1,2] = v*np.cos(theta)

    I = np.eye(x1.shape[0])
    F = I+dt*A

    return F


def dubinsUniUJacobian(x1, u, dt, L):
    '''
    df/du Jacobian
    '''

    theta = x1[2]
    v = u[0]
    phi = u[1]

    B = np.zeros((x1.shape[0], u.shape[0]))
    B[0,0] = np.cos(theta)
    B[1,0] = np.sin(theta)
    B[2,1] = 1

    return dt*B