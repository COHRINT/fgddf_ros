"""Module for utilities.
This module contains auxiliary functions for factor graphs.
Functions:
"""
import itertools

from fglib import nodes, rv
from fglib import inference
import numpy as np
# import networkx as nx
from copy import deepcopy
from math import fabs, pi

def distributeFactor(fnode, vars, agent, splitFlag = False):
    """Break a factor into componenets
        vars is a list of the node objects
    """
    dims = fnode.factor.dim

    # sparseJointInfMatA=buildJointMatrix(agent)
    infMat = fnode.factor._W
    infVec = fnode.factor._Wm
    tmpinfMat = deepcopy(infMat)
    tmpinfVec = deepcopy(infVec)
    factorCounter = agent.factorCounter
    vnodes = []
    idx = dict()
    instances = dict()
    # From the dense factor split extract the main diagonal into unary factors:
    for v in vars:
        instances[v] = []
        idx[v] = [i for i, d in enumerate(dims) if d == v]
        for i in idx[v]:
            instances[v].append(dims[i])
        vnodes.append(dims[idx[v][0]])

        # Check if the matrix is empty:
        Mat_all_zeros = np.all(tmpinfMat[idx[v][0]:idx[v][-1]+1,idx[v][0]:idx[v][-1]+1]==0)


        if (not Mat_all_zeros):
            f_i = nodes.FNode("f_"+str(factorCounter),
                              rv.Gaussian.inf_form(tmpinfMat[idx[v][0]:idx[v][-1]+1,idx[v][0]:idx[v][-1]+1], tmpinfVec[idx[v][0]:idx[v][-1]+1,:], *instances[v]))
            agent.fg.set_node(f_i)
            agent.fg.set_edge(dims[idx[v][0]], f_i)
            factorCounter+=1

            infMat[idx[v][0]:idx[v][-1]+1,idx[v][0]:idx[v][-1]+1] = 0
            infVec[idx[v][0]:idx[v][-1]+1,:] = 0

    # If one wishes to break into binary factors, i.e. connecting only two variables instead of multi-variables - splitFlag has to be True
    if splitFlag and len(vars)>2:
        combinations =  list(itertools.combinations(idx,2))
        for i in range(0, len(combinations)):
            newMat = np.zeros((len(idx[combinations[i][0]])+len(idx[combinations[i][1]]), len(idx[combinations[i][0]])+len(idx[combinations[i][1]])))
            newVec = np.zeros((len(idx[combinations[i][0]])+len(idx[combinations[i][1]]), 1))
            newInst = []
            n = 0; # counter

            for v1 in combinations[i]:
                n += 1
                m = 0; # counter
                for v2 in combinations[i]:
                    m += 1
                    newMat[(n-1)*len(idx[v1]):n*len(idx[v1]), (m-1)*len(idx[v2]):m*len(idx[v2])] = infMat[idx[v1][0]:idx[v1][-1]+1, idx[v2][0]:idx[v2][-1]+1]
                newInst = newInst + instances[v1]

            # Build new binary factor
            f_i = nodes.FNode("f_"+str(factorCounter),
                              rv.Gaussian.inf_form(newMat, newVec, *newInst))
            agent.fg.set_node(f_i)
            agent.fg.set_edge(combinations[i][0], f_i)
            agent.fg.set_edge(combinations[i][1], f_i)
            factorCounter+=1

        agent.fg.remove_node(fnode)


    agent.factorCounter = factorCounter
    # sparseJointInfMatB=buildJointMatrix(agent)
    return agent

def mergeFactors(agent, vList):
    """
    This function combines factors connected to the same nodes
    input: agent
    vList - list of updated variables (with new factors)
    Returns updated agent with updated factor graph
    """

    fList = agent.fg.get_fnodes()
    vNeighbors = dict()
    f_dict = dict()
    for var in vList:
        # find existing factors:

        try:
            vNeighbors[var] = list(agent.fg.neighbors(agent.varList[var]))
        except:
            for v in agent.fg.get_vnodes():
                if str(v)==var:
                    vNeighbors[var] = list(agent.fg.neighbors(v))


    for f in fList:
        for key, value in vNeighbors.items():
            if f in value:
                if f not in f_dict:
                    f_dict[f] = [key]
                else:
                    f_dict[f].append(key)

    combine = {}

    # This loop organizes into a dictionary with keys being tuples of nodes and values
    # are all the factors joint to the nodes in the tuple
    for key, value in f_dict.items():
        if tuple(value) not in combine:
            combine[tuple(value)] = [key]
        else:
            combine[tuple(value)].append(key)


    for key in combine:
        sumFactor = 0
        j=0
        while sumFactor == 0:

            if len(key) == len(list(agent.fg.neighbors(combine[key][j]))):
                sumFactor = combine[key][j]
            j+=1

            if j==len(combine[key]):
                break
        sumFactor_dim_list = []
        for ii in list(sumFactor.factor._dim):
            sumFactor_dim_list.append(str(ii))
        for i in range(j, len(combine[key])):
            f2_dim_list = []
            if len(key) == len(list(agent.fg.neighbors(combine[key][i]))):
                # check that dimensions are aligned
                for ii in list(combine[key][i].factor._dim):
                    f2_dim_list.append(str(ii))

                if sumFactor_dim_list!=f2_dim_list:  #dimensions are not ordered the same
                    combine[key][i] = sortFactorDims(combine[key][i], sumFactor.factor._dim)  # orders dimensions same as in sumFactor

                sumFactor.factor = sumFactor.factor.__mul__(combine[key][i].factor)
                agent.fg.remove_node(combine[key][i])
    # print('sumFactor',sumFactor )
    return agent

def findVNode(fg, nodeName):
    for n in list(fg.get_vnodes()):
        if str(n)==nodeName:
            nodeObj = n




    return nodeObj


def buildJointMatrix(agent):
    """ This function builds the full information matrix"""

    fList = list(agent.fg.get_fnodes())



    # infMat = deepcopy(fList[0])
    #
    # infMat.factor._W = infMat.factor._W*0
    # infMat.factor._Wm = infMat.factor._Wm*0
    #
    # infMat.factor._dim=fList[0].factor._dim

    infMat =nodes.FNode("infMat",rv.Gaussian.inf_form(fList[0].factor._W*0, fList[0].factor._Wm*0, *fList[0].factor._dim))
    for f in fList:
        #tmp_f = deepcopy(f)
        tmp_f =nodes.FNode("infMat",rv.Gaussian.inf_form(f.factor._W, f.factor._Wm, *f.factor._dim))
        #tmp_f.factor._dim=f.factor._dim
        infMat.factor = infMat.factor.__mul__(tmp_f.factor)

        del tmp_f
        #rv.remove_node(tmp_f)
    return infMat

def sortFactorDims(factor, refDims):
    """ This function sorts / orders the factor dimensions to be in the same order as refDims"""

    # Create string lists to compare:
    selfDimList = []
    refDimList = []
    for d in factor.factor._dim:
        selfDimList.append(str(d))
    for d in refDims:
        refDimList.append(str(d))

    if (not selfDimList==refDimList): # check if dim order is not the same
        Tmat = np.zeros((len(selfDimList), len(selfDimList)))
        i = 0;
        ind = []
        newDims = []
        for d in refDimList:
            ind.append(selfDimList.index(d))
            selfDimList[ind[i]] = ''
            Tmat[i, ind[i]] = 1
            newDims.append(factor.factor._dim[ind[i]])
            i += 1

        factor.factor._W = np.dot(Tmat, np.dot(factor.factor._W,Tmat.T))
        factor.factor._Wm = np.dot(Tmat, factor.factor._Wm)
        factor.factor._dim = newDims

    return factor


def wrapToPi(input_angle):
    '''
    wrap to pi
    Taken from: https://mika-s.github.io/python/control-theory/kinematics/2017/12/18/transformation-to-pipi.html
    '''

    #revolutions = int((input_angle + np.sign(input_angle) * pi) / (2 * pi))
    p1 = truncated_remainder(input_angle + np.sign(input_angle) * pi, 2 * pi)
    p2 = (np.sign(np.sign(input_angle)
                  + 2 * (np.sign(fabs((truncated_remainder(input_angle + pi, 2 * pi))
                                      / (2 * pi))) - 1))) * pi

    output_angle = p1 - p2
    return output_angle
    #return input_angle
def truncated_remainder(dividend, divisor):
    '''
    Helper function forwrapToPi
    Taken from: https://mika-s.github.io/python/control-theory/kinematics/2017/12/18/transformation-to-pipi.html
    '''
    divided_number = dividend / divisor
    divided_number = \
        -int(-divided_number) if divided_number < 0 else int(divided_number)

    remainder = dividend - divisor * divided_number

    return remainder

def correctQuadrant(input_angle, y, x):
    '''
    Checks if the angle is calculated in the correct quadrant.
    If not, corrects it in the interval [-pi, pi]
    '''

    if y>=0 and x>=0:   # quadrant 1
        if 0 <= input_angle <= np.pi/2: # correct quadrant 1
            output_angle = input_angle
        elif np.pi/2 < input_angle <=np.pi: # quadrant 2
            output_angle = np.pi-input_angle
        elif -np.pi < input_angle < -np.pi/2: # quadrant 3
            output_angle = np.pi+input_angle
        else: # quadrant 4
            output_angle = -input_angle
    elif y >= 0 > x:  # quadrant 2
        if np.pi/2 < input_angle <=np.pi: # correct quadrant 2
            output_angle = input_angle
        elif 0 <= input_angle <= np.pi/2: # quadrant 1
            output_angle = np.pi-input_angle
        elif -np.pi < input_angle < -np.pi/2: # quadrant 3
            output_angle = -input_angle
        else: # quadrant 4
            output_angle = np.pi+input_angle
    elif y<0 and x<0:  # quadrant 3
        if -np.pi < input_angle < -np.pi/2: # correct quadrant 3
            output_angle = input_angle
        elif 0 <= input_angle <= np.pi/2: # quadrant 1
            output_angle = input_angle-np.pi
        elif np.pi/2 < input_angle <=np.pi: #  quadrant 2
            output_angle = -input_angle
        else:  # quadrant 4
            output_angle = -(np.pi+input_angle)
    else: # quadrant 4
        if -np.pi < input_angle < -np.pi/2: # quadrant 3
            output_angle = -(np.pi+input_angle)
        elif 0 <= input_angle <= np.pi/2: # quadrant 1
            output_angle = -input_angle
        elif np.pi/2 < input_angle <=np.pi: #  quadrant 2
            output_angle = input_angle-np.pi
        else: # correct quadrant 4
            output_angle = input_angle

    output_angle = input_angle

    return output_angle

def dis_union(G, H, factorCounter):
    ''' Union of disjoint factor graphs G and H
        This function adds all factors, variables and edges of factor graph H to factor graph G
    '''

    v_list = H.get_vnodes()
    f_list = H.get_fnodes()
    e_list = H.edges()

    node_dict = dict()


    for v in v_list:
        node_dict[str(v)] = nodes.VNode(str(v), rv.Gaussian)
        G.set_node(node_dict[str(v)])
        G.nodes[node_dict[str(v)]]['comOrLoc'] = H.nodes[v]['comOrLoc']


    for f in f_list:
        newDims = []
        for d in f.factor._dim:
            newDims.append((node_dict[str(d)]))
        node_dict[str(f)] = nodes.FNode("f_"+str(factorCounter),rv.Gaussian.inf_form(f.factor._W, f.factor._Wm, *newDims))
        G.set_node(node_dict[str(f)])
        factorCounter += 1


    for e in e_list:
         G.set_edge(node_dict[str(e[0])], node_dict[str(e[1])])




    return factorCounter

def inferState(agents, dynamicList, nAgents, m , firstRunFlag, saveFlag):
    """
    This function infers all variable marginals
    """
    tmpGraph = dict()

    for a in range(nAgents):
        if firstRunFlag:
            agents[a]['results'][m] = dict()

        tmpGraph[a] = agents[a]['agent'].add_factors_to_clique_fg()

        if agents[a]['agent'].clique_fg.graph['cliqueFlag']==0:
            # for n in agents[a]['agent'].fg.get_vnodes():
            for n in tmpGraph[a].get_vnodes():
                #print('agent', a)
                varStr = str(n)
                for v in list(dynamicList):
                    if varStr.find(v) != -1:
                        varStr = v
                        break

                # belief = inference.sum_product(agents[a]['agent'].fg, n);
                belief = inference.sum_product(tmpGraph[a], n)
                myu = belief.mean
                myu[2] = wrapToPi(myu[2]) # wrapping angle to [-pi pi]

                agents[a]['filter'].x_hat[varStr] = myu

                if saveFlag:
                    if firstRunFlag:
                        agents[a]['results'][m][(varStr+'_mu')] = myu
                        agents[a]['results'][m][(varStr+'_cov')] = np.array(belief.cov)
                    else:
                        agents[a]['results'][m][(varStr+'_mu')] = np.append(agents[a]['results'][m][(varStr+'_mu')], myu, axis = 1)
                        agents[a]['results'][m][(varStr+'_cov')] = np.append(agents[a]['results'][m][(varStr+'_cov')], np.array(belief.cov), axis = 1)



        else:
            for n in tmpGraph[a].get_vnodes():
                varCount = 0
                belief = inference.sum_product(tmpGraph[a], n)
                try:
                    vNames = tmpGraph[a].nodes[n]['dims']
                    for d in range(len(vNames)):
                        if d<varCount:
                            continue

                        currentDims = [i for i, d in enumerate(vNames) if d == vNames[varCount]]
                        varStr = vNames[varCount]
                        for v in list(dynamicList):
                            if varStr.find(v) != -1:
                                varStr=v
                                break
                        myu = np.array(belief.mean)[currentDims[0]:currentDims[-1]+1]
                        myu[2] = wrapToPi(myu[2])

                        agents[a]['filter'].x_hat[varStr] = myu

                        if saveFlag:
                            agents[a]['results'][m][(varStr+'_mu')] = myu
                            agents[a]['results'][m][(varStr+'_cov')] = np.array(belief.cov)[currentDims[0]:currentDims[-1]+1,currentDims[0]:currentDims[-1]+1]


                        varCount = varCount+len(currentDims)

                except:
                    varStr=str(n)
                    for v in list(dynamicList):
                        if varStr.find(v) != -1:
                            varStr=v
                            break
                    myu = belief.mean
                    myu[2] = wrapToPi(myu[2]) # wrapping angle to [-pi pi]

                    agents[a]['filter'].x_hat[varStr] = myu

                    if firstRunFlag:
                        agents[a]['results'][m][(varStr+'_mu')] = myu
                        agents[a]['results'][m][(varStr+'_cov')] = np.array(belief.cov)
                    else:
                        agents[a]['results'][m][(varStr+'_mu')] = np.append(agents[a]['results'][m][(varStr+'_mu')], myu, axis = 1)
                        agents[a]['results'][m][(varStr+'_cov')] = np.append(agents[a]['results'][m][(varStr+'_cov')], np.array(belief.cov), axis = 1)



    del tmpGraph


    return agents