#!/usr/bin/env python

from typing import Counter
from fglib import graphs, nodes, inference, rv, utils
import networkx as nx
import numpy as np
import scipy.linalg
import scipy.io as sio
from scipy.io import savemat
import itertools
from copy import deepcopy

import rospy
import rospkg
import os.path as path
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from fgddf_ros.msg import ChannelFilter
from fgddf_ros.msg import Results


class FG_KF(object):
    """docstring for ."""

    def __init__(self, variables, varSet, measurementData, uData):
        """
        varSet: the set of variables in agent i's interest
        variables: a dictionary containing dynamic quantities for each dynamic variable: F, G, Q, u,
        measurementData: a dictionary containing measurement definitions for agent i

        """

        self.varSet = varSet
        self.dynamicList = variables["dynamicList"]
        self.measurementData = measurementData
        self.var = dict()

        for var in varSet:
            if var in variables["dynamicList"]:  # dynamic variable

                variables[var]["Qinv"] = np.linalg.inv(variables[var]["Q"])
                variables[var]["u"] = uData[variables[var]["uInd"], :]
                setattr(self, var, variables[var])
            else:
                setattr(self, var, variables[var])

    def add_Prediction(self, agent):
        """Add linear Gaussian prediction factor and new prediction nodes
            works in canonical / information form
        Args:

        """
        for var in agent.varSet:
            if var in self.dynamicList:
                currentNode = getattr(agent, var + "_Current")
                predNum = int(currentNode[currentNode.index("_") + 1 :]) + 1
                predictedNode = currentNode[: currentNode.index("_") + 1] + str(predNum)
                pastNode = currentNode

                factorCounter = agent.factorCounter
                X = getattr(self, var)
                F = X["F"]
                G = X["G"]
                Qinv = X["Qinv"]
                n = X["n"]

                try:
                    u = np.reshape(X["u"][:, predNum - 1], (len(X["u"]), 1))
                    predVec_kp1 = np.dot(np.dot(Qinv, G), u)  # next time step
                    predVec_k = -np.dot(
                        np.dot(np.dot(F.T, Qinv), G), u
                    )  # current time step
                except KeyError:
                    predVec_kp1 = np.zeros([n, 1], dtype=np.float64)  # next time step
                    predVec_k = np.zeros([n, 1], dtype=np.float64)  # current time step

                # Matrices:
                predMat_kp1 = Qinv
                predMat_k = np.dot(np.dot(F.T, Qinv), F)
                predMat_kp1_k = np.zeros([2 * n, 2 * n], dtype=np.float64)
                predMat_kp1_k[0:n, n : 2 * n + 1] = -np.dot(Qinv, F)
                predMat_kp1_k = predMat_kp1_k + predMat_kp1_k.T

                # add predicted node:
                xkp1 = nodes.VNode(predictedNode, rv.Gaussian)  # with n states

                instances_k = []
                instances_kp1 = []
                instances_kp1_k = []

                if var in agent.varList.keys():
                    varStr = var
                else:
                    varStr = pastNode

                for i in range(n):
                    instances_k.append(agent.varList[varStr])
                    instances_kp1.append(xkp1)
                    instances_kp1_k.append(xkp1)

                for i in range(n):
                    instances_kp1_k.append((agent.varList[varStr]))

                agent.varList[varStr] = xkp1

                f_k = nodes.FNode(
                    "f_" + str(factorCounter),
                    rv.Gaussian.inf_form(predMat_k, predVec_k, *instances_k),
                )
                factorCounter = factorCounter + 1

                f_kp1_k = nodes.FNode(
                    "f_" + str(factorCounter),
                    rv.Gaussian.inf_form(
                        predMat_kp1_k,
                        np.zeros([2 * n, 1], dtype=np.float64),
                        *instances_kp1_k,
                    ),
                )
                factorCounter = factorCounter + 1

                f_kp1 = nodes.FNode(
                    "f_" + str(factorCounter),
                    rv.Gaussian.inf_form(predMat_kp1, predVec_kp1, *instances_kp1),
                )
                factorCounter = factorCounter + 1
                # Add nodes to factor graph edges
                agent.fg.set_node(xkp1)
                agent.fg.set_nodes([f_k, f_kp1_k, f_kp1])

                # Add edges to factor graph
                agent.fg.set_edge(xkp1, f_kp1)
                agent.fg.set_edge(xkp1, f_kp1_k)

                list_vnodes = agent.fg.get_vnodes()
                x = []
                for i in range(
                    len(list_vnodes)
                ):  # You can display the name of all nodes

                    if str(list_vnodes[i]) == currentNode:
                        x = list_vnodes[i]

                agent.fg.set_edge(x, f_k)
                agent.fg.set_edge(x, f_kp1_k)

                agent.fg.nodes[xkp1]["comOrLoc"] = agent.fg.nodes[x]["comOrLoc"]

                setattr(agent, var + "_Current", predictedNode)
                setattr(agent, var + "_Past", pastNode)
                agent.factorCounter = factorCounter
        return agent

    def add_Measurement(self, agent, measData):
        """Add linear Gaussian measurement factor
            works in canonical / information form
        Args:
            measData: a dictionary containing current time step measurements:  y
        """
        factorCounter = agent.factorCounter
        for key in measData:

            H = self.measurementData[key]["H"]
            invR = self.measurementData[key]["invR"]
            y = measData[key]
            measMat = np.dot(np.dot(H.T, invR), H)
            measVec = np.dot(np.dot(H.T, invR), y)
            tmpMeasMat = deepcopy(measMat)

            varList = []
            # factorList=[]
            # commonFactor=None
            varDict = dict()
            l = 0
            for var in self.measurementData[key]["measuredVars"]:
                X = getattr(self, var)
                n = X["n"]
                varDict[var] = []
                for i in range(n):
                    varList.append(agent.varList[var])
                    varDict[var].append(l)
                    l += 1
                # add unary factors
                f_i = nodes.FNode(
                    "f_" + str(factorCounter),
                    rv.Gaussian.inf_form(
                        measMat[
                            varDict[var][0] : varDict[var][-1] + 1,
                            varDict[var][0] : varDict[var][-1] + 1,
                        ],
                        measVec[varDict[var][0] : varDict[var][-1] + 1],
                        *varList[varDict[var][0] : varDict[var][-1] + 1],
                    ),
                )
                tmpMeasMat[
                    varDict[var][0] : varDict[var][-1] + 1,
                    varDict[var][0] : varDict[var][-1] + 1,
                ] = 0
                factorCounter += 1
                agent.fg.set_node(f_i)
                agent.fg.set_edge(agent.varList[var], f_i)
                # del f_i

            # add binary factor if needed:
            if len(self.measurementData[key]["measuredVars"]) > 1:
                f_i = nodes.FNode(
                    "f_" + str(factorCounter),
                    rv.Gaussian.inf_form(tmpMeasMat, measVec * 0, *varList),
                )
                factorCounter += 1

                agent.fg.set_node(f_i)

                for var in self.measurementData[key]["measuredVars"]:
                    agent.fg.set_edge(agent.varList[var], f_i)

            agent = mergeFactors(agent, self.measurementData[key]["measuredVars"])

        agent.factorCounter = factorCounter
        return agent

    def marginalizeNodes(self, agent_i, nodesToMarginalize):
        """Marginalize out variable nodes - for heterogeneous fusion algorithm
            works in canonical / information form
        Args:
            nodesToMarginalize - string list of variable nodes

        Returns: marginalized copy of the factor graph
        """
        # jointInfMat=buildJointMatrix(agent_i)
        tmpGraph = deepcopy(agent_i.fg)
        # tmpGraph.varList=agent_i.varList
        factorCounter = agent_i.factorCounter

        nodeList = tmpGraph.get_vnodes()
        nodesDict = dict()
        for n in nodeList:
            nodesDict[str(n)] = n

        for var in nodesToMarginalize:
            S = []  # Separator variables
            i = 1
            factorFlag = False
            fList = tmpGraph[nodesDict[var]]  # get all factor connected to var

            factorsToRemove = []
            # multiply / sum all factors adjacent to var
            for f in fList:
                factorFlag = True

                if i == 1:
                    f1 = f
                    i += 1
                else:

                    f1.factor = f1.factor.__mul__(f.factor)

                # Build separator
                for s in tmpGraph[f]:
                    if s is not nodesDict[var]:
                        S.append(s)

                factorsToRemove.append(f)

            for j in factorsToRemove:
                tmpGraph.remove_node(j)

            # Create new factor by marginalizing out node var
            if factorFlag:
                tmpfactor = f1.factor.marginalize(nodesDict[var])

                if len(tmpfactor.dim) > 0:
                    # add new factor node to the graph:
                    tmpFnode = nodes.FNode(
                        "f_" + str(factorCounter),
                        rv.Gaussian.inf_form(
                            tmpfactor._W, tmpfactor._Wm, *tmpfactor.dim
                        ),
                    )

                    factorCounter += 1
                    tmpGraph.set_node(tmpFnode)
                    for s in S:
                        tmpGraph.set_edge(s, tmpFnode)

            tmpGraph.remove_node(nodesDict[var])
            tmpGraph.factorCounter = factorCounter

        return tmpGraph

    def filterPastState(self, agent_i, nodesToMarginalize):
        """Marginalize out variable nodes - for heterogeneous fusion algorithm
             works in canonical / information form
             Args:
               nodesToMarginalize - string list of variable nodes

        Returns: filtered graph
        """
        tmpGraph = agent_i.fg
        factorCounter = agent_i.factorCounter

        nodeList = tmpGraph.get_vnodes()
        nodesDict = dict()
        for n in nodeList:
            nodesDict[str(n)] = n

        for var in {nodesToMarginalize}:
            S = []  # Separator variables
            i = 1
            factorFlag = False
            fList = list(tmpGraph[nodesDict[var]])  # get all factor connected to var

            factorsToRemove = []
            # multiply / sum all factors adjacent to var
            for f in fList:
                factorFlag = True

                if i == 1:
                    f1 = f
                    i += 1
                else:
                    f1.factor = f1.factor.__mul__(f.factor)

                    # Build separator
                for s in tmpGraph[f]:
                    if s is not nodesDict[var]:
                        S.append(s)

                factorsToRemove.append(f)

            for j in factorsToRemove:
                tmpGraph.remove_node(j)

                # Create new factor by marginalizing out node var
            if factorFlag:
                tmpfactor = f1.factor.marginalize(nodesDict[var])

                if len(tmpfactor.dim) > 0:
                    # add new factor node to the graph:
                    tmpFnode = nodes.FNode(
                        "f_" + str(factorCounter),
                        rv.Gaussian.inf_form(
                            tmpfactor._W, tmpfactor._Wm, *tmpfactor.dim
                        ),
                    )

                    factorCounter += 1
                    tmpGraph.set_node(tmpFnode)
                    for s in S:
                        tmpGraph.set_edge(s, tmpFnode)

            tmpGraph.remove_node(nodesDict[var])
            agent_i.factorCounter = factorCounter

        return agent_i

    def consMarginalizeNodes(self, agent_i):
        """Conservative marginalization of nodes

        This function uses filterPastState to marginalize, but prior to that marginalization it deflates the
        information matrix while keeping the mean the same.
        The new graph / information matrix represents a new structure s.t conditional independence is maintained.

        n2m - the VNode itself that is to be marginalized

        """
        # n2m = dict()
        nodesToMarginalize = []
        # factorsToRemove = []
        localVars = []
        commonVars = []

        oldTrueGraph = deepcopy(agent_i)
        tmpGraph = deepcopy(agent_i)

        list_vnodes = agent_i.fg.get_vnodes()
        strListNodes = []
        for var in list_vnodes:
            strListNodes.append(str(var))

        for var in agent_i.varSet:
            if var in agent_i.dynamicList:
                nodesToMarginalize.append(getattr(agent_i, var + "_Past"))

        agent_i = mergeFactors(agent_i, strListNodes)

        # get pre-marginalization graph structure
        # Amat = deepcopy(nx.adjacency_matrix(agent_i.fg).todense())
        Amat = nx.adjacency_matrix(agent_i.fg).todense()
        A2mat = np.dot(
            Amat, Amat
        )  # path of length 2 (b.c of factors that are between variable nodes)
        adjNodeList = list(
            agent_i.fg.adj._atlas
        )  # The order of the adjacency matrix corresponds to the order of atlas

        # step 1: detach local nodes:
        # This step approximates the distribution by the marginals

        # split into common variables and local variables
        commonDict = (
            dict()
        )  # Dictionary of dictionaries containing data to build new adjacency matrix
        lenAdj = len(adjNodeList)
        commonList = []
        commonListKeep = []  # common nodes that are not marginalized out
        for v in list_vnodes:
            if agent_i.fg.nodes[v]["comOrLoc"] == "local":
                localVars.append(str(v))
            else:
                commonVars.append(str(v))
                if str(v) in nodesToMarginalize:
                    timeInd = str(v).find("_") + 1
                    k = str(v)[timeInd:]
                    nextNode = str(v)[0:timeInd] + str(int(k) + 1)
                    m = np.zeros((1, lenAdj), dtype=int)
                    m[0, adjNodeList.index(v)] = 1
                    commonDict[str(v)] = {
                        "node": v,
                        "nextNode": nextNode,
                        "adjIndex": adjNodeList.index(v),
                        "matRow": m,
                    }
                    commonList.append(str(v))
                    commonListKeep.append(nextNode)

        agent_i.fg = self.marginalizeNodes(agent_i, commonVars)
        tmpGraph.fg = self.marginalizeNodes(tmpGraph, localVars)

        # plt.figure(100006)
        # nx.draw(agent_i.fg, with_labels=True)

        # new graph out of the marginals of the two sets
        # agent_i.fg = nx.compose(agent_i.fg, tmpGraph.fg)
        agent_i.fg = nx.union(agent_i.fg, tmpGraph.fg)
        V_list = agent_i.fg.get_vnodes()
        for v in V_list:
            v.graph = agent_i.fg

        # Step 2 - Marginalize past nodes
        agent_i = mergeFactors(agent_i, strListNodes)

        for n in nodesToMarginalize:
            agent_i = self.filterPastState(agent_i, n)
            oldTrueGraph = self.filterPastState(oldTrueGraph, n)
            strListNodes.remove(n)

        # plt.figure(10006)
        # nx.draw(agent_i.fg, with_labels=True)
        #
        # plt.figure(10007)
        # nx.draw(oldTrueGraph.fg, with_labels=True)

        # Transformation matrix
        T_mat = commonDict[commonList[0]]["matRow"]
        for i in range(1, len(commonList)):
            T_mat = np.append(T_mat, commonDict[commonList[i]]["matRow"], axis=0)

        # Build new adjacency matrix (with only relevant parts):
        newAmat = np.dot(T_mat, np.dot(A2mat, T_mat.T))

        # update varList
        list_vnodesNew = agent_i.fg.get_vnodes()
        strListNodes = []
        for var in list_vnodesNew:
            strListNodes.append(str(var))

        for key in agent_i.varList:
            index = strListNodes.index(str(agent_i.varList[key]))
            agent_i.varList[key] = list_vnodesNew[index]

        # sparseJointInfMatB=buildJointMatrix(agent_i)
        # 3. Regain conditional independence:
        # 3.1 Break "coupling" multi-factor into unary and binary factors:
        for i in range(0, len(commonListKeep)):
            # check if there are no previous connections to other nodes
            varIndex = np.where(np.asarray(newAmat[i])[0] == 0)
            if (
                len(varIndex[0]) > 0
            ):  # ==> there shouldn't be an edge b/w those variables
                v = [v for v in list_vnodesNew if str(v) == commonListKeep[i]]
                nf = list(agent_i.fg[v[0]])
                for f in nf:
                    nv = list(agent_i.fg[f])  # neighbor variables
                    if len(nv) > 1:
                        agent_i = distributeFactor(f, nv, agent_i, True)

            agent_i = mergeFactors(agent_i, strListNodes)

        # 3.2 Remove binary factor to regain conditional independence:

        for i in range(0, len(commonListKeep)):
            # check if there are no previous connections to other nodes
            varIndex = np.where(np.asarray(newAmat[i])[0] == 0)
            if (
                len(varIndex[0]) > 0
            ):  # ==> there shouldn't be an edge b/w those variables
                v = [v for v in list_vnodesNew if str(v) == commonListKeep[i]]
                nf = list(agent_i.fg[v[0]])
                for f in nf:
                    nv = list(agent_i.fg[f])

                    nv.remove(
                        v[0]
                    )  # now nv only has the other variables, without the current one.
                    for v2 in nv:  # the current variable is coupled to other variables
                        # check if should be decoupled
                        ind = commonListKeep.index(str(v2))
                        if ind in varIndex[0]:  # variables should be decoupled
                            agent_i.fg.remove_node(f)

        # 4. Deflate factors to make conservative
        TargetJointInfMat = buildJointMatrix(oldTrueGraph)
        sparseJointInfMat = buildJointMatrix(agent_i)

        sparseJointInfMat = sortFactorDims(
            sparseJointInfMat, TargetJointInfMat.factor._dim
        )

        Q = np.dot(
            np.linalg.inv(scipy.linalg.sqrtm(sparseJointInfMat.factor._W)),
            np.dot(
                TargetJointInfMat.factor._W,
                np.linalg.inv(scipy.linalg.sqrtm(sparseJointInfMat.factor._W)),
            ),
        )
        lamdaVec, lamdaMat = np.linalg.eig(Q)

        # get deflation factor:
        lamdaMin = lamdaVec.min().real

        # Deflate the sparse information matrix:
        sparseJointInfMat.factor._W = sparseJointInfMat.factor._W * lamdaMin

        # Update information vector s.t the mean is unchanged:
        sparseJointInfMat.factor._Wm = np.dot(
            sparseJointInfMat.factor._W,
            np.dot(
                np.linalg.inv(TargetJointInfMat.factor._W), TargetJointInfMat.factor._Wm
            ),
        )

        # agent_i=mergeFactors(agent_i, strListNodes)

        # plt.figure(100007)
        # nx.draw(agent_i.fg, with_labels=True)
        # plt.savefig("check_graph_a.png")
        agent_i = mergeFactors(agent_i, strListNodes)
        # plt.figure(100007)
        # nx.draw(agent_i.fg, with_labels=True)
        # Deflate factors
        list_fnodes = list(agent_i.fg.get_fnodes())

        for f in list_fnodes:
            f.factor._W = lamdaMin * f.factor._W

            if len(agent_i.fg[f]) == 1:
                dimInd = [
                    i
                    for i, d in enumerate(sparseJointInfMat.factor.dim)
                    if d in f.factor.dim
                ]
                f.factor._Wm = sparseJointInfMat.factor._Wm[dimInd]
            else:
                f.factor._Wm = f.factor._Wm * 0

        agent_i.lamdaMin = lamdaMin
        # check:
        # newSparseJointInfMat=buildJointMatrix(agent_i)
        # A = newSparseJointInfMat.factor._W-sparseJointInfMat.factor._W
        # a = newSparseJointInfMat.factor._Wm-sparseJointInfMat.factor._Wm
        # eVal, eVec = np.linalg.eig(TargetJointInfMat.factor._W-newSparseJointInfMat.factor._W)
        # zeta = np.dot(newSparseJointInfMat.factor._W, np.dot(np.linalg.inv(TargetJointInfMat.factor._W),TargetJointInfMat.factor._Wm))
        # meanCheck = newSparseJointInfMat.factor.mean - TargetJointInfMat.factor.mean

        return agent_i


class agent(object):
    """docstring for agent class.
    input:
    varSet - set of variables in agent i inference task
    priors - a dictionary containing prior definitions for each variable in varSet

    """

    def __init__(
        self,
        varSet,
        dynamicList,
        filter,
        fusionAlgorithm,
        id,
        condVar=None,
        variables=None,
    ):
        self.fg = graphs.FactorGraph()
        self.varSet = varSet
        self.factorCounter = 0
        self.filter = filter
        self.id = id
        self.dynamicList = dynamicList
        self.fusionAlgorithm = fusionAlgorithm
        self.varList = dict()
        self.fusion = globals()[fusionAlgorithm](fusionAlgorithm)
        self.condVar = condVar  # need to be automated later

        for var in varSet:
            if var in dynamicList:
                """need to set the instances to be automatic and correct"""
                x = nodes.VNode(var + "_0", rv.Gaussian.inf_form(None, None))
                self.varList[var] = x
                setattr(self, var + "_Current", var + "_0")
            else:
                """need to set the instances to be automatic and correct"""
                x = nodes.VNode(var, rv.Gaussian)
                self.varList[var] = x
                setattr(self, var + "_Current", var)

            self.fg.set_node(x)
            self.fg.nodes[x]["comOrLoc"] = variables[var]["Type"]

    def set_prior(self, prior):
        """add prior factor to the graph

        input:
            varNode - the variable node that the prior factor is defined on
            prior - a dictionary containing definitions for the prior factor,
                    currently only infMAt and infVec for a information form Gaussian pdf
                    and: inst - Instances of the class VNode representing the variables of
                    the mean vector and covariance matrix, respectively. The number
                    of the positional arguments must match the number of dimensions
                    of the Numpy arrays.

        for now the factors are defined as Gaussian in information form

        """
        self.prior = prior
        list_vnodes = self.fg.get_vnodes()
        varNode = []
        for i in range(len(list_vnodes)):
            if str(list_vnodes[i]) in prior:
                varNode = list_vnodes[i]

            instances = []
            try:
                for j in range(prior[str(list_vnodes[i])]["dim"]):
                    instances.append(varNode)

                f = nodes.FNode(
                    "f_" + str(self.factorCounter),
                    rv.Gaussian.inf_form(
                        prior[str(list_vnodes[i])]["infMat"],
                        prior[str(list_vnodes[i])]["infVec"],
                        *instances,
                    ),
                )
                self.factorCounter = self.factorCounter + 1
                self.fg.set_node(f)
                self.fg.set_edge(varNode, f)
            except:
                print("No prior defined for variable ", str(list_vnodes[i]))

    def set_fusion(self, agent_j, variables):
        self.fusion.set_channel(self, agent_j)

        if "CF" in self.fusionAlgorithm:
            commonVars = self.fusion.commonVars[agent_j.id]
            dynamicList = self.dynamicList & commonVars
            self.fusion.fusionLib[agent_j.id] = agent(
                commonVars,
                dynamicList,
                self.filter,
                self.fusionAlgorithm,
                agent_j.id,
                None,
                variables,
            )
            self.fusion.fusionLib[agent_j.id].set_prior(self.prior)

    def sendMsg(self, agents, agent_i, agent_j_id):
        """
        returns a dictionary of with keys: dims, infMat, infVec
        dims is a list of names of variables
        """
        commonVars = self.fusion.commonVars[agent_j_id]
        msg = self.fusion.prepare_msg(
            self,
            agents[agent_i]["filter"],
            commonVars,
            agent_j_id,
        )
        return msg

    def fuseMsg(self):
        self.fusion.fuse(self)

    def build_semiclique_tree(self):
        """
        This function takes a factor graph and returns an additional factor graph that contains
        the minimum cliques that will keep the graph without cycles so the sum product algorithm can
        run on it.

        cliqueFlag - if 0: no cliques, graph is identical to the agents' graph (with no loops)
                     if 1: there are cliques and new factors need to be built
        """
        self.clique_fg = graphs.FactorGraph()
        self.clique_fg.graph["cliqueFlag"] = 0
        Vars = deepcopy(self.varSet)
        cliques = dict()
        separator = set()
        commonVars = []
        i = 1
        j = 1
        # 1. reason about cliques:
        for key in self.fusion.commonVars:
            [commonVars.append(c) for c in self.fusion.commonVars[key]]
            if len(self.fusion.commonVars[key]) > 1:
                for k in cliques:
                    if cliques[k] & self.fusion.commonVars[key]:
                        tmp = list(cliques[k] & self.fusion.commonVars[key])
                        for n in tmp:
                            separator.add(n)
                        [cliques[k].remove(s) for s in separator if s in cliques[k]]

                cliques[i] = self.fusion.commonVars[key] - separator
                [Vars.remove(v) for v in cliques[i] if v in Vars]
                i += 1

        # need to check independence
        separationClique = self.condVar | separator  # union
        [Vars.remove(v) for v in separationClique if v in Vars]

        vnodes = self.fg.get_vnodes()
        # find and change the dynamic variable name in cliques:
        nodesToRemove = []
        for n in range(1, i):
            varName = []
            for var in cliques[n]:
                for v in vnodes:
                    if str(v).find(var) != -1:
                        varName.append(str(v))
                        nodesToRemove.append(v)
                        break
            cliques[n] = set(varName)

        for v in nodesToRemove:
            vnodes.remove(v)
        # find and change the dynamic variable name in Vars:
        nodesToRemove = []
        varName = []
        for var in Vars:
            for v in vnodes:
                if str(v).find(var) != -1:
                    varName.append(str(v))
                    nodesToRemove.append(v)
                    break
        Vars = set(varName)

        for v in nodesToRemove:
            vnodes.remove(v)

        # find and change the dynamic variable name in separationClique:

        varName = []
        for var in separationClique:
            for v in vnodes:
                if str(v).find(var) != -1:
                    varName.append(str(v))
                    break
        separationClique = set(varName)

        # 2. add nodes to graph
        c_counter = 1
        self.clique_fg.graph["unaryVars"] = []
        self.clique_fg.graph["Cliques"] = []
        for v in Vars:
            x = nodes.VNode(v, rv.Gaussian)
            self.clique_fg.set_node(x)
            self.clique_fg.nodes[x]["vars"] = {v}
            self.clique_fg.graph["unaryVars"].append(v)

        for key, value in cliques.items():
            self.clique_fg.graph["cliqueFlag"] = 1
            x = nodes.VNode("C" + str(c_counter), rv.Gaussian)

            self.clique_fg.set_node(x)
            self.clique_fg.nodes[x]["vars"] = value
            self.clique_fg.graph["Cliques"].append("C" + str(c_counter))
            c_counter += 1

        if len(separationClique) == 1:
            x = nodes.VNode("Sep", rv.Gaussian)
            self.clique_fg.set_node(x)
            self.clique_fg.nodes[x]["vars"] = separationClique
        elif len(separationClique) > 1:
            x = nodes.VNode("Sep", rv.Gaussian)
            self.clique_fg.set_node(x)
            self.clique_fg.nodes[x]["vars"] = separationClique
            self.clique_fg.graph["cliqueFlag"] = 1

        self.clique_fg.graph["separationClique"] = separationClique

    def add_factors_to_clique_fg(self):
        """
        This function builds and adds factors to the semiclique_tree factor graph

        """
        # if the graph has no cliques, its structure is identical to the original graph
        if self.clique_fg.graph["cliqueFlag"] == 0:
            # tmpCliqueGraph  = deepcopy(self.fg)
            return self.fg

        # else - need to build factors
        tmpCliqueGraph = deepcopy(self.clique_fg)
        tmpMainFg = deepcopy(self.fg)
        # tmpMainFg = self.fg

        factorCounter = 1
        fList = []
        cFactor = []
        # vnodes=tmpMainFg.get_vnodes()
        factorFlag = []
        s = findVNode(tmpCliqueGraph, "Sep")
        # add factors between cliques and separator
        for cName in tmpCliqueGraph.graph["Cliques"]:
            c = findVNode(tmpCliqueGraph, cName)
            i = 1
            # loop over all variables in the clique
            for n in tmpCliqueGraph.nodes[c]["vars"]:
                # find node in agents' fg:
                factorFlag = False
                # summarize all factors connected to nodes in agents' fg
                for f in list(tmpMainFg[findVNode(tmpMainFg, n)]):
                    if f not in fList:
                        factorFlag = True
                        if i == 1:
                            cFactor = f
                            i = 0
                        else:
                            cFactor.factor = cFactor.factor.__mul__(f.factor)
                        fList.append(f)

            instances = []
            instances0 = []
            tmpCliqueGraph.nodes[c]["dims"] = []
            for d in cFactor.factor.dim:
                if str(d) in tmpCliqueGraph.nodes[c]["vars"]:
                    instances.append(c)
                    instances0.append(c)
                    tmpCliqueGraph.nodes[c]["dims"].append(str(d))  # save dims:
                else:
                    instances.append(s)
            if factorFlag:
                # add factor connecting all variables in vars:
                f_i = nodes.FNode(
                    "f_" + str(factorCounter),
                    rv.Gaussian.inf_form(
                        cFactor.factor._W, cFactor.factor._Wm, *instances
                    ),
                )
                # set dims correctly:

                tmpCliqueGraph.set_node(f_i)
                factorCounter += 1

                tmpCliqueGraph.set_edge(c, f_i)
                if s in instances:
                    tmpCliqueGraph.set_edge(s, f_i)

            f_j = nodes.FNode(
                "f_100" + str(factorCounter),
                rv.Gaussian.inf_form(
                    np.zeros((len(instances0), len(instances0))),
                    np.zeros((len(instances0), 1)),
                    *instances0,
                ),
            )
            tmpCliqueGraph.set_node(f_j)
            tmpCliqueGraph.set_edge(c, f_j)

        i = 1
        sFactor = []
        # loop over all variables in the clique
        for n in tmpCliqueGraph.nodes[s]["vars"]:
            # summarize all factors connected to nodes in agents' fg
            for f in list(tmpMainFg[findVNode(tmpMainFg, n)]):
                memberFlag = 1
                for j in tmpMainFg[f]:
                    if str(j) not in tmpCliqueGraph.nodes[s]["vars"]:
                        memberFlag = 0
                        break

                if f not in fList and memberFlag == 1:
                    if i == 1:
                        sFactor = f
                        i = 0
                    else:
                        sFactor.factor = sFactor.factor.__mul__(f.factor)
                    fList.append(f)

        instances = []
        tmpCliqueGraph.nodes[s]["dims"] = []
        for d in sFactor.factor.dim:
            instances.append(s)
            tmpCliqueGraph.nodes[s]["dims"].append(str(d))  # save dims:

        # add factor connecting all variables in vars:
        f_i = nodes.FNode(
            "f_" + str(factorCounter),
            rv.Gaussian.inf_form(sFactor.factor._W, sFactor.factor._Wm, *instances),
        )

        tmpCliqueGraph.set_node(f_i)
        factorCounter += 1

        tmpCliqueGraph.set_edge(s, f_i)

        f_j = nodes.FNode(
            "f_1000" + str(factorCounter),
            rv.Gaussian.inf_form(
                np.zeros((len(instances), len(instances))),
                np.zeros((len(instances), 1)),
                *instances,
            ),
        )
        tmpCliqueGraph.set_node(f_j)
        tmpCliqueGraph.set_edge(s, f_j)

        # add factors of unary variables:
        for uName in tmpCliqueGraph.graph["unaryVars"]:
            u = findVNode(tmpCliqueGraph, uName)
            i = 1
            # loop over all variables in the clique
            for n in tmpCliqueGraph.nodes[u]["vars"]:
                factorFlag = False

                # find node in agents' fg:
                uFactor = []
                # summarize all factors connected to nodes in agents' fg
                for f in list(tmpMainFg[findVNode(tmpMainFg, n)]):

                    if f not in fList:
                        factorFlag = True
                        if i == 1:
                            uFactor = f
                            i = 0
                        else:
                            uFactor.factor = uFactor.factor.__mul__(f.factor)
                        fList.append(f)

                if factorFlag:
                    instances = []
                    instances0 = []
                    for d in uFactor.factor.dim:
                        if str(d) in tmpCliqueGraph.nodes[u]["vars"]:
                            instances.append(u)
                            instances0.append(u)
                        else:
                            instances.append(s)

                    # add factor connecting all variables in vars:
                    f_i = nodes.FNode(
                        "f_" + str(factorCounter),
                        rv.Gaussian.inf_form(
                            uFactor.factor._W, uFactor.factor._Wm, *instances
                        ),
                    )

                    tmpCliqueGraph.set_node(f_i)
                    factorCounter += 1

                    tmpCliqueGraph.set_edge(u, f_i)

                    if s in instances:
                        tmpCliqueGraph.set_edge(s, f_i)

                    f_j = nodes.FNode(
                        "f_10000" + str(factorCounter),
                        rv.Gaussian.inf_form(
                            np.zeros((len(instances0), len(instances0))),
                            np.zeros((len(instances0), 1)),
                            *instances0,
                        ),
                    )
                    tmpCliqueGraph.set_node(f_j)
                    tmpCliqueGraph.set_edge(u, f_j)

        return tmpCliqueGraph


class fusionAlgo(object):
    """docstring for fusion class.
    input:
    agent_i - communicating agent
    """

    def __init__(self, fusionAlgorithm):

        self.fusionAlgorithm = fusionAlgorithm
        self.commonVars = dict()

    def set_channel(self, agent_i, agent_j):
        self.commonVars[agent_j.id] = agent_i.varSet & agent_j.varSet
        if "CF" in self.fusionAlgorithm:
            try:
                self.fusionLib
            except:
                self.fusionLib = dict()


class HS_CF(fusionAlgo):
    """Heterogeneous State Channel Filter"""

    def __init__(self, fusionAlgorithm):
        super(HS_CF, self).__init__(fusionAlgorithm)

    def prepare_msg(self, agent_i, filter, commonVars, agent_j_id):
        """prepare a message to send to neighbor
        In general that means marginalizing common variables

        It accounts for common information by removing it using the CF

        input:
        commonVars

        """

        list_vnodes = agent_i.fg.get_vnodes()
        # find nodes to marginalize out
        strVnodes = []
        for v in list_vnodes:
            strVnodes.append(str(v))
        diff = []

        vToRemove = []
        varDict = {}
        for var in strVnodes:
            for c in commonVars:
                if var.find(c) != -1:
                    vToRemove.append(var)
                    varDict[var] = self.fusionLib[agent_j_id].varList[c]
                    break

        for r in vToRemove:
            strVnodes.remove(r)
        diff = strVnodes

        # marginalized graph:
        tmpGraph = deepcopy(agent_i)
        tmpGraph.fg = filter.marginalizeNodes(tmpGraph, diff)

        tmpGraph = mergeFactors(tmpGraph, vToRemove)
        msgGraph = tmpGraph.fg

        # CF graph between agent i and j:
        CFagent = deepcopy(self.fusionLib[agent_j_id])

        CFgraph = CFagent.fg

        CF_fcounter = self.fusionLib[agent_j_id].factorCounter

        f_dict = dict()  # dictionary for factors to send to agent j
        counter = 1
        list_fnodes_msgGraph = msgGraph.get_fnodes()
        list_fnodes_CFgraph = CFgraph.get_fnodes()

        for F1 in list_fnodes_msgGraph:
            matchFlag = 0
            f1_dims = F1.factor.dim
            f1_dim_list = []
            flag = 0
            F2_remove = []
            for i in f1_dims:
                f1_dim_list.append(str(i))

            for F2 in list_fnodes_CFgraph:  # go over factors in CF graph
                # preliminary check for dimensions
                if len(f1_dims) == len(F2.factor.dim):
                    f2_dims = F2.factor.dim
                    f2_dim_list = []
                    for i in f2_dims:
                        f2_dim_list.append(str(i))
                    if len(set(f2_dim_list) - set(f1_dim_list)) == 0:
                        if (
                            f1_dim_list != f2_dim_list
                        ):  # dimensions are not ordered the same
                            F2 = sortFactorDims(F2, F1.factor._dim)

                        if flag == 0:  # first equivalent factor
                            f_dict[counter] = dict()
                            f_dict[counter]["dims"] = f1_dim_list

                            f_dict[counter]["infMat"] = F1.factor._W - F2.factor._W
                            f_dict[counter]["infVec"] = F1.factor._Wm - F2.factor._Wm
                            flag = 1

                        else:

                            f_dict[counter]["infMat"] = (
                                f_dict[counter]["infMat"] - F2.factor._W
                            )
                            f_dict[counter]["infVec"] = (
                                f_dict[counter]["infVec"] - F2.factor._Wm
                            )

                        F2_remove.append(F2)
                        matchFlag = 1

            for r in F2_remove:
                list_fnodes_CFgraph.remove(r)
                CFgraph.remove_node(r)

            if matchFlag == 0:  # only factor over this node
                f_dict[counter] = dict()
                f_dict[counter]["dims"] = f1_dim_list
                f_dict[counter]["infMat"] = F1.factor._W
                f_dict[counter]["infVec"] = F1.factor._Wm
                counter += 1
                zeroCheck = 0
                # print(f_dict[counter]['infMat'])
            else:
                Mat_all_zeros = np.all(f_dict[counter]["infMat"] == 0)
                Vec_all_zeros = np.all(f_dict[counter]["infVec"] == 0)
                if (not Mat_all_zeros) and (not Vec_all_zeros):
                    counter += 1
                    zeroCheck = 0
                else:  # nothing to send
                    zeroCheck = 1
                    del f_dict[counter]

            # Update CF graph:
            if zeroCheck == 0:
                # find instances in CF graph
                instances = []
                for inst in f1_dim_list:
                    instances.append(findVNode(self.fusionLib[agent_j_id].fg, inst))

                f = nodes.FNode(
                    "f_" + str(CF_fcounter),
                    rv.Gaussian.inf_form(
                        f_dict[counter - 1]["infMat"],
                        f_dict[counter - 1]["infVec"],
                        *instances,
                    ),
                )
                CF_fcounter += 1
                self.fusionLib[agent_j_id].fg.set_node(f)
                CFvarList = []
                for var in f_dict[counter - 1]["dims"]:
                    if varDict[var] not in CFvarList:
                        CFvarList.append(varDict[var])

                for n in CFvarList:
                    self.fusionLib[agent_j_id].fg.set_edge(n, f)

            msgGraph.remove_node(F1)
            self.fusionLib[agent_j_id].factorCounter = CF_fcounter
        return f_dict

    def fuse(self, agent_i):
        # TODO: Change this so that f_key is removed, can be replaced by list of messages
        """Fuse all incoming messages into the agents factor graph"""
        factorCounter = agent_i.factorCounter

        for key in self.fusionLib:
            CFfactorCounter = self.fusionLib[key].factorCounter
            inMsg = self.fusionLib[key].inMsg
            # print('in fusion:\n from:',key, 'to: ',agent_i.id, inMsg)
            CFvarList = []
            if inMsg is not None:
                # go over all sent factors:
                for f_key in inMsg:
                    varList = []
                    CFvarList = []
                    instances = []
                    CFinstances = []

                    for d in inMsg[f_key]["dims"]:
                        for v in agent_i.varList:
                            if str(d).find(v) != -1:
                                var = v
                                break
                        instances.append(agent_i.varList[var])
                        CFinstances.append(self.fusionLib[key].varList[var])
                        if agent_i.varList[var] not in varList:
                            varList.append(agent_i.varList[var])
                            CFvarList.append(self.fusionLib[key].varList[var])

                    f = nodes.FNode(
                        "f_" + str(factorCounter),
                        rv.Gaussian.inf_form(
                            inMsg[f_key]["infMat"], inMsg[f_key]["infVec"], *instances
                        ),
                    )
                    f_CF = nodes.FNode(
                        "f_" + str(CFfactorCounter),
                        rv.Gaussian.inf_form(
                            inMsg[f_key]["infMat"], inMsg[f_key]["infVec"], *CFinstances
                        ),
                    )

                    factorCounter += 1
                    CFfactorCounter += 1

                    agent_i.fg.set_node(f)
                    self.fusionLib[key].fg.set_node(f_CF)
                    for n in varList:
                        agent_i.fg.set_edge(n, f)
                    for n in CFvarList:
                        self.fusionLib[key].fg.set_edge(n, f_CF)
                # Delete message after fusion
                self.fusionLib[key].inMsg = None
                self.fusionLib[key].factorCounter = CFfactorCounter

                self.fusionLib[key] = mergeFactors(self.fusionLib[key], CFvarList)

        agent_i.factorCounter = factorCounter

        return agent_i

    def updateGraph(self, agent_j_id, lamdaMin):
        """
        This function updates the CF graph to correspond to conservative marginalization - i.e. deflates by lamdaMin
        :param agent_i: Current agent holding the CF graph
        :param commonVars: variables common to both agents
        :param agent_j_id: The "other" agent, with whom the CF graph represents the common estimate
        :return:
        """

        tmpCFagent = self.fusionLib[agent_j_id]

        list_fnodes = tmpCFagent.fg.get_fnodes()
        # oldSparseJointInfMat=buildJointMatrix(tmpCFagent)
        for f in list_fnodes:
            f.factor._W = lamdaMin * f.factor._W
            f.factor._Wm = lamdaMin * f.factor._Wm

        # newSparseJointInfMat=buildJointMatrix(tmpCFagent)

        return


def distributeFactor(fnode, vars, agent, splitFlag=False):
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
        Mat_all_zeros = np.all(
            tmpinfMat[idx[v][0] : idx[v][-1] + 1, idx[v][0] : idx[v][-1] + 1] == 0
        )

        if not Mat_all_zeros:
            f_i = nodes.FNode(
                "f_" + str(factorCounter),
                rv.Gaussian.inf_form(
                    tmpinfMat[idx[v][0] : idx[v][-1] + 1, idx[v][0] : idx[v][-1] + 1],
                    tmpinfVec[idx[v][0] : idx[v][-1] + 1, :],
                    *instances[v],
                ),
            )
            agent.fg.set_node(f_i)
            agent.fg.set_edge(dims[idx[v][0]], f_i)
            factorCounter += 1

            infMat[idx[v][0] : idx[v][-1] + 1, idx[v][0] : idx[v][-1] + 1] = 0
            infVec[idx[v][0] : idx[v][-1] + 1, :] = 0

    # If one wishes to break into binary factors, i.e. connecting only two variables instead of multi-variables - splitFlag has to be True
    if splitFlag and len(vars) > 2:
        combinations = list(itertools.combinations(idx, 2))
        for i in range(0, len(combinations)):
            newMat = np.zeros(
                (
                    len(idx[combinations[i][0]]) + len(idx[combinations[i][1]]),
                    len(idx[combinations[i][0]]) + len(idx[combinations[i][1]]),
                )
            )
            newVec = np.zeros(
                (len(idx[combinations[i][0]]) + len(idx[combinations[i][1]]), 1)
            )
            newInst = []
            n = 0
            # counter

            for v1 in combinations[i]:
                n += 1
                m = 0
                # counter
                for v2 in combinations[i]:
                    m += 1
                    newMat[
                        (n - 1) * len(idx[v1]) : n * len(idx[v1]),
                        (m - 1) * len(idx[v2]) : m * len(idx[v2]),
                    ] = infMat[
                        idx[v1][0] : idx[v1][-1] + 1, idx[v2][0] : idx[v2][-1] + 1
                    ]
                newInst = newInst + instances[v1]

            # Build new binary factor
            f_i = nodes.FNode(
                "f_" + str(factorCounter),
                rv.Gaussian.inf_form(newMat, newVec, *newInst),
            )
            agent.fg.set_node(f_i)
            agent.fg.set_edge(combinations[i][0], f_i)
            agent.fg.set_edge(combinations[i][1], f_i)
            factorCounter += 1

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
                if str(v) == var:
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
        j = 0
        while sumFactor == 0:

            if len(key) == len(list(agent.fg.neighbors(combine[key][j]))):
                sumFactor = combine[key][j]
            j += 1

            if j == len(combine[key]):
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

                if (
                    sumFactor_dim_list != f2_dim_list
                ):  # dimensions are not ordered the same
                    combine[key][i] = sortFactorDims(
                        combine[key][i], sumFactor.factor._dim
                    )  # orders dimensions same as in sumFactor

                sumFactor.factor = sumFactor.factor.__mul__(combine[key][i].factor)
                agent.fg.remove_node(combine[key][i])
    # print('sumFactor',sumFactor )
    return agent


def findVNode(fg, nodeName):
    for n in list(fg.get_vnodes()):
        if str(n) == nodeName:
            nodeObj = n

    return nodeObj


def buildJointMatrix(agent):
    """This function builds the full information matrix"""

    fList = list(agent.fg.get_fnodes())

    infMat = deepcopy(fList[0])

    infMat.factor._W = infMat.factor._W * 0
    infMat.factor._Wm = infMat.factor._Wm * 0

    infMat.factor._dim = fList[0].factor._dim

    for f in fList:
        tmp_f = deepcopy(f)

        tmp_f.factor._dim = f.factor._dim
        infMat.factor = infMat.factor.__mul__(tmp_f.factor)

        del tmp_f

    return infMat


def sortFactorDims(factor, refDims):
    """This function sorts / orders the factor dimensions to be in the same order as refDims"""

    # Create string lists to compare:
    selfDimList = []
    refDimList = []
    for d in factor.factor._dim:
        selfDimList.append(str(d))
    for d in refDims:
        refDimList.append(str(d))

    if not selfDimList == refDimList:  # check if dim order is not the same
        Tmat = np.zeros((len(selfDimList), len(selfDimList)))
        i = 0
        ind = []
        newDims = []
        for d in refDimList:
            ind.append(selfDimList.index(d))
            selfDimList[ind[i]] = ""
            Tmat[i, ind[i]] = 1
            newDims.append(factor.factor._dim[ind[i]])
            i += 1

        factor.factor._W = np.dot(Tmat, np.dot(factor.factor._W, Tmat.T))
        factor.factor._Wm = np.dot(Tmat, factor.factor._Wm)
        factor.factor._dim = newDims

    return factor


def callback(data, agent):
    if data.recipient == agent["agent"].id:
        agent["agent"].fusion.fusionLib[data.sender].inMsg = convertMsgToDict(data)


def boss_callback(msg):
    pass


def convertMsgToDict(msg):
    msgs = {}
    data = {}
    data["dims"] = msg.dims
    n = msg.matrixDim
    data["infMat"] = np.array(msg.infMat).reshape((n, n))
    data["infVec"] = np.array(msg.infVec).reshape((n, 1))
    msgs[1] = data
    return msgs

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

# def meas_callback(msg,ag,meas_data):
#     x = msg.pose.pose.position.x
#     y = msg.pose.pose.position.y
#     mu = np.array([0,0])
#     R0 = ag["measData"][0]["R"]
#     R1 = ag["measData"][1]["R"]
#     # bias = np.array([5,5])
#     noise1 = np.random.multivariate_normal(mu, R0)
#     noise2 = np.random.multivariate_normal(mu,R1)
#     m1 = np.array([x,y]) + bias + noise1
#     m2 = bias + noise2
#     meas_data.data = [m1,m2]
#     # print(meas_data)
#     return 

# def meas_callback_1(msg,ag,meas_data,l):
#     print(type(meas_data))
#     x = msg.pose.position.x
#     y = msg.pose.position.y
#     mu = np.array([0,0])
#     R0 = ag["measData"][0]["R"]
#     R1 = ag["measData"][1]["R"]
#     # bias = np.array([5,5])
#     noise1 = np.random.multivariate_normal(mu, R0)
#     noise2 = np.random.multivariate_normal(mu,R1)
#     m1 = np.array([x,y]) + bias + noise1
#     m2 = bias + noise2
#     meas_data.data[l] = [m1,m2]
#     # print(meas_data)
#     return 

np.set_printoptions(precision=3)

#### BEGINNING OF INPUT FILE

"""
Dynamic target estimation example - 2 agent, 2 target
The agent has East and north bias (s) and the target has East and North position states (x)
This example uses a linear observation model
"""

DEBUG = 0
dt = 0.1
nAgents = 4   # number of agents

prior = dict()

variables = dict()
agents = dict()
varSet = dict()
condVar =  dict() # variables for conditional independence (will need to be automated later)
varList = dict()
commonVars = dict()
localVars = dict()

# define local agent variable sets in dictionaries:
localVars = {"S1", "S2", "T1"}
varSet = [set({"T1", "T2", "S1"}), set({"T2", "S2"})]
condVar = [{"S1"},{"S2"}]

S1 = {'n' : 2}
S2 = {'n' : 2}  #{'n' : 2}
T1 = {'n' : 4}
T2 = {'n' : 4}

variables["T1"], variables["T2"]= T1, T2
variables["S1"], variables["S2"]  = S1, S2

for var in variables:
    if var in localVars:
        variables[var]['Type'] = 'local'
    else:
        variables[var]['Type']= 'common'

dynamicList = {"T1", "T2"}
variables["dynamicList"] = dynamicList

# Define Linear observations:
for a in range(1, nAgents+1):
    agents[a] = dict()
    agents[a]['measData'] = dict()
    agents[a]['measData'][1] = dict()
    agents[a]['measData'][2] = dict()
    agents[a]['currentMeas'] = dict()
    agents[a]['neighbors'] = dict()
    agents[a]['results'] = dict()

agents[1]['measData'][3]=dict()

# agent 1:
agents[0]['measData'][0]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[0]['measData'][0]['R'] = np.diag([1.0, 10.0])
agents[0]['measData'][0]['invR'] = np.linalg.inv(agents[1]['measData'][1]['R'])
agents[0]['measData'][0]['measuredVars'] = ['T1','S1']   # has to be in the order of the variable vector
agents[0]["measData"][0]["measType"] = "targetPos"

# agent 1:
agents[0]['measData'][1]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[0]['measData'][1]['R'] = np.diag([1.0, 10.0])
agents[0]['measData'][1]['invR'] = np.linalg.inv(agents[1]['measData'][2]['R'])
agents[0]['measData'][1]['measuredVars'] = ['T2','S1']   # has to be in the order of the variable vector
agents[0]["measData"][1]["measType"] = "targetPos"


agents[0]['measData'][2]['H'] = np.array([[ 1, 0], [ 0, 1]], dtype=np.float64)
agents[0]['measData'][2]['R'] = np.diag([3.0, 3.0])
agents[0]['measData'][2]['invR'] = np.linalg.inv(agents[1]['measData'][3]['R'])
agents[0]['measData'][2]['measuredVars'] = ['S1']   # has to be in the order of the variable vector
agents[0]["measData"][2]["measType"] = "agentBias"

# agent 2:
agents[1]['measData'][0]['H'] = np.array([[1, 0, 0, 0, 1, 0],
                                          [0, 0,  1, 0,  0, 1]], dtype=np.float64)
agents[1]['measData'][0]['R'] = np.diag([3.0, 3.0])
agents[1]['measData'][0]['invR'] = np.linalg.inv(agents[2]['measData'][1]['R'])
agents[1]['measData'][0]['measuredVars'] = ['T2','S2']   # has to be in the order of the variable vector
agents[1]["measData"][0]["measType"] = "targetPos" 

# agent 2:
agents[1]['measData'][1]['H'] = np.array([[ 1, 0], [ 0, 1]], dtype=np.float64)
agents[1]['measData'][1]['R'] = np.diag([3.0, 3.0])
agents[1]['measData'][1]['invR'] = np.linalg.inv(agents[2]['measData'][2]['R'])
agents[1]['measData'][1]['measuredVars'] = ['S2']   # has to be in the order of the variable vector
agents[1]["measData"][1]["measType"] = "agentBias"



# Define neighbors:
agents[1]['neighbors'] = [2]
agents[2]['neighbors'] = [1]


# Create factor nodes for prior:
x0 = np.array([[0], [0], [0], [0]])
X0 = np.diag([100.0, 100.0, 100.0, 100.0])

s0 = np.array([[5], [5]])
S0 = np.diag([10.0, 10.0])

prior['T1_0'] = prior['T2_0']   \
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

variables["T1"]["uInd"] = [0,1]
variables["T2"]["uInd"] = [2,3]

# Agent bias
bias = np.array([5,5])

# Target names
target1 = "cohrint_tycho_bot_1"
target2 = "cohrint_tycho_bot_2"
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

#### END OF INPUT FILE

rospack = rospkg.RosPack()
p = rospack.get_path("fgddf_ros")
matFile = sio.loadmat(path.join(p, "measurements_TRO_1T_2A_Dynamic_250.mat"))

class MeasData:
    def __init__(self) -> None:
        self.data = []

rospy.init_node("talker", anonymous=True)
meas_data = MeasData()
# print(meas_data)
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

# meas_callback_lambda = lambda x: meas_callback(x,ag,meas_data,l)
# meas_callback_lambda_1 = lambda x: meas_callback_1(x,ag,meas_data,target1_idx)

# meas_sub = rospy.Subscriber(meas_topic,PoseWithCovarianceStamped,meas_callback_lambda)

# target_counter = 0
# for var in ag["measData"][0]["measuredVars"]:
#     if var == "T1":
#         meas_data.data.append([None,None])
#         target1_idx = target_counter
#         meas_sub1 = rospy.Subscriber("vrpn_client_node/"+target1+"/pose",PoseStamped,meas_callback_lambda_1)
#         target_counter = target_counter + 1

# meas_sub = rospy.Subscriber(meas_topic,PoseWithCovarianceStamped,meas_callback)
sub = rospy.Subscriber("chatter", ChannelFilter, callback, (ag))
boss_sub = rospy.Subscriber("boss", String, boss_callback)
pub = rospy.Publisher("chatter", ChannelFilter, queue_size=10)
data = ChannelFilter()
# meas_topic = 'cohrint_tycho_bot_1/vicon_pose'

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
        if (a["measData"][l]["measType"] == "targetPos"):
            a["currentMeas"][l] = get_target_pos(a["measData"][l])
        elif (a["measData"][l]["measType"] == "agentBias"):
            a["currentMeas"][l] = get_agent_bias(a["measData"][l])

        # a["currentMeas"][l] = meas_data.data[l].reshape((len(meas_data.data[l]),1))
        

    a["agent"] = a["filter"].add_Measurement(a["agent"], a["currentMeas"])

# set initial fusion defenitions:
for i, a in enumerate(agents):
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
    print("agent", i)
    for n in a["neighbors"]:
        # recieve message (a dictionary of factors):
        print("neighbor", n)
        msg = agents[n]["agent"].sendMsg(agents, n, i)
        a["agent"].fusion.fusionLib[n].inMsg = msg

# Fuse incoming messages, time step 1:
for a in agents:
    # if len(agents[a]['neighbors'])>0:
    a["agent"].fuseMsg()
    for key in a["agent"].fusion.commonVars:
        a["agent"] = mergeFactors(a["agent"], a["agent"].fusion.commonVars[key])
        # TODO check if mergeFactors works with dynamic variables

for a in agents:
    a["agent"].build_semiclique_tree()


tmpGraph = dict()
# Inference
for i, a in enumerate(agents):
    tmpGraph[i] = a["agent"].add_factors_to_clique_fg()

    a["results"][0] = dict()  # m is the MC run number

    # Calculate full covariance for NEES test
    jointInfMat = buildJointMatrix(a["agent"])
    # jointCovMat=jointInfMat.factor.cov

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
        for n in tmpGraph[i].get_vnodes():
            varCount = 0
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
    nM = len(a["measData"])  # number of measurements
    for l in range(nM):
        # p = a["measData"][l]["H"].shape[0]  # number of vector elements
        if (a["measData"][l]["measType"] == "targetPos"):
            a["currentMeas"][l] = get_target_pos(a["measData"][l])
        elif (a["measData"][l]["measType"] == "agentBias"):
            a["currentMeas"][l] = get_agent_bias(a["measData"][l])

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

    rospy.wait_for_message("boss", String)  # Wait for go ahead
    ag["agent"].fuseMsg()

    for key in ag["agent"].fusion.commonVars:
        ag["agent"] = mergeFactors(ag["agent"], ag["agent"].fusion.commonVars[key])

    ag["agent"].build_semiclique_tree()
    tmpGraph = dict()
    # inference
    jointInfMat = buildJointMatrix(ag["agent"])
    tmpGraph[ag["agent"].id] = ag["agent"].add_factors_to_clique_fg()
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
                ag["results"][0][(varStr + "_mu")] = np.array(belief.mean)
                ag["results"][0][(varStr + "_cov")] = np.array(belief.cov)

        ag["results"][0]["FullCov"] = np.array(jointInfMat.factor.cov)
        ag["results"][0]["FullMu"] = np.array(jointInfMat.factor.mean)
    del tmpGraph
    k += 1

    for var in ag["agent"].varSet:
        ag_tag = "S" + str(ag_idx + 1)
        if var != ag_tag:
            agent_results = Results()
            agent_results.TimeStep = k-1
            agent_results.Agent = ag_tag
            agent_results.Bias = bias
            agent_results.Target = var
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