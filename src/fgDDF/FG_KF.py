"""
Kalman filter (KF) implemented on a Factor graph (FG) class

"""
# from fglib import graphs, nodes, inference, rv, utils
import matplotlib.pyplot as plt
import networkx as nx
# import numpy as np
import scipy.linalg
# import scipy.io as sio
# from copy import deepcopy
from fgDDF.factor_utils import *
# from operator import itemgetter

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

        # print(varSet)

        for var in varSet:
            if var in variables["dynamicList"]:  # dynamic variable

                variables[var]["Qinv"] = np.linalg.inv(variables[var]["Q"])
                # print(variables[var]["uInd"])
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
        nodesToMarginalize=[]
        # factorsToRemove = []
        localVars = []
        commonVars = []

        oldTrueGraph = deepcopy(agent_i)
        tmpGraph = deepcopy(agent_i)

        list_vnodes = agent_i.fg.get_vnodes( )
        strListNodes = []
        for var in list_vnodes:
            strListNodes.append(str(var))

        for var in agent_i.varSet:
            if var in agent_i.dynamicList:
                nodesToMarginalize.append(getattr(agent_i, var+"_Past"))


        agent_i = mergeFactors(agent_i,strListNodes)


        # get pre-marginalization graph structure
        Amat = nx.adjacency_matrix(agent_i.fg).todense()
        A2mat = np.dot(Amat, Amat)      # path of length 2 (b.c of factors that are between variable nodes)
        adjNodeList = list(agent_i.fg.adj._atlas)   # The order of the adjacency matrix corresponds to the order of atlas

        # step 1: detach local nodes:
        # This step approximates the distribution by the marginals

        # split into common variables and local variables
        commonDict = dict()        # Dictionary of dictionaries containing data to build new adjacency matrix
        lenAdj = len(adjNodeList)
        commonList = []
        commonListKeep = []   # common nodes that are not marginalized out
        for v in list_vnodes:
            if agent_i.fg.nodes[v]['comOrLoc'] == 'local':
                localVars.append(str(v))
            else:
                commonVars.append(str(v))
                if str(v) in nodesToMarginalize:
                    timeInd = str(v).find('_')+1
                    k = str(v)[timeInd:]
                    nextNode = str(v)[0:timeInd]+str(int(k)+1)
                    m = np.zeros((1, lenAdj), dtype=int)
                    m[0, adjNodeList.index(v)] = 1
                    commonDict[str(v)] = {'node': v, 'nextNode': nextNode, 'adjIndex': adjNodeList.index(v), 'matRow': m}
                    commonList.append(str(v))
                    commonListKeep.append(nextNode)


        agent_i.fg = self.marginalizeNodes(agent_i, commonVars )
        tmpGraph.fg = self.marginalizeNodes(tmpGraph, localVars )

        # new graph out of the marginals of the two sets
        agent_i.factorCounter = dis_union(agent_i.fg, tmpGraph.fg, agent_i.factorCounter)

        del tmpGraph
        # Step 2 - Marginalize past nodes
        agent_i = mergeFactors(agent_i,strListNodes)

        for n in nodesToMarginalize:
            agent_i = self.filterPastState(agent_i, n)
            oldTrueGraph = self.filterPastState(oldTrueGraph, n)
            strListNodes.remove(n)

        # Transformation matrix
        T_mat = commonDict[commonList[0]]['matRow']
        for i in range(1, len(commonList)):
            T_mat = np.append(T_mat, commonDict[commonList[i]]['matRow'], axis= 0)

        # Build new adjacency matrix (with only relevant parts):
        newAmat = np.dot(T_mat, np.dot(A2mat,T_mat.T))

        # update varList
        list_vnodesNew = agent_i.fg.get_vnodes( )
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
            varIndex = np.where(np.asarray(newAmat[i])[0]==0)
            if len(varIndex[0]) > 0:    # ==> there shouldn't be an edge b/w those variables
                v = [v for v in list_vnodesNew if str(v)==commonListKeep[i]]
                nf = list(agent_i.fg[v[0]])
                for f in nf:
                    nv=list(agent_i.fg[f])  # neighbor variables
                    if len(nv)>1:
                        agent_i = distributeFactor(f, nv, agent_i, True)

            agent_i = mergeFactors(agent_i,strListNodes)

        # 3.2 Remove binary factor to regain conditional independence:

        for i in range(0, len(commonListKeep)):
            # check if there are no previous connections to other nodes
            varIndex = np.where(np.asarray(newAmat[i])[0]==0)
            if len(varIndex[0]) > 0:    # ==> there shouldn't be an edge b/w those variables
                v = [v for v in list_vnodesNew if str(v)==commonListKeep[i]]
                nf = list(agent_i.fg[v[0]])
                for f in nf:
                    nv=list(agent_i.fg[f])

                    nv.remove(v[0])   # now nv only has the other variables, without the current one.
                    for v2 in nv: # the current variable is coupled to other variables
                        # check if should be decoupled
                        ind = commonListKeep.index(str(v2))
                        if ind in varIndex[0]:    # variables should be decoupled
                            agent_i.fg.remove_node(f)

        # 4. Deflate factors to make conservative
        TargetJointInfMat=buildJointMatrix(oldTrueGraph)
        del oldTrueGraph
        sparseJointInfMat=buildJointMatrix(agent_i)

        sparseJointInfMat = sortFactorDims(sparseJointInfMat, TargetJointInfMat.factor._dim)

        Q = np.dot(np.linalg.inv(scipy.linalg.sqrtm(sparseJointInfMat.factor._W)), np.dot(TargetJointInfMat.factor._W, np.linalg.inv(scipy.linalg.sqrtm(sparseJointInfMat.factor._W))))
        lamdaVec, lamdaMat  = np.linalg.eig(Q)

        # get deflation factor:
        lamdaMin = lamdaVec.min().real

        # Deflate the sparse information matrix:
        sparseJointInfMat.factor._W = (sparseJointInfMat.factor._W+sparseJointInfMat.factor._W.T)/2   # make sure that symmetric
        sparseJointInfMat.factor._W = sparseJointInfMat.factor._W*lamdaMin

        # Update information vector s.t the mean is unchanged:
        sparseJointInfMat.factor._Wm = np.dot(sparseJointInfMat.factor._W, np.dot(np.linalg.inv(TargetJointInfMat.factor._W),TargetJointInfMat.factor._Wm))

        agent_i = mergeFactors(agent_i,strListNodes)

        # Deflate factors
        list_fnodes=list(agent_i.fg.get_fnodes())

        for f in list_fnodes:
            f.factor._W = lamdaMin*f.factor._W

            if len(agent_i.fg[f]) == 1:
                dimInd = [i for i, d in enumerate(sparseJointInfMat.factor.dim) if d in f.factor.dim]
                f.factor._Wm=sparseJointInfMat.factor._Wm[dimInd]
            else:
                f.factor._Wm=f.factor._Wm*0

        agent_i.lamdaMin = lamdaMin
        del sparseJointInfMat, TargetJointInfMat
        return agent_i