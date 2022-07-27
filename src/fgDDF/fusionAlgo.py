"""
fusion class for decentralized data fusion (DDF)
agent is the base class for all agents in the network
"""
# from fglib import graphs, nodes, inference, rv, utils
# from copy import deepcopy
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
from fgDDF.factor_utils import *

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