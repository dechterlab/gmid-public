"""
# influence diagrams for sys admin problem with a ring topology.
[SysAdmin](https://jair.org/index.php/jair/article/view/10341/24723)
Carlos Guestrin, et.al, "efficient solution algorithms for factored mdps", JAIR 2003

* illustration
  * factored state represents the state of a server
  * factored decision represents a switch that turns on and off a server
  that a single switch can intervene K consecutive servers in a ring  (switch does not overlap)
  * reward function is defined per the state of each server
    * +1 when a server is turned on.
    * extend reward function depending on decision later
  * transition probability is defined in page 7.
  * time stage T is also a parameter

* script
  * generate ID-UAI files (*.uai, *.id, *.pvo)

* TODO curernt gmid assumes 1 decision variable per stage... changes required in many places... fileio, and ??
"""
PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.constants import *
from gen_mdp import create_cpt, make_fh_id_multi_dec


def make_2t_sysadmin(n_s = 4, n_d = 2, trans_scope=2):
    """
    :param n_s: num state vars, servers
    :param n_d: num decision vars, switches that divide servers n_d < n_s
    :param trans_scope: scope of transitions, trans < n_s
    :return: 2 time stage MDP in nx Directed graph
    """
    mdp_2t = nx.DiGraph()

    node_label = 0      # label inside influence diagram
    # init state variables
    for n in range(node_label, node_label + n_s):
        mdp_2t.add_node(n)
        mdp_2t.node[n]['step'] = 0
        mdp_2t.node[n]['type'] = 'initial_state'
        mdp_2t.node[n]['domain_size'] = 2
        mdp_2t.node[n]['parents'] = []   # empty
        mdp_2t.node[n]['table'] = create_cpt(mdp_2t, mdp_2t.node[n]['domain_size'], [])

    node_label += n_s
    decision_labels = []
    # add decision variables
    for n in range(node_label, node_label + n_d):
        decision_labels.append(n)
        mdp_2t.add_node(n)
        mdp_2t.node[n]['step'] = 1
        mdp_2t.node[n]['type'] = 'decision'
        mdp_2t.node[n]['domain_size'] = 2
        mdp_2t.node[n]['parents'] = list(range(n_s))     # todo all initial_state vars/it's only informational arc
        mdp_2t.node[n]['table'] = []                     # no table for a decision variable

    node_label += n_d
    # create utility nodes per state variable
    for n in range(node_label, node_label + n_s):
        mdp_2t.add_node(n)
        mdp_2t.node[n]['step'] = 2
        mdp_2t.node[n]['type'] = 'utility'
        mdp_2t.node[n]['domain_size'] = None  # this is a node for a function
        mdp_2t.node[n]['parents'] =  [n - n_d - n_s]        # single parent
        mdp_2t.add_edge(n-n_d-n_s, n)                       # directed edge from state to reward node
        mdp_2t.node[n]['table'] = [0, 1]                    # 0 for turned off, 1 for turn on

    node_label += n_s
    # create state transitions
    for n in range(node_label, node_label + n_s):
        mdp_2t.add_node(n)
        mdp_2t.node[n]['step'] = 3
        mdp_2t.node[n]['type'] = 'state'
        init_state_label = n - node_label  # match current node id to the init state node id
        mdp_2t.node[n]['domain_size'] = mdp_2t.node[init_state_label]['domain_size']
        parents = [k % n_s for k in range(init_state_label, init_state_label-trans_scope, -1)]
        parents.append(decision_labels[0])  # todo there is only 1 decision variable
        # parents.append( decision_labels[int(init_state_label/n_d)] )
        mdp_2t.node[n]['parents'] = sorted(parents)
        mdp_2t.node[n]['table'] = create_cpt(mdp_2t, mdp_2t.node[n]['domain_size'], mdp_2t.node[n]['parents'])     # use random cpt
        # todo later change table to actual numbers if reqruied, now only observe the graph
    return mdp_2t


if __name__ == "__main__":
    n_s = 4
    n_d = 1
    trans_scope = 2
    time_horizon = 4
    file_name = "sysadmin-" + '_'.join([str(n_s), str(n_d), str(trans_scope), str(time_horizon)])
    mdp_2t = make_2t_sysadmin(n_s, n_d, trans_scope)
    id_diagram = make_fh_id_multi_dec(mdp_2t, time_horizon, os.path.join(BETA_PATH, file_name))












