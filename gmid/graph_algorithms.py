from constants import *
import numpy as np
from numpy import inf
import itertools, bisect, copy, random
from functools import reduce
from collections import defaultdict
from sortedcontainers import SortedSet
from pyGM.varset_py import Var, VarSet
from pyGM.factor import Factor
from pyGM.graphmodel import factorSet
from valuation import Valuation
if IMPORT_PYGRAPHVIZ:
    import pygraphviz as pgv


########################################################################################################################
# helper functions
########################################################################################################################
def find_closest_var(varset, order_list):
    """ Find the earliest variable shown in order_list from a set of variables

    Parameters
    ----------
    varset      :   a set of variables
    order_list  :   an elimination order

    Returns:
    --------
    variable    :   the earliest eliminating variable in varset
                    if all variables in varset are not present in order_list, return None

    """
    for el in order_list:
        if el in varset:
            return el


def toposort2(data):
    """ Find a topological sort of DAG provided as data

    Parameters
    ----------
    data        :   dict of DAG { node : set(parents node) }

    Returns
    -------
    generator   : generator producing list of parallel executable nodes
                  (nodes that are at the same topological order)

    Notes
    -----
    taken from http://rosettacode.org/wiki/Topological_sort#Python

    """
    for k, v in data.items():
        v.discard(k)  # Ignore self dependencies
    extra_items_in_deps = reduce(set.union, data.values()) - set(data.keys())
    data.update({item: set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.items() if not dep)
        if not ordered:
            break
        yield sorted(ordered)
        data = {item: (dep - ordered) for item, dep in data.items()
                if item not in ordered}
    assert not data, "A cyclic dependency exists amongst %r" % data


########################################################################################################################
# functions for drawing nx graph by graphviz
########################################################################################################################
def draw_nx_graph(file_name, nx_graph, node_fill=[]):
    """ Draw undirected networkX graph by graphviz

    Parameters
    ----------
    file_name       :   file name for storing a graphviz graph
    nx_graph        :   networkX Graph()
    node_fill       :   node that will be filled with a color yellow

    Output
    ------
    produces a png file showing the graph

    Notes
    -----
    From input undirected graph, internally build a dag to sort nodes by labels
    use undirected arrows when drawing the graph

    c.f. is there a better and easier way to control
    the layout and sorting the nodes in graphviz?

    """
    draw_graph = nx.DiGraph()
    draw_graph.add_nodes_from(nx_graph.nodes_iter())

    is_directed = isinstance(nx_graph, nx.DiGraph)

    for nn in draw_graph.nodes_iter():
        draw_graph.node[nn]["shape"] = "circle"
        if nn in node_fill:
            draw_graph.node[nn]["style"] = "filled"
            draw_graph.node[nn]["fillcolor"] = "yellow"

    for u, v in sorted(nx_graph.edges_iter()):
        if is_directed:
            draw_graph.add_edge(u, v)
        else:
            if u < v:       # add edges from lower_id -> larger_id for sorting the layout
                if not draw_graph.has_edge(u, v):
                    draw_graph.add_edge(u, v, dir="none")
            else:
                if not draw_graph.has_edge(v, u):
                    draw_graph.add_edge(u, v, dir="none")

    draw_graph.graph['graph'] = {'rankdir': "LR", 'ordering': "out", 'splines':'line'}
    if not is_directed:
        draw_graph.graph['graph']['arrowType'] = "None"
    A = nx.nx_agraph.to_agraph(draw_graph)
    A.layout('dot')
    A.draw(file_name + ".png")


def draw_influence_diagram(file_name, nx_influence_diagram, decision_nodes):
    n_chance = []
    n_dec = []
    n_util = []
    for n in sorted(nx_influence_diagram.nodes_iter()):
        if nx_influence_diagram.node[n]["node_type"] == "C":
            n_chance.append(n)
        elif nx_influence_diagram.node[n]["node_type"] == "D":
            n_dec.append(n)
        else:
            n_util.append(n)
    e_info = []
    e_value = []
    e_prob = []
    n_obs = set()
    for u,v in sorted(nx_influence_diagram.edges()):
        if nx_influence_diagram.node[v]["node_type"] == "D":
            e_info.append((u,v))
            assert u in n_chance, "observed variable should be a chance variable"
            n_obs.add(u)
        elif nx_influence_diagram.node[v]["node_type"] == "U":
            e_value.append((u,v))
        else:
            e_prob.append((u,v))
    draw_graph = nx.DiGraph()
    # http: // www.graphviz.org / doc / info / attrs.html
    # https: // stackoverflow.com / questions / 39657395 / how - to - draw - properly - networkx - graphs
    draw_graph.add_nodes_from(n_chance, shape="circle")
    for nn in n_obs:
        draw_graph.node[nn]["style"] = "filled"
        draw_graph.node[nn]["fillcolor"] = "yellow"
    draw_graph.add_nodes_from(n_dec, shape="box")
    for nn in n_dec:
        draw_graph.node[nn]["style"] = "filled"
        draw_graph.node[nn]["fillcolor"] = "magenta"
    draw_graph.add_nodes_from(n_util, shape="diamond")
    for nn in n_util:
        draw_graph.node[nn]["style"] = "filled"
        draw_graph.node[nn]["fillcolor"] = "cyan"

    draw_graph.add_edges_from(e_info, color="red", label="info", style="dashed", penwidth=3.0)
    draw_graph.add_edges_from(e_prob, color="black")
    draw_graph.add_edges_from(e_value, color="blue")

    for d_ind in range(len(decision_nodes)-1):
        draw_graph.add_edge(decision_nodes[d_ind], decision_nodes[d_ind+1], style="invisible", color="green", dir="none")

    # introduce invisible arcs from util to the next obs
    from networkx.algorithms.dag import descendants
    for d_ind in range(len(decision_nodes)-1):
        util_descendant = [nn for nn in descendants(draw_graph, decision_nodes[d_ind])
                           if nx_influence_diagram.node[nn]["node_type"] == "U"]
        next_ch = None
        for ch in sorted(draw_graph.successors(decision_nodes[d_ind]), reverse=False):
            if ch not in util_descendant:
                next_ch = ch
                break

        for nn in util_descendant:
            draw_graph.add_edge(nn, decision_nodes[d_ind+1], style="invisible", color="green", dir="none")
            if next_ch:
                draw_graph.add_edge(nn, next_ch, style="invisible", color="green", dir="none")

    draw_graph.graph['graph'] = {'rankdir': "LR", 'overlap' : "ortho_xy", 'splines':"line", 'esep':10}
    A = nx.nx_agraph.to_agraph(draw_graph)
    # add subgraph to graphviz to control layout
    # for d in decision_nodes:
    #     current_step = set()
    #     current_step.add(d)
    #     obs = nx_influence_diagram.predecessors(d)
    #     current_step.update(obs)
    #     states = set()
    #     for nn in obs:
    #         states.update(nx_influence_diagram.predecessors(nn))
    #     current_step.update(states)
    #     values = set()
    #     for nn in n_util:
    #         if set(nx_influence_diagram.predecessors(nn)) & current_step:     # parent of utility
    #             values.add(nn)
    #     current_step.update(values)
    #     if states:
    #         A.add_subgraph(states, rank="same")
    #     if obs:
    #         A.add_subgraph(obs, rank="same")
    #     A.add_subgraph([d] + list(values), rank="same")
    #     # A.add_subgraph(list(values), rank="same")
    A.layout('dot')
    A.draw(file_name +".png")


# todo here... draw factored mdp
def draw_fh_mdp_diagram(file_name, nx_influence_diagram, decision_nodes):
    pass

########################################################################################################################
# NxDiagram, PrimalGraph, InfluenceDiagram, Limid...
########################################################################################################################
class NxDiagram(object):
    """
    A base class for a diagram for a graphical model.

    **Usages**

    use graph algorithms for reasoning about topology of graphical model and
    visualization of the graph and graph algorithms

    """
    def __init__(self, directed):
        if directed:
            self.nx_diagram = nx.DiGraph()
        else:
            self.nx_diagram = nx.Graph()


class PrimalGraph(NxDiagram):
    def __init__(self, graphical_model):
        super(PrimalGraph, self).__init__(directed=False)           # self.nx_diagram = nx.Graph()
        self.create_pg_from_gm(graphical_model)

    def create_pg_from_gm(self, graphical_model):
        self.nx_diagram.add_nodes_from(range(graphical_model.nvar))
        for sc in graphical_model.scopes:
            nodes = [int(v) for v in sc]
            self.nx_diagram.add_edges_from([(int(src), int(tgt)) for src, tgt in itertools.combinations(nodes, 2)])

    @property
    def pg(self):
        return self.nx_diagram

    @property
    def G(self):
        return self.nx_diagram

    def draw_diagram(self, file_name, node_fill = []):
        draw_nx_graph(file_name, self.nx_diagram, node_fill)


class NxInfluenceDiagram(NxDiagram):
    """
    A wrapper class for influence diagram.

    An Influence Diagram stores a networkx Digraph encoding the topology of the influence diagram.

    **Usages**

    create_id_from_scopes()     :   creates NxInfluenceDiagram from nvar, var_types, scopes, etc

    create_id_from_nxgraph()    :   creates NxInfluenceDiagram from other nxgraph

    node_type(node_id)          :   returns "C" or "D" depending on the type of the node

    parents_of(node_id)         :   returns a list of sorted node ids that are parent of the node

    id                          :   returns networkX object as a property

    Note
    ----

    Diagram only shows topology, factor information cannot be mapped to uai due to possible re-ordering of scopes.
    The scope implicitly encodes directions, so it should not be sorted.
    The main use cases of this class are computing relaxed partial variable ordering and drawing diagrams.

    """

    def __init__(self):
        super(NxInfluenceDiagram, self).__init__(directed=True)     # self.nx_diagram = nx.DiGraph()
        self.decision_nodes = []
        self.nvar = 0
        self.var_types = []
        self.scopes = []
        self.scope_types = []
        self.partial_elim_order = []

    def create_id_from_nxgraph(self, nx_graph):
        """ Create an influence diagram from another nx_graph defining an influence diagram

        Parameters
        ----------
        nx_graph            :   networkX DiGraph defining an influence diagram

        **Behavior**

        take input nx_graph as self.nx_diagram and recover the basic information from the nx_graph

        this function behaves in an opposite way compared to create_id_from_scopes()

        Note
        ----
        the scope information only reflects the topology of the diagram

        """
        from networkx.algorithms.dag import topological_sort
        self.nx_diagram = nx_graph
        vars = sorted([v for v in nx_graph.nodes_iter() if nx_graph.node[v]["node_type"] in ['C', 'D']])
        self.nvar = len(vars)
        assert self.nvar == vars[-1]+1, "uai format assumes variables ids are from 0 to nvar-1"
        self.var_types = [nx_graph.node[v]["node_type"] for v in vars]
        self.scopes = []
        self.scope_types = []
        for node_id in sorted(nx_graph.nodes_iter()):
            if nx_graph.node[node_id]["node_type"] == 'C':
                self.scopes.append(sorted(nx_graph.predecessors(node_id)) + [node_id])
                self.scope_types.append('P')
            elif nx_graph.node[node_id]["node_type"] == 'U':
                self.scopes.append(sorted(nx_graph.predecessors(node_id)))
                self.scope_types.append('U')

        self.decision_nodes = []
        for node_id in topological_sort(nx_graph):
            if nx_graph.node[node_id]["node_type"] == 'D':
                self.decision_nodes.append(node_id)

        partial_temporal_order = []
        nodes_covered = set(self.decision_nodes)
        for d in self.decision_nodes:
            obs = nx_graph.predecessors(d)
            if obs:
                partial_temporal_order.append(sorted(obs))
                nodes_covered.update(obs)
            partial_temporal_order.append([d])
        if len(nodes_covered) < self.nvar:
            hidden_nodes = [nn for nn in nx_graph.nodes_iter() if nn not in nodes_covered and nx_graph.node[nn]["node_type"]!= "U"]
            partial_temporal_order.append(hidden_nodes)
        self.partial_elim_order = list(reversed(partial_temporal_order))

    def create_id_from_scopes(self, nvar, var_types, scopes, scope_types, partial_elim_order):
        """ Create an influence diagram from basic input data

        Parameters
        ----------
        nvar                :   the total number of variables
                                this is sum of the number of chance variables and decision variables
        var_types           :   a list of characters
                                "C" for a chance, "D" for a decision
        scopes              :   a list of lists of ints reflecting the direction in the influence diagram
                                defines an edge from scopes[i] to scopes[-1]
        scope_types         :   a list of characters
                                "P" for a probability, "U" for a utility
                                this class assumes that decision scope is defined by partial_elim_order
        partial_elim_order  :   a list of list of variables defining partial elimination order

        ** Behavior**

        take basic data for defining influence diagram and generate networkX DiGraph

        """
        def check_input():
            assert nvar == len(var_types)
            assert len(scopes) == len(scope_types)
            vars = set()
            for bk in partial_elim_order:
                vars.update(bk)
            assert len(vars) == nvar, "variables shown in partial elim order does not match with the input nvar"
            nutil = len([st for st in scope_types if st == 'U'])
            ndec = len([v for v in var_types if v == 'D'])
            assert len(scopes) == nutil + nvar - ndec, "scopes should define only probability and utility functions"
        if debug:
            check_input()
        for n_id, n_type in enumerate(var_types):
            self.nx_diagram.add_node(n_id, node_type=n_type)    # each node has node_type attribute, C, D, U
        nutil = len([st for st in scope_types if st == 'U'])      # count utility functions
        n_id = nvar
        for sc_ind, sc in enumerate(scopes):                                       # scopes pa -> sc[-1]
            if scope_types[sc_ind] == "P":
                self.nx_diagram.add_edges_from((pa, sc[-1]) for pa in sc[:-1])
            elif scope_types[sc_ind] == "U":
                self.nx_diagram.add_node(n_id, node_type='U')
                self.nx_diagram.add_edges_from((pa, n_id) for pa in sc)
                n_id += 1
        assert n_id == nvar+nutil
        obs = None
        self.decision_nodes = []
        for block in reversed(partial_elim_order):              # read blocks in temporal order
            if block:                                           # non empty block
                if var_types[block[0]] == 'C':                  # obs block
                    obs = block
                else:                                           # decision block
                    assert len(block) == 1, "basic influence diagram only allows total ordering on decisions"
                    self.decision_nodes.append(block[0])
                    if obs is not None:                         # obs available to the current decision
                        self.nx_diagram.add_edges_from((pa, block[0]) for pa in obs)
                    obs = None
        self.nvar = nvar
        self.var_types = var_types
        self.scopes = scopes
        self.scope_types = scope_types
        self.partial_elim_order = partial_elim_order

    def node_type(self, node_id):
        if self.nx_diagram.has_node(node_id):
            return self.nx_diagram.node[node_id]['node_type']

    def parents_of(self, node_id):
        if self.nx_diagram.has_node(node_id):
            return sorted(self.nx_diagram.predecessors(node_id))

    @property
    def id(self):
        return self.nx_diagram

    @property
    def G(self):
        return self.nx_diagram

    def draw_diagram(self, file_name):
        draw_influence_diagram(file_name, self.nx_diagram, self.decision_nodes)



# todo add NxFactoredMDP class
# allow multiple decision variables per stage for factored MDP
class NxFactoredMDP(NxInfluenceDiagram):
    def __init__(self):
        super(NxFactoredMDP, self).__init__(directed=True)
        self.total_elim_order = []      # required ???

    def draw_diagram(self, file_name):
        draw_fh_mdp_diagram(file_name, self.nx_diagram, self.decision_nodes)

    def create_id_from_scopes(self, nvar, var_types, scopes, scope_types, partial_elim_order):
        pass


########################################################################################################################
# algorithms for finding minimum separating information set in DAG and testing d-separation
########################################################################################################################
def get_msis_from_dag(dag, source_nodes, terminal_nodes, fixed_nodes):
    """ Find a minimum separating information set MSIS from a dag
        that satisfying I<source_nodes | fixed_nodes union MSIS | terminal_nodes>
        with the minimum size

    Parameters
    ----------
    dag             :   NetworkX DiGraph
    source_nodes    :   a set or list of source nodes
    terminal_nodes  :   a set or list of terminal nodes
    fixed_nodes     :   a set or list of pre-fixed nodes

    Returns
    -------
    cutset          :   a set of nodes of MSIS
                        ensuring    I<source_nodes | fixed_nodes union MSIS | terminal_nodes>

    Notes
    -----
    Implementation of this function only intended to ensure the correctness.
        1. create an ancestral graph from the input dag with respect to the input nodes
        2. moralize the ancestral graph
        3. if the cardinality of the source_nodes (terminal_nodes) is greater than 1, introduce a super node
        4. connect the super node with all nodes that are adjacent to source_nodes (target_nodes)
        5. remove fixed_nodes
        6. call nx.minimum_st_node_cut(processed_undirected_graph, source_node, target_node)
        7. return the cutset returned by the minimum_st_node_cut

    """
    def add_super_node(undirected_graph, nodes):
        super_node = max(undirected_graph.nodes_iter()) + 1
        neighbor_nodes = set()
        for nn in nodes:
            neighbor_nodes.update(undirected_graph.neighbors(nn))
            undirected_graph.add_edges_from([u, super_node] for u in neighbor_nodes)
        return super_node

    from networkx.algorithms.connectivity import minimum_st_node_cut          # local networkx stored under libs
    all_nodes = set(source_nodes) | set(terminal_nodes) | set(fixed_nodes)
    anc_graph = ancestor_dag(dag, all_nodes)
    moral_graph = moralize_dag(anc_graph)

    if len(source_nodes) > 1:
        super_source = add_super_node(moral_graph, source_nodes)
    else:
        super_source = source_nodes[0]
    if len(terminal_nodes) > 1:
        super_terminal = add_super_node(moral_graph, terminal_nodes)
    else:
        super_terminal = terminal_nodes[0]

    moral_graph.remove_nodes_from(fixed_nodes)      # remove fixed_nodes
    msis = minimum_st_node_cut(moral_graph, super_source, super_terminal)
    msis = msis - (set(source_nodes) | set(terminal_nodes))
    return msis


def get_relaxed_influence_diagram(influence_diagram):
    """ Find an information relaxed influence diagram.

    Parameters
    ----------
    influence_diagram   :   NxInfluenceDiagram object
                            defining an networkX object for influence diagram together with helper methods

    Returns
    -------
    relaxed_influence_diagram   : NxInfluenceDiagram object
                                  generated by information relaxation scheme

    """
    from networkx.algorithms.dag import descendants, ancestors

    decision_nodes = influence_diagram.decision_nodes
    nx_id = influence_diagram.id.copy()     # this function changes topology of influence diagram
    requisite_nodes = set()
    for d_ind in range(len(decision_nodes)-1, -1, -1):
        current_decision = decision_nodes[d_ind]
        descendants_of_current_decision = descendants(nx_id, current_decision)

        history_nodes = []
        for i in range(d_ind+1):
            history_nodes.append(decision_nodes[i])
            history_nodes.extend(nx_id.predecessors(decision_nodes[i]))
        assert len(set(history_nodes)) == len(history_nodes)
        if history_nodes == [current_decision]:
            continue

        descendant_utils_of_current_decision = []
        for nn in descendants_of_current_decision:
            if nx_id.node[nn]["node_type"] == "U":
                descendant_utils_of_current_decision.append(nn)

        ancestor_utils = set(descendant_utils_of_current_decision)
        for nn in descendant_utils_of_current_decision:
            ancestor_utils.update(ancestors(nx_id, nn))
        y1  = ancestor_utils & descendants_of_current_decision
        y2 = ancestor_utils & requisite_nodes
        influenced_nodes = list(y1 | y2)

        # get msis between influence nodes and history nodes if both are non-empty
        # if not influenced_nodes and not (set(history_nodes) - set([current_decision])):
        msis = get_msis_from_dag(nx_id, influenced_nodes, history_nodes, [current_decision])
        nx_id.add_edges_from([nn, current_decision] for nn in msis if nx_id.node[nn]["node_type"] == "C")

        current_info_nodes = nx_id.predecessors(current_decision)
        new_info_set = set(current_info_nodes)
        for nn in current_info_nodes:
            if dsep_in_dag(nx_id, {nn}, descendant_utils_of_current_decision, new_info_set - {nn} | {current_decision}):
                new_info_set = new_info_set - {nn}
                nx_id.remove_edge(nn, current_decision)
        requisite_nodes.update(new_info_set)

    relaxed_id = NxInfluenceDiagram()
    relaxed_id.create_id_from_nxgraph(nx_id)
    return relaxed_id


def dsep_in_dag(dag, x, y, z):
    """ Testing D-separation from the input dag by  I <X, Z, Y> (Z is conditional)

    Parameters
    ----------
    dag     :       networkX.DiGraph that is connected
    x       :       iterable of nodes
    y       :       iterable of nodes
    z       :       iterable of nodes

    Returns
    -------
    bool    :       True if X and Y are d-separated by Z in dag (disconnected)

    Notes
    -----
    Testing d-separation by
        0. assert dag is connected
        1. find ancestral subgraph of x, y, z
        2. moralize the subgraph
        3. remove nodes z
        4. if moralized graph disconnected then z d-separates x and y

    """
    assert nx.is_connected(dag.to_undirected()), "input dag is not connected graph"
    all_nodes = set(x) | set(y) | set(z)
    ans_dag = ancestor_dag(dag, all_nodes)
    moral_graph = moralize_dag(ans_dag)
    moral_graph.remove_nodes_from(z)
    for s in set(x):
        for t in set(y):
            if nx.has_path(moral_graph, s, t):
                return False
    return True


def ancestor_dag(dag, nodes, copy=False):
    """ Create a subgraph with ancestor nodes from the input dag

    Parameters
    ----------
    dag     :       networkX.DiGraph
    nodes   :       an iterable of nodes
    copy    :       if set True, return a deep copy of subgraph that copies all the attributes
                    o.w. function is still retuning a shallow copy, i.e., attributes points original

    """
    from networkx.algorithms.dag import ancestors
    ancestor_nodes = set()
    for nn in nodes:
        ancestor_nodes.update(ancestors(dag, nn))
        ancestor_nodes.add(nn)
    if copy:
        return dag.subgraph(ancestor_nodes).copy()
    else:
        return dag.subgraph(ancestor_nodes)


def moralize_dag(dag):
    """ Create a moral graph from the input dag

    Parameters
    ----------
    dag     :       networkX.DiGraph

    """
    moral_graph = dag.to_undirected()
    moral_edges = set()
    for nn in dag.nodes_iter():
        parents = dag.predecessors(nn)
        moral_edges.update(list(itertools.combinations(parents, 2)))
    moral_graph.add_edges_from(moral_edges)
    return moral_graph


########################################################################################################################
# algorithms for finding variable elimination ordering
########################################################################################################################
def get_induced_width_from_ordering(primal_graph, ordering):
    import copy
    pg = copy.deepcopy(primal_graph.G)
    w = 0
    for v in ordering:
        nhd = pg.neighbors(v)
        w = max(w, len(nhd))  # width is max cluster size -1 (exclude v)
        pg.remove_node(v)
        for u,v in itertools.combinations(nhd, 2):
            pg.add_edge(u, v)
    return w


def greedy_variable_order(primal_graph, pool_size=8, pick_e=-1, cutoff=inf, pvo=None):
    def fill_count_and_init_scores(current_graph, processing):
        adj = {k: list(v.keys()) for k, v in current_graph.adjacency_iter() if k in processing}
        try:
            scores = [sum( map(lambda x: 1 if not current_graph.has_edge(x[0], x[1]) else 0, itertools.combinations(adj[b], 2)) ) for b in processing]
        except:
            assert False, "greedy_variable_order encountered error, it's fragile"
        return sorted([[score, processing[ind]] for ind, score in enumerate(scores)], key=lambda x: x[0])

    def add_fill_edges(current_graph, rm_node):
        nhd = current_graph.neighbors(rm_node)
        current_graph.remove_node(rm_node)
        for u,v in itertools.combinations(nhd, 2):
            current_graph.add_edge(u, v)

    G = copy.deepcopy(primal_graph)
    if pvo is None:
        pvo = [ G.nodes() ] # list of list, 1 block contains all variables
    ordering = []
    induced_width = 0;

    for each_block in pvo:
        processing_nodes = each_block
        for v_iter in xrange(len(each_block)):
            if induced_width > cutoff:
                return [], induced_width  # escape
            scores = fill_count_and_init_scores(G, processing_nodes)
            if scores[0][0] == 0:  # lowest score
                selected = scores.pop(0)[1]
            else:
                score_var_pool=scores[:pool_size]
                prob_pool = [pow(sc[0], pick_e) for sc in score_var_pool]
                prob_pool = [sum(prob_pool[:i+1]) for i in range(len(prob_pool))]
                prob = np.random.uniform(0, prob_pool[-1])
                picked_ind = bisect.bisect_right(prob_pool, prob)
                selected = score_var_pool[picked_ind][1]
            ordering.append(selected)
            processing_nodes.remove(selected)
            if len(G.neighbors(selected)) > induced_width:
                induced_width = len(G.neighbors(selected))
            add_fill_edges(current_graph=G, rm_node=selected)
    return ordering, induced_width


def iterative_greedy_variable_order(iter_limit, pg, ps=8, pe=-1, ct=inf, pv=None):
    import copy
    for i in xrange(iter_limit):
        pv_copy = copy.deepcopy(pv)  # pv is a list
        ordering, iw = greedy_variable_order(pg, ps, pe, ct, pv_copy)
        if iw < ct:
            ct = iw
            order_found = ordering
            if debug:
                print('it={} iw={}, order found:{}'.format(i+1, iw, ordering))
    return order_found, ct


########################################################################################################################
# Message Graph
########################################################################################################################
class MessageGraphError(Exception):
    """ error in MessageGraph class """


class MessageGraph(object):
    def __init__(self, region_graph, elim_order):
        self.elim_order = elim_order
        self.region_graph = region_graph # undirected graph
        self.message_graph = None
        self.is_directed = None

    def init_msg_propagation_graph(self, graph_type):
        if graph_type in ['elim', 'directed']:
            self.message_graph = self.region_graph.to_directed()  # edges in both directions for FW/BW message passing
            self.is_directed = True
        elif graph_type in ['dec', 'undirected']:
            self.message_graph = copy.deepcopy(self.region_graph) # undirected edge
            self.is_directed = False
        else:
            raise MessageGraphError

    def destroy_propagation_graph(self):
        del(self.message_graph)

    def reset_msg(self, src, target):
        if not self.message_graph:
            raise MessageGraphError
        self.message_graph.edge[src][target]['msg'] = None

    def pull_msg(self, node, next_node):
        """ collect msg from edges incident to the current node except next_node
            this method can be used in CTE/IJGP algorithms, which use directed message passing graph"""
        if not self.message_graph:
            raise MessageGraphError
        pulled_msg = factorSet()
        for u, v, attr in self.message_graph.in_edges([node], data=True) if self.is_directed else self.message_graph.edges([node], data=True):
            if u == next_node:
                continue
            if self.message_graph.edge[u][v]['msg']:
                pulled_msg.add(attr['msg'])
        return pulled_msg if pulled_msg else None

    def pull_msg_from(self, node_from, node_to):
        """ pull a message at an edge (node_from, node_to) to the cluster node_to"""
        if not self.message_graph:
            raise MessageGraphError
        pulled_msg = factorSet()
        pulled_msg.add( self.message_graph.edge[node_from][node_to]['msg'])
        return pulled_msg if pulled_msg else None

    def push_msg(self, node, next_node, msg):
        """ send msg from node to the next node by storing msg at the edge """
        if next_node is None:
            # if self.message_graph.node[node]['msg'] is None:
            self.message_graph.node[node]['msg'] = msg
            # else:
            #     self.message_graph.node[node]['msg'] = self.message_graph.node[node]['msg'] * msg
        else:
            self.message_graph.edge[node][next_node]['msg'] = msg
            self.message_graph.edge[node][next_node]['ct'] += 1

    def set_node_attr(self, node, attr, value):
        self.message_graph.node[node][attr] = value

    def set_factor(self, node, f):
        """ set a factor/valuation f as a factor set at the node """
        self.message_graph.node[node]['fs'] = factorSet({f})

    def set_factor_rg(self, node, f):
        self.region_graph.node[node]['fs'] = factorSet({f})

    def combine_factors(self, node, factor_set, is_log=True):
        """ combine factors in current node and factor from factor_set """
        if factor_set:
            all_factors = factorSet(self.message_graph.node[node]['fs']) | factor_set # both are factorSet()
            # all_factors = self.message_graph.node[node]['fs'] | factor_set # both are factorSet()
        else:
            all_factors = factorSet(self.message_graph.node[node]['fs'])
            # all_factors = self.message_graph.node[node]['fs']
        if len(all_factors) == 0:
            return None

        combined_factor = all_factors[0].copy() # must copy
        if is_log:
            for f in all_factors[1:]:
                combined_factor = combined_factor + f
        else:
            for f in all_factors[1:]:
                combined_factor = combined_factor * f
        return combined_factor

    elim_op_f = {
        'max': lambda factor, vars: factor.max(vars),
        'lse': lambda factor, vars: factor.lse(vars),
        'sum': lambda factor, vars: factor.sum(vars),
        'min': lambda factor, vars: factor.min(vars)
    }

    def marginalize_factors(self, node, next_node, combined_factor, weights, is_log, normalize=False):
        """ marginalize factors and create a message from node to next_node for CTE/IJGP """
        if next_node:
            eliminator = self.message_graph.node[node]['sc']-self.message_graph.node[next_node]['sc'] # set of int
        else: # eliminate all vars in current scope
            # eliminator = self.message_graph.node[node]['sc'] # todo changed recently
            # eliminator = [el for el in self.elim_order if el in self.message_graph.node[node]['sc']]
            eliminator = set(self.elim_order) & self.message_graph.node[node]['sc']

        for v in sorted(eliminator, key=lambda x: self.elim_order.index(x)): # v is int, a label for Var
            if weights[v] == 0.0:
                elim_op = 'max'
            elif weights[v] == 1.0:
                elim_op = 'lse' if is_log else 'sum'
            else:
                raise MessageGraphError # ignore other marginal operators
            combined_factor = MessageGraph.elim_op_f[elim_op](combined_factor, [v])

        # if normalize:
        #     #  TODO normalization is wrong, IJGP produces wrong result, require damping?
        #     f_temp = combined_factor.copy()
        #     vars = f_temp.vars
        #     for v in vars:
        #         if weights[v] == 0.0:
        #             elim_op = 'max'
        #         elif weights[v] == 1.0:
        #             elim_op = 'lse' if is_log else 'sum'
        #         else:
        #             raise MessageGraphError
        #         f_temp = MessageGraph.elim_op_f[elim_op](f_temp, [v])
        #     combined_factor = (combined_factor - f_temp) if is_log else (combined_factor / f_temp)
        return combined_factor

    def marginalize_factors_max_sum(self, node, next_node, combined_factor, weights, is_log, normalize=False):
        """ marginalize factors and create a message from node to next_node for CTE/IJGP """
        ### for mini bucket elimination, if mini-bucket# is not 0 use max /// node id (var, mini-bucket#)
        if next_node:
            eliminator = self.message_graph.node[node]['sc']-self.message_graph.node[next_node]['sc'] # set of int
        else: # eliminate all vars in current scope
            eliminator = self.message_graph.node[node]['sc']
        for v in sorted(eliminator, key=lambda x: self.elim_order.index(x)): # v is int, a label for Var
            if weights[v] == 0.0:
                elim_op = 'max'
            elif weights[v] == 1.0 and (node[1] != 0):
                elim_op = 'max'
            elif weights[v] == 1.0 and (node[1] == 0):
                elim_op = 'lse' if is_log else 'sum'
            else:
                raise MessageGraphError # ignore other marginal operators
            combined_factor = MessageGraph.elim_op_f[elim_op](combined_factor, [v])
        return combined_factor

    def set_fs_from_mg_to_rg(self):
        for node in self.message_graph.nodes_iter():
            self.region_graph.node[node]['fs'] = self.message_graph.node[node]['fs']


########################################################################################################################
# Mini Bucket Tree and Join graph
########################################################################################################################
def mini_bucket_tree(graphical_model, elim_order, ibound, ignore_msg=False, random_partition=False):
    """ build mini bucket tree with an ibound
        when ignore_msg, result is not valid mini bucket tree. it assigns functions to mini-buckets for decomposition """
    def create_mb(v, sc, f):
        if mini_buckets[v]:
            mb_id = (v, mini_buckets[v][-1][-1] + 1)
        else:
            mb_id = (v, 0)
        mini_buckets[v].add(mb_id)
        mbt.add_node(mb_id, sc=set(), fs=set())
        add_factor_to_mb(mb_id, sc, f)
        return mb_id

    def add_factor_to_mb(mb_id, sc, f):
        mbt.node[mb_id]['sc'].update(sc)
        if f is not None:
            mbt.node[mb_id]['fs'].add(f)

    def allocate_factor(v, sc, f, ibound):
        if random_partition:
            mb_key = lambda x: random.random()
        else:
            mb_key = lambda x: (x[0], x[1])

        for mb_id in sorted(mini_buckets[v], key=mb_key ):
            if ignore_msg: # ignore scope of message when building a mini-bucket tree
                if type(f) is Factor:
                    f_sc = {int(v) for v in f.vars}
                else:
                    f_sc = set()
                mb_sc = {int(v) for ff in mbt.node[mb_id]['fs'] for v in ff.vars}
                if len( f_sc | mb_sc) <= ibound+1:
                    add_factor_to_mb(mb_id, sc, f)
                    return mb_id
            else:
                if mbt.node[mb_id]['sc'] <= sc or len(sc | mbt.node[mb_id]['sc']) <=ibound+1:
                    add_factor_to_mb(mb_id, sc, f)
                    return mb_id
        return create_mb(v, sc, f)

    mbt = nx.Graph() # node id is a tuple (var id, mini-bucket id)
    mini_buckets = defaultdict(SortedSet)
    fall = graphical_model.factors
    for v in elim_order:
        fs = graphical_model.factors_with(v) & fall
        fall = fall - fs
        while fs:
            if random_partition:
                f_pick = fs.pop(random.randint(0, len(fs)-1))
            else:
                f_pick = fs.pop()
            allocate_factor(v, {int(el) for el in f_pick.vars}, f_pick, ibound)

        for mb_id in mini_buckets[v]:
            sc_msg = {int(el) for el in mbt.node[mb_id]['sc']} - {mb_id[0]} # sc: eliminate var and send msg
            vto = find_closest_var(sc_msg, elim_order) # scope of message
            if vto is not None:
                mb_dest = allocate_factor(vto, sc_msg, None, ibound)
                mbt.add_edge(mb_id, mb_dest, sc= sc_msg, ct=0, msg=None)
    return MessageGraph(mbt, elim_order), mini_buckets


########################################################################################################################
# Join graph
########################################################################################################################
def join_graph(mbe_tree, mini_buckets, elim_order, connect_mb_only=False, make_copy=True):
    """ build a join graph from mini bucket tree
        input mini bucket tree
        output minimal edge-labeled join graph from mbt,
            (1) connect mini-buckets in chain
            (2) merge connected mini-buckets and make mini-bucket maximal
                (i,j adjacent & sc(Bi) subset sc(Bj) then Bj = Bj | Bi ) """
    def update_node(node_old, node_new):
        jg.node[node_new]['fs'].update(jg.node[node_old]['fs'])

        for nhd in jg.neighbors(node_old):
            if nhd == node_new: continue
            sc = jg.node[node_new]['sc'] & jg.node[nhd]['sc']
            jg.add_edge(nhd, node_new, sc=sc, ct=0, msg=None) # neighbors of old node becomes nhd of new node
        jg.remove_node(node_old) # remove node aborbed inside node_new

    if make_copy:
        jg = mbe_tree.region_graph.copy() # deep copy of existing graph
    else:
        jg = mbe_tree.region_graph

    for mb_list in mini_buckets.values(): # connect mini-buckets to make join graph
        for v,i in mb_list:
            if (v, i+1) in mb_list:
                jg.add_edge((v,i),(v,i+1), sc={v}, ct=0, msg=None)

    if not connect_mb_only: # make maximal cliques
        for node in sorted( jg.nodes(), key= lambda x: (elim_order.index(x[0]), x[1]) ,reverse=True):
            candidates = []
            for nhd in jg.neighbors(node):
                if jg.node[node]['sc'].issubset(jg.node[nhd]['sc']):
                    candidates.append(nhd)
            candidates.sort(key=lambda x: (elim_order.index(x[0]), x[1]), reverse=True)
            if candidates:
                update_node(node, candidates[0])

    if make_copy:
        return MessageGraph(jg, elim_order)
    else:
        # mbe_tree.region_graph = jg
        mbe_tree.message_graph = None       # possible loss of memory factors because set_fs_from_mg_to_rg ??
        return mbe_tree


########################################################################################################################
# GDD graph (take clusters of Join graph and connect edges)
########################################################################################################################
def gdd_graph(message_graph, elim_order, make_copy=True):
    """ take a node from message graph and create a message passing graph for gdd
        given two clusters, connect them by an edge if there two node shares a scope """
    if make_copy:
        input_graph = message_graph.region_graph.copy() # get a copy of join graph
    else:
        input_graph = message_graph.region_graph

    for node1 in input_graph.nodes_iter():
        sc1 = input_graph.node[node1]['sc']
        for node2 in input_graph.nodes_iter():
            if node1 == node2:
                continue
            if input_graph.has_edge(node1, node2):
                continue
            sc2 = input_graph.node[node2]['sc']
            if sc1 & sc2: # intersection of two scope is not empty
                input_graph.add_edge(node1, node2, sc=sc1 & sc2, ct=0, msg=None)
    if make_copy:
        return MessageGraph(input_graph, elim_order)
    else:
        message_graph.region_graph = input_graph
        return message_graph


########################################################################################################################
# functions defining additional node/ edge attributes for Region graphs
########################################################################################################################
def get_const_with_vars(var_labels, variables, is_valuation, is_log):
    var_set = {var for var in variables if var.label in var_labels}
    if is_valuation:
        const_linear = Valuation(Factor(var_set.copy(), ONE), Factor(var_set.copy(), ZERO))
    else:
        const_linear = Factor(var_set.copy(), ONE)
    if is_log:
        return const_linear.log()
    else:
        return const_linear


def get_const_factor_with_v(v, variables, is_log):
    """ return a constant factor with scope v, all ones (in linear) or all zero (in log) """
    for var in variables:
        if var.label == v:
            return Factor({var}, 0.0) if is_log else Factor({var}, 1.0)


def get_const_valuation_with_v(v, variables, is_log):
    """ return a constant valuation with scope v, (all ones, all zero) """
    for var in variables:
        if var.label == v:
            if is_log: # transform back to log scale
                # return Valuation(Factor({var}, ONE), Factor({var}, ONE)).log()
                return Valuation(Factor({var}, ONE), Factor({var}, ZERO)).log()
            else:
                # return Valuation(Factor({var}, ONE), Factor({var}, ONE))
                return Valuation(Factor({var}, ONE), Factor({var}, ZERO))


def add_mg_attr_to_nodes(MG):
    """ adding additional attributes to the nodes in a message graph
        that can be used in various algorithms
    """
    for n in MG.region_graph.nodes_iter():
        MG.region_graph.node[n]['fsc'] = {int(v) for f in MG.region_graph.node[n]['fs'] for v in f.vars}
        MG.region_graph.node[n]['w'] = {sc: 0 for sc in MG.region_graph.node[n]['sc']}
        MG.region_graph.node[n]['w_old'] = MG.region_graph.node[n]['w'].copy()
        MG.region_graph.node[n]['fs_old'] = None
        MG.region_graph.node[n]['fs_backup'] = None     # used by local search
        MG.region_graph.node[n]['pseudo_belief'] = None
        MG.region_graph.node[n]['bound'] = None
        MG.region_graph.node[n]['util_const'] = 0.0
        MG.region_graph.node[n]['util_const_old'] = 0.0
        MG.region_graph.node[n]['msg'] = None
        # MG.region_graph.node[n]['bound_old'] = None
        # MG.region_graph.node[n]['entropy'] = None
        # MG.region_graph.node[n]['cost'] = None # Factor or Valuation
        # MG.region_graph.node[n]['unit_cost'] = None  # Factor or Valuation


def add_mg_attr_to_edges(MG, variables, is_log, is_valuation=False):
    """ adding additional attributes to the edges in a message graph
        msg, msg_shape, msg_len is for recording the shape and dimension of a single factor in a message
        to be used with scipy
    """
    for n1, n2 in MG.region_graph.edges_iter():
        sc = MG.region_graph.edge[n1][n2]['sc']
        f_temp = factorSet()
        for v in sc:
            if is_valuation:
                f_temp.add(get_const_valuation_with_v(v, variables, is_log))
            else:
                f_temp.add(get_const_factor_with_v(v, variables, is_log))
        combined_factor = f_temp[0]
        if is_log:
            for f in f_temp[1:]:
                combined_factor = combined_factor + f
        else:
            for f in f_temp[1:]:
                combined_factor = combined_factor * f
        MG.region_graph.edge[n1][n2]['msg'] = combined_factor
        MG.region_graph.edge[n1][n2]['msg_shape'] = combined_factor.vars.dims()
        MG.region_graph.edge[n1][n2]['msg_len'] = reduce(lambda  x,y:x*y, combined_factor.vars.dims())


def add_const_factors(MG, variables, is_valuation, is_log):
    """ add const factors to the nodes if some vars are absent in it fsc """
    for n in sorted(MG.region_graph.nodes_iter()):
        var_labels = []
        for v in MG.region_graph.node[n]['sc']:
            # if v not in MG.region_graph.node[n]['fsc']:
            var_labels.append(v)
        if var_labels:
            MG.region_graph.node[n]['fs'].add(get_const_with_vars(var_labels, variables, is_valuation, is_log))
