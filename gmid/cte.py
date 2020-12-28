from constants import *
from message_passing import *
from graph_algorithms import toposort2


class CTE(MessagePassing):
    def __init__(self, verbose_level, message_graph, elim_order, weights, is_log, is_valuation, log_file_name):
        super(CTE, self).__init__(verbose_level, message_graph, elim_order, weights, is_log, log_file_name)
        self.is_valuation = is_valuation

    ####################################################################################################################
    def schedule(self):
        """ produce a message passing schedule for CTE using topological sort for DAG  """
        sg = self.mg.region_graph.copy().to_directed() # copy undirected graph as directed graph
        for u,v in sg.edges(): # make a DAG, arrows following elim order
            if self.elim_order.index(u[0]) > self.elim_order.index(v[0]):
                sg.remove_edge(u,v)
            elif self.elim_order.index(u[0]) == self.elim_order.index(v[0]) and self.elim_order.index(u[1]) > self.elim_order.index(v[1]):
                sg.remove_edge(u,v)
        # make a dag of nodes to call toposort2
        node_parents = {node : set([pa for pa, me in sg.in_edges([node])]) for node in sg.nodes()}
        # self.node_schedule = list(toposort2(node_parents))
        self.node_schedule = []
        for nodes in list(toposort2(node_parents)):
            # sort nodes wrt elim order
            # nodes.sort(key=lambda x: (self.elim_order.index(x[0]), x[1]))
            self.node_schedule.append(sorted(nodes, key=lambda x:(self.elim_order.index(x[0]), x[1] ) ))
        self.schedule_graph = sg
        return self.node_schedule

    def _get_next_node_to_eliminate(self, node, direction='fw'):
        if direction == 'fw': # get a list of next nodes from current node
            return [ch for me, ch in self.schedule_graph.out_edges_iter([node])]
        elif direction == 'bw':
            return [pa for pa, me in self.schedule_graph.in_edges_iter([node])]
        else:
            raise MessagePassingError

    def _get_roots_of_schedule_graph(self):
        root_nodes = []
        for node, deg in self.schedule_graph.out_degree_iter():
            if deg == 0:
                root_nodes.append(node)
        if root_nodes:
            return root_nodes
        else:
            raise MessagePassingError

    ####################################################################################################################
    def init_propagate(self):
        self.mg.init_msg_propagation_graph('directed')
        for n in self.mg.message_graph.nodes_iter():  # pre-combine all factors in each node
            combined_factor = self.mg.combine_factors(n, None, self.is_log)
            self.mg.set_factor(n, combined_factor)

    ####################################################################################################################
    def propagate(self, propagation_type='ve', max_sum=False):
        """ max_sum for mini bucket elimination with weight 1 and 0"""
        self.time_0 = time.time()
        if propagation_type == 've':
            self._propagate_one_pass('fw', max_sum)
            # if not self.is_valuation:  # fixme cannot do bw passing in ID elimination sequence is constrained
            #     self._propagate_one_pass('bw')
        elif propagation_type == 'mbe_cost_shifting':  # used for gdd initialization
            self._propagate_mbe_cost_shifting(max_sum)

        current_bound = self.bounds(root_only=True)
        # self.print_log('\tupdate:time={:8.2f}\tbound={}'.format(time.time() - self.time_0, str(current_bound)))
        if self.is_valuation:
            return self.obj_from_bounds(current_bound), self.Z_from_bounds(current_bound)
        else:
            return self.factor_bounds(current_bound)


    ####################################################################################################################
    def _propagate_one_pass(self, direction, max_sum=False):
        if direction == 'fw':   # fw refers to the direction of inference
            node_schedule = self.node_schedule
        else:
            node_schedule = reversed(self.node_schedule)
        for nodes in node_schedule:
            for node in nodes:  # parallel propagation possible
                next_nodes = self._get_next_node_to_eliminate(node, direction)
                for next_node in next_nodes:
                    msg_pulled = self.mg.pull_msg(node, next_node)
                    combined_msg = self.mg.combine_factors(node, msg_pulled, self.is_log)
                    if max_sum:
                        msg_to_send = self.mg.marginalize_factors_max_sum(node, next_node, combined_msg, self.weights, self.is_log)
                    else:
                        msg_to_send = self.mg.marginalize_factors(node, next_node, combined_msg, self.weights, self.is_log)
                    self.mg.push_msg(node, next_node, msg_to_send)  # push message on the edge

    def _propagate_mbe_cost_shifting(self, max_sum=False):
        for nodes in self.node_schedule:
            for node in nodes:  # parallel propagation possible
                next_nodes = self._get_next_node_to_eliminate(node, 'fw')
                if len(next_nodes) == 0:
                    continue
                if debug:
                    assert len(next_nodes) == 1
                    assert len(self.mg.message_graph.node[node]['fs']) == 1
                    assert not self.is_log
                next_node = next_nodes[0]

                current_factor = self.mg.message_graph.node[node]['fs'][0]
                if max_sum:
                    msg_to_send = self.mg.marginalize_factors_max_sum(node, next_node, current_factor, self.weights, self.is_log)
                else:
                    msg_to_send = self.mg.marginalize_factors(node, next_node, current_factor, self.weights, self.is_log)
                next_factor = self.mg.message_graph.node[next_node]['fs'][0]

                cs_current_factor = current_factor/msg_to_send
                self.mg.set_factor(node, cs_current_factor)
                cs_next_factor = next_factor*msg_to_send
                self.mg.set_factor(next_node, cs_next_factor)


    ####################################################################################################################
    def bounds(self, root_only=True):
        if root_only:
            bound_nodes = self._get_roots_of_schedule_graph()
        else:
            bound_nodes = self.schedule_graph.nodes()
        marginals = []
        for node in sorted(bound_nodes, key=lambda x: self.elim_order.index( x[0] )):
            in_msgs = self.mg.pull_msg(node, None)
            combined_msg = self.mg.combine_factors(node, in_msgs, self.is_log)
            marginal = self.mg.marginalize_factors(node, None, combined_msg, self.weights, self.is_log)
            marginals.append(marginal)
        return marginals

    def obj_from_bounds(self, marginals):
        tot = marginals[0].copy()
        for el in marginals[1:]:
            if self.is_log:
                tot = tot + el
            else:
                tot= tot * el
        return tot.util

    def Z_from_bounds(self, marginals):
        tot = marginals[0].copy()
        for el in marginals[1:]:
            if self.is_log:
                tot = tot + el
            else:
                tot= tot * el
        return tot.prob

    def factor_bounds(self, marginals):
        tot = marginals[0].copy()
        for el in marginals[1:]:
            if self.is_log:
                tot = tot + el
            else:
                tot = tot * el
        return tot
