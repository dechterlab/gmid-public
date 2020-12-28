from constants import *
from message_passing import *
from graph_algorithms import toposort2

from pyGM.factor import Factor
from valuation import Valuation

import random
import time
from collections import defaultdict
from scipy.optimize import minimize
from scipy.optimize import Bounds

class WeightedMBE2(MessagePassing):
    def __init__(self, verbose_level, message_graph, elim_order, weights, is_log, epsilon, log_file_name, ibound,
                 mini_buckets, gm_variables):
        super(WeightedMBE2, self).__init__(verbose_level, message_graph, elim_order, weights, is_log, log_file_name)
        self.epsilon = epsilon  # WEPS smallest weight
        self.ibound = ibound
        self.mini_buckets = mini_buckets    # defaultdict(SortedSet) mini_buckets[var] : SortedSet( [(var, m_id), ...] )
        self.prob_const_per_layer = {v: 1.0 for v in self.elim_order}
        self.global_meu_bound = 1.0     # meu is in linear scale
        self.gm_variables = gm_variables        # list of Var objects from gm class     Var0 at index 0 of list
        self.processed_nodes = set()

    ####################################################################################################################
    def schedule(self):
        pass

    ####################################################################################################################
    def init_propagate(self):
        self.mg.init_msg_propagation_graph('directed')
        for n in self.mg.message_graph.nodes_iter():  # pre-combine all factors in each node
            combined_factor = self.mg.combine_factors(n, None, self.is_log)
            self.mg.set_factor(n, combined_factor)
        self.sum_vars = [v for v in self.elim_order if self.weights[v] > 0]

    def _reset_uniform_weights(self, nodes=[]):
        if not nodes:
            nodes = self.mg.message_graph.nodes()
        for v, weight in enumerate(self.weights):       # var/weight
            if v not in self.sum_vars:
                continue
            count = 0
            tot = 0.0
            for n in nodes:# self.mg.message_graph.nodes_iter():
                if n in self.processed_nodes:
                    continue
                if v in self.mg.message_graph.node[n]['sc']:
                    count += 1

            for n in nodes:# self.mg.message_graph.nodes_iter():
                if n in self.processed_nodes:
                    continue
                if v in self.mg.message_graph.node[n]['sc']:
                    self.mg.message_graph.node[n]['w'][v] = float(weight)/count
                    tot += self.mg.message_graph.node[n]['w'][v]

            for n in nodes:# self.mg.message_graph.nodes_iter():
                if n in self.processed_nodes:
                    continue

                if v in self.mg.message_graph.node[n]['sc']:
                    self.mg.message_graph.node[n]['w'][v] = self.mg.message_graph.node[n]['w'][v]/tot
                    self.mg.message_graph.node[n]['w_old'][v] = self.mg.message_graph.node[n]['w'][v]
                    if debug:
                        self.print_log("_init_uniform_weight")
                        self.print_log("node:{n}, var:{v}, wgt:{w}".format(n=n, v=v, w=str(self.mg.message_graph.node[n]['w'][v])))

    def _reset_node_bounds(self, nodes = []):
        if not nodes:
            nodes = self.mg.message_graph.nodes()
        for node in nodes: #self.mg.message_graph.nodes_iter():
            if node in self.processed_nodes:
                continue
            self._update_bounds_at_node(node)

    def _reset_pseudo_beliefs(self, nodes=[]):
        if not nodes:
            nodes = self.mg.message_graph.nodes_iter()
        for node in nodes:
            self._eval_pseudo_belief_node(node)


    ####################################################################################################################
    def propagate(self, time_limit, iter_limit, optimize_weight, optimize_cost):
        return self._propagate_one_pass(time_limit, iter_limit, optimize_weight, optimize_cost)

    ####################################################################################################################
    def _propagate_one_pass(self, time_limit, iter_limit, optimize_weight, optimize_cost):
        """
        wmbe_id: initialize uniform weights -> inherit weights while passing mbe messages
        """
        const_valuations = []
        self._reset_uniform_weights()
        self._reset_node_bounds()
        self._reset_pseudo_beliefs()

        for var in self.elim_order:
            var_obj = self.gm_variables[var]
            nodes_at_the_current_layer = self.mini_buckets[var]
            # self._reset_uniform_weights()
            # self._reset_node_bounds()
            # self._reset_pseudo_beliefs() --> do this for message received below

            # pull message and combine before cost shifting
            self.print_log("process var:{}\nmini_buckets:{}".format(var, nodes_at_the_current_layer))
            for node in nodes_at_the_current_layer:
                if debug: assert node[0] == var
                for n_from, n_cur  in self.mg.message_graph.in_edges_iter([node]):
                    if debug:
                        assert n_cur == node
                    if self.elim_order.index( n_from[0] ) < self.elim_order.index( n_cur[0] ):
                        in_msgs = self.mg.pull_msg_from(n_from, node)
                        combined_factor = self.mg.combine_factors(node, in_msgs, self.is_log)
                        self.mg.set_factor(node, combined_factor)

            # optimize bound
            time_0, iter, cost_updated, weight_updated = time.time(), 1, False, False
            while len(nodes_at_the_current_layer) > 1 and iter <= iter_limit and (time.time() - time_0) < time_limit:
                if optimize_cost:
                    cost_updated = self._optimize_costs(var_obj, nodes_at_the_current_layer)
                if optimize_weight:
                    weight_updated = self._optimize_weights(var_obj, nodes_at_the_current_layer)
                if not cost_updated and not weight_updated: # both not updated within iter limit escape
                    self.print_log("\toptimizing mini_buckets iter:{} no improvement".format(iter))
                    break
                else:
                    self.print_log("\toptimizing mini_buckets iter:{}\tbound:{}".format(iter, self.read_decomposed_bounds()))
                    self.print_log("\t\tc_updated:{}\tw_updated:{}".format(cost_updated, weight_updated))
                    iter += 1

            # push message downward along the edge
            nodes_changed = []
            for node in nodes_at_the_current_layer:
                # if node == (20,0):
                #     print("watch")        first mini bucket split in mdp1
                w = self.mg.message_graph.node[node]['w'][var]
                w_inv = np.inf if w <= self.epsilon else ONE/w
                F = self.mg.message_graph.node[node]['fs'][0].abs()
                lnF = F.log()
                lnmsg = lnF.lsePower(elim={var_obj}, power=w_inv)
                msg = lnmsg.exp()
                next_node = self._get_next_node(node)
                if next_node:
                    nodes_changed.append(next_node)
                    self.print_log("\tsend msg from {} to {}".format(node, next_node))
                    self.mg.push_msg(node, next_node, msg)
                    ### inherit weights, recompute bounds, pseudo belief at receiving nodes
                    sc_from = self.mg.message_graph.node[node]['sc']
                    sc_to = self.mg.message_graph.node[next_node]['sc']
                    w_from = self.mg.message_graph.node[node]['w']
                    w_to = self.mg.message_graph.node[next_node]['w']
                    if debug:
                        sc_msg = sc_from - {var}
                        assert sc_msg.issubset(sc_to)
                    for vv in sc_from:
                        if vv == var:   # skip weight for the current layer
                            continue
                        # inherit weigths
                        self.mg.message_graph.node[next_node]['w'][vv] += self.mg.message_graph.node[node]['w'][vv]
                else:
                    self.print_log("\tconstant msg from {} with valuation:{}".format(node, msg))
                    self.mg.push_msg(node, None, msg)
                    const_valuations.append(msg)
            if nodes_changed:
                self._reset_node_bounds(nodes_changed)
                self._reset_pseudo_beliefs(nodes_changed)
            self.processed_nodes.update(nodes_at_the_current_layer)

            # if var == 19:     for mdp1 problem debugging first decision
            #     exit()
        bound_valuation = Valuation(1.0, 0.0)
        print(const_valuations)
        for val in const_valuations:
            bound_valuation = bound_valuation * val
        return float(bound_valuation.util), float(bound_valuation.prob)

    def _update_uniform_weights_to_minibuckets(self, var_obj, nodes_at_the_current_layer):
        for node in nodes_at_the_current_layer:
            if var_obj.label in self.sum_vars:
                self.mg.message_graph.node[node]['w'][var_obj.label] = 1.0/len(nodes_at_the_current_layer)
            else:
                self.mg.message_graph.node[node]['w'][var_obj.label] = 0.0

    def _get_next_node(self, node):
        if debug:
            dest = []
            for node_cur, node_to in self.mg.message_graph.out_edges_iter([node]):
                assert node == node_cur
                if self.elim_order.index( node_cur[0] ) < self.elim_order.index( node_to[0] ):
                    dest.append(node_to)
            if dest:
                assert len(dest) == 1
                return dest[0]
            else:
                return None
        else:
            for node_cur, node_to in self.mg.message_graph.out_edges_iter([node]):
                if self.elim_order.index(node_cur[0]) < self.elim_order.index(node_to[0]):
                    return node_to
            return None

    def _optimize_costs(self, var_obj, nodes_at_the_current_layer):
        if debug:
            print("start updating cost at var {} on {}:".format(var_obj, nodes_at_the_current_layer))
        edges = []
        for ind_to, node_to in enumerate(nodes_at_the_current_layer[1:]):
            node_from = nodes_at_the_current_layer[ind_to]
            edges.append((node_from, node_to))

        # define constraints
        obj_pre = self.read_decomposed_bounds()
        x0 = self._init_costs(var_obj, edges)
        prob_lb = np.zeros( var_obj.states * len(edges) ) + WEPS
        prob_ub = np.array( [np.inf]* (var_obj.states*len(edges)) )
        util_lb, util_ub = np.array([]), np.array([])
        for edge in edges:
            lb, ub = self._box_util_bound(var_obj, edge)
            util_lb = np.concatenate( (util_lb, lb))
            util_ub = np.concatenate( (util_ub, ub) )
        bounds = Bounds(np.concatenate( (prob_lb, util_lb) ), np.concatenate( (prob_ub, util_ub) ))

        # call optimzier
        res = minimize(self._cost_obj, x0, args=(var_obj, edges), method='SLSQP',        # SLSQP
                                      jac='2-point', bounds=bounds)#options={"maxiter":2})
        if debug:
            print("\nafter minimize")
            print("\tscipy success:{}, status:{}, message:{}, niter:{}".format(res.success, res.status, res.message, res.nit))
            print("\tcost prev:{}\tnew:{}\tdecreased:{}".format(obj_pre, res.fun, res.fun < obj_pre))
        if res.fun < obj_pre:
            self._update_costs(var_obj, res.x, edges)
            return True
        return False

    def _update_costs(self, var_obj, sol, edges):
        valuations = self._array_to_val(var_obj, sol, edges)
        for ind_edge, edge in enumerate(edges):
            val = valuations[ind_edge]
            node_from, node_to = edge
            self.mg.message_graph.edge[node_from][node_to]['shift'] = val
            temp1 = self.mg.message_graph.node[node_from]['fs'][0] /  val
            self.mg.set_factor(node_from,  temp1)
            temp2 = self.mg.message_graph.node[node_to]['fs'][0] * val
            self.mg.set_factor(node_to, temp2)
            self._update_bounds_at_node(node_from)
            self._update_bounds_at_node(node_to)
        objective = self.read_decomposed_bounds()
        return objective

    def _init_costs(self, var_obj, edges):
        """ pack [ prob || util ]
            for each block, concatenate table per edge
        """
        prob_tables = np.ones( var_obj.states* len(edges))
        util_tables = np.zeros( var_obj.states*len(edges))
        return np.concatenate( (prob_tables, util_tables) )

    def _array_to_val(self, var_obj, x, edges):
        prob_tables, util_tables = np.split(x.copy(), 2)
        valuations = []
        for p, u in zip(np.split(prob_tables, len(edges)), np.split(util_tables, len(edges))):
            valuations.append(Valuation(Factor(var_obj, p), Factor(var_obj, u)))
        return valuations

    def _val_to_array(self, var_obj, valuations, edges):
        prob_tables, util_tables = np.array([]), np.array([])
        for ind_edge in range(len(edges)):
            val = valuations[ind_edge]
            prob_tables = np.concatenate( (prob_tables, val.prob.t))
            util_tables = np.concatenate( (util_tables, val.util.t))
        return np.concatenate( (prob_tables, util_tables) )

    def _box_util_bound(self, var_obj, edge):
        node_from, node_to = edge
        val_from = self.mg.message_graph.node[node_from]['fs'][0].copy()
        u_from = val_from.util/val_from.prob
        u_from = u_from.min( u_from.vars - {var_obj} )

        val_to = self.mg.message_graph.node[node_to]['fs'][0].copy()
        u_to = val_to.util/val_to.prob
        u_to = u_to.min( u_to.vars - {var_obj})

        lb = -u_to.t
        ub = u_from.t
        ub[lb >= ub] = lb[lb >= ub]
        return lb, ub

    def _cost_obj(self, x, var_obj, edges):
        valuations = self._array_to_val(var_obj, x, edges)
        objective = self.read_decomposed_bounds()
        if debug: print("\nbefore objective:{}".format(objective))

        if debug: print("\nevaluate:{}".format(x))
        for ind_edge, edge in enumerate(edges):
            val = valuations[ind_edge]
            node_from, node_to = edge
            if debug:
                print("\nold valuations at edge:{}".format(edge))
                print("node_from fs:{}".format(self.mg.message_graph.node[node_from]['fs'][0]))
                print("node_from bounds:{}".format(self.mg.message_graph.node[node_from]['bound']))
                print("node_to fs:{}".format(self.mg.message_graph.node[node_to]['fs'][0]))
                print("node_to bounds:{}".format(self.mg.message_graph.node[node_to]['bound']))

            self.mg.message_graph.edge[node_from][node_to]['cost'] = val
            temp1 = self.mg.message_graph.node[node_from]['fs'][0] /  val
            self.mg.set_factor(node_from,  temp1)
            temp2 = self.mg.message_graph.node[node_to]['fs'][0] * val
            self.mg.set_factor(node_to, temp2)
            self._update_bounds_at_node(node_from)
            self._update_bounds_at_node(node_to)
            if debug:
                print("\nnew valuations at edge:{}".format(edge))
                print("node_from fs:{}".format(self.mg.message_graph.node[node_from]['fs'][0]))
                print("node_from bounds:{}".format(self.mg.message_graph.node[node_from]['bound']))
                print("node_to fs:{}".format(self.mg.message_graph.node[node_to]['fs'][0]))
                print("node_to bounds:{}".format(self.mg.message_graph.node[node_to]['bound']))

        objective = self.read_decomposed_bounds()
        if debug:
            print("\nafter objective:{}".format(objective))

        # recover the original state
        for edge in edges:
            node_from, node_to = edge
            val = self.mg.message_graph.edge[node_from][node_to]['cost']
            temp1 = self.mg.message_graph.node[node_from]['fs'][0] * val
            self.mg.set_factor(node_from, temp1)
            temp2 = self.mg.message_graph.node[node_to]['fs'][0] / val
            self.mg.set_factor(node_to, temp2)

        for edge in edges:
            node_from, node_to = edge
            self._update_bounds_at_node(node_from)
            self._update_bounds_at_node(node_to)
            if debug:
                print("\nrecovered valuations at edge:{}".format(edge))
                print("node_from fs:{}".format(self.mg.message_graph.node[node_from]['fs'][0]))
                print("node_from bounds:{}".format(self.mg.message_graph.node[node_from]['bound']))
                print("node_to fs:{}".format(self.mg.message_graph.node[node_to]['fs'][0]))
                print("node_to bounds:{}".format(self.mg.message_graph.node[node_to]['bound']))
        return objective

    ####################################################################################################################
    def _optimize_weights(self, var_obj, nodes_at_the_current_layer):
        if debug:
            print("start updating weight at var {} on {}:".format(var_obj, nodes_at_the_current_layer))

        obj_0, obj_1 = None, self.read_decomposed_bounds()
        gd_updated, gd_steps, ls_steps, tol, ls_updated = False, 10, 30, TOL, False
        for s in range(gd_steps):
            gradient = self._eval_weight_gradients(var_obj, nodes_at_the_current_layer)
            abs_grad = map(abs, gradient)
            L0 = max(abs_grad)
            L1 = sum(abs_grad)
            L2 = sum(el * el for el in abs_grad)
            L2= np.sqrt(L2)
            self.grad_weight_l2 = L2
            if debug:
                print("\t\t\tGradient Norms L0={}\t\tL1={}\t\tL2={}".format(L0, L1, L2))
            if L0 < tol:
                if debug:
                    print("\t\t\t\tGradient Descent Terminated Too Small L0 < TOL at step {s}".format(s=s + 1))
                    print("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))
                return gd_updated

            step = min(1.0, 1.0 / L1) if obj_0 is None else min(1.0, 2 * (obj_0 - obj_1) / L1)
            step = step if step > 0 else 1.0
            if debug:
                print("\t\t\t\tGradient Descent with Line Search Initial Step Size {}".format(step))
                print("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))
            obj_0 = obj_1
            new_weights = self._line_search_weights(var_obj, nodes_at_the_current_layer, obj_0, gradient, step, L0, L2)
            ls_updated = False if new_weights is None else True
            if ls_updated:
                self._update_weights(var_obj, new_weights, nodes_at_the_current_layer)
            obj_1 = self.read_decomposed_bounds()

            gd_updated |= ls_updated
            if not ls_updated:  # no more line search, escape
                if debug:
                    print("\t\t\t\tGradient Descent Finished, line Search No Improvement")
                    print("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))
                return gd_updated
        else:   # no more gradient move
            if debug:
                print("\t\t\t\tGradient Descent Finished Iteration Limit {}".format(gd_steps))
                print("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))
        return gd_updated

    def _line_search_weights(self, var_obj, nodes, obj_0, gradient, step, L0, L2):
        ls_steps = 30
        armijo_thr = 1e-4
        armijo_step_back = 0.5
        ls_tol = TOL

        if debug:
            print("\t\t\t\t\t\tLine Search on nodes: {}".format(str(nodes)))

        for l in range(ls_steps):
            new_weights = self._normalize_weights(var_obj, nodes, gradient, step)
            obj_1 = self._weight_obj(new_weights, var_obj, nodes)
            if debug:
                print("\t\t\t\t\t\t\tLine search iteration step:{}".format(l + 1))
                print("\t\t\t\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))

            if obj_0 - obj_1 > step * armijo_thr * L2:
                if debug:
                    print("\t\t\t\t\t\t\tLine Search Improved Objective")
                return new_weights
            else:
                step *= armijo_step_back
                if step * L0 < ls_tol:
                    if debug:
                        print("\t\t\t\t\t\t\tLine Search Failed step*L0 < ls_tol")
                    return None

    def _normalize_weights(self, var_obj, nodes, gradient, step):
        nw_list = []
        for ind, n in enumerate(nodes):
            cw = self.mg.message_graph.node[n]['w'][var_obj.label]
            # self.mg.message_graph.node[n]['w_old'][v_w] = cw
            nw = cw * np.exp(-step * gradient[ind])
            if nw > WINF:
                nw = WINF
            if nw < WEPS:
                nw = WEPS
            # if not np.isfinite(nw):
            #     nw = WINF
            if debug:
                assert np.isfinite(nw)
            nw_list.append(nw)
        nw_tot = sum(nw_list)
        for ind, nw in enumerate(nw_list):
            nw_list[ind] = nw / nw_tot
        return nw_list

    def _eval_weight_gradients(self, var_obj, nodes):
        wgts, Hcond_u, Hcond_p = [], [], []
        for node in nodes:
            sc = self.mg.message_graph.node[node]['sc']
            mu_p = self.mg.message_graph.node[node]['pseudo_belief'].prob.marginal(sc)      # todo need pseudo belifs at each node
            mu_u = self.mg.message_graph.node[node]['pseudo_belief'].util.marginal(sc)
            elimvars_in_node = sorted(sc, key=lambda x: self.elim_order.index(x))
            v_w_ind = elimvars_in_node.index(var_obj.label)

            mu_p_temp1 = mu_p.marginal(elimvars_in_node[v_w_ind:])  # P(xi, xi+1~)
            if type(mu_p_temp1) is not Factor:
                mu_p_temp1 = Factor({}, mu_p_temp1)
            H1_p = mu_p_temp1.entropy()

            mu_u_temp1 = mu_u.marginal(elimvars_in_node[v_w_ind:])  # P(xi, xi+1~)
            if type(mu_u_temp1) is not Factor:
                mu_u_temp1 = Factor({}, mu_u_temp1)
            if np.all(mu_u_temp1.t == 0):  # when mu_u_temp1 is all zero factor, raise assertion error
                H1_u = 0.0
            else:
                H1_u = mu_u_temp1.entropy()

            if v_w_ind + 1 < len(elimvars_in_node):
                mu_p_temp2 = mu_p.marginal(elimvars_in_node[v_w_ind + 1:])  # P(xi+1 , xi+2 ~)
                if type(mu_p_temp2) is not Factor:
                    mu_p_temp2 = Factor({}, mu_p_temp2)
                H2_p = mu_p_temp2.entropy()

                mu_u_temp2 = mu_u.marginal(elimvars_in_node[v_w_ind + 1:])  # P(xi+1 , xi+2 ~)
                if type(mu_u_temp2) is not Factor:
                    mu_u_temp2 = Factor({}, mu_u_temp2)
                if np.all(mu_u_temp2.t == 0):
                    H2_u = 0.0
                else:
                    H2_u = mu_u_temp2.entropy()
            else:
                H2_p = 0.0
                H2_u = 0.0

            wgts.append(self.mg.message_graph.node[node]['w'][var_obj.label])
            H_p_cond = H1_p - H2_p
            H_u_cond = H1_u - H2_u
            Hcond_p.append(H_p_cond)  # conditional entropy
            Hcond_u.append(H_u_cond)  # conditional entropy

        Hbar_u = 0.0
        Hbar_p = 0.0
        for m, node in enumerate(nodes):
            Hbar_u += wgts[m] * Hcond_u[m] * self._eval_util_bound_norms(node)
            Hbar_p += wgts[m] * Hcond_p[m] * (self._eval_util_bound_norms() - self._eval_util_bound_norms(node))

        gradient = []
        for m, node in enumerate(nodes):
            grad = Hcond_u[m] * self._eval_util_bound_norms(node) - Hbar_u
            grad += Hcond_p[m] * (self._eval_util_bound_norms() - self._eval_util_bound_norms(node)) - Hbar_p
            grad *= wgts[m]
            grad *= self._eval_prob_bound_norms()       # todo pull out this term with objective?
            gradient.append(grad)

        if self.verbose:
            print('wgts:', wgts)
            print('Hcond_p:', Hcond_p)
            print('Hcond_u:', Hcond_p)
            print('Hbar_u:', Hbar_u)
            print('Hbar_p:', Hbar_p)
            print('grad:', gradient)

        if debug:
            assert np.all(np.isfinite(gradient))
        return gradient

    def _weight_obj(self, x, var_obj, nodes):
        # set new weights to evaluate objective
        for node_ind, node in enumerate(nodes):
            self.mg.message_graph.node[node]['w_old'][var_obj.label] = self.mg.message_graph.node[node]['w'][var_obj.label]
            self.mg.message_graph.node[node]['w'][var_obj.label] = x[node_ind]
            self._update_bounds_at_node(node)
        objective = self.read_decomposed_bounds()

        # recover the previous states
        for node_ind, node in enumerate(nodes):
            self.mg.message_graph.node[node]['w'][var_obj.label] = self.mg.message_graph.node[node]['w_old'][var_obj.label]
            self._update_bounds_at_node(node)
        return objective

    def _update_weights(self, var_obj, sol, nodes):
        for node_ind, node in enumerate(nodes):
            self.mg.message_graph.node[node]['w_old'][var_obj.label] = self.mg.message_graph.node[node]['w'][
                var_obj.label]
            self.mg.message_graph.node[node]['w'][var_obj.label] = sol[node_ind]
            self._update_bounds_at_node(node)
        objective = self.read_decomposed_bounds()
        return objective

    ####################################################################################################################
    def bounds(self):
        return self.read_decomposed_bounds()

    def _update_bounds_at_node(self, node):
        val = self.mg.message_graph.node[node]['fs'][0].copy()
        removable_vars = val.vars.labels
        for v in (el for el in self.elim_order if el in removable_vars):
            w_exponent = self.mg.message_graph.node[node]['w'][v]
            if w_exponent <= self.epsilon:
                # np.clip(val.util.t, a_min=ZERO, a_max=None, out=val.util.t)
                val = val.max([v])  # evaluate using max
            else:
                # np.clip(val.util.t, a_min=ZERO, a_max=None, out=val.util.t)
                val = val.abs()
                val = val.log()
                val = val.lsePower([v], ONE / w_exponent)
                val = val.exp()
        self.mg.message_graph.node[node]['bound'] = val
        return val

    def read_decomposed_bounds(self, nodes=[]):
        if not nodes:
            nodes = self.mg.message_graph.nodes()
        temp = Valuation(1.0, 0.0)
        for node in nodes: #self.mg.message_graph.nodes_iter():
            if node in self.processed_nodes:
                continue

            val = self.mg.message_graph.node[node]['bound']
            temp = temp * val  # combine bounds
            assert not np.isnan(temp.util)
        return temp.util

    def _eval_pseudo_belief_node(self, node):
        pr = self.mg.message_graph.node[node]['fs'][0].prob
        pr = pr.log()
        mu_p = self._eval_pseudo_belief(node, pr)

        eu = self.mg.message_graph.node[node]['fs'][0].util.copy()
        # np.clip(eu.t, a_min=ZERO, a_max=None, out=eu.t)
        eu = eu.abs()
        eu = eu.log()
        mu_u = self._eval_pseudo_belief(node, eu)

        if self.mg.message_graph.node[node]['pseudo_belief'] is None:
            self.mg.message_graph.node[node]['pseudo_belief'] = Valuation(mu_p, mu_u)
        else:
            self.mg.message_graph.node[node]['pseudo_belief'].prob = mu_p
            self.mg.message_graph.node[node]['pseudo_belief'].util = mu_u

    def _eval_pseudo_belief(self, node, lnZ0):
        removable_vars = [el for el in self.elim_order if el in lnZ0.vars.labels]
        lnmu = 0.0
        for v in (el for el in self.elim_order if el in removable_vars):
            if self.mg.message_graph.node[node]['w'][v] <= self.epsilon:
                w_exponent = self.epsilon
            else:
                w_exponent = self.mg.message_graph.node[node]['w'][v]

            if w_exponent <= self.epsilon:
                lnZ1 = lnZ0.lsePower([v], np.inf)  # np.inf will evaluate max
            else:
                lnZ1 = lnZ0.lsePower([v], ONE / w_exponent)  # pass 1/w as power to lsePower() it takes inv weight
            lnZ0 -= lnZ1  # this is point where inf became nan -inf - -inf
            lnZ0.t[np.isnan(lnZ0.t)] = -np.inf  # todo here pseudo marginal
            lnZ0 *= ONE / w_exponent  # this number is big when w_exponent is small, underflow unavoidable
            lnmu += lnZ0
            lnZ0 = lnZ1
        mu = lnmu.exp()
        return mu


    def _eval_util_bound_norms(self, node=None):
        if node:
            return self.mg.message_graph.node[node]['bound'].util / self.mg.message_graph.node[node]['bound'].prob
        else:
            u_bound = 0.0
            for node in self.mg.message_graph.nodes_iter():
                u_bound += self.mg.message_graph.node[node]['bound'].util / self.mg.message_graph.node[node]['bound'].prob
            return u_bound

    def _eval_prob_bound_norms(self, node=None):
        if node:
            return self.mg.message_graph.node[node]['bound'].prob
        else:
            p_bound_log = 0.0
            for node in self.mg.message_graph.nodes_iter():
                p_bound_log += np.log(self.mg.message_graph.node[node]['bound'].prob)
            return np.exp(p_bound_log)