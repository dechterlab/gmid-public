from constants import *
from pyGM.factor import Factor
from valuation import Valuation
from message_passing import MessagePassing
from graph_algorithms import toposort2
import random
import time

class GddIdHingeShift(MessagePassing):
    def __init__(self, verbose_level, message_graph, elim_order, weights, is_log, epsilon, log_file_name):
        super(GddIdHingeShift, self).__init__(verbose_level, message_graph, elim_order, weights, is_log, log_file_name)
        self.epsilon = epsilon  # WEPS smallest weight
        self.old_eu = np.inf
        self.new_eu = np.inf
        self.prob_const = 1.0  # accumulated product of max of prob functions in linear scale
        self.util_const = 0.0
        self.weight_schedule = None
        self.cost_schedule = None
        self.message_graph_nodes = None
        self.sum_vars = None
        self.best_expected_utility = np.inf
        self.time_0 = None
        self.total_diff = np.inf
        self.grad_weight_l2 = np.inf
        self.grad_cost_l2 = np.inf
        self.grad_shift_l2 = np.inf

    ####################################################################################################################
    def schedule(self):
        self.weight_schedule = self._schedule_per_var() # list of lists of nodes
        self.cost_schedule = self._schedule_mbe()   # list of edges; follow mbe tree

    def _schedule_per_var(self):
        """ a list of list of nodes, each list of nodes associated with a variable following elim order """
        # sg = self.mg.region_graph.copy()      #   no mutation
        sg = self.mg.region_graph
        var_elim_schedule = []
        for v in self.elim_order:
            nodes_with_v = [node for node in sg.nodes_iter() if v in sg.node[node]['sc']]
            var_elim_schedule.append(nodes_with_v)
        return var_elim_schedule

    def _schedule_random_edges_from_region(self):
        self.message_graph_nodes = self.mg.region_graph.nodes()
        return sorted(self.mg.region_graph.edges(), key=lambda x: random.randint())

    def _schedule_sorted_edges_from_region(self):
        self.message_graph_nodes = self.mg.region_graph.nodes()
        return sorted(self.mg.region_graph.edges())

    def _schedule_mbe(self):
        """ schedule of edges following the directions in mbe tree """
        sg = self.mg.region_graph.copy().to_directed()  # copy undirected graph as directed graph
        for u, v in sg.edges():  # make a DAG, arrows following elim order
            if self.elim_order.index(u[0]) > self.elim_order.index(v[0]):
                sg.remove_edge(u, v)
            elif self.elim_order.index(u[0]) == \
                    self.elim_order.index(v[0]) and self.elim_order.index(u[1]) > self.elim_order.index(v[1]):
                sg.remove_edge(u, v)
        node_parents = {node: set([pa for pa, me in sg.in_edges([node])]) for node in sg.nodes()}
        mbe_schedule = []
        # for nodes in reversed(list(toposort2(node_parents))):   # reverse ve order
        for nodes in list(toposort2(node_parents)):  # reverse ve order
            # mbe_schedule.append(sorted(nodes, key=lambda x: (self.elim_order.index(x[0]), x[1]), reverse=True))
            mbe_schedule.append(sorted(nodes, key=lambda x: (self.elim_order.index(x[0]), x[1]), reverse=False))

        self.message_graph_nodes = []
        for nodes in mbe_schedule:
            self.message_graph_nodes.extend(nodes)
        self.message_graph_nodes.reverse()

        # given a node, find source of in-edges
        jg_edge_schedule = []
        for nodes in mbe_schedule:
            for node in nodes:
                for pa, me in sg.in_edges([node]):
                    # jg_edge_schedule.append((me, pa))
                    jg_edge_schedule.append((pa, me))
        return jg_edge_schedule

    ####################################################################################################################
    def init_propagate(self):
        self.mg.init_msg_propagation_graph('undirected')        # copy message graph mg from rg
        for n in self.mg.message_graph.nodes_iter():
            combined_factor = self.mg.combine_factors(n, None, self.is_log)
            self.mg.set_factor(n, combined_factor)
            # self.mg.set_factor_rg(n, combined_factor.copy())      # no need to copy at region graph here gdd don't use rg at all

        self.sum_vars = [v for v in self.elim_order if self.weights[v] > 0]
        self._init_uniform_weight()

    def _init_uniform_weight(self):
        for v, weight in enumerate(self.weights):
            if v not in self.sum_vars:
                continue
            count = 0
            tot = 0.0
            for n in self.mg.message_graph.nodes_iter():
                if v in self.mg.message_graph.node[n]['sc']:
                    count += 1
            for n in self.mg.message_graph.nodes_iter():
                if v in self.mg.message_graph.node[n]['sc']:
                    self.mg.message_graph.node[n]['w'][v] = float(weight)/count
                    tot += self.mg.message_graph.node[n]['w'][v]
            for n in self.mg.message_graph.nodes_iter():
                if v in self.mg.message_graph.node[n]['sc']:
                    self.mg.message_graph.node[n]['w'][v] = self.mg.message_graph.node[n]['w'][v]/tot
                    self.mg.message_graph.node[n]['w_old'][v] = self.mg.message_graph.node[n]['w'][v]
                    if self.verbose:
                        self.print_log("_init_uniform_weight")
                        self.print_log("node:{n}, var:{v}, wgt:{w}".format(n=n, v=v, w=str(self.mg.message_graph.node[n]['w'][v])))

    def _init_pseudo_beliefs(self):
        for node in self.mg.message_graph.nodes_iter():
            self._eval_pseudo_belief_node(node)

    def _extract_prob_max(self):
        for n in self.mg.message_graph.nodes_iter():
            p = self.mg.message_graph.node[n]['fs'][0].prob
            eu = self.mg.message_graph.node[n]['fs'][0].util
            u = eu/p
            m = np.max(p.t)
            self.prob_const *= m    # in linear
            p.t = p.t/m             # subtract m from prob table
            eu.t = u.t*p.t          # adjust changes to the expected utility

    ####################################################################################################################
    # evaluations related to lp norms
    ####################################################################################################################
    def _eval_pseudo_belief_node(self, node):
        pr = self.mg.message_graph.node[node]['fs'][0].prob
        pr = pr.log()
        mu_p = self._eval_pseudo_belief(node, pr)

        eu = self.mg.message_graph.node[node]['fs'][0].util.copy()
        np.clip(eu.t, a_min=ZERO, a_max=None, out=eu.t)
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
            w_exponent = self.epsilon if self.mg.message_graph.node[node]['w'][v] <= self.epsilon else \
            self.mg.message_graph.node[node]['w'][v]
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

    ####################################################################################################################
    def bounds(self, changed_nodes=[]):
        bounds_at_nodes = []
        for node in self.message_graph_nodes:
            if node in changed_nodes:   # recompute bounds for relevant nodes
                combined_valuation = self.mg.message_graph.node[node]['fs'][0].copy()
                if debug:
                    assert np.all(np.isfinite(combined_valuation.prob.t))
                    assert np.all(np.isfinite(combined_valuation.util.t))

                removable_vars = combined_valuation.vars.labels
                for v in (el for el in self.elim_order if el in removable_vars):
                    w_exponent = self.mg.message_graph.node[node]['w'][v]
                    if debug:
                        assert np.isfinite(w_exponent)
                    if w_exponent <= self.epsilon:
                        np.clip(combined_valuation.util.t, a_min=ZERO, a_max=None, out=combined_valuation.util.t)
                        combined_valuation = combined_valuation.max([v])   # evaluate using max
                    # elif (1-self.epsilon)  <= w_exponent:
                    #     np.clip(combined_valuation.util.t, a_min=ZERO, a_max=None, out=combined_valuation.util.t)
                    #     combined_valuation = combined_valuation.sum([v])    # evaluate using sum
                    #     if debug:
                    #         if type(combined_valuation.prob) is Factor :
                    #             assert np.all(np.isfinite(combined_valuation.prob.t))
                    #         if type(combined_valuation.util) is Factor:
                    #             assert np.all(np.isfinite(combined_valuation.util.t))
                    else:
                        np.clip(combined_valuation.util.t, a_min=ZERO, a_max=None, out=combined_valuation.util.t)
                        combined_valuation = combined_valuation.log()
                        combined_valuation = combined_valuation.lsePower([v], ONE / w_exponent)
                        combined_valuation = combined_valuation.exp()
                if debug:
                    assert np.all(np.isfinite(combined_valuation.prob))
                    assert np.all(np.isfinite(combined_valuation.util))

                bounds_at_nodes.append(combined_valuation)
                self.mg.message_graph.node[node]['bound'] = combined_valuation
            else:
                bounds_at_nodes.append(self.mg.message_graph.node[node]['bound']) # read most recent bound
        return bounds_at_nodes  # elements are valuations

    def eu_from_bounds(self, bounds, scale_prob_max=True):
        temp = Valuation(1.0, 0.0) # bounds[0].copy()  # first valuation
        for el in bounds:
            temp = temp * el    # combine bounds
        if scale_prob_max:
            return temp.util * self.prob_const  # result is in linear scale
        else:
            return temp.util

    def util_shifted_bounds(self, bounds):
        return self.eu_from_bounds(bounds, scale_prob_max=True) - self.util_const

    def _eval_util_bound_norms(self, node=None):
        if node:
            return self.mg.message_graph.node[node]['bound'].util / self.mg.message_graph.node[node]['bound'].prob
        else:
            u_bound = 0.0
            for node in self.message_graph_nodes:
                u_bound += self.mg.message_graph.node[node]['bound'].util / self.mg.message_graph.node[node]['bound'].prob
            return u_bound

    def _eval_prob_bound_norms(self, node=None):
        if node:
            return self.mg.message_graph.node[node]['bound'].prob
        else:
            p_bound_log = 0.0
            for node in self.message_graph_nodes:
                p_bound_log += np.log(self.mg.message_graph.node[node]['bound'].prob)
            return np.exp(p_bound_log)

    ####################################################################################################################
    # prints
    ####################################################################################################################
    def print_bounds(self):
        self.print_log("summary local bounds")
        bounds_at_nodes = self.bounds(self.mg.message_graph.nodes())
        for ind, node in enumerate(self.message_graph_nodes):
            self.print_log("node:{}, bound:{}".format(str(node), str(bounds_at_nodes[ind])))
        self.print_log("prob scale pulled out:{}".format(self.prob_const))
        self.print_log("eu from bound={}".format(self.eu_from_bounds(bounds_at_nodes)))
        self.print_log("util shifted bound={}".format(self.util_shifted_bounds(bounds_at_nodes)))

    def print_weights(self):
        self.print_log("summary weights")
        for node in sorted(self.mg.message_graph.nodes_iter()):
            for v_w in sorted(self.mg.message_graph.node[node]['w']):
                self.print_log(
                    "node:{}, var:{}, w:{}".format(node, v_w, self.mg.message_graph.node[node]['w'][v_w]))

    def print_prob_const(self):
        self.print_log("prob const pulled out:{}".format(self.prob_const))

    def print_util_const(self):
        self.print_log("summary util consts")
        self.print_log("util const pulled out:{}".format(self.util_const))
        for node in sorted(self.mg.message_graph.nodes_iter()):
            self.print_log("node:{}, util const:{}".format(node, self.mg.message_graph.node[node]['util_const']))


    ####################################################################################################################
    def propagate(self, time_limit, iter_limit, cost_options, weight_options, util_options):
        self.time_0 = time.time()
        self.time_limit = time_limit
        self.print_weights()
        self.util_const_updated_count = 0
        for iter in range(iter_limit):
            self._extract_prob_max()
            self._init_pseudo_beliefs()
            self.total_diff = self._propagate_one_pass(iter, cost_options, weight_options, util_options)
            if (iter + 1) % 10 == 0:
                self.print_weights()
                self.print_bounds()
                self.print_util_const()
            if self.util_const_updated_count > 1 and iter > 100 and (self.total_diff < TOL):
                print("gdd terminating due to diff:{}".format(self.total_diff))
                break
            if self.util_const_updated_count > 1 and iter > 100 and (time.time() - self.time_0 > self.time_limit):
                print("gdd terminating due to time_limit:{}".format(self.time_limit))
                break
        else:
            print("gdd terminating due to iter_limit:{}".format(iter_limit))

        self.print_weights()
        self.print_bounds()
        self.print_util_const()
        self.print_log("best expected utility so far:{}".format(self.best_expected_utility))
        return self.best_expected_utility

    ####################################################################################################################
    def _propagate_one_pass(self, it, cost_options, weight_options, util_options):
        self.new_eu = self.util_shifted_bounds(self.bounds(self.mg.message_graph.nodes()))
        one_pass_init = self.new_eu
        if it >= 0:
            lg_str = "iter:{}\tbefore update:\ttime={:.2f}\tlin={:2.12f}"
            self.print_log(lg_str.format(it + 1, time.time() - self.time_0, self.new_eu))

        #### weight update
        self.old_eu = self.new_eu
        for ind, nodes in enumerate(self.weight_schedule):
            v_w = self.elim_order[ind]
            if v_w not in self.sum_vars or len(nodes) <= 1:
                continue
            self._update_weights_per_var(v_w, nodes, weight_options)
        self.new_eu = self.util_shifted_bounds(self.bounds(self.mg.message_graph.nodes()))
        if self.best_expected_utility > self.new_eu:
            self.best_expected_utility = self.new_eu
        if it >= 0:
            lg_str = "\t\tweight update:\ttime={:.2f}\tlin={:2.12f}\tdiff={:+2.12f}\tdecreased={}"
            diff = self.new_eu - self.old_eu
            self.print_log(lg_str.format(time.time() - self.time_0, self.new_eu, diff, diff <= 0))

        #### cost update
        self.old_eu = self.new_eu
        for edge in self.cost_schedule:
            self._update_cost_per_edge(edge, cost_options)
        self.new_eu = self.util_shifted_bounds(self.bounds(self.mg.message_graph.nodes()))
        if self.best_expected_utility > self.new_eu:
            self.best_expected_utility = self.new_eu
        if it >= 0:
            lg_str = "\t\tcost update:\ttime={:.2f}\tlin={:2.12f}\tdiff={:+2.12f}\tdecreased={}"
            diff = self.new_eu - self.old_eu
            self.print_log(lg_str.format(time.time() - self.time_0, self.new_eu, diff, diff <= 0))

        ### util const update
        if it < 0 or (self.total_diff < 1e-1):# or (it > 50):
            self.util_const_updated_count += 1
            self.old_eu = self.new_eu
            for node in sorted(self.mg.message_graph.nodes_iter(), key=lambda x: self.elim_order.index(x[0]),
                               reverse=True):
                self.update_util_const_per_node(node, util_options)
            # self.update_util_consts_for_all(util_options)
            self.new_eu = self.util_shifted_bounds(self.bounds(self.mg.message_graph.nodes()))
            if self.best_expected_utility > self.new_eu:
                self.best_expected_utility = self.new_eu
            if it >= 0:
                lg_str = "\t\tutil update:\ttime={:.2f}\tlin={:2.12f}\tdiff={:+2.12f}\tdecreased={}"
                diff = self.new_eu - self.old_eu
                self.print_log(lg_str.format(time.time() - self.time_0, self.new_eu, diff, diff <= 0))

        #### finish one pass
        one_pass_end = self.new_eu
        self.total_diff = one_pass_init - one_pass_end
        if it >= 0:
            lg_str = "\t\ttotal diff={:2.12f}"
            self.print_log(lg_str.format(self.total_diff))
        return self.total_diff


    ####################################################################################################################
    # weight optimization; exponentiated gradient descent
    ####################################################################################################################
    def _update_weights_per_var(self, v_w, nodes, options):
        if self.terse:
            self.print_log("\n\t[Updating weights var={}\tnodes:{}]".format(v_w, str(nodes)))
        gd_steps = options['gd_steps']
        tol = options['tol']
        obj_0, obj_1 = None, self._obj_weight_per_var(nodes)

        ################################################################################################################
        gd_updated = False
        for s in range(gd_steps):
            if self.terse:
                self.print_log("\t\tGradient Descent iteration step:{s}".format(s=s + 1))

            gradient = self._eval_weight_gradients_per_var(v_w, nodes)
            abs_grad = map(abs, gradient)
            L0 = max(abs_grad)
            L1 = sum(abs_grad)
            L2 = sum(el * el for el in abs_grad)
            L2= np.sqrt(L2)
            self.grad_weight_l2 = L2
            if self.terse:
                self.print_log("\t\t\tGradient Norms L0={}\t\tL1={}\t\tL2={}".format(L0, L1, L2))
            if L0 < tol:
                if self.terse:
                    self.print_log("\t\t\t\tGradient Descent Terminated Too Small L0 < TOL at step {s}".format(s=s + 1))
                    self.print_log("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))
                return gd_updated

            step = min(1.0, 1.0 / L1) if obj_0 is None else min(1.0, 2 * (obj_0 - obj_1) / L1)
            step = step if step > 0 else 1.0
            if self.terse:
                self.print_log("\t\t\t\tGradient Descent with Line Search Initial Step Size {}".format(step))
                self.print_log("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))
            obj_0 = obj_1
            ls_updated = self._line_search_weights(v_w, nodes, obj_0, gradient, step, L0, L2, options)
            obj_1 = self._obj_weight_per_var(nodes)
            if not gd_updated:
                gd_updated = ls_updated
            if not ls_updated:
                if self.terse:
                    self.print_log("\t\t\t\tGradient Descent Finished, line Search No Improvement")
                    self.print_log("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))
                return gd_updated
        else:
            if self.terse:
                self.print_log("\t\t\t\tGradient Descent Finished Iteration Limit {}".format(gd_steps))
                self.print_log("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))
        return gd_updated

    def _line_search_weights(self, v_w, nodes, obj_0, gradient, step, L0, L2, options):
        ls_steps = options['ls_steps']
        armijo_thr = options['armijo_thr']
        armijo_step_back = options['armijo_step_back']
        ls_tol = options['ls_tol']

        if self.terse:
            self.print_log("\t\t\t\t\t\tLine Search on nodes: {}".format(str(nodes)))

        for l in range(ls_steps):
            self._set_weights_per_var(v_w, nodes, gradient, step)
            obj_1 = self._obj_weight_per_var(nodes)

            if self.verbose:
                self.print_log("\t\t\t\t\t\t\tLine search iteration step:{}".format(l + 1))
                self.print_log("\t\t\t\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))

            if obj_0 - obj_1 > step * armijo_thr * L2:
                for n in nodes:
                    self._eval_pseudo_belief_node(n)
                if self.verbose:
                    self.print_log("\t\t\t\t\t\t\tLine Search Improved Objective")
                return True
            else:
                self._reset_weights_per_var(v_w, nodes)
                step *= armijo_step_back
                if step * L0 < ls_tol:
                    if self.verbose:
                        self.print_log("\t\t\t\t\t\t\tLine Search Failed step*L0 < ls_tol")
                    return False

    def _obj_weight_per_var(self, nodes):
        bound = self.eu_from_bounds(self.bounds(nodes), scale_prob_max=False)
        return bound  # self.eu_from_bounds(self.bounds(nodes), scale_prob_max=False)

    def _eval_weight_gradients_per_var(self, v_w, nodes):
        wgts = []
        Hcond_u = []
        Hcond_p = []

        for node in nodes:
            sc = self.mg.message_graph.node[node]['sc']
            mu_p = self.mg.message_graph.node[node]['pseudo_belief'].prob.marginal(sc)
            mu_u = self.mg.message_graph.node[node]['pseudo_belief'].util.marginal(sc)
            elimvars_in_node = sorted(sc, key=lambda x: self.elim_order.index(x))
            v_w_ind = elimvars_in_node.index(v_w)

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

            wgts.append(self.mg.message_graph.node[node]['w'][v_w])
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

    def _assign_weights_per_var(self, v_w, nodes, weights):
        if debug:
            assert np.all(np.isfinite(weights))

        for ind, n in enumerate(nodes):
            current_weight = self.mg.message_graph.node[n]['w'][v_w]
            self.mg.message_graph.node[n]['w_old'][v_w] = current_weight
            self.mg.message_graph.node[n]['w'][v_w] = weights[ind]

    def _set_weights_per_var(self, v_w, nodes, gradient, step):
        nw_list = []
        for ind, n in enumerate(nodes):
            cw = self.mg.message_graph.node[n]['w'][v_w]
            self.mg.message_graph.node[n]['w_old'][v_w] = cw
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
        for ind, n in enumerate(nodes):
            self.mg.message_graph.node[n]['w'][v_w] = nw_list[ind] / nw_tot     # avoid inf / inf

    def _reset_weights_per_var(self, v_w, nodes):
        for n in nodes:
            self.mg.message_graph.node[n]['w'][v_w] = self.mg.message_graph.node[n]['w_old'][v_w]

    ####################################################################################################################
    # cost optimization; simple gradient descent
    ####################################################################################################################
    def _update_cost_per_edge(self, edge, options):
        if self.terse:
            self.print_log("\n\t[Updating cost\tnodes:{}]".format(str(edge)))
        gd_steps = options['gd_steps']
        tol = options['tol']
        node_from = edge[0]; node_to = edge[1]
        obj_0, obj_1 = None, self._obj_cost(node_from, node_to)
        if debug:
            assert np.isfinite(obj_1), "objective is not finite!"
        ################################################################################################################
        gd_updated = False
        for s in range(gd_steps):
            if self.terse:
                self.print_log("\t\tGradient Descent iteration step:{s}".format(s=s+1))

            #### eval gradient
            prob_gradient = self._eval_prob_gradients(node_from, node_to)
            util_gradient = self._eval_util_gradients(node_from, node_to)

            abs_gradient = prob_gradient.abs()
            L0_prob = abs_gradient.max()
            L1_prob = abs_gradient.sum()
            L2_prob = (abs_gradient*abs_gradient).sum()

            abs_gradient = util_gradient.abs()
            L0_util = abs_gradient.max()
            L1_util = abs_gradient.sum()
            L2_util = (abs_gradient*abs_gradient).sum()

            L0 = max(L0_util, L0_prob)
            L1 = L1_util + L1_prob
            L2 = L2_util + L2_prob
            L2 = np.sqrt(L2)

            self.grad_cost_l2 = L2

            if self.terse:
                self.print_log("\t\t\tGradient Norms L0={}\t\tL1={}\t\tL2={}".format(L0, L1, L2))

            if L0 < tol:
                if self.terse:
                    self.print_log("\t\t\t\tGradient Descent Finished, L0 < tol at:{s}".format(s=s+1))
                    self.print_log("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0>=obj_1))
                return gd_updated

            #### line search
            step = min(1.0, 1.0 / L1) if obj_0 is None else min(1.0, 2 * (obj_0 - obj_1) / L1)
            step = step if step > 0 else 1.0
            # step = 1.0
            if self.terse:
                self.print_log("\t\t\t\tGradient Descent with Line Search Initial Step Size {}".format(step))
                self.print_log("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0 >= obj_1))

            obj_0 = obj_1
            ls_updated = self._line_search_cost(node_from, node_to, obj_0, util_gradient, prob_gradient, step, L0, L2, options)
            obj_1 = self._obj_cost(node_from, node_to)
            if not gd_updated and ls_updated:
                gd_updated = True

            if not ls_updated: # no improvement from current gradient [perturb? or terminiate current gd iteration]
                if self.terse:
                    self.print_log("\t\t\t\tGradient Descent Finished, line Search No Improvement")
                    self.print_log("\tedge{:<30}{:<6}\tgd step:{}\tdiff:{:e}\t\tL0:{:e}\tL2:{:e}".format([node_from, node_to], 'L', s + 1, obj_1 - obj_0, L0, L2))
                    self.print_log("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0>=obj_1))
                return gd_updated
        else: # gd iteration limit
            if self.terse:
                self.print_log("\t\t\t\tGradient Descent Finished Iteration Limit {}".format(gd_steps))
                self.print_log("\tedge{:<30}{:<6}\tgd step:{}\tdiff:{:e}\t\tL0:{:e}\tL2:{:e}".format([node_from, node_to], 'T', s + 1, obj_1 - obj_0, L0, L2))
                self.print_log("\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0>=obj_1))
        return gd_updated

    def _obj_cost(self, node_from, node_to):
        bound = self.eu_from_bounds(self.bounds(changed_nodes=[node_from, node_to]), scale_prob_max=False)    # gradient ignores prob_max
        if debug:
            assert np.isfinite(bound), "bound should be finite"
        return bound  # compute total expected utility divided by scale [excluded by gradient computation]

    def _eval_util_gradients(self, node_from, node_to):
        eu_from = self.mg.message_graph.node[node_from]['bound'].util
        eu_to = self.mg.message_graph.node[node_to]['bound'].util
        prob_bound_tot = self._eval_prob_bound_norms()

        sc = self.mg.message_graph.edge[node_from][node_to]['sc']
        pm_eu_from = self.mg.message_graph.node[node_from]['pseudo_belief'].util
        pm_eu_to = self.mg.message_graph.node[node_to]['pseudo_belief'].util

        eucomp_from = self.mg.message_graph.node[node_from]['fs'][0].util
        pcomp_from =  self.mg.message_graph.node[node_from]['fs'][0].prob
        eucomp_to = self.mg.message_graph.node[node_to]['fs'][0].util
        pcomp_to =  self.mg.message_graph.node[node_to]['fs'][0].prob

        if eu_to == 0 and eu_from == 0: # both zero no update
            temp = pm_eu_to.marginal(sc)
            gradient = Factor(temp.v, 0.0)
        elif eu_to == 0: # eu_to zero A = 0
            neg_eu_position = eucomp_from.t <= 0.0
            B = pm_eu_from * pcomp_from / eucomp_from
            B.t[neg_eu_position] = 0.0
            finite_positions = np.isfinite(B.t)
            B.t[~finite_positions] = 0.0
            B = B.marginal(sc)
            B *= self._eval_util_bound_norms(node_from)
            gradient = -B   # a unit of cost
        elif eu_from == 0:   # eu_from zero B = 0
            neg_eu_position = eucomp_to.t <= 0.0
            A = pm_eu_to * pcomp_to / eucomp_to
            A.t[neg_eu_position] = 0.0
            finite_positions = np.isfinite(A.t)
            A.t[~finite_positions] = 0.0
            A = A.marginal(sc)
            A *= self._eval_util_bound_norms(node_to)
            gradient = A  # a unit of cost
        else:   # both non-zero
            neg_eu_position = eucomp_to.t <= 0.0
            A = pm_eu_to * pcomp_to / eucomp_to
            A.t[neg_eu_position] = 0.0
            finite_positions = np.isfinite(A.t)
            A.t[~finite_positions] = 0.0
            A = A.marginal(sc)
            A *= self._eval_util_bound_norms(node_to)

            neg_eu_position = eucomp_from.t <= 0.0
            B = pm_eu_from * pcomp_from / eucomp_from
            B.t[neg_eu_position] = 0.0
            finite_positions = np.isfinite(B.t)
            B.t[~finite_positions] = 0.0
            B = B.marginal(sc)
            B *= self._eval_util_bound_norms(node_from)
            gradient = A - B
        return gradient * prob_bound_tot  # negative of gradient

    def _eval_prob_gradients(self, node_from, node_to):
        sc = self.mg.message_graph.edge[node_from][node_to]['sc']
        pm_p_from = self.mg.message_graph.node[node_from]['pseudo_belief'].prob
        pm_eu_from = self.mg.message_graph.node[node_from]['pseudo_belief'].util
        pm_p_to = self.mg.message_graph.node[node_to]['pseudo_belief'].prob
        pm_eu_to = self.mg.message_graph.node[node_to]['pseudo_belief'].util

        A = pm_eu_to.marginal(sc) - pm_p_from.marginal(sc)
        A *= self._eval_util_bound_norms(node_to)
        B = pm_p_to.marginal(sc) - pm_eu_from.marginal(sc)
        B *= self._eval_util_bound_norms(node_from)
        C = pm_p_to.marginal(sc) - pm_p_from.marginal(sc)
        C *= (self._eval_util_bound_norms() - self._eval_util_bound_norms(node_to) - self._eval_util_bound_norms(node_from))

        gradient = A + B + C
        return self._eval_prob_bound_norms() * gradient

    def _line_search_cost(self, node_from, node_to, obj_0, util_gradient, prob_gradient, step, L0, L2, options):
        ls_steps = options['ls_steps']
        armijo_thr = options['armijo_thr']
        armijo_step_back = options['armijo_step_back']
        ls_tol = options['ls_tol']

        if self.terse:
            self.print_log("\t\t\t\t\t\tLine Search on nodes: {}".format(str([node_from, node_to])))

        for l in range(ls_steps):
            self._set_cost(node_from, node_to, prob_gradient, util_gradient, step)
            obj_1 = self._obj_cost(node_from, node_to)

            if self.verbose:
                self.print_log("\t\t\t\t\t\t\tLine search iteration step:{}".format(l+1))
                self.print_log("\t\t\t\t\t\t\t\tf0={}\t\tf1={}\t\tdec:{}".format(obj_0, obj_1, obj_0>=obj_1))

            if obj_0 - obj_1 > step*armijo_thr*L2:
                self._eval_pseudo_belief_node(node_from)
                self._eval_pseudo_belief_node(node_to)
                if self.verbose:
                    self.print_log("\t\t\t\t\t\t\tLine Search Improved Objective")
                return True
            else:
                self._reset_cost(node_from, node_to)
                step *= armijo_step_back
                if step*L0 < ls_tol:
                    if self.terse:
                        print('line search step too small step:{} obj_1:{} obj_0:{} diff:{}'.format(step, obj_1, obj_0, obj_1-obj_0))
                    if self.verbose:
                        self.print_log("\t\t\t\t\t\t\tLine Search Failed step*L0 < ls_tol")
                    return False
        else:
            if self.terse:
                print('line search step limit step:{} obj_1:{} obj_0:{} diff:{}'.format(step, obj_1, obj_0, obj_1 - obj_0))
            if self.verbose:
                self.print_log("\t\t\t\t\t\t\tLine Search Failed iteration limit {}".format(ls_steps))
            return False

    def _set_cost(self, node_from, node_to, prob_gradient, util_gradient, step):
        prob_shift = -step*prob_gradient
        util_shift = -step*util_gradient
        va_shift = Valuation(prob_shift.exp(), util_shift)
        np.clip(va_shift.prob.t, a_min=TOL, a_max=None, out=va_shift.prob.t)  # fix numerical error cannot div by zero
        va_from = self.mg.message_graph.node[node_from]['fs'][0]
        va_to = self.mg.message_graph.node[node_to]['fs'][0]
        self.mg.message_graph.node[node_from]['fs_old'] = va_from
        self.mg.message_graph.node[node_to]['fs_old'] = va_to
        self.mg.set_factor(node_from, va_from/va_shift)
        self.mg.set_factor(node_to, va_to*va_shift)

    def _set_cost_by_shift(self, node_from, node_to, va_shift):
        va_from = self.mg.message_graph.node[node_from]['fs'][0]
        va_to = self.mg.message_graph.node[node_to]['fs'][0]
        self.mg.message_graph.node[node_from]['fs_old'] = va_from
        self.mg.message_graph.node[node_to]['fs_old'] = va_to
        self.mg.set_factor(node_from, va_from/va_shift)
        self.mg.set_factor(node_to, va_to*va_shift)

    def _reset_cost(self, node_from, node_to):
        # self.mg.set_factor(node_from, self.mg.message_graph.node[node_from]['fs_old'].copy())
        self.mg.set_factor(node_from, self.mg.message_graph.node[node_from]['fs_old'])
        # self.mg.set_factor(node_to, self.mg.message_graph.node[node_to]['fs_old'].copy())
        self.mg.set_factor(node_to, self.mg.message_graph.node[node_to]['fs_old'])
        self.mg.message_graph.node[node_from]['fs_old'] = None
        self.mg.message_graph.node[node_to]['fs_old'] = None

    ####################################################################################################################
    # util optimization; simple gradient descent
    ####################################################################################################################
    def _obj_util_const(self, node):
        return self.util_shifted_bounds(self.bounds([node]))

    def update_util_const_per_node(self, node, options):
        obj_0, obj_1 = None, self._obj_util_const(node)
        if debug:
            assert np.isfinite(obj_1), "objective is not finite!"
        gd_updated = False
        for s in range(options['gd_steps']):
            util_gradient = self._eval_util_const_gradient(node)
            L0 = L1 = L2 = abs(util_gradient)
            self.grad_shift_l2 = L2
            if L0 < options['tol']:
                return gd_updated
            step = min(1.0, 1.0 / L1) if obj_0 is None else min(1.0, 2 * (obj_0 - obj_1) / L1)
            step = step if step > 0 else 1.0
            obj_0 = obj_1
            ls_updated = self._line_search_util_const(node, obj_0, util_gradient, step, L0, L2, options)
            obj_1 = self._obj_util_const(node)
            if not gd_updated and ls_updated:
                gd_updated = True
            if not ls_updated:
                return gd_updated
        else:
            return gd_updated

    def _eval_util_const_gradient(self, node):
        prob = self.mg.message_graph.node[node]['fs'][0].prob
        eu = self.mg.message_graph.node[node]['fs'][0].util
        pm_eu = self.mg.message_graph.node[node]['pseudo_belief'].util

        neg_eu_position = eu.t <= 0.0
        A = pm_eu * prob / eu           # does the scope agree yes
        A.t[neg_eu_position] = 0.0
        finite_positions = np.isfinite(A.t)
        A.t[~finite_positions] = 0.0
        A = A.marginal([])              # scalar, marginalize all variables
        gradient = self.prob_const* A * self._eval_prob_bound_norms() * self._eval_util_bound_norms(node)
        gradient -= 1.0
        return gradient

    def _line_search_util_const(self, node, obj_0, util_gradient, step, L0, L2, options):
        for l in range(options['ls_steps']):
            self._set_util_const(node, util_gradient, step)
            obj_1 = self._obj_util_const(node)
            if obj_0 - obj_1 > step * options['armijo_thr'] * L2:
                self._eval_pseudo_belief_node(node)
                return True
            else:
                self._reset_util_const(node)
                step *= options['armijo_step_back']
                if step * L0 < options['ls_tol']:
                    return False
        else:
            return False

    def _set_util_const(self, node, util_gradient, step):
        util_const_shift = -step * util_gradient
        self.mg.message_graph.node[node]['util_const_old'] = self.mg.message_graph.node[node]['util_const']
        self.mg.message_graph.node[node]['util_const'] += util_const_shift
        self.mg.message_graph.node[node]['fs_old'] = self.mg.message_graph.node[node]['fs'][0]

        p = self.mg.message_graph.node[node]['fs'][0].prob
        eu = self.mg.message_graph.node[node]['fs'][0].util
        u_shifted = eu/p + util_const_shift     # creates a new object
        eu_shifted = p*u_shifted                # creates a new object
        self.mg.set_factor(node, Valuation(p, eu_shifted))
        self.util_const += util_const_shift

    def _reset_util_const(self, node):
        self.mg.set_factor(node, self.mg.message_graph.node[node]['fs_old'])
        delta = self.mg.message_graph.node[node]['util_const'] - self.mg.message_graph.node[node]['util_const_old']
        self.util_const -= delta
        self.mg.message_graph.node[node]['util_const'] = self.mg.message_graph.node[node]['util_const_old']
        self.mg.message_graph.node[node]['util_const_old'] = None

    def _obj_util_consts_for_all(self):
        return self.util_shifted_bounds(self.bounds(self.mg.message_graph.nodes()))

    def update_util_consts_for_all(self, options):
        obj_0, obj_1 = None, self._obj_util_consts_for_all()
        gd_updated = False
        for s in range(options['gd_steps']):
            util_gradient = self._eval_util_consts_gradients_for_all()
            abs_gradient = np.abs(util_gradient)
            L0 = np.max(abs_gradient)
            L1 = np.sum(abs_gradient)
            L2 = np.sqrt(np.sum(abs_gradient * abs_gradient))
            if L0 < options['tol']:
                return gd_updated
            step = min(1.0, 1.0 / L1) if obj_0 is None else min(1.0, 2 * (obj_0 - obj_1) / L1)
            step = step if step > 0 else 1.0
            obj_0 = obj_1
            ls_updated = self._line_search_util_consts_for_all(obj_0, util_gradient, step, L0, L2, options)
            obj_1 = self._obj_util_consts_for_all()
            if not gd_updated and ls_updated:
                gd_updated = True
            if not ls_updated:
                return gd_updated
        else:
            return gd_updated

    def _eval_util_consts_gradients_for_all(self):
        gradients = []
        for node in sorted(self.mg.message_graph.nodes_iter()):
            gradients.append(self._eval_util_const_gradient(node))
        return gradients

    def _line_search_util_consts_for_all(self, obj_0, gradients, step, L0, L2, options):
        for l in range(options['ls_steps']):
            self._set_util_consts_for_all(gradients, step)
            obj_1 = self._obj_util_consts_for_all()
            if obj_0 - obj_1 > step * options['armijo_thr'] * L2:
                for node in self.mg.message_graph.nodes_iter():
                    self._eval_pseudo_belief_node(node)
                return True
            else:
                for node in self.mg.message_graph.nodes_iter():
                    self._reset_util_const(node)
                step *= options['armijo_step_back']
                if step * L0 < options['ls_tol']:
                    return False
        else:
            return False

    def _set_util_consts_for_all(self, gradients, step):
        for ind, node in enumerate(sorted(self.mg.message_graph.nodes_iter())):
            self._set_util_const(node, gradients[ind], step)
