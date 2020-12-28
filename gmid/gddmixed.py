from constants import *
from pyGM.factor import Factor
from valuation import Valuation
from message_passing import MessagePassing
from gddid_hinge_shift import GddIdHingeShift
from graph_algorithms import toposort2
import random
import time


# mixed inference in log scale

class GddMixed(MessagePassing):
    def __init__(self, verbose_level, message_graph, elim_order, weights, is_log, epsilon, log_file_name):
        super(GddMixed, self).__init__(verbose_level, message_graph, elim_order, weights, is_log, log_file_name)
        self.epsilon = epsilon
        if is_log:
            self.prob_const = 0.0       # log scale
        else:
            self.prob_const = 1.0
        self.best_mmap = np.inf
        self.total_diff = np.inf
        self.new_mmap = None
        self.old_mmap = None

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
        # print("schedule_mbe::self.message_graph_nodes={}".format(self.message_graph_nodes))

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

    # todo different
    def _extract_prob_max(self):
        for n in self.mg.message_graph.nodes_iter():
            p = self.mg.message_graph.node[n]['fs'][0]
            m = np.max(p.t)
            if self.is_log:
                self.prob_const += m
                p.t = p.t - m
            else:
                self.prob_const *= m
                p.t = p.t/m

    ####################################################################################################################
    # evaluations related to lp norms
    ####################################################################################################################
    # todo different
    def _eval_pseudo_belief_node(self, node):

        if self.is_log:
            pr = self.mg.message_graph.node[node]['fs'][0].copy()
        else:
            pr = self.mg.message_graph.node[node]['fs'][0].log()
        mu_p = self._eval_pseudo_belief(node, pr)
        self.mg.message_graph.node[node]['pseudo_belief'] = mu_p

    def _eval_pseudo_belief(self, node, lnZ0):
        removable_vars = [el for el in self.elim_order if el in lnZ0.vars.labels]
        lnmu = 0.0
        for v in (el for el in self.elim_order if el in removable_vars):
            w_exponent = self.epsilon if self.mg.message_graph.node[node]['w'][v] <= self.epsilon else \
            self.mg.message_graph.node[node]['w'][v]
            if w_exponent <= self.epsilon:
                # lnZ1 = lnZ0.lsePower([v], np.inf)  # np.inf will evaluate max
                lnZ1 = lnZ0.max([v])
            else:
                lnZ1 = lnZ0.lsePower([v], ONE / w_exponent)  # pass 1/w as power to lsePower() it takes inv weight
            lnZ0 -= lnZ1  # this is the point where inf became nan -inf - -inf
            lnZ0.t[np.isnan(lnZ0.t)] = -np.inf  # todo here pseudo marginal
            lnZ0 *= ONE / w_exponent  # this number is big when w_exponent is small, underflow unavoidable
            lnmu += lnZ0
            lnZ0 = lnZ1
        mu = lnmu.exp()
        return mu

    ####################################################################################################################
    # todo different
    def bounds(self, changed_nodes=[]):
        bounds_at_nodes = []
        for node in self.message_graph_nodes:
            if node in changed_nodes:   # recompute bounds for relevant nodes
                combined_factor = self.mg.message_graph.node[node]['fs'][0].copy()
                removable_vars = combined_factor.vars.labels
                for v in (el for el in self.elim_order if el in removable_vars):
                    w_exponent = self.mg.message_graph.node[node]['w'][v]

                    if w_exponent <= self.epsilon:
                        combined_factor = combined_factor.max([v])  # evaluate using max
                    else:
                        if self.is_log:
                            combined_factor = combined_factor.lsePower([v], ONE / w_exponent)
                        else:
                            combined_factor = combined_factor.log()
                            combined_factor = combined_factor.lsePower([v], ONE / w_exponent)
                            if type(combined_factor) is Factor:
                                combined_factor = combined_factor.exp()
                            else:
                                combined_factor = np.exp(combined_factor)

                # print("bounds::combined_factor={}".format(combined_factor))
                bounds_at_nodes.append(combined_factor)
                self.mg.message_graph.node[node]['bound'] = combined_factor
            else:
                bounds_at_nodes.append(self.mg.message_graph.node[node]['bound']) # read most recent bound
        return bounds_at_nodes

    # todo new
    def mmap_from_bounds(self, bounds, scale_prob_max=True):
        temp = bounds[0].copy()
        for el in bounds[1:]:
            if self.is_log:
                temp = temp + el      # combine factors
            else:
                temp = temp * el
        if scale_prob_max:
            if self.is_log:
                return temp + self.prob_const       # in log scale
            else:
                return temp * self.prob_const
        else:
            return temp

    ####################################################################################################################
    # prints
    ####################################################################################################################
    # todo different
    def print_bounds(self):
        self.print_log("summary local bounds")
        bounds_at_nodes = self.bounds(self.mg.message_graph.nodes())
        for ind, node in enumerate(self.message_graph_nodes):
            self.print_log("node:{}, bound:{}".format(str(node), str(bounds_at_nodes[ind])))
        self.print_log("prob scale pulled out:{}".format(self.prob_const))
        self.print_log("mmap from bound={}".format(self.mmap_from_bounds(bounds_at_nodes)))

    def print_weights(self):
        self.print_log("summary weights")
        for node in sorted(self.mg.message_graph.nodes_iter()):
            for v_w in sorted(self.mg.message_graph.node[node]['w']):
                self.print_log(
                    "node:{}, var:{}, w:{}".format(node, v_w, self.mg.message_graph.node[node]['w'][v_w]))

    def print_prob_const(self):
        self.print_log("prob const pulled out:{}".format(self.prob_const))

    ####################################################################################################################
    # todo different
    def propagate(self, time_limit, iter_limit, cost_options, weight_options):
        self.time_0 = time.time()
        self.time_limit = time_limit
        self.print_weights()
        for iter in range(iter_limit):
            self._extract_prob_max()
            self._init_pseudo_beliefs()
            self.total_diff = self._propagate_one_pass(iter, cost_options, weight_options)
            if (iter + 1) % 10 == 0:
                self.print_weights()
                self.print_bounds()
            if self.total_diff < TOL:
                print("gdd terminating due to diff:{}".format(self.total_diff))
                break
            if time.time() - self.time_0 > self.time_limit:
                print("gdd terminating due to time_limit:{}".format(self.time_limit))
                break
        else:
            print("gdd terminating due to iter_limit:{}".format(iter_limit))

        self.print_weights()
        self.print_bounds()
        self.print_log("best mmap value so far:{}".format(np.exp(self.best_mmap)))
        return np.exp(self.best_mmap)

    # todo different
    def _propagate_one_pass(self, it, cost_options, weight_options):
        self.new_mmap = self.mmap_from_bounds(self.bounds(self.mg.message_graph.nodes()))
        # print("propagate_one_pass::self.new_mmap={}".format(self.new_mmap))
        one_pass_init = self.new_mmap
        if it >= 0:
            lg_str = "iter:{}\tbefore update:\ttime={:.2f}\tlin={:2.12f}"
            self.print_log(lg_str.format(it + 1, time.time() - self.time_0, np.exp(self.new_mmap)))

        #### weight update
        self.old_mmap = self.new_mmap
        for ind, nodes in enumerate(self.weight_schedule):
            v_w = self.elim_order[ind]
            if v_w not in self.sum_vars or len(nodes) <= 1:
                continue
            self._update_weights_per_var(v_w, nodes, weight_options)
        self.new_mmap = self.mmap_from_bounds(self.bounds(self.mg.message_graph.nodes()))
        if self.best_mmap > self.new_mmap:
            self.best_mmap = self.new_mmap
        if it >= 0:
            lg_str = "\t\tweight update:\ttime={:.2f}\tlin={:2.12f}\tdiff={:+2.12f}\tdecreased={}"
            diff = self.new_mmap - self.old_mmap
            self.print_log(lg_str.format(time.time() - self.time_0, np.exp(self.new_mmap), diff, diff <= 0))

        #### cost update
        self.old_mmap = self.new_mmap
        for edge in self.cost_schedule:
            self._update_cost_per_edge(edge, cost_options)
        self.new_mmap = self.mmap_from_bounds(self.bounds(self.mg.message_graph.nodes()))
        if self.best_mmap > self.new_mmap:
            self.best_mmap = self.new_mmap
        if it >= 0:
            lg_str = "\t\tcost update:\ttime={:.2f}\tlin={:2.12f}\tdiff={:+2.12f}\tdecreased={}"
            diff = self.new_mmap - self.old_mmap
            self.print_log(lg_str.format(time.time() - self.time_0, np.exp(self.new_mmap), diff, diff <= 0))

        #### finish one pass
        one_pass_end = self.new_mmap
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

    # todo different
    def _obj_weight_per_var(self, nodes):
        bound = self.mmap_from_bounds(self.bounds(nodes), scale_prob_max=False)
        return bound  # self.eu_from_bounds(self.bounds(nodes), scale_prob_max=False)

    # todo different
    def _eval_weight_gradients_per_var(self, v_w, nodes):
        wgts = []
        Hcond = []

        for node in nodes:
            sc = self.mg.message_graph.node[node]['sc']
            mu_p = self.mg.message_graph.node[node]['pseudo_belief'].marginal(sc)
            # mu_p = self.mg.message_graph.node[node]['pseudo_belief']
            elimvars_in_node = sorted(sc, key=lambda x: self.elim_order.index(x))
            v_w_ind = elimvars_in_node.index(v_w)

            mu_p_temp1 = mu_p.marginal(elimvars_in_node[v_w_ind:])  # P(xi, xi+1~)
            if type(mu_p_temp1) is not Factor:
                mu_p_temp1 = Factor({}, mu_p_temp1)

            if np.all(mu_p_temp1.t == 0):  # when mu_u_temp1 is all zero factor, raise assertion error
                H1_p = 0.0
            else:
                H1_p = mu_p_temp1.entropy()

            if v_w_ind + 1 < len(elimvars_in_node):
                mu_p_temp2 = mu_p.marginal(elimvars_in_node[v_w_ind + 1:])  # P(xi+1 , xi+2 ~)
                if type(mu_p_temp2) is not Factor:
                    mu_p_temp2 = Factor({}, mu_p_temp2)
                if np.all(mu_p_temp2.t == 0):
                    H2_p = 0.0
                else:
                    H2_p = mu_p_temp2.entropy()
            else:
                H2_p = 0.0

            wgts.append(self.mg.message_graph.node[node]['w'][v_w])
            Hcond.append(H1_p - H2_p)  # conditional entropy

        Hbar = 0.0
        for i in range(len(wgts)):
            Hbar += wgts[i] * Hcond[i]
        gradient = [w_i*(Hcond[ind] - Hbar) for ind, w_i in enumerate(wgts)]

        if self.verbose:
            print('wgts:', wgts)
            print('Hcond:', Hcond)
            print('Hbar:', Hbar)
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
    # todo different
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

            abs_gradient = prob_gradient.abs()
            L0 = abs_gradient.max()
            L1 = abs_gradient.sum()
            L2 = (abs_gradient*abs_gradient).sum()
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
            ls_updated = self._line_search_cost(node_from, node_to, obj_0, prob_gradient, step, L0, L2, options)
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

    # todo different
    def _obj_cost(self, node_from, node_to):
        bound = self.mmap_from_bounds(self.bounds(changed_nodes=[node_from, node_to]), scale_prob_max=False)    # gradient ignores prob_max
        if debug:
            assert np.isfinite(bound), "bound should be finite"
        return bound  # compute total expected utility divided by scale [excluded by gradient computation]

    # todo different
    def _eval_prob_gradients(self, node_from, node_to):
        sc = self.mg.message_graph.edge[node_from][node_to]['sc']
        pm_p_from = self.mg.message_graph.node[node_from]['pseudo_belief']
        pm_p_to = self.mg.message_graph.node[node_to]['pseudo_belief']
        return pm_p_to.marginal(sc) - pm_p_from.marginal(sc)

    # todo different
    def _line_search_cost(self, node_from, node_to, obj_0, prob_gradient, step, L0, L2, options):
        ls_steps = options['ls_steps']
        armijo_thr = options['armijo_thr']
        armijo_step_back = options['armijo_step_back']
        ls_tol = options['ls_tol']

        if self.terse:
            self.print_log("\t\t\t\t\t\tLine Search on nodes: {}".format(str([node_from, node_to])))

        for l in range(ls_steps):
            self._set_cost(node_from, node_to, prob_gradient, step)
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

    # todo different
    def _set_cost(self, node_from, node_to, prob_gradient, step):
        prob_shift = -step*prob_gradient
        factor_from = self.mg.message_graph.node[node_from]['fs'][0]
        factor_to = self.mg.message_graph.node[node_to]['fs'][0]
        self.mg.message_graph.node[node_from]['fs_old'] = factor_from
        self.mg.message_graph.node[node_to]['fs_old'] = factor_to
        self.mg.set_factor(node_from, factor_from - prob_shift)
        self.mg.set_factor(node_to, factor_to + prob_shift)

    def _reset_cost(self, node_from, node_to):
        # self.mg.set_factor(node_from, self.mg.message_graph.node[node_from]['fs_old'].copy())
        self.mg.set_factor(node_from, self.mg.message_graph.node[node_from]['fs_old'])
        # self.mg.set_factor(node_to, self.mg.message_graph.node[node_to]['fs_old'].copy())
        self.mg.set_factor(node_to, self.mg.message_graph.node[node_to]['fs_old'])
        self.mg.message_graph.node[node_from]['fs_old'] = None
        self.mg.message_graph.node[node_to]['fs_old'] = None