from constants import *
import pyGM.graphmodel as gm
from valuation import *
import copy


class GraphicalModel(object):
    def __init__(self, factor_list, weight_list, is_log = False, elim_order = None):
        self.variables = []
        self.factors = gm.factorSet()
        self.factors_by_var = [] # useful when building graph data structures
        self.scope_list = []
        self.weights = []
        self.add_factors(factor_list)
        self.set_weights(weight_list)
        self.elim_order = elim_order
        self.is_log = is_log
        if self.is_log:
            self.to_log()

    def copy(self):
        return copy.deepcopy(self)

    def add_factors(self, factor_list):
        """update factor_set, var_set, scopes from list of factors/valuations"""
        import copy
        factor_list = copy.deepcopy(factor_list)
        self.factors.update(factor_list) # scope_list follows factor_list order // factors follows the same order

        for f in factor_list:
            for v in reversed(f.vars):
                if (v.label >= len(self.variables)):
                    self.variables.extend( [gm.Var(i,0) for i in range(len(self.variables), v.label+1)] )
                    self.factors_by_var.extend( [gm.factorSet() for i in range(len(self.factors_by_var), v.label+1)] )
                if self.variables[v].states == 0: self.variables[v] = v
                assert self.variables[v].states == v.states, '# states for a variable should match'
                self.factors_by_var[v].add(f) # add f to the v-th factor set in factors_by_var list; shared factor obj
            self.scope_list.append(f.vars)

    def remove_factors(self, factor_list):
        self.factors.difference_update(factor_list)
        for f in factor_list:
            for v in f.vars:
                self.factors_by_var[v].discard(f)

    def factors_with(self, v, copy=False):
        return self.factors_by_var[v].copy() if copy else self.factors_by_var[v]

    def factors_with_any(self, vars):
        factors = gm.factorSet()
        for v in vars:
            factors.update(self.factors_by_var[v])
        return factors

    def factors_with_all(self, vars):
        if len(vars) == 0:
            return self.factors.copy()
        factors = self.factors_by_var[vars[0]].copy()
        for v in vars:
            factors.intersection_update( self.factors_by_var[v] )
        return factors

    def set_weights(self, weight_list):
        if type(weight_list) in [int, float]:
            self.weights = [float (weight_list) ] * len(self.variables)
        elif len(weight_list) == len(self.variables):
            self.weights = weight_list
        else:
            assert False, 'weight list should be either const or list having same len as variables'

    @property
    def vars(self):
        return self.variables

    @property
    def vars(self, ind):
        return self.variables[ind]

    @property
    def nvar(self):
        return len(self.variables)

    @property
    def nfactor(self):
        return len(self.factors)

    @property
    def scopes(self):
        return self.scope_list

    def to_log(self):
        """Convert internal factors to log form (if not already).  May use 'isLog' to check."""
        for f in self.factors:
            f.logIP()
        # if not self.is_log:
        #     for f in self.factors: f.logIP()
        #     self.is_log = True
        # return self

    def to_exp(self):
        """Convert internal factors to exp form (product of probabilities) if not.  May use 'isLog' to check."""
        for f in self.factors:
            f.expIP()
        # if self.is_log:
        #     for f in self.factors: f.expIP()
        #     self.is_log = False
        # return self