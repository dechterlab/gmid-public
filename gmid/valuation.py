from constants import *
from pyGM.factor import Factor

def factor_to_valuation(factor, factor_type, const_shift=False):
    if const_shift:
        # (P, P) = (P,0)*(1,1), (1, U+1) = (1,U)*(1,1) shift valuation by (1,1) adding utility by 1
        return Valuation(factor.copy(), factor.copy()) if factor_type =='P' else Valuation(Factor(factor.vars.copy(), 1.0), factor.copy()+Factor(factor.vars.copy(), 1.0))

    else:
        # (P,0) := (P, 1e-300), (1,U)
        return Valuation(factor.copy(), Factor(factor.vars.copy(), ZERO)) if factor_type == 'P' else Valuation(Factor(factor.vars.copy(), 1.0), factor.copy())

########################################################################################################################
# local helper for Valuation class
########################################################################################################################
def exp_v(term):
    if type(term) in {Valuation, Factor}:
        return term.exp()
    else:
        return np.exp(term)


def log_v(term):
    if type(term) in {Valuation, Factor}:
        return term.log()
    else:
        # if debug and term == 0:
        #     return np.log(ZERO)
        # else:
        #     return np.log(term)
        return np.log(term)


def abs_v(term):
    if type(term) in {Valuation, Factor}:
        return term.abs()
    else:
        return np.abs(term)


def sign_v(term):
    if type(term) in {Factor}:
        return term.sign()
    else:
        return np.sign(term)

########################################################################################################################
class Valuation(object):
    def __init__(self, p, u):
        self.prob = p
        self.util = u

    def __repr__(self):
        return repr((repr(self.prob), repr(self.util)))

    def __str__(self):
        return str(('P:' + str(self.prob), 'U:' + str(self.util)))

    @property
    def t(self):
        return str(self)

    @property
    def vars(self):
        return self.prob.v | self.util.v

    @property
    def nvar(self):
        return len(self.prob.v | self.util.v)

    # @property
    # def u_div_p_from_log(self):
    #     A = exp_v(self.util)
    #     B = exp_v(self.prob)
    #     C = A/B
    #     if isinstance(C, Factor):
    #         C.t = np.clip(C.t, a_min=-np.exp(EXPLIM), a_max=np.exp(EXPLIM))
    #     else:
    #         if C < -np.exp(EXPLIM):
    #             C = -np.exp(EXPLIM)
    #         elif C > np.exp(EXPLIM):
    #             C = np.exp(EXPLIM)
    #     # return exp_v(self.util) / exp_v(self.prob)
    #     return C
    # @property
    # def u_div_p(self):
    #     return self.util / self.prob

    def copy(self):
        return Valuation(self.prob.copy(), self.util.copy() )

    def mul(self, other):
        return Valuation(self.prob*other.prob, self.prob*other.util + self.util*other.prob)
        # new_p = exp_v(log_v(self.prob) +log_v( other.prob))
        # new_eu = other.util.sign()*exp_v(self.prob.log() + other.util.abs().log()) + self.util.sign()*exp_v(self.util.abs().log() + other.prob.log())
        # new_eu = sign_v(other.util) * exp_v( log_v(self.prob) + log_v( abs_v(other.util))) + sign_v(self.util) * exp_v( log_v(abs_v(self.util)) + log_v(other.prob))
        # return Valuation(new_p, new_eu)

    def __mul__(self, other):
        return self.mul(other) # V1 * V2 --> combination in linear scale

    def div(self, other):
        return Valuation(self.prob/other.prob, self.util/other.prob - self.prob*other.util/other.prob.power(2))
        # new_p = exp_v(self.prob.log() - other.prob.log())
        # new_p = exp_v(log_v(self.prob) - log_v(other.prob))
        # new_eu = self.util.sign()*exp_v(self.util.abs().log() - other.prob.log()) - other.util.sign()*exp_v(self.prob.log()+other.util.abs().log() - 2*other.prob.log())
        # new_eu = sign_v(self.util) * exp_v(log_v(abs_v(self.util)) - log_v(other.prob)) - sign_v(other.util) * exp_v(log_v(self.prob) + log_v(abs_v(other.util)) - 2 * log_v(other.prob))
        # return Valuation(new_p, new_eu)

    def __div__(self, other):
        return self.div(other) # V1 / V2 = V1 * inv(V2) --> division in linear scale

    # def mul_log(self, other):
    #     # combination in log scale
    #     return Valuation(self.prob+other.prob, log_v(exp_v(self.prob + other.util)+exp_v(other.prob+self.util)))
    #
    # def __add__(self, other):
    #     return self.mul_log(other) # V1 + V2 --> combination of Valuations in log scale

    # def div_log(self, other):
    #     # division in log scale
    #     # if debug: # check if expected utility >= 0
    #     #     check = exp_v(self.util - other.prob) - exp_v(self.prob + other.util - 2 * other.prob)
    #     #     assert np.all(check.t >= 0.0)
    #     return Valuation(self.prob - other.prob, log_v( exp_v(self.util - other.prob) - exp_v(self.prob + other.util - 2 * other.prob)))
    #
    # def __sub__(self, other):
    #     return self.div_log(other) # V1 - V2 = V1 * inv(V2) --> division in log scale

    def logIP(self): # in place transformation to log
        self.prob.logIP()
        # self.prob.t = np.clip(self.prob.t, a_min=-EXPLIM, a_max=EXPLIM)
        self.util.logIP()
        # self.util.t = np.clip(self.util.t, a_min=-EXPLIM, a_max=EXPLIM)

    def expIP(self): # in place transformation to linear
        self.prob.expIP()
        # self.prob.t = np.clip(self.prob.t, a_min=-np.exp(EXPLIM), a_max=np.exp(EXPLIM))
        self.util.expIP()
        # self.util.t = np.clip(self.util.t, a_min=-np.exp(EXPLIM), a_max=np.exp(EXPLIM))

    def exp(self): # create a new valuation in linear scale
        return Valuation(self.prob.exp() if type(self.prob) is Factor else np.exp(self.prob), self.util.exp() if type(self.util) is Factor else np.exp(self.util))

    def log(self): # create a new valuation in log scale
        return Valuation(self.prob.log() if type(self.prob) is Factor else np.log(self.prob), self.util.log() if type(self.util) is Factor else np.log(self.util))

    def abs(self):
        return Valuation(self.prob, self.util.abs())

    def clip_eu_IP(self):
        np.clip(self.util.t, a_min=ZERO, out=self.util.t)

    def max(self, elim=None, out=None):
        return Valuation(self.prob.max(elim, out), self.util.max(elim, out))

    def sum(self, elim=None, out=None):
        return Valuation(self.prob.sum(elim, out), self.util.sum(elim, out))

    def lse(self, elim=None, out=None):
        return Valuation(self.prob.lse(elim, out), self.util.lse(elim, out))

    def min(self, elim=None, out=None):
        return Valuation(self.prob.min(elim, out), self.util.min(elim, out))

    def sumPower(self, elim=None, power=1.0, out=None):
        return Valuation(self.prob.sumPower(elim, power, out), self.util.sumPower(elim, power, out))

    def lsePower(self, elim=None, power=1.0, out=None):
        return Valuation(self.prob.lsePower(elim, power, out), self.util.lsePower(elim, power, out))

    def marginal(self, target, out=None):
        return Valuation(self.prob.marginal(target, out), self.util.marginal(target, out))

    def maxmarginal(self, target, out=None):
        return Valuation(self.prob.maxmarginal(target, out), self.util.maxmarginal(target, out))

    def minmarginal(self, target, out=None):
        return Valuation(self.prob.minmarginal(target, out), self.util.minmarginal(target, out))