import sys
PRJ_PATH = "/home/junkyul/git/gmid"
sys.path.append(PRJ_PATH)
from gmid.constants import *
import unittest
from unittest import TestCase
import os

from libs.pyGM.factor import *
from libs.pyGM.varset_py import *

class FactorOpTest(unittest.TestCase):
    def _test_lse(self):
        x = [Var(0, 2), Var(1, 3), Var(2, 2), Var(3, 2), Var(4, 5)]
        vals = [np.log(1.0), np.log(2.0), np.log(3.0), np.log(4.0), np.log(5.0), np.log(6.0)]
        F0 = Factor([x[0], x[1]], vals=vals)
        print(F0)
        F1 = F0.lse(elim=None)
        print(np.exp(F1))
        print(1+2+3+4+5+6)

    def _test_lse2(self):
        x = [Var(0, 2), Var(1, 3), Var(2, 2), Var(3, 2), Var(4, 5)]
        vals = [np.log(1.0), np.log(2.0), np.log(3.0), np.log(4.0), np.log(5.0), np.log(6.0)]
        vals_C = [np.log(1.0), np.log(3.0), np.log(5.0), np.log(2.0), np.log(4.0), np.log(6.0)]
        F0 = Factor([x[0], x[1]], vals=vals)
        print(np.exp(F0[0,0]), np.exp(F0[0,1]), np.exp(F0[0,2]))
        # (1.0, 3.0000000000000004, 4.999999999999999)   processed in Fortran order
        # reverse the scope from C, assign values to that reversed combination; happens while reading from file
        print(F0)
        F1 = F0.lse(elim=VarSet([x[0]]))
        print(np.exp(F1))

    def test_lsePower(self):
        x = [Var(0, 2), Var(1, 3)]
        vals = [np.log(1.0), np.log(2.0), np.log(3.0), np.log(4.0), np.log(5.0), np.log(6.0)]
        F0 = Factor([x[0], x[1]], vals=vals)
        print(F0)
        # Factor([[0.         1.09861229 1.60943791]
        #         [0.69314718 1.38629436 1.79175947]])

        F1 = F0.lsePower(VarSet([x[0]]), power=2.0)
        print(F1)
        # Factor([0.80471896 1.60943791 2.05543693])


if __name__ == '__main__':
	unittest.main()

