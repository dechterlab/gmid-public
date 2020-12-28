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
    def test_lse(self):
        x = [Var(0, 2), Var(1, 3), Var(2, 2), Var(3, 2), Var(4, 5)]
        vals = [1,2,3,4,5,6]
        F0 = Factor([x[0], x[1]], vals=vals)
        print(F0)
        ax = tuple( (slice(None), 0) )
        print(ax)
        print( F0.t[ax] )
        print( F0.t[ax].max() )
        print(F0.t[ax].argmax())
        print(np.unravel_index(F0.t[ax].argmax(), (2,3)))
        k2 = F0.argmax(evidence={Var(0,2):0})
        print(k2)        # (1,2)
        print(F0[k2])



        # Factor([0.80471896 1.60943791 2.05543693])


if __name__ == '__main__':
	unittest.main()

