import sys
PRJ_PATH = "/home/junkyul/git/gmid"
sys.path.append(PRJ_PATH)
from gmid.constants import *
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.cte import *
import unittest
from unittest import TestCase
import os


class TestLimidByMaua(TestCase):
    def setUp(self):
        self.file_name = "chain-3-2-0.maua"
        self.file_path = os.path.join(TEST_PATH, self.file_name)

    def test_read_maua(self):
        file_info = read_maua(self.file_path)
        self.assertEquals(file_info['nvar'], 6)
        self.assertEquals(file_info['nprob'], 3)
        self.assertEquals(file_info['ndec'], 3)
        self.assertEquals(file_info['nutil'], 1)
        self.assertTrue(np.all(file_info['factors'][0].t.ravel(order='C') ==
                               np.array([0.811931775397, 0.188068224603, 0.0470909862676, 0.952909013732 ])))
        self.assertTrue(np.all(file_info['factors'][1].t.ravel(order='C') ==
                               np.array([0.714540766661, 0.285459233339, 0.0460354881443, 0.953964511856,
                                         0.0553602303241, 0.944639769676, 0.12317141498, 0.87682858502])))

    def test_convert_maua_to_uai(self):
        convert_maua_to_uai(self.file_path, self.file_path)
        file_info = read_uai_id(self.file_path, sort_scope=False)
        self.assertEquals(file_info['nvar'], 6)
        self.assertEquals(file_info['nprob'], 3)
        self.assertEquals(file_info['ndec'], 3)
        self.assertEquals(file_info['nutil'], 1)
        self.assertEquals(file_info['domains'], [2,2,2,2,2,2])
        self.assertTrue(np.all(file_info['factors'][0].t.ravel(order='C') ==
                               np.array([0.811931775397, 0.188068224603, 0.0470909862676, 0.952909013732 ])))
        self.assertTrue(np.all(file_info['factors'][1].t.ravel(order='C') ==
                               np.array([0.714540766661, 0.285459233339, 0.0460354881443, 0.953964511856,
                                         0.0553602303241, 0.944639769676, 0.12317141498, 0.87682858502])))
        self.assertEquals([sorted(el) for el in file_info['blocks']], [[0,1,2], [5], [4], [3]])


if __name__ == "__main__":
    unittest.main()
