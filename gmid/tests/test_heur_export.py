import sys
PRJ_PATH = "/home/junkyul/conda/gmid"
sys.path.append(PRJ_PATH)

from gmid.constants import *
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.weighted_mbe3 import WeightedMBE3
from gmid.search import message_graph_as_heuristic
import unittest
from unittest import TestCase

class TestExportHeuristic(TestCase):
    def setUp(self):
        self.name = "car"
        self.ibound = 1
        self.optimize_weight = 1
        self.optimize_cost = 1
        self.pseudo_tree = [-1, 5, 5, 0, 3, 4]

    def run_weighted_mbe(self):
        print("read problem")
        problem_path = os.path.join(TEST_PATH, self.name)
        file_info = read_uai_id(problem_path)
        ordering_path = problem_path + '.vo'
        factors = file_info['factors']
        var_types = file_info['var_types']
        fun_types = file_info['factor_types']
        self.num_var = file_info['nvar']
        self.num_fun = len(file_info['factors'])
        print("convert to valuations")
        valuations = [factor_to_valuation(factor, factor_type, False) for factor, factor_type in
                      zip(factors, fun_types)]
        weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
        is_log = False
        is_valuation = True
        print("create graphical model")
        gm = GraphicalModel(valuations, weights, is_log=is_log)
        self.ordering, iw = read_vo(ordering_path)
        print("create mini bucket tree")
        mbtd, self.mini_buckets = mini_bucket_tree(gm, self.ordering, self.ibound, ignore_msg=False, random_partition=False)
        jgd = join_graph(mbtd, self.mini_buckets, self.ordering, connect_mb_only=True, make_copy=False)
        add_const_factors(jgd, gm.variables, is_valuation, is_log)
        add_mg_attr_to_nodes(jgd)
        wmb_str = "_".join([self.name, "WeightedMBE", "iw=" + str(iw), "ibd=" + str(self.ibound)])
        log_file_name = os.path.join(LOG_PATH_ID, wmb_str)
        print("start WeightedMBE")
        self.wmb_heur = WeightedMBE3(0, jgd, self.ordering, weights, is_log, WEPS, log_file_name,
                                    self.ibound, self.mini_buckets, gm.variables)
        self.wmb_heur.schedule()
        self.wmb_heur.init_propagate()
        self.optimize_weight = 0
        self.optimize_cost = 0
        best_eu, final_Z = self.wmb_heur.propagate(time_limit=3600, iter_limit=5, optimize_weight=self.optimize_weight,
                                                   optimize_cost=self.optimize_cost)
        print("end WeightedMBE")

    def test_heur_info(self):
        self.run_weighted_mbe()

        heur_info = message_graph_as_heuristic(self.wmb_heur.mg.message_graph, self.ordering, self.pseudo_tree,
                                               self.mini_buckets, self.num_var, self.num_fun)
        print("num_var:{}".format(heur_info['num_var']))
        print("msg_id_start:{}".format(heur_info['msg_id_start']))
        self.assertEquals(heur_info['num_var'], self.num_var)
        self.assertEquals(heur_info['msg_id_start'], self.num_fun)
        self.assertEquals(len(heur_info['bucket_msg']), self.num_var)
        self.assertEquals(heur_info['num_msg'], 7)

        print("bucket{}:{}".format(1, sorted(heur_info['bucket_msg'][1])))
        print("bucket{}:{}".format(2, sorted(heur_info['bucket_msg'][2])))
        print("bucket{}:{}".format(5, sorted(heur_info['bucket_msg'][5])))
        print("bucket{}:{}".format(4, sorted(heur_info['bucket_msg'][4])))
        print("bucket{}:{}".format(3, sorted(heur_info['bucket_msg'][3])))
        print("bucket{}:{}".format(0, sorted(heur_info['bucket_msg'][0])))

        print(heur_info['msg_indexer'])
        print(heur_info['msg_indexer'][7].prob)
        print("print scope")
        print(" ".join([str(el) for el in heur_info['msg_indexer'][7].prob.vars]))
        print("print table")
        print(heur_info['msg_indexer'][7].prob.table)
        print("\n".join(map(str, heur_info['msg_indexer'][7].prob.table.ravel(order='C'))))

        write_mini_bucket_heuristic_from_info( os.path.join(TEST_PATH, self.name), heur_info )


if __name__ == "__main__":
    unittest.main()