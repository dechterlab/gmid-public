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


def run_ctemixed(file_info, is_log):
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']
    fun_types = file_info['factor_types']
    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_valuation = False
    gm = GraphicalModel(factors, weights, is_log=is_log)
    pg = PrimalGraph(gm)
    ordering, iw = iterative_greedy_variable_order(100, pg.nx_diagram, ps=8, pe=-1, ct=inf, pv=blocks)
    mbtd, mini_buckets = mini_bucket_tree(graphical_model=gm, elim_order=ordering, ibound=iw, ignore_msg=False,
                                          random_partition=False)
    add_mg_attr_to_nodes(mbtd)
    add_const_factors(mbtd, gm.variables, is_valuation, is_log)
    verbose_level = 1
    log_file_name = os.path.join(LOG_PATH, "test.mixed")
    tree_mp = CTE(verbose_level, mbtd, ordering, weights, is_log, is_valuation, log_file_name)
    tree_mp.schedule()
    tree_mp.init_propagate()
    mmap_value = tree_mp.propagate(propagation_type='ve', max_sum=False)
    print("is_log:{}\t{}".format(is_log, mmap_value))
    return mmap_value


class TestRunCteMixed(TestCase):
    def setUp(self):
        self.name = "mdp0-2_2_2_3"
        self.test_path = os.path.join(TEST_PATH, self.name + ".mixed")

    def test_run_ctemixed(self):
        self.file_info1 = read_uai_mixed(self.test_path)
        mmap_linear = run_ctemixed(self.file_info1, False)
        self.file_info2 = read_uai_mixed(self.test_path)
        mmap_log = run_ctemixed(self.file_info2, True)

        self.assertAlmostEqual(np.exp(mmap_log), mmap_linear)


if __name__ == "__main__":
    unittest.main()
