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


def factor_cte(file_info):
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']
    fun_types = file_info['factor_types']
    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_log = False
    is_valuation = False
    gm = GraphicalModel(factors, weights, is_log=is_log)
    pg = PrimalGraph(gm)
    ordering, iw = iterative_greedy_variable_order(100, pg.nx_diagram, ps=8, pe=-1, ct=inf, pv=blocks)
    mbtd, mini_buckets = mini_bucket_tree(graphical_model=gm, elim_order=ordering, ibound=iw, ignore_msg=False,
                                          random_partition=False)
    add_mg_attr_to_nodes(mbtd)
    add_const_factors(mbtd, gm.variables, is_valuation, is_log)
    verbose_level = 1
    log_file_name = os.path.join(LOG_PATH, "id_mixed_trans")
    tree_mp = CTE(verbose_level, mbtd, ordering, weights, is_log, is_valuation, log_file_name)
    tree_mp.print_log("CTE")
    tree_mp.print_log("iw={}, elim_order={}".format(iw, ordering))
    tree_mp.print_log("weights={}".format(weights))
    tree_mp.print_log("number of clusters={}".format(len(mbtd.region_graph.nodes())))
    tree_mp.schedule()
    tree_mp.init_propagate()
    mmap_value = tree_mp.propagate(propagation_type='ve', max_sum=False)
    return mmap_value


def valuation_cte(file_info):
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']
    fun_types = file_info['factor_types']

    valuations = [factor_to_valuation(factor, factor_type, False) for factor, factor_type in zip(factors, fun_types)]
    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_log = False
    is_valuation = True
    gm = GraphicalModel(valuations, weights, is_log=is_log)
    pg = PrimalGraph(gm)
    ordering, iw = iterative_greedy_variable_order(100, pg.nx_diagram, ps=8, pe=-1, ct=inf, pv=blocks)

    mbtd, mini_buckets = mini_bucket_tree(graphical_model=gm, elim_order=ordering, ibound=iw, ignore_msg=False, random_partition=False)
    add_mg_attr_to_nodes(mbtd)
    add_const_factors(mbtd, gm.variables, is_valuation, is_log)

    verbose_level = 1
    log_file_name = os.path.join(LOG_PATH, "ID with valuation")
    tree_mp = CTE(verbose_level, mbtd, ordering, weights, is_log, is_valuation, log_file_name)
    tree_mp.print_log("CTE")
    tree_mp.print_log("iw={}, elim_order={}".format(iw, ordering))
    tree_mp.print_log("weights={}".format(weights))
    tree_mp.print_log("number of clusters={}".format(len(mbtd.region_graph.nodes())))
    tree_mp.schedule()
    tree_mp.init_propagate()
    bound_at_root, Z_at_root = tree_mp.propagate(propagation_type='ve', max_sum=True)
    marginals = tree_mp.bounds(root_only=False)
    clusters = sorted(tree_mp.schedule_graph.nodes(), key=lambda x: tree_mp.elim_order.index(x[0]))
    tree_mp.print_log("final MEU:{}".format(bound_at_root))
    tree_mp.print_log("final Z:{}".format(Z_at_root))
    print("{}\tMEU={}\tZ={}\n\n".format("ID with valuation", bound_at_root, Z_at_root))
    return bound_at_root, Z_at_root


class TestId2Mixed(TestCase):
    def setUp(self):
        self.created_files = []
        self.test_name = "mdp0-2_2_2_3"
        self.test_path = os.path.join(TEST_PATH, self.test_name)
        self.id_file_info = read_uai_id(self.test_path, sort_scope=False)
        self.mmap_info = translate_uai_id_to_mixed(self.id_file_info)
        write_uai_mixed(self.test_path + ".transtest", self.mmap_info, uai_type="ID")
        self.created_files.append(self.test_path + ".transtest.uai")
        self.created_files.append(self.test_path + ".transtest.mi")
        self.created_files.append(self.test_path + ".transtest.pvo")
        self.MEU, self.Z = valuation_cte(self.id_file_info)

    def test_read_mi(self):
        num_vars, var_types = read_mi(self.test_path + ".transtest.mi")
        self.assertEquals(num_vars, 10)
        self.assertEquals(var_types, ['C', 'C', 'D', 'C', 'C', 'D', 'C', 'C', 'D', 'C' ])

    def test_read_pvo(self):
        blocks, nblocks, nvars = read_pvo(self.test_path + ".transtest.pvo")
        self.assertEquals(nvars, 10)
        self.assertEquals(nblocks, 7)
        self.assertEquals(sorted(blocks[0]), [9])
        self.assertEquals(sorted(blocks[1]), [8])
        self.assertEquals(sorted(blocks[2]), [6, 7])
        self.assertEquals(sorted(blocks[3]), [5])
        self.assertEquals(sorted(blocks[4]), [3, 4])

    def test_read_uai_mixed(self):
        mixed_info = read_uai_mixed(self.test_path + ".transtest")
        mmap_value = factor_cte(mixed_info)
        self.assertAlmostEqual(mmap_value, self.MEU)

    def test_convert_uai_id_to_mixed(self):
        convert_uai_id_to_mixed(self.test_path, self.test_path + ".converted")
        self.created_files.append(self.test_path + ".converted.uai")
        self.created_files.append(self.test_path + ".converted.mi")
        self.created_files.append(self.test_path + ".converted.pvo")
        converted_mixed_info = read_uai_mixed(self.test_path + ".converted")
        mmap_value = factor_cte(converted_mixed_info)
        self.assertAlmostEqual(mmap_value, self.MEU)

    def test_convert_mmap_to_mixed(self):
        convert_mmap_to_mixed(self.test_path + ".mmap", self.test_path + ".from_mmap")
        self.created_files.append(self.test_path + ".from_mmap.uai")
        self.created_files.append(self.test_path + ".from_mmap.mi")
        self.created_files.append(self.test_path + ".from_mmap.pvo")
        from_mmap_info = read_uai_mixed(self.test_path + ".from_mmap")
        mmap_value = factor_cte(from_mmap_info)
        self.assertAlmostEqual(mmap_value, self.MEU)


    def tearDown(self):
        for f in self.created_files:
            print("test done remove:{}".format(f))
            os.remove(f)



if __name__ == "__main__":
    unittest.main()
