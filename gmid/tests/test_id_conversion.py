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


def test_ve_id(test_name, file_info):
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']
    fun_types = file_info['factor_types']
    print("read {} format".format(test_name))
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
    log_file_name = os.path.join(LOG_PATH, test_name)
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
    print("{}\tMEU={}\tZ={}\n\n".format(test_name, bound_at_root, Z_at_root))
    return bound_at_root, Z_at_root


def test_cte(test_path):
    file_info = read_uai_id(test_path)
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']
    fun_types = file_info['factor_types']

    # factors = read_uai(test_path + ".uai")
    # blocks, num_blocks, num_vars = read_pvo(test_path + ".pvo")
    # num_vars, var_types, num_funcs, fun_types = read_id(test_path + ".id")

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
    log_file_name = os.path.join(LOG_PATH, "uai previous")
    tree_mp = CTE(verbose_level, mbtd, ordering, weights, is_log, is_valuation, log_file_name=log_file_name)
    tree_mp.schedule()
    tree_mp.init_propagate()
    bound_at_root, Z_at_root = tree_mp.propagate(propagation_type='ve', max_sum=True)
    marginals = tree_mp.bounds(root_only=False)
    clusters = sorted(tree_mp.schedule_graph.nodes(), key=lambda x: tree_mp.elim_order.index(x[0]))
    print("{}\tMEU={}\tZ={}\n\n".format("uai previous", bound_at_root, Z_at_root))
    return bound_at_root, Z_at_root


class TestFileIOFromUAI(TestCase):
    def setUp(self):
        # self.test_name1 = "pomdp1-4_2_2_2_3"
        self.created_files= []
        self.test_name1 = "mdp0-2_2_2_3"
        self.test_path1 = os.path.join(TEST_PATH, self.test_name1)
        self.MEU, self.Z = test_cte(self.test_path1)        # from base line reader

    def tearDown(self):
        for f in self.created_files:
            print("test done remove:{}".format(f))
            os.remove(f)

    def test_read_uai_id(self):
        file_info_uai_id = read_uai_id(self.test_path1)
        MEU, Z = test_ve_id("uai_id", file_info_uai_id)
        self.assertAlmostEqual(MEU, self.MEU)
        self.assertAlmostEqual(Z, self.Z)

    def test_convert_uai_to_limid(self):
        convert_uai_to_limid(self.test_path1, self.test_path1)
        self.created_files.append(self.test_path1 + ".limid")
        file_info_limid = read_limid(self.test_path1 + ".limid")
        MEU, Z = test_ve_id("limid", file_info_limid)
        self.assertAlmostEqual(MEU, self.MEU)
        self.assertAlmostEqual(Z, self.Z)
        file_info_old = read_limid(self.test_path1 + ".limid.uai")
        MEU2, Z2 = test_ve_id("limid.uai", file_info_old)
        self.assertAlmostEqual(MEU2, self.MEU)
        self.assertAlmostEqual(Z2, self.Z)

    def test_convert_uai_to_erg(self):
        convert_uai_to_erg(self.test_path1, self.test_path1 +".vo", self.test_path1)
        self.created_files.append(self.test_path1 + ".erg")
        file_info_erg = read_erg(self.test_path1 + ".erg")
        MEU, Z = test_ve_id("erg", file_info_erg)
        self.assertAlmostEqual(MEU, self.MEU)
        self.assertAlmostEqual(Z, self.Z)
        file_info_old = read_erg(self.test_path1 + ".old.uai")
        MEU2, Z2 = test_ve_id("old.uai", file_info_old)
        self.assertAlmostEqual(MEU2, self.MEU)
        self.assertAlmostEqual(Z2, self.Z)


class TestFileIOFromErg(TestCase):
    def setUp(self):
        self.created_files = []
        self.test_name1 = "rand-c20d2o1-01"
        self.test_path1 = os.path.join(TEST_PATH, self.test_name1)
        file_info_erg = read_erg(self.test_path1 + ".erg")
        self.MEU, self.Z = test_ve_id("erg", file_info_erg)

    def tearDown(self):
        for f in self.created_files:
            print("test done remove:{}".format(f))
            os.remove(f)

    def test_convert_erg_to_uai(self):
        convert_erg_to_uai(self.test_path1 + ".erg", self.test_path1 + "_from_erg")
        self.created_files.append(self.test_path1 + "_from_erg.uai")
        self.created_files.append(self.test_path1 + "_from_erg.pvo")
        self.created_files.append(self.test_path1 + "_from_erg.id")
        file_info_uai = read_uai_id(self.test_path1 + "_from_erg")
        MEU, Z = test_ve_id("uai_id", file_info_uai)
        self.assertAlmostEqual(MEU, self.MEU)
        self.assertAlmostEqual(Z, self.Z)

    def test_convert_erg_to_limid(self):
        convert_erg_to_limid(self.test_path1 + ".erg", self.test_path1 + "_from_erg")
        self.created_files.append(self.test_path1 + "_from_erg.limid")
        file_info_limid = read_limid(self.test_path1 + "_from_erg.limid")
        MEU, Z = test_ve_id("limid", file_info_limid)
        self.assertAlmostEqual(MEU, self.MEU)
        self.assertAlmostEqual(Z, self.Z)


class TestFileFromLIMID(TestCase):
    def setUp(self):
        self.created_files = []
        self.test_name = "ID_from_BN_78_w18d3"
        self.test_path = os.path.join(TEST_PATH, self.test_name)
        file_info = read_limid(self.test_path + ".limid")
        self.MEU, self.Z = test_ve_id("limid", file_info)

    def tearDown(self):
        for f in self.created_files:
            print("test done remove:{}".format(f))
            os.remove(f)

    def test_convert_limid_to_uai(self):
        convert_limid_to_uai(self.test_path + ".limid", self.test_path + "_from_limid")
        self.created_files.append(self.test_path + "_from_limid.uai")
        self.created_files.append(self.test_path + "_from_limid.pvo")
        self.created_files.append(self.test_path + "_from_limid.id")
        file_info_uai = read_uai_id(self.test_path + "_from_limid")
        MEU, Z = test_ve_id("uai_id", file_info_uai)
        self.assertAlmostEqual(MEU, self.MEU)
        self.assertAlmostEqual(Z, self.Z)


if __name__ == '__main__':
    unittest.main()
