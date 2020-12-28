import sys
PRJ_PATH = "/home/junkyul/git/gmid"
sys.path.append(PRJ_PATH)
from gmid.constants import *
IMPORT_PYGRAPHVIZ=True
from gmid.fileio import *
from gmid.graph_algorithms import *
import unittest
from unittest import TestCase
import os


class TestNxInfluenceDiagram(TestCase):
    def setUp(self):
        self.file_name = os.path.join(TEST_PATH, "pomdp1-4_2_2_2_3")
        self.file_info = read_uai_id(self.file_name, False)

    def tearDown(self):
        pass

    def test_id_creation(self):
        test_id_obj = NxInfluenceDiagram()
        nvar = self.file_info["nvar"]
        var_types = self.file_info["var_types"]
        scopes = [[v.label for v in sc] for sc in self.file_info["scopes"]]
        scope_types = self.file_info["scope_types"]
        partial_elim_order = self.file_info["blocks"]
        test_id_obj.create_id_from_scopes(nvar, var_types, scopes, scope_types, partial_elim_order)
        test_id = test_id_obj.id
        test_id_obj.draw_diagram(self.file_name.split("/")[-1])

        # create a new NxInfluenceDiagram and create class with the nxgraph defined earlier
        test_id_obj2 = NxInfluenceDiagram()
        test_id_obj2.create_id_from_nxgraph(test_id.copy())
        test_id_obj2.draw_diagram(self.file_name.split("/")[-1] + "_copy")

        # two influence diagram should show the same topology information
        self.assertEquals(test_id_obj.nvar, test_id_obj2.nvar)
        self.assertEquals(test_id_obj.decision_nodes, test_id_obj2.decision_nodes)
        self.assertEquals(test_id_obj.var_types, test_id_obj2.var_types)
        self.assertEquals([sorted(bk) for bk in test_id_obj.partial_elim_order],
                          [sorted(bk) for bk in test_id_obj2.partial_elim_order])
        sc1 = [sorted(sc) for sc in test_id_obj.scopes]
        sc1 = sorted(sc1, key=lambda x: len(x))
        sc1 = [sc for sc in sorted(sc1)]
        sc2 = [sorted(sc) for sc in test_id_obj2.scopes]
        sc2 = sorted(sc2, key=lambda x: len(x))
        sc2 = [sc for sc in sorted(sc2)]
        self.assertEquals(sc1, sc2)

    def test_dag_drawings(self):
        test_id_obj = NxInfluenceDiagram()
        nvar = self.file_info["nvar"]
        var_types = self.file_info["var_types"]
        scopes = [[v.label for v in sc] for sc in self.file_info["scopes"]]
        scope_types = self.file_info["scope_types"]
        partial_elim_order = self.file_info["blocks"]
        test_id_obj.create_id_from_scopes(nvar, var_types, scopes, scope_types, partial_elim_order)
        test_id = test_id_obj.id

        # ancestral graph of [21, 22, 4, 5, 6]
        anc_id = ancestor_dag(test_id, [4, 5, 6, 21, 22])
        draw_influence_diagram(self.file_name.split("/")[-1] + "_ancestor_test", anc_id, [6])
        # check by looking at picture

        moral_id = moralize_dag(anc_id)
        draw_nx_graph(self.file_name.split("/")[-1] + "_moralize_test", moral_id, node_fill=[0,1,2])
        # check by looking at picture       moral graph -> cluster graph? dual graph?

    def test_dsep_in_dag(self):
        test_id_obj = NxInfluenceDiagram()
        nvar = self.file_info["nvar"]
        var_types = self.file_info["var_types"]
        scopes = [[v.label for v in sc] for sc in self.file_info["scopes"]]
        scope_types = self.file_info["scope_types"]
        partial_elim_order = self.file_info["blocks"]
        test_id_obj.create_id_from_scopes(nvar, var_types, scopes, scope_types, partial_elim_order)
        test_id = test_id_obj.id

        self.assertTrue(dsep_in_dag(test_id, [0,1,2,3], [23, 24], [7,8,9,10,13]))
        self.assertTrue(dsep_in_dag(test_id, [0, 1, 2, 3], [26, 24], [7, 8, 9, 10, 13]))
        self.assertFalse(dsep_in_dag(test_id, [0,1], [10, 11, 12, 13], [23, 25]))
        # looks OK

class TestRelaxation(TestCase):
    def setUp(self):
        self.file_name = os.path.join(TEST_PATH, "pomdp1-4_2_2_2_3")
        self.file_info = read_uai_id(self.file_name, False)

    def test_get_relaxed_influence_diagram(self):
        test_id_obj = NxInfluenceDiagram()
        nvar = self.file_info["nvar"]
        var_types = self.file_info["var_types"]
        scopes = [[v.label for v in sc] for sc in self.file_info["scopes"]]
        scope_types = self.file_info["scope_types"]
        partial_elim_order = self.file_info["blocks"]
        test_id_obj.create_id_from_scopes(nvar, var_types, scopes, scope_types, partial_elim_order)
        test_id_obj.draw_diagram(self.file_name.split("/")[-1])
        print(test_id_obj.decision_nodes)
        print(test_id_obj.partial_elim_order)


        relaxed_id = get_relaxed_influence_diagram(test_id_obj)
        relaxed_id.draw_diagram(self.file_name.split("/")[-1] + "_relaxed")
        print(relaxed_id.decision_nodes)
        print(relaxed_id.partial_elim_order)


if __name__ == "__main__":
    unittest.main()
