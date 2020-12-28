import sys
PRJ_PATH = "/home/junkyul/git/gmid"
sys.path.append(PRJ_PATH)
from gmid.constants import *
from gmid.fileio import *
import unittest
from unittest import TestCase
import os

def get_var_ids_from_scope(scope):
    return [el.label for el in scope]


class TestSimpleReadWrite(TestCase):
    def setUp(self):
        self.created_files = []
        self.test_name1 = "mdp0-2_2_2_3"
        self.test_path1 = os.path.join(TEST_PATH, self.test_name1)
        self.test_name2 = "test"
        self.test_path2 = os.path.join(TEST_PATH, self.test_name2)

    def test_read_mi(self):
        num_vars, var_types = read_mi(self.test_path1 + ".mi")
        self.assertEqual(num_vars, 9)
        self.assertEqual(var_types, ['C', 'C', 'D', 'C', 'C', 'D', 'C', 'C', 'D'])

    def test_read_id(self):
        num_vars, var_types, num_funcs, factor_types = read_id(self.test_path1 + ".id")
        self.assertEqual(num_vars, 9)
        self.assertEqual(var_types, ['C', 'C', 'D', 'C', 'C', 'D', 'C', 'C', 'D'])
        self.assertEqual(num_funcs, 12)
        self.assertEqual(factor_types, ['P', 'P', 'U', 'U', 'P', 'P', 'U', 'U', 'P', 'P', 'U', 'U'])

    def test_read_standard_uai(self):
        file_info = read_standard_uai(self.test_path1 + ".uai", sort_scope=True)
        factor_list = read_uai(self.test_path1 + ".uai")
        for f_ind, f in enumerate(file_info['factors']):
            self.assertEqual(f.t.shape, factor_list[f_ind].t.shape)
            self.assertTrue(np.all(f.t.ravel(order='C') == factor_list[f_ind].t.ravel(order='C')))
        self.assertEqual(file_info['domains'], [2, 2, 2, 2, 2, 2, 2, 2, 2])
        self.assertEqual(len(file_info['scopes']), 12)
        self.assertEqual(get_var_ids_from_scope(file_info['scopes'][0]), [0])
        self.assertEqual(get_var_ids_from_scope(file_info['scopes'][1]), [1])
        self.assertEqual(get_var_ids_from_scope(file_info['scopes'][2]), [0, 1, 2])

    def test_write_standard_uai(self):
        file1_info = read_standard_uai(self.test_path1 + ".uai", sort_scope=False)
        write_standard_uai(self.test_path1 + ".writetest.uai", file1_info, file_type="ID")
        self.created_files.append(self.test_path1 + ".writetest.uai")
        file2_info = read_standard_uai(self.test_path1 + ".writetest.uai", sort_scope=False)
        factors1 = file1_info['factors']
        factors2 = file2_info['factors']
        self.assertEqual(len(factors1), len(factors2))
        for f_ind, f1 in enumerate(factors1):
            self.assertEquals(f1.t.shape, factors2[f_ind].t.shape)
            self.assertTrue(np.all(f1.t.ravel(order='C') == factors2[f_ind].t.ravel(order='C')))

    def test_write_pvo_from_partial_elim_order(self):
        blocks, nblocks, nvars = read_pvo(self.test_path1 + ".pvo")
        write_pvo_from_partial_elim_order(self.test_path1 + ".writetest.pvo", blocks)
        self.created_files.append(self.test_path1 + ".writetest.pvo")
        blocks2, nblocks2, nvars2 = read_pvo(self.test_path1 + ".writetest.pvo")
        self.assertEquals(blocks, blocks2)
        self.assertEquals(nblocks, nblocks2)
        self.assertEquals(nvars, nvars2)

    def test_write_id_from_types(self):
        num_vars, var_types, num_funcs, factor_types = read_id(self.test_path1 + ".id")
        write_id_from_types(self.test_path1 + ".writetest.id", var_types, factor_types)
        self.created_files.append(self.test_path1 + ".writetest.id")
        num_vars2, var_types2, num_funcs2, factor_types2 = read_id(self.test_path1 + ".writetest.id")
        self.assertEquals(num_vars, num_vars2)
        self.assertEquals(var_types, var_types2)
        self.assertEquals(num_funcs, num_funcs)
        self.assertEquals(factor_types, factor_types2)

    def test_write_mi_from_types(self):
        num_vars, var_types = read_mi(self.test_path1 + ".mi")
        write_mi_from_types(self.test_path1 + ".writetest.mi", var_types)
        self.created_files.append(self.test_path1 + ".writetest.mi")
        num_vars2, var_types2 = read_mi(self.test_path1 + ".writetest.mi")
        self.assertEquals(num_vars, num_vars2)
        self.assertEquals(var_types, var_types2)

    def test_write_map_from_list(self):
        var_list = read_map(os.path.join(TEST_PATH, "mdp1-4_2_2_5.map"))
        write_map_from_list(os.path.join(TEST_PATH, "mdp1-4_2_2_5.writetest.map"), var_list)
        self.created_files.append(os.path.join(TEST_PATH, "mdp1-4_2_2_5.writetest.map"))
        var_list2 = read_map(os.path.join(TEST_PATH, "mdp1-4_2_2_5.writetest.map"))
        self.assertEquals(var_list, var_list2)

    def test_read_write_mmap(self):
        mmap_file_info = read_uai_mmap(os.path.join(TEST_PATH, "test"), sort_scope=False)
        uai_info = read_standard_uai(os.path.join(TEST_PATH, "test"), sort_scope=False)
        map_vars = read_map(os.path.join(TEST_PATH, "test.map"))
        self.assertEquals(mmap_file_info['blocks'][-1], map_vars)
        for f_ind, f in enumerate(uai_info['factors']):
            self.assertTrue(np.all(f.t.ravel(order='C')==mmap_file_info['factors'][f_ind].t.ravel(order='C')))
        write_uai_mmap(os.path.join(TEST_PATH, "test.writetest.uai"), mmap_file_info)
        self.created_files.append(os.path.join(TEST_PATH, "test.writetest.uai"))
        self.created_files.append(os.path.join(TEST_PATH, "test.writetest.map"))
        mmap_file_info2 = read_uai_mmap(os.path.join(TEST_PATH, "test.writetest"), sort_scope=False)
        for f_ind, f in enumerate(mmap_file_info['factors']):
            self.assertTrue(np.all(f.t.ravel(order='C')==mmap_file_info2['factors'][f_ind].t.ravel(order='C')))

    def test_read_uai_mixed(self):
        mixed_info = read_uai_mixed(self.test_path2, sort_scope=False)
        self.assertEquals(mixed_info['nvar'], 6)
        self.assertEquals(len(mixed_info['factors']), 12)
        self.assertEquals(sorted(mixed_info['blocks'][0]), [1, 3, 5])
        self.assertEquals(sorted(mixed_info['blocks'][1]), [0, 2, 4])
        self.assertTrue(np.all(mixed_info['factors'][0].t.ravel(order='C') == np.array([0.1, 0.2, 0.3, 0.4])))
        self.assertTrue(np.all(mixed_info['factors'][1].t.ravel(order='C') == np.array([0.7, 0.4, 0.5, 0.6])))
        self.assertTrue(np.all(mixed_info['factors'][2].t.ravel(order='C') == np.array([0.3, 0.6, 0.4, 0.2])))
        self.assertEquals(mixed_info['var_types'], ['D', 'C', 'D', 'C', 'D', 'C'])
        self.assertEquals(get_var_ids_from_scope(mixed_info['scopes'][0]), [0, 1])
        self.assertEquals(get_var_ids_from_scope(mixed_info['scopes'][1]), [0, 2])
        self.assertEquals(get_var_ids_from_scope(mixed_info['scopes'][2]), [0, 3])

    def test_write_uai_mixed(self):
        mixed_info_read = read_uai_mixed(self.test_path2, sort_scope=False)
        write_uai_mixed(self.test_path2 + ".writetest", mixed_info_read)
        self.created_files.append(self.test_path2 + ".writetest.uai")
        self.created_files.append(self.test_path2 + ".writetest.mi")
        self.created_files.append(self.test_path2 + ".writetest.pvo")
        mixed_info = read_uai_mixed(self.test_path2 + ".writetest", sort_scope=False)
        self.assertEquals(mixed_info['nvar'], 6)
        self.assertEquals(len(mixed_info['factors']), 12)
        self.assertEquals(sorted(mixed_info['blocks'][0]), [1, 3, 5])
        self.assertEquals(sorted(mixed_info['blocks'][1]), [0, 2, 4])
        self.assertTrue(np.all(mixed_info['factors'][0].t.ravel(order='C') == np.array([0.1, 0.2, 0.3, 0.4])))
        self.assertTrue(np.all(mixed_info['factors'][1].t.ravel(order='C') == np.array([0.7, 0.4, 0.5, 0.6])))
        self.assertTrue(np.all(mixed_info['factors'][2].t.ravel(order='C') == np.array([0.3, 0.6, 0.4, 0.2])))
        self.assertEquals(mixed_info['var_types'], ['D', 'C', 'D', 'C', 'D', 'C'])
        self.assertEquals(get_var_ids_from_scope(mixed_info['scopes'][0]), [0, 1])
        self.assertEquals(get_var_ids_from_scope(mixed_info['scopes'][1]), [0, 2])
        self.assertEquals(get_var_ids_from_scope(mixed_info['scopes'][2]), [0, 3])

    def tearDown(self):
        for f in self.created_files:
            print("test done remove:{}".format(f))
            os.remove(f)

if __name__ == '__main__':
    unittest.main()
