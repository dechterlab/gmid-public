PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.cte import *
import os
import datetime


def run_ctemixed(name="ID_from_BN_0_w28d6.mixed", ibound=20):
    print("Start JOB:{},{} at {}".format(name, ibound, datetime.datetime.now()))
    problem_path = os.path.join(TEST_PATH, name)
    print("read problem files:{}".format(problem_path))
    file_info = read_uai_mixed(problem_path)
    ordering_path = problem_path + '.vo'
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']
    fun_types = file_info['factor_types']
    num_var = file_info['nvar']
    num_fun = len(file_info['factors'])
    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_log = True
    is_valuation = False
    gm = GraphicalModel(factors, weights, is_log=is_log)
    pg = PrimalGraph(gm)

    # ordering, iw = iterative_greedy_variable_order(1, pg.nx_diagram, ps=16, pe=-1, ct=inf, pv=blocks)
    # write_vo_from_elim_order(ordering_path, ordering, iw)
    ordering, iw = read_vo(problem_path + '.vo')
    iw = get_induced_width_from_ordering(pg, ordering)

    mbtd, mini_buckets = mini_bucket_tree(gm, ordering, ibound, False, True)
    add_mg_attr_to_nodes(mbtd)
    add_mg_attr_to_edges(mbtd, gm.variables, is_log)
    add_const_factors(mbtd, gm.variables, is_valuation, is_log)
    verbose_level = 0
    cte_str = '_Mbe_i=' + str(ibound)
    log_file_name = os.path.join(LOG_PATH_MIXED, name + cte_str)
    tree_mp = CTE(verbose_level, mbtd, ordering, weights, is_log, is_valuation, log_file_name)
    tree_mp.schedule()
    tree_mp.init_propagate()
    best_mmap = tree_mp.propagate(propagation_type='ve', max_sum=True)
    tree_mp.print_log("START")
    tree_mp.print_log("name:{}".format(name))
    tree_mp.print_log("num vars:{}".format(file_info['nvar']))
    tree_mp.print_log("num factors:{}".format(len(file_info['factors'])))
    tree_mp.print_log("max domain:{}".format(max(file_info['domains'])))
    tree_mp.print_log("max scope:{}".format(max([len(s) for s in file_info['scopes']])))
    tree_mp.print_log('induced width:{}'.format(iw))
    tree_mp.print_log("connected components:{}".format(nx.number_connected_components(pg.G)))
    tree_mp.print_log("final MEU:{}".format(np.exp(best_mmap)))
    tree_mp.print_log("END")
    print("Finish JOB:{},{} at {}".format(name, ibound, datetime.datetime.now()))



if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(sys.argv)
        name=str(sys.argv[1])
        if not name.endswith(".mixed.uai"):
            assert False, name + " is wrong input file"
        ibound = int(sys.argv[2])
        name=name.replace(".uai", "")
        run_ctemixed(name, ibound)
    else:
        run_ctemixed()
